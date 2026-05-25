"""
Loader helpers for tpu-perf:memory-profile.

Reads an xplane.pb file, locates the /host:CPU plane, and turns its
MemoryAllocation / MemoryDeallocation XEvents into flat dataclasses.
Parent chain is derived inside each XLine by sweeping events ordered by
(offset_ps, -duration_ps) with a containment stack.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Optional

# Reuse profile-anatomy's xplane_pb2 — explicit dependency, not vendored.
_HERE = pathlib.Path(__file__).resolve().parent
_PROFILE_ANATOMY_PROTO = (
    _HERE.parent.parent / "profile-anatomy" / "scripts" / "_proto"
)
sys.path.insert(0, str(_PROFILE_ANATOMY_PROTO))
import xplane_pb2  # noqa: E402


# XLA dtype enum int → string. Mirrors xla.proto's PrimitiveType. Truncated to
# the dtypes that show up in TPU captures; unknown ints fall back to f"dt{n}".
_DTYPE_NAMES = {
    0: "INVALID", 1: "PRED",
    2: "s8", 3: "s16", 4: "s32", 5: "s64",
    6: "u8", 7: "u16", 8: "u32", 9: "u64",
    10: "f16", 11: "f32", 12: "f64",
    16: "bf16",
    19: "f8e5m2", 20: "f8e4m3fn", 23: "f8e4m3", 24: "f8e5m2fnuz",
    26: "f8e4m3fnuz", 28: "f8e3m4",
}


def _dtype_str(n: int) -> str:
    return _DTYPE_NAMES.get(n, f"dt{n}")


@dataclasses.dataclass(slots=True)
class AllocEvent:
    ts_ns: int
    addr: int
    pool_id: int
    requested_bytes: int
    allocation_bytes: int
    bytes_allocated: int          # allocator-reported pool occupancy after this event
    peak_bytes_in_use: int        # allocator-reported running peak after this event
    fragmentation: float          # allocator-reported pool fragmentation
    shape: str
    tf_op: str
    data_type: str
    parent_chain: list[str]
    line_name: str


@dataclasses.dataclass(slots=True)
class DeallocEvent:
    ts_ns: int
    addr: int
    bytes_allocated: int
    peak_bytes_in_use: int
    fragmentation: float
    line_name: str


@dataclasses.dataclass(slots=True)
class HostAllocatorEvents:
    allocs: list[AllocEvent]
    deallocs: list[DeallocEvent]
    pool_capacity: dict[int, int]
    host_plane_present: bool
    n_planes: int


def load_xspace(profile_dir: str | pathlib.Path) -> Optional[xplane_pb2.XSpace]:
    """Load the first *.xplane.pb in profile_dir. Returns None if none."""
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        return None
    xs = xplane_pb2.XSpace()
    xs.ParseFromString(pbs[0].read_bytes())
    return xs


def load_host_allocator_events(
    xs: xplane_pb2.XSpace,
) -> tuple[Optional[HostAllocatorEvents], Optional[str]]:
    """Extract MemoryAllocation/Deallocation events from /host:CPU plane.

    Returns (events, None) on success, (None, reason_code) when absent.
    Reason codes: 'host_plane_absent', 'no_memory_events'.
    """
    host = next((p for p in xs.planes if p.name == "/host:CPU"), None)
    if host is None:
        return None, "host_plane_absent"

    sm = {meta.id: meta.name for _, meta in host.stat_metadata.items()}
    em = {meta.id: meta.name for _, meta in host.event_metadata.items()}

    name_to_stat_id = {v: k for k, v in sm.items()}
    alloc_event_ids = {eid for eid, name in em.items() if name == "MemoryAllocation"}
    dealloc_event_ids = {eid for eid, name in em.items() if name == "MemoryDeallocation"}
    if not alloc_event_ids and not dealloc_event_ids:
        return None, "no_memory_events"

    def stat_value(ev, stat_name: str):
        sid = name_to_stat_id.get(stat_name)
        if sid is None:
            return None
        for st in ev.stats:
            if st.metadata_id != sid:
                continue
            v = st.WhichOneof("value")
            return getattr(st, v) if v else None
        return None

    allocs: list[AllocEvent] = []
    deallocs: list[DeallocEvent] = []
    pool_capacity: dict[int, int] = {}

    for line in host.lines:
        # Build parent chain via containment sweep.
        ordered = sorted(
            range(len(line.events)),
            key=lambda i: (line.events[i].offset_ps, -line.events[i].duration_ps),
        )
        # stack: list of (end_offset_ps, name) currently containing the cursor.
        stack: list[tuple[int, str]] = []
        # For each event, compute its parent chain (outermost-first) of all
        # events whose [start, start+dur] strictly contains its start.
        parent_chains: dict[int, list[str]] = {}
        for idx in ordered:
            ev = line.events[idx]
            start = ev.offset_ps
            # Pop entries whose end <= start.
            stack = [(end, n) for (end, n) in stack if end > start]
            parent_chains[idx] = [n for (_end, n) in stack]
            # Push self onto stack (only if it has positive duration; zero-dur
            # events do not contain anything).
            if ev.duration_ps > 0:
                stack.append((start + ev.duration_ps, em.get(ev.metadata_id, "")))

        for idx, ev in enumerate(line.events):
            ev_name = em.get(ev.metadata_id, "")
            if ev.metadata_id in alloc_event_ids:
                pool_id = int(stat_value(ev, "id") or 0)
                br = int(stat_value(ev, "bytes_reserved") or 0)
                if br > pool_capacity.get(pool_id, 0):
                    pool_capacity[pool_id] = br
                allocs.append(AllocEvent(
                    ts_ns=line.timestamp_ns + ev.offset_ps // 1000,
                    addr=int(stat_value(ev, "addr") or 0),
                    pool_id=pool_id,
                    requested_bytes=int(stat_value(ev, "requested_bytes") or 0),
                    allocation_bytes=int(stat_value(ev, "allocation_bytes") or 0),
                    bytes_allocated=int(stat_value(ev, "bytes_allocated") or 0),
                    peak_bytes_in_use=int(stat_value(ev, "peak_bytes_in_use") or 0),
                    fragmentation=float(stat_value(ev, "fragmentation") or 0.0),
                    shape=str(stat_value(ev, "shape") or ""),
                    tf_op=str(stat_value(ev, "tf_op") or ""),
                    data_type=_dtype_str(int(stat_value(ev, "data_type") or 0)),
                    parent_chain=parent_chains.get(idx, []),
                    line_name=line.name,
                ))
            elif ev.metadata_id in dealloc_event_ids:
                br = int(stat_value(ev, "bytes_reserved") or 0)
                # Capacity may surface only on dealloc events on some lines.
                # Track largest seen.
                if br > pool_capacity.get(0, 0):
                    pool_capacity[0] = br
                deallocs.append(DeallocEvent(
                    ts_ns=line.timestamp_ns + ev.offset_ps // 1000,
                    addr=int(stat_value(ev, "addr") or 0),
                    bytes_allocated=int(stat_value(ev, "bytes_allocated") or 0),
                    peak_bytes_in_use=int(stat_value(ev, "peak_bytes_in_use") or 0),
                    fragmentation=float(stat_value(ev, "fragmentation") or 0.0),
                    line_name=line.name,
                ))

    if not allocs and not deallocs:
        return None, "no_memory_events"

    allocs.sort(key=lambda e: e.ts_ns)
    deallocs.sort(key=lambda e: e.ts_ns)
    return HostAllocatorEvents(
        allocs=allocs, deallocs=deallocs,
        pool_capacity=pool_capacity, host_plane_present=True,
        n_planes=len(xs.planes),
    ), None


class StepPolicyError(ValueError):
    """Raised when the requested step policy cannot be satisfied."""


@dataclasses.dataclass(slots=True)
class StepWindow:
    id: Optional[str]                 # event metadata.name (e.g. "step_3"); None for all-trace
    range_ns: tuple[int, int]
    source: str                       # "steps_line" | "execute_event" | "all_trace"
    policy_used: str                  # "explicit" | "peak" | "last" | "first" | "all_trace"


def _steps_line_intervals(xs: xplane_pb2.XSpace) -> list[tuple[str, int, int]]:
    """Return [(name, start_ns, end_ns)] for /device:TPU:0 'Steps' events."""
    for plane in xs.planes:
        if plane.name != "/device:TPU:0":
            continue
        em = {meta.id: meta.name for _, meta in plane.event_metadata.items()}
        for line in plane.lines:
            if line.name != "Steps":
                continue
            out: list[tuple[str, int, int]] = []
            for ev in line.events:
                start_ns = line.timestamp_ns + ev.offset_ps // 1000
                end_ns = start_ns + ev.duration_ps // 1000
                out.append((em.get(ev.metadata_id, ""), start_ns, end_ns))
            return out
    return []


def _execute_event_intervals(xs: xplane_pb2.XSpace) -> list[tuple[str, int, int]]:
    """Return [(name, start_ns, end_ns)] for outer 'Execute (jit_*)' events on /host:CPU."""
    for plane in xs.planes:
        if plane.name != "/host:CPU":
            continue
        em = {meta.id: meta.name for _, meta in plane.event_metadata.items()}
        out: list[tuple[str, int, int]] = []
        for line in plane.lines:
            for ev in line.events:
                name = em.get(ev.metadata_id, "")
                if "Execute (jit_" not in name:
                    continue
                if ev.duration_ps <= 0:
                    continue
                start_ns = line.timestamp_ns + ev.offset_ps // 1000
                end_ns = start_ns + ev.duration_ps // 1000
                out.append((name, start_ns, end_ns))
        out.sort(key=lambda x: x[1])
        return out
    return []


def pick_step_window(
    xs: xplane_pb2.XSpace, *, all_trace: bool, policy: str,
    explicit: Optional[int], peak_ts_ns_hint: Optional[int],
) -> Optional[StepWindow]:
    if all_trace:
        return StepWindow(id=None, range_ns=(0, (1 << 63) - 1),
                          source="all_trace", policy_used="all_trace")

    steps = _steps_line_intervals(xs)
    if explicit is not None:
        if not steps:
            raise StepPolicyError("explicit --step requires a 'Steps' line on /device:TPU:0")
        if explicit < 0 or explicit >= len(steps):
            raise StepPolicyError(
                f"--step {explicit} out of range; Steps line has {len(steps)} events"
            )
        name, start, end = steps[explicit]
        return StepWindow(id=name, range_ns=(start, end),
                          source="steps_line", policy_used="explicit")

    if steps:
        if policy == "first":
            name, start, end = steps[0]
        elif policy == "last":
            name, start, end = steps[-1]
        elif policy == "peak":
            if peak_ts_ns_hint is None:
                name, start, end = steps[0]
            else:
                hit = next(
                    ((n, s, e) for (n, s, e) in steps if s <= peak_ts_ns_hint <= e),
                    None,
                )
                if hit is None:
                    # Pick the closest step.
                    hit = min(steps, key=lambda t: min(
                        abs(t[1] - peak_ts_ns_hint), abs(t[2] - peak_ts_ns_hint)
                    ))
                name, start, end = hit
        else:
            raise StepPolicyError(f"unknown policy: {policy}")
        return StepWindow(id=name, range_ns=(start, end),
                          source="steps_line", policy_used=policy)

    # Fallback: outer Execute (jit_*) events on /host:CPU.
    execs = _execute_event_intervals(xs)
    if not execs:
        return None
    if policy == "first":
        name, start, end = execs[0]
    elif policy == "last":
        name, start, end = execs[-1]
    elif policy == "peak":
        if peak_ts_ns_hint is None:
            name, start, end = execs[0]
        else:
            hit = next(
                ((n, s, e) for (n, s, e) in execs if s <= peak_ts_ns_hint <= e),
                None,
            )
            if hit is None:
                hit = min(execs, key=lambda t: min(
                    abs(t[1] - peak_ts_ns_hint), abs(t[2] - peak_ts_ns_hint)
                ))
            name, start, end = hit
    else:
        raise StepPolicyError(f"unknown policy: {policy}")
    return StepWindow(id=name, range_ns=(start, end),
                      source="execute_event", policy_used=policy)


@dataclasses.dataclass(slots=True)
class TimelineSample:
    ts_ns: int
    bytes_allocated: int
    live_count: int
    fragmentation: float


@dataclasses.dataclass(slots=True)
class AliveBuffer:
    addr: int
    pool_id: int
    size_bytes: int
    alloc_bytes: int
    shape: str
    tf_op: str
    data_type: str
    alloc_ts_ns: int
    age_ns_at_peak: int
    crossed_step_boundaries: int
    parent_chain: list[str]
    lifetime_class: str
    deallocated: bool


@dataclasses.dataclass(slots=True)
class FirstPassResult:
    global_peak_ts_ns: int
    global_peak_bytes: int
    timeline_samples: list[TimelineSample]
    alloc_accounting_drift_pct: float
    unmatched_dealloc_count: int
    unmatched_alloc_count: int
    trace_end_live_bytes: int
    pool_max_peak_in_use: dict[int, int]
    n_alloc: int
    n_dealloc: int


@dataclasses.dataclass(slots=True)
class PeakSnapshot:
    peak_ts_ns: int
    bytes_total: int
    bytes_by_pool: dict[int, int]
    fragmentation_at_peak: float
    is_global_peak: bool
    alive: list[AliveBuffer]
    alive_total_bytes: int


def _merged_event_stream(events: HostAllocatorEvents):
    """Yield (ts_ns, kind, payload) ordered by ts_ns, allocs before deallocs at ties."""
    a_iter = iter(events.allocs)
    d_iter = iter(events.deallocs)
    a = next(a_iter, None)
    d = next(d_iter, None)
    while a is not None or d is not None:
        if d is None or (a is not None and a.ts_ns <= d.ts_ns):
            yield a.ts_ns, "A", a
            a = next(a_iter, None)
        else:
            yield d.ts_ns, "D", d
            d = next(d_iter, None)


def sweep_first_pass(events: HostAllocatorEvents, *, time_samples_n: int) -> FirstPassResult:
    live: dict[tuple[int, int], AllocEvent] = {}
    bytes_now_by_pool: dict[int, int] = {}
    pool_max_peak: dict[int, int] = dict(events.pool_capacity)
    pool_max_peak.update({pid: 0 for pid in pool_max_peak})  # init counters
    pool_max_peak_in_use: dict[int, int] = {}

    global_peak_bytes = 0
    global_peak_ts_ns = 0

    last_fragmentation = 0.0
    drift_max = 0.0
    drift_seen = False

    unmatched_dealloc_count = 0

    # Linear scan of (ts_ns, bytes_allocated_total, fragmentation, live_count).
    samples: list[tuple[int, int, float, int]] = []

    for ts, kind, payload in _merged_event_stream(events):
        if kind == "A":
            a = payload
            key = (a.pool_id, a.addr)
            live[key] = a
            bytes_now_by_pool[a.pool_id] = bytes_now_by_pool.get(a.pool_id, 0) + a.requested_bytes
            last_fragmentation = a.fragmentation
            if a.peak_bytes_in_use > pool_max_peak_in_use.get(a.pool_id, 0):
                pool_max_peak_in_use[a.pool_id] = a.peak_bytes_in_use
            # Drift between our running sum and allocator's report (single pool only).
            if len(bytes_now_by_pool) == 1:
                ours = bytes_now_by_pool[a.pool_id]
                allocator = a.bytes_allocated
                if allocator > 0:
                    drift_seen = True
                    rel = abs(ours - allocator) / allocator
                    if rel > drift_max:
                        drift_max = rel
        else:  # "D"
            d = payload
            # Dealloc events do not carry pool_id; match by addr in any pool.
            match_key = next((k for k in live if k[1] == d.addr), None)
            if match_key is None:
                unmatched_dealloc_count += 1
                last_fragmentation = d.fragmentation
            else:
                a = live.pop(match_key)
                bytes_now_by_pool[a.pool_id] = bytes_now_by_pool[a.pool_id] - a.requested_bytes
                last_fragmentation = d.fragmentation
                if d.peak_bytes_in_use > pool_max_peak_in_use.get(a.pool_id, 0):
                    pool_max_peak_in_use[a.pool_id] = d.peak_bytes_in_use

        total = sum(bytes_now_by_pool.values())
        if total > global_peak_bytes:
            global_peak_bytes = total
            global_peak_ts_ns = ts
        samples.append((ts, total, last_fragmentation, len(live)))

    # Down-sample to time_samples_n equally-spaced wallclock points.
    if not samples:
        timeline_samples: list[TimelineSample] = []
    else:
        t0 = samples[0][0]
        t1 = samples[-1][0]
        if t1 == t0 or time_samples_n <= 1:
            timeline_samples = [TimelineSample(ts_ns=samples[-1][0],
                                               bytes_allocated=samples[-1][1],
                                               fragmentation=samples[-1][2],
                                               live_count=samples[-1][3])]
        else:
            step = (t1 - t0) / (time_samples_n - 1)
            timeline_samples = []
            i = 0
            for k in range(time_samples_n):
                target = t0 + step * k
                while i + 1 < len(samples) and samples[i + 1][0] <= target:
                    i += 1
                ts_k, b_k, f_k, l_k = samples[i]
                timeline_samples.append(TimelineSample(
                    ts_ns=int(target), bytes_allocated=b_k,
                    fragmentation=f_k, live_count=l_k,
                ))

    return FirstPassResult(
        global_peak_ts_ns=global_peak_ts_ns,
        global_peak_bytes=global_peak_bytes,
        timeline_samples=timeline_samples,
        alloc_accounting_drift_pct=(drift_max * 100.0) if drift_seen else 0.0,
        unmatched_dealloc_count=unmatched_dealloc_count,
        unmatched_alloc_count=len(live),
        trace_end_live_bytes=sum(bytes_now_by_pool.values()),
        pool_max_peak_in_use=pool_max_peak_in_use,
        n_alloc=len(events.allocs),
        n_dealloc=len(events.deallocs),
    )


def _count_step_boundaries_crossed(alloc_ts_ns: int, end_ts_ns: int,
                                   boundaries_ns: list[tuple[int, int]]) -> int:
    if not boundaries_ns:
        return 0
    n = 0
    for _i, (s, _e) in enumerate(boundaries_ns):
        if alloc_ts_ns <= s <= end_ts_ns:
            n += 1
    return n


def snapshot_at_peak(events: HostAllocatorEvents, *, peak_ts_ns: int,
                     step_range_ns: tuple[int, int],
                     step_boundaries_ns: list[tuple[int, int]],
                     persistent_threshold_steps: int) -> PeakSnapshot:
    # Re-run the sweep but stop computing at peak_ts_ns to capture the live set.
    live: dict[tuple[int, int], AllocEvent] = {}
    bytes_now_by_pool: dict[int, int] = {}
    last_fragmentation = 0.0

    # Find addr → dealloc_ts_ns map for lifetime classification.
    dealloc_ts_by_addr: dict[int, int] = {}
    for d in events.deallocs:
        dealloc_ts_by_addr.setdefault(d.addr, d.ts_ns)

    for ts, kind, payload in _merged_event_stream(events):
        if ts > peak_ts_ns:
            break
        if kind == "A":
            a = payload
            live[(a.pool_id, a.addr)] = a
            bytes_now_by_pool[a.pool_id] = bytes_now_by_pool.get(a.pool_id, 0) + a.requested_bytes
            last_fragmentation = a.fragmentation
        else:
            d = payload
            match_key = next((k for k in live if k[1] == d.addr), None)
            if match_key is not None:
                a = live.pop(match_key)
                bytes_now_by_pool[a.pool_id] = bytes_now_by_pool[a.pool_id] - a.requested_bytes
                last_fragmentation = d.fragmentation

    bytes_total = sum(bytes_now_by_pool.values())

    # Trace end ts_ns is the max event ts in either stream — used to compute
    # crossed boundaries when an alloc was never deallocated. We also take the
    # max step-boundary end into account: a never-freed buffer is alive through
    # the rest of the trace, which the Steps line bounds even if no allocator
    # event landed past the last step.
    last_event_ts = max(
        (events.allocs[-1].ts_ns if events.allocs else 0),
        (events.deallocs[-1].ts_ns if events.deallocs else 0),
        max((e for _s, e in step_boundaries_ns), default=0),
    )

    alive_buffers: list[AliveBuffer] = []
    for (_pool, addr), a in live.items():
        dealloc_ts = dealloc_ts_by_addr.get(addr)
        end_ts = dealloc_ts if dealloc_ts is not None else last_event_ts
        crossed = _count_step_boundaries_crossed(a.ts_ns, end_ts, step_boundaries_ns)
        deallocated = dealloc_ts is not None and dealloc_ts >= peak_ts_ns
        # Lifetime classification.
        same_step = False
        if dealloc_ts is not None:
            for s, e in step_boundaries_ns:
                if s <= a.ts_ns <= e and s <= dealloc_ts <= e:
                    same_step = True
                    break
        if not deallocated and crossed >= persistent_threshold_steps and dealloc_ts is None:
            cls = "persistent"
        elif same_step and dealloc_ts is not None:
            cls = "transient"
        else:
            cls = "unknown"
        alive_buffers.append(AliveBuffer(
            addr=a.addr, pool_id=a.pool_id,
            size_bytes=a.requested_bytes, alloc_bytes=a.allocation_bytes,
            shape=a.shape or "<no shape>",
            tf_op=a.tf_op or "<no tf_op>",
            data_type=a.data_type,
            alloc_ts_ns=a.ts_ns,
            age_ns_at_peak=peak_ts_ns - a.ts_ns,
            crossed_step_boundaries=crossed,
            parent_chain=list(a.parent_chain),
            lifetime_class=cls,
            deallocated=deallocated,
        ))

    alive_buffers.sort(key=lambda b: b.size_bytes, reverse=True)
    return PeakSnapshot(
        peak_ts_ns=peak_ts_ns,
        bytes_total=bytes_total,
        bytes_by_pool=dict(bytes_now_by_pool),
        fragmentation_at_peak=last_fragmentation,
        is_global_peak=False,  # caller sets this against FirstPassResult.global_peak_ts_ns
        alive=alive_buffers,
        alive_total_bytes=sum(b.size_bytes for b in alive_buffers),
    )


def pick_parent_jit(parent_chain: list[str]) -> str:
    if not parent_chain:
        return "<no parent>"
    for n in parent_chain:
        if "jit_" in n:
            return n
    return parent_chain[0]


_LIFETIME_KEYS = ("persistent", "transient", "unknown")


def _empty_mix() -> dict[str, int]:
    return {k: 0 for k in _LIFETIME_KEYS}


def _group_with_mix(buffers: list[AliveBuffer], key_fn,
                    *, top_k: int, total_bytes: int,
                    truncate: bool) -> list[dict]:
    groups: dict[str, dict] = {}
    for b in buffers:
        k = key_fn(b)
        g = groups.setdefault(k, {
            "key": k, "n_buffers": 0, "total_bytes": 0,
            "lifetime_mix": _empty_mix(),
        })
        g["n_buffers"] += 1
        g["total_bytes"] += b.size_bytes
        g["lifetime_mix"][b.lifetime_class] += b.size_bytes
    rows = sorted(groups.values(), key=lambda r: r["total_bytes"], reverse=True)
    for r in rows:
        r["pct_of_peak"] = (r["total_bytes"] / total_bytes * 100.0) if total_bytes else 0.0
    if truncate and len(rows) > top_k:
        head = rows[:top_k]
        tail_rows = rows[top_k:]
        tail = {
            "key": "<tail>",
            "n_buffers": sum(r["n_buffers"] for r in tail_rows),
            "total_bytes": sum(r["total_bytes"] for r in tail_rows),
            "lifetime_mix": {
                k: sum(r["lifetime_mix"][k] for r in tail_rows)
                for k in _LIFETIME_KEYS
            },
        }
        tail["pct_of_peak"] = (
            tail["total_bytes"] / total_bytes * 100.0 if total_bytes else 0.0
        )
        return head + [tail]
    return rows


def _group_simple(buffers: list[AliveBuffer], key_fn,
                  *, total_bytes: int) -> list[dict]:
    """Rollup without lifetime_mix or Top-K (used for by_lifetime_class, by_dtype)."""
    groups: dict[str, dict] = {}
    for b in buffers:
        k = key_fn(b)
        g = groups.setdefault(k, {"key": k, "n_buffers": 0, "total_bytes": 0})
        g["n_buffers"] += 1
        g["total_bytes"] += b.size_bytes
    rows = sorted(groups.values(), key=lambda r: r["total_bytes"], reverse=True)
    for r in rows:
        r["pct_of_peak"] = (r["total_bytes"] / total_bytes * 100.0) if total_bytes else 0.0
    return rows


def build_rollups(alive: list[AliveBuffer], *, top_k: int,
                  total_bytes: int) -> dict[str, list[dict]]:
    return {
        "by_lifetime_class": _group_simple(alive, lambda b: b.lifetime_class,
                                           total_bytes=total_bytes),
        "by_shape": _group_with_mix(alive, lambda b: b.shape,
                                    top_k=top_k, total_bytes=total_bytes,
                                    truncate=True),
        "by_tf_op": _group_with_mix(alive, lambda b: b.tf_op or "<no tf_op>",
                                    top_k=top_k, total_bytes=total_bytes,
                                    truncate=True),
        "by_parent_jit": _group_with_mix(alive, lambda b: pick_parent_jit(b.parent_chain),
                                         top_k=top_k, total_bytes=total_bytes,
                                         truncate=True),
        "by_dtype": _group_simple(alive, lambda b: b.data_type,
                                  total_bytes=total_bytes),
    }
