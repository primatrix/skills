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
