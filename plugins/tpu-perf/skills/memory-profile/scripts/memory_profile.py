"""
Entry point for tpu-perf:memory-profile.

Locates HBM peak occupancy within a chosen step (or across the full
trace) and emits one JSON object on stdout describing the peak moment
plus every buffer alive at that moment, rollups, timeline samples, and
diagnostics. Single-mode skill — no --mode flag. See spec
docs/superpowers/specs/2026-05-25-tpu-perf-memory-profile-design.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys


SKILL_NAME = "memory-profile"
SCHEMA_VERSION = 1


_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _loader  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_profile.py",
        description=(
            "Locate HBM peak occupancy within a step and list every buffer "
            "alive at that moment."
        ),
    )
    p.add_argument("profile_dir", help="Profile directory containing *.xplane.pb")

    step_group = p.add_mutually_exclusive_group()
    step_group.add_argument("--step", type=int, default=None,
                            help="Explicit step index on /device:TPU:0 'Steps' line (0-based).")
    step_group.add_argument("--all-trace", action="store_true",
                            help="Disable step scoping; analyse the full trace window.")

    p.add_argument("--step-policy", choices=("peak", "last", "first"),
                   default="peak", help="Default step picker (default: peak).")
    p.add_argument("--top", type=int, default=30,
                   help="Top-K applied to alive_at_peak.buffers and rollup tables.")
    p.add_argument("--persistent-threshold-steps", type=int, default=2,
                   help="Min crossed step boundaries for lifetime_class=persistent.")
    p.add_argument("--include-host-pools", action="store_true",
                   help="Include allocator pools other than HBM (id != 0).")
    p.add_argument("--time-samples", type=int, default=200,
                   help="Number of equally-spaced timeline samples.")
    return p


def _emit_absent(profile_dir: str, reason: str, **extra) -> dict:
    return {
        "status": "absent",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "reason": reason,
        "inputs": {"profile_dir": profile_dir, **extra},
    }


def _alive_to_json(b: _loader.AliveBuffer) -> dict:
    return {
        "addr": b.addr, "pool_id": b.pool_id,
        "size_bytes": b.size_bytes, "alloc_bytes": b.alloc_bytes,
        "shape": b.shape, "tf_op": b.tf_op, "data_type": b.data_type,
        "alloc_ts_ns": b.alloc_ts_ns,
        "age_ns_at_peak": b.age_ns_at_peak,
        "crossed_step_boundaries": b.crossed_step_boundaries,
        "parent_chain": b.parent_chain,
        "lifetime_class": b.lifetime_class,
        "deallocated": b.deallocated,
    }


def _step_boundaries_for_classification(xs) -> list[tuple[int, int]]:
    boundaries = _loader._steps_line_intervals(xs)
    if boundaries:
        return [(s, e) for (_n, s, e) in boundaries]
    execs = _loader._execute_event_intervals(xs)
    return [(s, e) for (_n, s, e) in execs]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile_dir = args.profile_dir

    xs = _loader.load_xspace(profile_dir)
    if xs is None:
        json.dump(_emit_absent(profile_dir, "no_xplane_pb"), sys.stdout)
        sys.stdout.write("\n")
        return 0

    events, reason = _loader.load_host_allocator_events(xs)
    if events is None:
        json.dump(_emit_absent(profile_dir, reason or "no_memory_events",
                               n_planes=len(xs.planes)), sys.stdout)
        sys.stdout.write("\n")
        return 0

    if not args.include_host_pools:
        kept_alloc = [a for a in events.allocs if a.pool_id == 0]
        if kept_alloc:
            events = dataclasses.replace(
                events, allocs=kept_alloc,
                pool_capacity={0: events.pool_capacity.get(0, 0)},
            )

    first = _loader.sweep_first_pass(events, time_samples_n=args.time_samples)

    try:
        sw = _loader.pick_step_window(
            xs, all_trace=args.all_trace, policy=args.step_policy,
            explicit=args.step,
            peak_ts_ns_hint=first.global_peak_ts_ns if first.global_peak_bytes else None,
        )
    except _loader.StepPolicyError as e:
        json.dump(_emit_absent(profile_dir, "step_policy_error",
                               error=str(e)), sys.stdout)
        sys.stdout.write("\n")
        return 0

    if sw is None:
        json.dump(_emit_absent(profile_dir, "no_step_data",
                               n_planes=len(xs.planes)), sys.stdout)
        sys.stdout.write("\n")
        return 0

    # Find step-scoped peak: max bytes_now within sw.range_ns.
    s0, s1 = sw.range_ns
    live: dict[tuple[int, int], _loader.AllocEvent] = {}
    bytes_by_pool: dict[int, int] = {}
    peak_bytes = 0
    peak_ts = s0
    last_fragmentation = 0.0
    for ts, kind, payload in _loader._merged_event_stream(events):
        if ts > s1:
            break
        if kind == "A":
            a = payload
            key = (a.pool_id, a.addr)
            # Address-reuse without an intervening MemoryDeallocation event:
            # implicitly evict the prior buffer so the running sum is correct.
            prior = live.get(key)
            if prior is not None:
                bytes_by_pool[prior.pool_id] = (
                    bytes_by_pool.get(prior.pool_id, 0) - prior.requested_bytes
                )
            live[key] = a
            bytes_by_pool[a.pool_id] = bytes_by_pool.get(a.pool_id, 0) + a.requested_bytes
            last_fragmentation = a.fragmentation
        else:
            d = payload
            mk = next((k for k in live if k[1] == d.addr), None)
            if mk is not None:
                a = live.pop(mk)
                bytes_by_pool[a.pool_id] -= a.requested_bytes
                last_fragmentation = d.fragmentation
        if ts < s0:
            continue
        total = sum(bytes_by_pool.values())
        if total > peak_bytes:
            peak_bytes = total
            peak_ts = ts
    if peak_bytes == 0:
        peak_ts = first.global_peak_ts_ns
        peak_bytes = first.global_peak_bytes

    boundaries = _step_boundaries_for_classification(xs)
    snap = _loader.snapshot_at_peak(
        events, peak_ts_ns=peak_ts,
        step_range_ns=(s0, s1),
        step_boundaries_ns=boundaries,
        persistent_threshold_steps=args.persistent_threshold_steps,
    )
    snap.is_global_peak = (snap.peak_ts_ns == first.global_peak_ts_ns
                           and snap.bytes_total >= first.global_peak_bytes)

    rollups = _loader.build_rollups(
        snap.alive, top_k=args.top, total_bytes=snap.alive_total_bytes,
    )

    head = snap.alive[: args.top]
    tail_rows = snap.alive[args.top:]
    alive_payload = {
        "n_buffers": len(snap.alive),
        "total_bytes": snap.alive_total_bytes,
        "buffers": [_alive_to_json(b) for b in head],
        "tail": {
            "n_buffers": len(tail_rows),
            "total_bytes": sum(b.size_bytes for b in tail_rows),
        },
    }

    timeline_payload = {
        "samples": [
            {"ts_ns": s.ts_ns, "bytes_allocated": s.bytes_allocated,
             "live_count": s.live_count, "fragmentation": s.fragmentation}
            for s in first.timeline_samples
        ],
        "events_of_interest": [
            {"kind": "global_peak", "ts_ns": first.global_peak_ts_ns,
             "bytes": first.global_peak_bytes},
            {"kind": "step_start", "ts_ns": s0,
             "step_id": sw.id if sw.source != "all_trace" else None},
            {"kind": "step_end", "ts_ns": s1,
             "step_id": sw.id if sw.source != "all_trace" else None},
            {"kind": "step_local_peak", "ts_ns": snap.peak_ts_ns,
             "step_id": sw.id, "bytes": snap.bytes_total},
        ],
        "axis_units": {"ts_ns": "nanoseconds since epoch",
                       "bytes": "bytes (base-2)"},
    }

    pool_id = 0 if 0 in events.pool_capacity else next(iter(events.pool_capacity), 0)
    diagnostics = {
        "alloc_accounting_drift_pct": first.alloc_accounting_drift_pct,
        "unmatched_dealloc_count": first.unmatched_dealloc_count,
        "pretrace_dealloc_count": first.pretrace_dealloc_count,
        "unmatched_alloc_count": first.unmatched_alloc_count,
        "trace_end_live_bytes": first.trace_end_live_bytes,
        "n_pools_seen": len(events.pool_capacity),
        "pools_summary": [
            {"pool_id": pid,
             "n_alloc": sum(1 for a in events.allocs if a.pool_id == pid),
             "n_dealloc": first.n_dealloc,
             "max_peak_bytes_in_use": first.pool_max_peak_in_use.get(pid, 0)}
            for pid in sorted(events.pool_capacity.keys())
        ],
        "step_line_present": sw.source != "execute_event",
        "shape_missing_count": sum(1 for a in events.allocs if not a.shape),
        "tf_op_missing_count": sum(1 for a in events.allocs if not a.tf_op),
        "warnings": [],
    }
    if first.alloc_accounting_drift_pct > 1.0:
        diagnostics["warnings"].append(
            f"alloc_accounting_drift_pct={first.alloc_accounting_drift_pct:.3f}%"
            " exceeds 1% threshold; results may include alignment/metadata padding"
        )
    if first.unmatched_dealloc_count > 0:
        diagnostics["warnings"].append(
            f"{first.unmatched_dealloc_count} MemoryDeallocation event(s) had no matching alloc"
        )
    if first.pretrace_dealloc_count > 0:
        diagnostics["warnings"].append(
            f"{first.pretrace_dealloc_count} MemoryDeallocation event(s) reference"
            " pre-trace allocations (trace truncation; not a producer bug)"
        )
    if sw.source == "execute_event":
        diagnostics["warnings"].append(
            "/device:TPU:0 'Steps' line absent; using outer Execute (jit_*) event as step window"
        )

    output = {
        "status": "ok",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "inputs": {
            "profile_dir": profile_dir,
            "xplane_pb": str(sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))[0]),
            "n_planes": len(xs.planes),
            "host_plane_present": True,
        },
        "step": {
            "id": sw.id, "policy": sw.policy_used,
            "range_ns": [s0, s1], "source": sw.source,
        },
        "pool": {"id": pool_id,
                 "bytes_reserved": events.pool_capacity.get(pool_id, 0)},
        "peak": {
            "ts_ns": snap.peak_ts_ns,
            "bytes_total": snap.bytes_total,
            "bytes_by_pool": {str(k): v for k, v in snap.bytes_by_pool.items()},
            "fragmentation_at_peak": snap.fragmentation_at_peak,
            "is_global_peak": snap.is_global_peak,
        },
        "alive_at_peak": alive_payload,
        "rollups": rollups,
        "timeline": timeline_payload,
        "diagnostics": diagnostics,
    }
    json.dump(output, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
