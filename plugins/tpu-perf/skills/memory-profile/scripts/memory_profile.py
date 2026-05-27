"""
Entry point for tpu-perf:memory-profile.

Primary data source: the largest `*.hlo_proto.pb` in `profile_dir`. Its
`BufferAssignmentProto` is what TensorBoard's Memory Viewer renders, and
it is the authoritative source for HBM peak: every buffer XLA reserves at
compile time (weights, optimizer state, activations, communication
scratch) is enumerated with size, lifetime, and HLO instruction
attribution. The runtime allocator events on /host:CPU are kept as a
secondary signal because they are routinely truncated by the trace
window — they miss every buffer allocated before capture started.

Output: one JSON object on stdout, `status: ok | absent`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys


SKILL_NAME = "memory-profile"
SCHEMA_VERSION = 2


_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _loader  # noqa: E402
import _hlo_loader  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_profile.py",
        description=(
            "Locate HBM peak occupancy. Primary source is the static buffer "
            "assignment in *.hlo_proto.pb (Memory Viewer's data); runtime "
            "allocator events on /host:CPU are reported as a secondary signal."
        ),
    )
    p.add_argument("profile_dir", help="Profile directory containing *.xplane.pb and *.hlo_proto.pb")

    step_group = p.add_mutually_exclusive_group()
    step_group.add_argument("--step", type=int, default=None,
                            help="Explicit step index on /device:TPU:0 'Steps' line (0-based). Affects runtime block only.")
    step_group.add_argument("--all-trace", action="store_true",
                            help="Disable step scoping for the runtime block; analyse the full trace window.")

    p.add_argument("--step-policy", choices=("peak", "last", "first"),
                   default="peak", help="Default step picker (default: peak). Runtime block only.")
    p.add_argument("--top", type=int, default=30,
                   help="Top-K applied to alive_at_peak.buffers and rollup tables.")
    p.add_argument("--persistent-threshold-steps", type=int, default=2,
                   help="Min crossed step boundaries for lifetime_class=persistent. Runtime block only.")
    p.add_argument("--include-host-pools", action="store_true",
                   help="Include allocator pools other than HBM (id != 0). Runtime block only.")
    p.add_argument("--time-samples", type=int, default=200,
                   help="Number of equally-spaced timeline samples in the runtime block.")
    p.add_argument("--no-runtime", action="store_true",
                   help="Skip the runtime allocator block entirely; emit only the HLO block.")
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


def _hlo_alive_to_json(b: "_hlo_loader.HloAliveBuffer") -> dict:
    return {
        "logical_buffer_id": b.logical_buffer_id,
        "size_bytes": b.size_bytes,
        "allocation_index": b.allocation_index,
        "offset_in_allocation": b.offset_in_allocation,
        "instruction_id": b.instruction_id,
        "instruction_name": b.instruction_name,
        "opcode": b.opcode,
        "op_name": b.op_name,
        "shape_index": b.shape_index,
    }


def _step_boundaries_for_classification(xs) -> list[tuple[int, int]]:
    boundaries = _loader._steps_line_intervals(xs)
    if boundaries:
        return [(s, e) for (_n, s, e) in boundaries]
    execs = _loader._execute_event_intervals(xs)
    return [(s, e) for (_n, s, e) in execs]


def _build_hlo_block(profile_dir: str, top_k: int) -> tuple[dict | None, str | None]:
    """Build the HLO buffer-assignment block.

    Returns (block, reason). If no hlo_proto.pb is present, returns (None, reason).
    """
    a = _hlo_loader.analyse(profile_dir)
    if a is None:
        return None, "no_hlo_proto_pb"

    head = a.peak_alive_buffers[:top_k]
    tail = a.peak_alive_buffers[top_k:]
    peak_rollups = _hlo_loader.rollups_for_alive(
        a.peak_alive_buffers, a.peak_alive_bytes, top_k=top_k,
    )
    always_rollups = _hlo_loader.rollups_for_alive(
        a.always_alive_buffers, a.always_alive_bytes, top_k=top_k,
    )

    return {
        "hlo_proto_path": a.hlo_proto_path,
        "module_name": a.module_name,
        "static_peak_bytes": a.static_peak_bytes,
        "decomposition": {
            "entry_params_bytes": a.entry_param_bytes,
            "constants_bytes": a.constant_bytes,
            "thread_local_bytes": a.thread_local_bytes,
            "temp_pool_bytes": a.temp_pool_bytes,
            "temp_pool_alloc_index": a.temp_pool_alloc_index,
        },
        "n_logical_buffers": a.n_logical_buffers,
        "n_buffer_allocations": a.n_buffer_allocations,
        "schedule_sweep": {
            "schedule_present": a.schedule_present,
            "entry_schedule_length": a.entry_schedule_length,
            "peak_schedule_pos": a.peak_schedule_pos,
            "peak_instruction": {
                "id": a.peak_instruction_id,
                "name": a.peak_instruction_name,
                "opcode": a.peak_instruction_opcode,
                "op_name": a.peak_instruction_op_name,
            },
            "peak_alive_bytes_entry_level": a.peak_alive_bytes,
            "n_subcomputation_lbs_skipped": a.n_subcomputation_lbs_skipped,
            "scope_note": (
                "Entry-level sweep walks instructions of the entry computation. "
                "Logical buffers defined inside while-bodies / fusion / scan-bodies "
                "are counted as part of their wrapping while/call output buffer; "
                "individual per-iteration buffers inside such regions are skipped "
                "(see n_subcomputation_lbs_skipped). The static_peak_bytes total "
                "still bounds true HBM use."
            ),
        },
        "alive_at_peak": {
            "n_buffers": len(a.peak_alive_buffers),
            "total_bytes": a.peak_alive_bytes,
            "buffers": [_hlo_alive_to_json(b) for b in head],
            "tail": {
                "n_buffers": len(tail),
                "total_bytes": sum(b.size_bytes for b in tail),
            },
            "rollups": peak_rollups,
        },
        "always_alive": {
            "total_bytes": a.always_alive_bytes,
            "pct_of_temp_pool": (
                100.0 * a.always_alive_bytes / a.temp_pool_bytes
                if a.temp_pool_bytes else 0.0
            ),
            "buffers": [_hlo_alive_to_json(b) for b in a.always_alive_buffers[:top_k]],
            "rollups": always_rollups,
            "definition": (
                "Bytes inside the temp pool owned by exactly one logical buffer "
                "in the address space. By construction these bytes are alive at "
                "every schedule position — they are the static-residency floor "
                "that no remat policy can eliminate."
            ),
        },
        "top_allocations": a.top_allocations,
    }, None


def _build_runtime_block(args) -> tuple[dict | None, dict, str | None]:
    """Build the runtime-allocator block. Returns (block, diagnostics, absent_reason)."""
    profile_dir = args.profile_dir
    xs = _loader.load_xspace(profile_dir)
    if xs is None:
        return None, {}, "no_xplane_pb"

    events, reason = _loader.load_host_allocator_events(xs)
    if events is None:
        return None, {"n_planes": len(xs.planes)}, reason or "no_memory_events"

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
        return None, {}, f"step_policy_error: {e}"

    if sw is None:
        return None, {"n_planes": len(xs.planes)}, "no_step_data"

    s0, s1 = sw.range_ns
    live: dict[tuple[int, int], _loader.AllocEvent] = {}
    bytes_by_pool: dict[int, int] = {}
    peak_bytes = 0
    peak_ts = s0
    for ts, kind, payload in _loader._merged_event_stream(events):
        if ts > s1:
            break
        if kind == "A":
            a = payload
            key = (a.pool_id, a.addr)
            prior = live.get(key)
            if prior is not None:
                bytes_by_pool[prior.pool_id] = (
                    bytes_by_pool.get(prior.pool_id, 0) - prior.requested_bytes
                )
            live[key] = a
            bytes_by_pool[a.pool_id] = bytes_by_pool.get(a.pool_id, 0) + a.requested_bytes
        else:
            d = payload
            mk = next((k for k in live if k[1] == d.addr), None)
            if mk is not None:
                a = live.pop(mk)
                bytes_by_pool[a.pool_id] -= a.requested_bytes
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
            " exceeds 1% threshold; runtime alive_at_peak likely missing pre-trace"
            " allocations — trust the HLO block instead"
        )
    if first.unmatched_dealloc_count > 0:
        diagnostics["warnings"].append(
            f"{first.unmatched_dealloc_count} MemoryDeallocation event(s) had no matching alloc"
        )
    if first.pretrace_dealloc_count > 0:
        diagnostics["warnings"].append(
            f"{first.pretrace_dealloc_count} MemoryDeallocation event(s) reference"
            " pre-trace allocations (trace truncation; HLO block is authoritative)"
        )
    if sw.source == "execute_event":
        diagnostics["warnings"].append(
            "/device:TPU:0 'Steps' line absent; using outer Execute (jit_*) event as step window"
        )

    block = {
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
        "n_planes": len(xs.planes),
        "host_plane_present": True,
    }
    return block, diagnostics, None


def _consistency_check(hlo_block: dict | None, runtime_block: dict | None) -> list[str]:
    """Compare HLO static peak to runtime peak. Emit warnings on disagreement."""
    out: list[str] = []
    if hlo_block is None or runtime_block is None:
        return out
    static_peak = hlo_block["static_peak_bytes"]
    runtime_peak = runtime_block["peak"]["bytes_total"]
    if static_peak <= 0 or runtime_peak <= 0:
        return out
    drift = abs(static_peak - runtime_peak) / max(static_peak, runtime_peak)
    if drift > 0.05:
        out.append(
            f"runtime_peak={runtime_peak/1024**2:.1f} MiB disagrees with "
            f"static_peak={static_peak/1024**2:.1f} MiB by {drift*100:.1f}%; "
            f"this is the typical signature of trace truncation (runtime trace "
            f"started after model init). Trust the HLO block."
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile_dir = args.profile_dir

    hlo_block, hlo_absent_reason = _build_hlo_block(profile_dir, args.top)

    runtime_block = None
    runtime_diag: dict = {}
    runtime_absent_reason: str | None = None
    if not args.no_runtime:
        runtime_block, runtime_diag, runtime_absent_reason = _build_runtime_block(args)

    if hlo_block is None and runtime_block is None:
        json.dump(_emit_absent(
            profile_dir,
            "no_data_sources",
            hlo_reason=hlo_absent_reason,
            runtime_reason=runtime_absent_reason,
        ), sys.stdout)
        sys.stdout.write("\n")
        return 0

    consistency_warnings = _consistency_check(hlo_block, runtime_block)

    output = {
        "status": "ok",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "inputs": {
            "profile_dir": profile_dir,
            "xplane_pb": str(next(iter(sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))), ""))
                or None,
            "hlo_proto_pb": hlo_block["hlo_proto_path"] if hlo_block else None,
        },
        "primary_source": "hlo_buffer_assignment" if hlo_block else "runtime_allocator",
        "hlo": hlo_block,
        "hlo_absent_reason": hlo_absent_reason,
        "runtime": runtime_block,
        "runtime_diagnostics": runtime_diag if runtime_block else None,
        "runtime_absent_reason": runtime_absent_reason,
        "consistency_warnings": consistency_warnings,
    }
    json.dump(output, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
