#!/usr/bin/env python3
"""VPR liveness analysis and register pressure tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult

_TOTAL_VPRS = 32
_PRESSURE_THRESHOLD = 0.75


@dataclass
class VPRLiveness:
    vpr_id: int
    defined_by: str
    last_used_by: str
    live_start_ns: float
    live_end_ns: float


@dataclass
class VPROccupancy:
    liveness: list[VPRLiveness]
    peak_concurrent: int
    peak_time_ns: float
    utilization_ratio: float
    pressure_warnings: list[str]


def analyze_vpr_liveness(
    ops: list[PipelineOp], sched: ScheduleResult
) -> VPROccupancy:
    entries = sched.entries_by_id

    vpr_defs: dict[int, list[tuple[str, float]]] = {}
    vpr_last_use: dict[int, list[tuple[str, float]]] = {}

    for op in ops:
        entry = entries[op.op_id]
        for v in op.output_vprs:
            vpr_defs.setdefault(v, []).append((op.op_id, entry.end_ns))
        for v in op.input_vprs:
            vpr_last_use.setdefault(v, []).append((op.op_id, entry.end_ns))

    liveness: list[VPRLiveness] = []
    for vpr_id in sorted(set(list(vpr_defs.keys()) + list(vpr_last_use.keys()))):
        defs = vpr_defs.get(vpr_id, [])
        uses = vpr_last_use.get(vpr_id, [])
        if not defs:
            continue
        def_op, def_time = min(defs, key=lambda x: x[1])
        if uses:
            use_op, use_time = max(uses, key=lambda x: x[1])
            live_end = max(def_time, use_time)
        else:
            use_op = def_op
            live_end = def_time
        liveness.append(VPRLiveness(
            vpr_id=vpr_id,
            defined_by=def_op,
            last_used_by=use_op,
            live_start_ns=def_time,
            live_end_ns=live_end,
        ))

    total_time = sched.total_latency_ns
    events: list[tuple[float, int]] = []
    for lv in liveness:
        events.append((lv.live_start_ns, +1))
        events.append((lv.live_end_ns, -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    peak = 0
    peak_time = 0.0
    current = 0
    for t, delta in events:
        current += delta
        if current > peak:
            peak = current
            peak_time = t

    total_live = sum(max(0, lv.live_end_ns - lv.live_start_ns) for lv in liveness)
    util = total_live / (total_time * _TOTAL_VPRS) if total_time > 0 else 0.0

    warnings: list[str] = []
    if peak > _TOTAL_VPRS * _PRESSURE_THRESHOLD:
        warnings.append(
            f"VPR pressure critical: {peak}/{_TOTAL_VPRS} VPRs live "
            f"simultaneously at t={peak_time:.0f}ns "
            f"({peak/_TOTAL_VPRS*100:.0f}% utilization)"
        )

    return VPROccupancy(
        liveness=liveness,
        peak_concurrent=peak,
        peak_time_ns=peak_time,
        utilization_ratio=util,
        pressure_warnings=warnings,
    )
