#!/usr/bin/env python3
"""VPR timeline plot — data layer for register activity visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult
from vpr_analyzer import analyze_vpr_liveness


@dataclass
class VPRInterval:
    """A time interval where a VPR is in a specific state."""
    vpr_id: int
    start_ns: float
    end_ns: float
    unit: str        # DMA | MXU | VPU
    access: str      # write | read | live
    op_id: str


def build_vpr_activity(
    ops: list[PipelineOp], sched: ScheduleResult
) -> dict[int, list[VPRInterval]]:
    """Build per-VPR activity intervals from schedule results.

    For each VPR, produces intervals tagged with (unit, access_type):
    - "write": the op's time window when it writes this VPR (output_vprs)
    - "read":  the op's time window when it reads this VPR (input_vprs)
    - "live":  gaps between write-end and last-read-end where VPR holds data
    """
    entries = sched.entries_by_id
    raw: dict[int, list[VPRInterval]] = {}

    for op in ops:
        entry = entries[op.op_id]
        for v in op.output_vprs:
            raw.setdefault(v, []).append(VPRInterval(
                vpr_id=v, start_ns=entry.start_ns, end_ns=entry.end_ns,
                unit=op.unit, access="write", op_id=op.op_id,
            ))
        for v in op.input_vprs:
            raw.setdefault(v, []).append(VPRInterval(
                vpr_id=v, start_ns=entry.start_ns, end_ns=entry.end_ns,
                unit=op.unit, access="read", op_id=op.op_id,
            ))

    # Fill "live" gaps: VPR holds data between active intervals
    occ = analyze_vpr_liveness(ops, sched)
    result: dict[int, list[VPRInterval]] = {}
    for vpr_id, intervals in raw.items():
        intervals.sort(key=lambda i: i.start_ns)
        lv = next((l for l in occ.liveness if l.vpr_id == vpr_id), None)
        if not lv:
            result[vpr_id] = intervals
            continue

        filled: list[VPRInterval] = []
        # Merge overlapping active intervals to find gaps
        active_spans = sorted((i.start_ns, i.end_ns) for i in intervals)
        merged: list[tuple[float, float]] = []
        for s, e in active_spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Unit for live gaps comes from the first writer
        def_unit = intervals[0].unit

        # Insert live intervals in gaps within liveness range
        prev_end = lv.live_start_ns
        for ms, me in merged:
            if ms > prev_end and ms > lv.live_start_ns:
                gap_start = max(prev_end, lv.live_start_ns)
                if gap_start < ms:
                    filled.append(VPRInterval(
                        vpr_id=vpr_id, start_ns=gap_start, end_ns=ms,
                        unit=def_unit, access="live", op_id="",
                    ))
            prev_end = me
        # Trailing live gap
        if prev_end < lv.live_end_ns:
            filled.append(VPRInterval(
                vpr_id=vpr_id, start_ns=prev_end, end_ns=lv.live_end_ns,
                unit=def_unit, access="live", op_id="",
            ))

        filled.extend(intervals)
        filled.sort(key=lambda i: (i.start_ns, {"write": 0, "read": 1, "live": 2}[i.access]))
        result[vpr_id] = filled

    return result
