#!/usr/bin/env python3
"""Render micro-op schedules as text or JSON."""
from __future__ import annotations

import json

from micro_op_scheduler import ScheduleResult
from report import _format_ns


def _micro_op_rows(schedule: ScheduleResult) -> list[dict]:
    rows = []
    for op_id, timing in sorted(schedule.op_timings.items(), key=lambda item: (item[1].start_ns, item[0])):
        rows.append(
            {
                "op_id": op_id,
                "start_ns": timing.start_ns,
                "end_ns": timing.end_ns,
                "duration_ns": timing.end_ns - timing.start_ns,
            }
        )
    return rows


def _timeline_rows(schedule: ScheduleResult) -> list[dict]:
    return _micro_op_rows(schedule)


def _occupancy_dict(schedule: ScheduleResult) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for resource_id, intervals in schedule.resource_occupancy.items():
        data[resource_id] = [
            {
                "start_ns": interval.start_ns,
                "end_ns": interval.end_ns,
                "op_id": interval.op_id,
            }
            for interval in intervals
        ]
    return data


def _residency_dict(schedule: ScheduleResult) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for fragment_id, intervals in schedule.fragment_residency.items():
        data[fragment_id] = [
            {
                "level": interval.resource_id,
                "start_ns": interval.start_ns,
                "end_ns": interval.end_ns,
                "producer": interval.op_id,
            }
            for interval in intervals
        ]
    return data


def _optimization_hints(schedule: ScheduleResult) -> list[str]:
    dominant = max(schedule.stall_breakdown, key=schedule.stall_breakdown.get)
    if schedule.stall_breakdown[dominant] == 0:
        return ["No dominant stall class detected in the current schedule."]
    hints = {
        "WAIT_DATA": "Critical-path dependencies dominate. Reduce serial dependencies or keep reusable fragments live longer.",
        "WAIT_UNIT": "Execution units are saturated. Increase overlap or rebalance work across MXU/VPU stages.",
        "WAIT_VMEM": "VMEM slot pressure is dominant. Reduce live fragments or change buffer-slot reuse.",
        "WAIT_REG": "Register pressure is dominant. Release accumulators earlier or split fused work.",
    }
    return [hints[dominant]]


def micro_schedule_to_json(schedule: ScheduleResult, step_results: list[dict]) -> str:
    payload = {
        "summary": {
            "total_time_ns": schedule.total_time_ns,
            "micro_op_count": len(schedule.op_timings),
            "step_count": len(step_results),
        },
        "step_results": step_results,
        "micro_ops": _micro_op_rows(schedule),
        "timeline": _timeline_rows(schedule),
        "resource_occupancy": _occupancy_dict(schedule),
        "fragment_residency": _residency_dict(schedule),
        "critical_path": schedule.critical_path,
        "stall_breakdown": schedule.stall_breakdown,
        "optimization_hints": _optimization_hints(schedule),
    }
    return json.dumps(payload, indent=2)


def micro_schedule_to_text(schedule: ScheduleResult, step_results: list[dict]) -> str:
    lines = ["=== Macro Summary ==="]
    lines.append(f"Total schedule time: {_format_ns(schedule.total_time_ns)}")
    lines.append(f"Micro-ops: {len(schedule.op_timings)}")
    lines.append(f"Steps summarized: {len(step_results)}")
    lines.append("")

    lines.append("=== Micro-Op Schedule Summary ===")
    for row in _micro_op_rows(schedule):
        lines.append(
            f"{row['op_id']}: start={_format_ns(row['start_ns'])} "
            f"end={_format_ns(row['end_ns'])}"
        )
    lines.append("")

    lines.append("=== Timeline ===")
    for row in _timeline_rows(schedule):
        lines.append(
            f"t={_format_ns(row['start_ns'])} -> {row['op_id']} "
            f"({_format_ns(row['duration_ns'])})"
        )
    lines.append("")

    lines.append("=== Residency and Occupancy ===")
    for resource_id, intervals in sorted(schedule.resource_occupancy.items()):
        span = ", ".join(
            f"{interval.op_id}[{_format_ns(interval.start_ns)}, {_format_ns(interval.end_ns)}]"
            for interval in intervals
        )
        lines.append(f"{resource_id}: {span}")
    for fragment_id, intervals in sorted(schedule.fragment_residency.items()):
        span = ", ".join(
            f"{interval.resource_id}[{_format_ns(interval.start_ns)}, {_format_ns(interval.end_ns)}]"
            for interval in intervals
        )
        lines.append(f"{fragment_id}: {span}")
    lines.append("")

    lines.append("=== Critical Path and Optimization Hints ===")
    lines.append(f"Critical path: {' -> '.join(schedule.critical_path) if schedule.critical_path else '(empty)'}")
    lines.append(f"Stalls: {schedule.stall_breakdown}")
    for hint in _optimization_hints(schedule):
        lines.append(f"- {hint}")

    return "\n".join(lines)
