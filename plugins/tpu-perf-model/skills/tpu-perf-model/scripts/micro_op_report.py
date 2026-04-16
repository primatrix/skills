#!/usr/bin/env python3
"""Render micro-op schedules as text or JSON."""
from __future__ import annotations

import json
import re

from micro_op_ir import MicroOpGraph
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


def _tile_index_from_op_id(op_id: str) -> int | None:
    """Extract tile index from op_id like 's0_qk_matmul_load_q_tile3'."""
    match = re.search(r"tile(\d+)", op_id)
    return int(match.group(1)) if match else None


def _unit_from_op(graph: MicroOpGraph, op_id: str) -> str | None:
    """Return the primary execution unit (DMA, MXU, VPU) for a micro-op."""
    op = graph.micro_ops.get(op_id)
    if not op or not op.required_units:
        return None
    return op.required_units[0]


def _short_label(op_id: str) -> str:
    """Shorten op_id for Mermaid bar labels. Strip step prefix like 's0_'."""
    return re.sub(r"^s\d+_", "", op_id)


def micro_schedule_to_mermaid(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
    max_tiles: int = 3,
) -> str:
    """Render the micro-op schedule as a Mermaid Gantt pipeline diagram."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")
    # Determine total tile count
    all_tile_indices = set()
    for op_id in schedule.op_timings:
        idx = _tile_index_from_op_id(op_id)
        if idx is not None:
            all_tile_indices.add(idx)
    total_tiles = max(all_tile_indices, default=0) + 1 if all_tile_indices else 0

    # Collect step names for title
    step_names = []
    for op in graph.micro_ops.values():
        if op.step_name not in step_names:
            step_names.append(op.step_name)
    title = ", ".join(step_names)

    # Group ops by unit, filtered by tile range
    unit_ops: dict[str, list[tuple[str, float, float]]] = {}
    for op_id, timing in schedule.op_timings.items():
        tile_idx = _tile_index_from_op_id(op_id)
        if tile_idx is not None and tile_idx >= max_tiles:
            continue
        unit = _unit_from_op(graph, op_id)
        if unit is None:
            continue
        unit_ops.setdefault(unit, []).append(
            (op_id, timing.start_ns, timing.end_ns)
        )

    # Sort each unit's ops by start time
    for unit in unit_ops:
        unit_ops[unit].sort(key=lambda x: (x[1], x[0]))

    # Build Mermaid output
    lines = [
        "```mermaid",
        "gantt",
        f"    title Tile Pipeline: {title} (ns)",
        "    dateFormat x",
        "    axisFormat %Q",
    ]

    section_order = ["DMA", "MXU", "VPU"]
    for unit in section_order:
        ops = unit_ops.get(unit)
        if not ops:
            continue
        lines.append(f"    section {unit}")
        for op_id, start_ns, end_ns in ops:
            label = _short_label(op_id)
            lines.append(
                f"        {label} :{int(start_ns)}, {int(end_ns)}"
            )

    if total_tiles > max_tiles:
        lines.append(
            f"    %% ... tiles {max_tiles}-{total_tiles - 1} follow steady-state pattern ..."
        )

    lines.append("```")
    return "\n".join(lines) + "\n"
