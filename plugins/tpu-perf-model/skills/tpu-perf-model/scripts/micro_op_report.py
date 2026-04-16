#!/usr/bin/env python3
"""Render micro-op schedules as text or JSON."""
from __future__ import annotations

import json
import re

from micro_op_ir import MicroOpGraph
from micro_op_scheduler import ScheduleResult
from report import _format_ns


def _detect_op_stalls(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
) -> dict[str, list[str]]:
    """Re-derive per-op wait reasons from schedule timings.

    Approximate re-derivation for diagram annotation (not scheduling).
    Mirrors the scheduler's _classify_wait logic but works post-hoc from
    resource_occupancy intervals rather than the internal *_busy_until
    maps.  The unit-slot scan is an approximation since ScheduleResult
    does not record which specific slot was chosen per op.

    Categories: WAIT_DATA, WAIT_UNIT, WAIT_VMEM, WAIT_REG.
    Root ops (no dependencies) always get an empty list.
    """
    # Build per-unit busy-until timeline from resource occupancy.
    # For each op, find the latest end time of any resource occupancy
    # interval that ends at or before the op's start time.
    stalls: dict[str, list[str]] = {}
    for op_id, op in graph.micro_ops.items():
        if not op.depends_on:
            stalls[op_id] = []
            continue

        timing = schedule.op_timings[op_id]
        dep_ready = max(
            schedule.op_timings[dep].end_ns for dep in op.depends_on
        )

        # No stall if op starts exactly when deps are ready (no execution gap)
        if timing.start_ns <= dep_ready:
            stalls[op_id] = []
            continue

        # Determine unit readiness from resource occupancy
        unit_ready = dep_ready
        for unit in op.required_units:
            for res_id, intervals in schedule.resource_occupancy.items():
                if not res_id.startswith(unit + ":"):
                    continue
                for iv in intervals:
                    if iv.op_id != op_id and iv.end_ns <= timing.start_ns and iv.end_ns > unit_ready:
                        unit_ready = iv.end_ns

        # Determine VMEM slot readiness
        vmem_ready = dep_ready
        for slot in op.required_vmem_slots:
            res_id = f"VMEM:{slot}"
            for iv in schedule.resource_occupancy.get(res_id, []):
                if iv.op_id != op_id and iv.end_ns <= timing.start_ns and iv.end_ns > vmem_ready:
                    vmem_ready = iv.end_ns

        # Determine register group readiness
        reg_ready = dep_ready
        for reg in op.required_reg_groups:
            res_id = f"REG:{reg}"
            for iv in schedule.resource_occupancy.get(res_id, []):
                if iv.op_id != op_id and iv.end_ns <= timing.start_ns and iv.end_ns > reg_ready:
                    reg_ready = iv.end_ns

        # Find the binding constraint (latest ready time)
        binding = max(dep_ready, unit_ready, vmem_ready, reg_ready)
        if binding == dep_ready:
            stalls[op_id] = ["WAIT_DATA"]
        else:
            reasons: list[str] = []
            if unit_ready == binding and op.required_units:
                reasons.append("WAIT_UNIT")
            if vmem_ready == binding and op.required_vmem_slots:
                reasons.append("WAIT_VMEM")
            if reg_ready == binding and op.required_reg_groups:
                reasons.append("WAIT_REG")
            if not reasons:
                reasons.append("WAIT_DATA")
            stalls[op_id] = reasons
    return stalls


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


def _enhanced_label(op_id: str, graph: MicroOpGraph) -> str:
    """Build label with tile shape and resource annotations."""
    base = _short_label(op_id)
    op = graph.micro_ops.get(op_id)
    if not op:
        return base
    # Find tile shape from output fragments (fallback to input)
    shape_str = ""
    for frag_id in op.output_fragments:
        frag = graph.fragments.get(frag_id)
        if frag and frag.shape:
            shape_str = "[" + ",".join(str(d) for d in frag.shape) + "]"
            break
    if not shape_str:
        for frag_id in op.input_fragments:
            frag = graph.fragments.get(frag_id)
            if frag and frag.shape:
                shape_str = "[" + ",".join(str(d) for d in frag.shape) + "]"
                break
    # Resource annotations
    resources = []
    for slot in op.required_vmem_slots:
        resources.append(slot)
    for reg in op.required_reg_groups:
        resources.append(reg)
    res_str = ",".join(resources) if resources else ""
    parts = [base]
    if shape_str:
        parts.append(shape_str)
    if res_str:
        parts.append(res_str)
    return " ".join(parts)


def micro_schedule_to_mermaid(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
    max_tiles: int = 3,
) -> str:
    """Render the micro-op schedule as a resource-centric Mermaid Gantt."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")

    all_tile_indices = set()
    for op_id in schedule.op_timings:
        idx = _tile_index_from_op_id(op_id)
        if idx is not None:
            all_tile_indices.add(idx)
    total_tiles = max(all_tile_indices, default=0) + 1 if all_tile_indices else 0

    step_names = []
    for op in graph.micro_ops.values():
        if op.step_name not in step_names:
            step_names.append(op.step_name)
    title = ", ".join(step_names)

    stalls = _detect_op_stalls(schedule, graph)

    # Collect intervals grouped by resource type
    vmem_resources: dict[str, list] = {}
    reg_resources: dict[str, list] = {}
    for res_id, intervals in schedule.resource_occupancy.items():
        filtered = []
        for iv in intervals:
            tile_idx = _tile_index_from_op_id(iv.op_id)
            if tile_idx is not None and tile_idx >= max_tiles:
                continue
            filtered.append(iv)
        if not filtered:
            continue
        if res_id.startswith("VMEM:"):
            vmem_resources[res_id] = sorted(filtered, key=lambda iv: iv.start_ns)
        elif res_id.startswith("REG:"):
            reg_resources[res_id] = sorted(filtered, key=lambda iv: iv.start_ns)

    lines = [
        "```mermaid",
        "gantt",
        f"    title Resource Occupancy: {title} (ns)",
        "    dateFormat x",
        "    axisFormat %Q",
    ]

    # VMEM section
    if vmem_resources:
        lines.append("    section VMEM Slots")
        for res_id in sorted(vmem_resources):
            slot_name = res_id.split(":", 1)[1]
            intervals = vmem_resources[res_id]
            prev_end = None
            for iv in intervals:
                if prev_end is not None and iv.start_ns > prev_end:
                    wait_reasons = stalls.get(iv.op_id, [])
                    wait_label = ",".join(wait_reasons) if wait_reasons else "WAIT"
                    lines.append(f"        {wait_label} :crit, {int(prev_end)}, {int(iv.start_ns)}")
                label = f"{slot_name} [{_short_label(iv.op_id)}]"
                lines.append(f"        {label} :{int(iv.start_ns)}, {int(iv.end_ns)}")
                prev_end = iv.end_ns

    # REG section
    if reg_resources:
        lines.append("    section REG Groups")
        for res_id in sorted(reg_resources):
            reg_name = res_id.split(":", 1)[1]
            intervals = reg_resources[res_id]
            prev_end = None
            for iv in intervals:
                if prev_end is not None and iv.start_ns > prev_end:
                    wait_reasons = stalls.get(iv.op_id, [])
                    wait_label = ",".join(wait_reasons) if wait_reasons else "WAIT"
                    lines.append(f"        {wait_label} :crit, {int(prev_end)}, {int(iv.start_ns)}")
                label = f"{reg_name} [{_short_label(iv.op_id)}]"
                lines.append(f"        {label} :{int(iv.start_ns)}, {int(iv.end_ns)}")
                prev_end = iv.end_ns

    # Capacity comments
    peak_vmem = schedule.peak_vmem_slots
    peak_reg = schedule.peak_reg_groups
    lines.append(f"    %% Peak VMEM: {peak_vmem} slots")
    lines.append(f"    %% Peak REG: {peak_reg}/32 groups ({peak_reg * 100 // 32}%)")

    if total_tiles > max_tiles:
        lines.append(f"    %% ... tiles {max_tiles}-{total_tiles - 1} follow steady-state pattern ...")

    lines.append("```")
    return "\n".join(lines) + "\n"


def _sanitize_node_id(op_id: str) -> str:
    """Make op_id safe for Mermaid node IDs (alphanumeric + underscore)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", op_id)


def _flowchart_node_label(op_id: str, graph: MicroOpGraph) -> str:
    """Build multi-line Mermaid node label with shape, dtype, unit, resources."""
    op = graph.micro_ops.get(op_id)
    if not op:
        return _short_label(op_id)
    base = _short_label(op_id)
    parts = [base]
    for frag_id in op.output_fragments + op.input_fragments:
        frag = graph.fragments.get(frag_id)
        if frag and frag.shape:
            shape_str = "[" + ",".join(str(d) for d in frag.shape) + "] " + frag.dtype
            parts.append(shape_str)
            break
    if op.required_units:
        parts.append(" | ".join(op.required_units))
    if op.required_vmem_slots:
        parts.append(",".join(op.required_vmem_slots))
    if op.required_reg_groups:
        parts.append(",".join(op.required_reg_groups))
    return "<br/>".join(parts)


def micro_schedule_to_mermaid_flowchart(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
    max_tiles: int = 3,
) -> str:
    """Render per-tile flowcharts showing dependencies and stall edges."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")

    all_tile_indices = set()
    for op_id in schedule.op_timings:
        idx = _tile_index_from_op_id(op_id)
        if idx is not None:
            all_tile_indices.add(idx)
    total_tiles = max(all_tile_indices, default=0) + 1 if all_tile_indices else 0

    stalls = _detect_op_stalls(schedule, graph)
    blocks: list[str] = []

    for tile_idx in range(min(max_tiles, total_tiles)):
        tile_ops = [
            op_id for op_id in graph.micro_ops
            if _tile_index_from_op_id(op_id) == tile_idx
        ]
        if not tile_ops:
            continue

        lines = [
            "```mermaid",
            "flowchart TD",
        ]

        # Define nodes
        for op_id in tile_ops:
            node_id = _sanitize_node_id(op_id)
            label = _flowchart_node_label(op_id, graph)
            lines.append(f'    {node_id}["{label}"]')

        # Dependency edges (solid for non-stalled, dashed for stalled)
        for op_id in tile_ops:
            op = graph.micro_ops[op_id]
            reasons = stalls.get(op_id, [])
            for dep in op.depends_on:
                if dep in graph.micro_ops and _tile_index_from_op_id(dep) == tile_idx:
                    if reasons:
                        reason_str = ",".join(reasons)
                        lines.append(
                            f"    {_sanitize_node_id(dep)} -.{reason_str}.-> {_sanitize_node_id(op_id)}"
                        )
                    else:
                        lines.append(
                            f"    {_sanitize_node_id(dep)} --> {_sanitize_node_id(op_id)}"
                        )

        lines.append("```")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"
