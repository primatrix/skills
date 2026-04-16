#!/usr/bin/env python3
"""Schedule fragment-level micro-ops under resource constraints."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from hw_params import TPUParams
from micro_op_ir import MicroOp, MicroOpGraph


@dataclass
class OpTiming:
    start_ns: float
    end_ns: float


@dataclass
class OccupancyInterval:
    resource_id: str
    start_ns: float
    end_ns: float
    op_id: str


@dataclass
class ScheduleResult:
    op_timings: dict[str, OpTiming]
    resource_occupancy: dict[str, list[OccupancyInterval]]
    fragment_residency: dict[str, list[OccupancyInterval]]
    stall_breakdown: dict[str, int]
    critical_path: list[str]
    total_time_ns: float
    peak_vmem_slots: int
    peak_reg_groups: int


def _resource_capacity(hw: TPUParams) -> dict[str, int]:
    return {
        "DMA": hw.dma_engine_count,
        "MXU": hw.mxu_count,
        "VPU": hw.vpu_count,
    }


def _resource_ready_time(resource_busy_until: dict[str, list[float]], unit: str) -> tuple[int, float]:
    slots = resource_busy_until[unit]
    best_idx = min(range(len(slots)), key=lambda idx: slots[idx])
    return best_idx, slots[best_idx]


def _named_resource_ready_time(resource_busy_until: dict[str, float], resource_names: tuple[str, ...]) -> float:
    if not resource_names:
        return 0.0
    return max(resource_busy_until.get(resource_name, 0.0) for resource_name in resource_names)


def _record_interval(
    occupancy: dict[str, list[OccupancyInterval]],
    resource_id: str,
    start_ns: float,
    end_ns: float,
    op_id: str,
) -> None:
    occupancy.setdefault(resource_id, []).append(
        OccupancyInterval(
            resource_id=resource_id,
            start_ns=start_ns,
            end_ns=end_ns,
            op_id=op_id,
        )
    )


def _critical_path(graph: MicroOpGraph) -> list[str]:
    scores: dict[str, float] = {}
    parent: dict[str, str | None] = {}

    for op_id, op in graph.micro_ops.items():
        if op.depends_on:
            best_dep = max(op.depends_on, key=lambda dep: scores[dep])
            scores[op_id] = scores[best_dep] + op.latency_ns
            parent[op_id] = best_dep
        else:
            scores[op_id] = op.latency_ns
            parent[op_id] = None

    if not scores:
        return []

    end_op = max(scores, key=scores.get)
    path: list[str] = []
    while end_op is not None:
        path.append(end_op)
        end_op = parent[end_op]
    path.reverse()
    return path


def _fragment_residency(
    graph: MicroOpGraph,
    op_timings: dict[str, OpTiming],
    total_time_ns: float,
) -> dict[str, list[OccupancyInterval]]:
    residency: dict[str, list[OccupancyInterval]] = {}
    for fragment_id, fragment in graph.fragments.items():
        producer_ids = [
            op_id for op_id, op in graph.micro_ops.items()
            if fragment_id in op.output_fragments
        ]
        consumer_ids = [
            op_id for op_id, op in graph.micro_ops.items()
            if fragment_id in op.input_fragments
        ]
        start_ns = max((op_timings[op_id].end_ns for op_id in producer_ids), default=0.0)
        end_ns = max((op_timings[op_id].start_ns for op_id in consumer_ids), default=total_time_ns)
        end_ns = max(end_ns, start_ns)
        residency[fragment_id] = [
            OccupancyInterval(
                resource_id=fragment.home_level,
                start_ns=start_ns,
                end_ns=end_ns,
                op_id=producer_ids[-1] if producer_ids else "source",
            )
        ]
    return residency


def _classify_wait(
    op: MicroOp,
    dep_ready_ns: float,
    unit_ready_ns: float,
    slot_ready_ns: float,
    reg_ready_ns: float,
) -> list[str]:
    if dep_ready_ns > 0 and dep_ready_ns >= unit_ready_ns and dep_ready_ns >= slot_ready_ns and dep_ready_ns >= reg_ready_ns:
        return ["WAIT_DATA"]

    wait_reasons: list[str] = []
    start_ns = max(dep_ready_ns, unit_ready_ns, slot_ready_ns, reg_ready_ns)
    if start_ns > dep_ready_ns:
        if unit_ready_ns == start_ns and op.required_units:
            wait_reasons.append("WAIT_UNIT")
        if slot_ready_ns == start_ns and op.required_vmem_slots:
            wait_reasons.append("WAIT_VMEM")
        if reg_ready_ns == start_ns and op.required_reg_groups:
            wait_reasons.append("WAIT_REG")
    return wait_reasons


def schedule_micro_op_graph(graph: MicroOpGraph, hw: TPUParams) -> ScheduleResult:
    """Greedy list scheduler for dependency-ordered micro-ops."""
    capacities = _resource_capacity(hw)
    resource_busy_until = {
        unit: [0.0] * count
        for unit, count in capacities.items()
    }
    vmem_busy_until: dict[str, float] = {}
    reg_busy_until: dict[str, float] = {}
    op_timings: dict[str, OpTiming] = {}
    resource_occupancy: dict[str, list[OccupancyInterval]] = {}
    stall_breakdown = {
        "WAIT_DATA": 0,
        "WAIT_UNIT": 0,
        "WAIT_VMEM": 0,
        "WAIT_REG": 0,
    }
    dependency_count = {
        op_id: len(op.depends_on)
        for op_id, op in graph.micro_ops.items()
    }
    children: dict[str, list[str]] = {op_id: [] for op_id in graph.micro_ops}
    for op_id, op in graph.micro_ops.items():
        for dep in op.depends_on:
            children[dep].append(op_id)
    ready_ops = deque(
        op_id for op_id, dep_count in dependency_count.items()
        if dep_count == 0
    )

    scheduled_ops = 0
    while ready_ops:
        op_id = ready_ops.popleft()
        op = graph.micro_ops[op_id]

        dep_ready_ns = max((op_timings[dep].end_ns for dep in op.depends_on), default=0.0)
        unit_ready_ns = dep_ready_ns
        chosen_units: list[tuple[str, int]] = []
        for unit in op.required_units:
            unit_idx, ready_ns = _resource_ready_time(resource_busy_until, unit)
            chosen_units.append((unit, unit_idx))
            unit_ready_ns = max(unit_ready_ns, ready_ns)
        slot_ready_ns = _named_resource_ready_time(vmem_busy_until, op.required_vmem_slots)
        reg_ready_ns = _named_resource_ready_time(reg_busy_until, op.required_reg_groups)

        start_ns = max(dep_ready_ns, unit_ready_ns, slot_ready_ns, reg_ready_ns)
        end_ns = start_ns + op.latency_ns
        op_timings[op_id] = OpTiming(start_ns=start_ns, end_ns=end_ns)

        for reason in _classify_wait(op, dep_ready_ns, unit_ready_ns, slot_ready_ns, reg_ready_ns):
            stall_breakdown[reason] += 1

        for unit, unit_idx in chosen_units:
            resource_busy_until[unit][unit_idx] = end_ns
            _record_interval(resource_occupancy, f"{unit}:{unit_idx}", start_ns, end_ns, op_id)

        for slot_name in op.required_vmem_slots:
            vmem_busy_until[slot_name] = end_ns
            _record_interval(resource_occupancy, f"VMEM:{slot_name}", start_ns, end_ns, op_id)

        for reg_name in op.required_reg_groups:
            reg_busy_until[reg_name] = end_ns
            _record_interval(resource_occupancy, f"REG:{reg_name}", start_ns, end_ns, op_id)

        scheduled_ops += 1
        for child_id in children[op_id]:
            dependency_count[child_id] -= 1
            if dependency_count[child_id] == 0:
                ready_ops.append(child_id)

    if scheduled_ops != len(graph.micro_ops):
        raise ValueError("Micro-op graph contains unresolved dependencies or unsupported resource constraints")

    total_time_ns = max((timing.end_ns for timing in op_timings.values()), default=0.0)

    # Compute peak concurrent resource usage via event-based sweep
    all_events: list[tuple[float, int, str]] = []  # (time, +1/-1, type)
    for resource_id, intervals in resource_occupancy.items():
        if resource_id.startswith("VMEM:"):
            for iv in intervals:
                all_events.append((iv.start_ns, +1, "vmem"))
                all_events.append((iv.end_ns, -1, "vmem"))
        elif resource_id.startswith("REG:"):
            for iv in intervals:
                all_events.append((iv.start_ns, +1, "reg"))
                all_events.append((iv.end_ns, -1, "reg"))
    all_events.sort(key=lambda e: (e[0], e[1]))
    peak_vmem = 0
    peak_reg = 0
    cur_vmem = 0
    cur_reg = 0
    for _, delta, rtype in all_events:
        if rtype == "vmem":
            cur_vmem += delta
            peak_vmem = max(peak_vmem, cur_vmem)
        else:
            cur_reg += delta
            peak_reg = max(peak_reg, cur_reg)

    return ScheduleResult(
        op_timings=op_timings,
        resource_occupancy=resource_occupancy,
        fragment_residency=_fragment_residency(graph, op_timings, total_time_ns),
        stall_breakdown=stall_breakdown,
        critical_path=_critical_path(graph),
        total_time_ns=total_time_ns,
        peak_vmem_slots=peak_vmem,
        peak_reg_groups=peak_reg,
    )
