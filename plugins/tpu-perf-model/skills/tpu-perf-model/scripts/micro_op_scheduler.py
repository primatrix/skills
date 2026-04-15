#!/usr/bin/env python3
"""Schedule fragment-level micro-ops under resource constraints."""
from __future__ import annotations

from dataclasses import dataclass

from hw_params import TPUParams
from micro_op_ir import MicroOpGraph


@dataclass
class OpTiming:
    start_ns: float
    end_ns: float


@dataclass
class ScheduleResult:
    op_timings: dict[str, OpTiming]
    total_time_ns: float


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


def schedule_micro_op_graph(graph: MicroOpGraph, hw: TPUParams) -> ScheduleResult:
    """Greedy list scheduler for dependency-ordered micro-ops."""
    capacities = _resource_capacity(hw)
    resource_busy_until = {
        unit: [0.0] * count
        for unit, count in capacities.items()
    }
    op_timings: dict[str, OpTiming] = {}

    pending = dict(graph.micro_ops)
    while pending:
        progress = False
        for op_id, op in list(pending.items()):
            if any(dep not in op_timings for dep in op.depends_on):
                continue

            dep_ready_ns = max((op_timings[dep].end_ns for dep in op.depends_on), default=0.0)
            unit_ready_ns = dep_ready_ns
            chosen_units: list[tuple[str, int]] = []
            for unit in op.required_units:
                unit_idx, ready_ns = _resource_ready_time(resource_busy_until, unit)
                chosen_units.append((unit, unit_idx))
                unit_ready_ns = max(unit_ready_ns, ready_ns)

            start_ns = unit_ready_ns
            end_ns = start_ns + op.latency_ns
            op_timings[op_id] = OpTiming(start_ns=start_ns, end_ns=end_ns)

            for unit, unit_idx in chosen_units:
                resource_busy_until[unit][unit_idx] = end_ns

            del pending[op_id]
            progress = True

        if not progress:
            raise ValueError("Micro-op graph contains unresolved dependencies or unsupported resource constraints")

    total_time_ns = max((timing.end_ns for timing in op_timings.values()), default=0.0)
    return ScheduleResult(op_timings=op_timings, total_time_ns=total_time_ns)
