#!/usr/bin/env python3
"""Greedy list scheduler for pipeline IR under hardware resource constraints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pipeline_ir import PipelineOp
from dependency_analyzer import analyze_dependencies, DependencyGraph


@dataclass
class ScheduleEntry:
    op_id: str
    start_ns: float
    end_ns: float
    unit: str
    wait_reason: str  # NONE | WAIT_DATA | WAIT_UNIT
    stall_ns: float


@dataclass
class ScheduleResult:
    entries: list[ScheduleEntry]
    total_latency_ns: float
    critical_path: list[str]
    stall_total_ns: float
    entries_by_id: dict[str, ScheduleEntry] = field(default_factory=dict)

    def __post_init__(self):
        self.entries_by_id = {e.op_id: e for e in self.entries}


def schedule(ops: list[PipelineOp]) -> ScheduleResult:
    dep_graph = analyze_dependencies(ops)
    op_map = {op.op_id: op for op in ops}
    topo_order = dep_graph.topo_sort()

    successors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, set[str]] = defaultdict(set)
    for e in dep_graph.edges:
        successors[e.from_op].add(e.to_op)
        predecessors[e.to_op].add(e.from_op)

    unit_busy_until: dict[str, float] = {"DMA": 0.0, "MXU": 0.0, "VPU": 0.0}
    op_end_time: dict[str, float] = {}
    entries: list[ScheduleEntry] = []

    for op_id in topo_order:
        op = op_map[op_id]
        dep_ready = 0.0
        for pred in predecessors[op_id]:
            dep_ready = max(dep_ready, op_end_time[pred])
        unit_ready = unit_busy_until[op.unit]
        start = max(dep_ready, unit_ready)
        end = start + op.latency_ns

        if start == 0.0 or (dep_ready == 0.0 and unit_ready == 0.0):
            wait_reason = "NONE"
            stall = 0.0
        elif dep_ready > unit_ready:
            wait_reason = "WAIT_DATA"
            stall = dep_ready - min(dep_ready, unit_ready)
        elif unit_ready > dep_ready:
            wait_reason = "WAIT_UNIT"
            stall = unit_ready - dep_ready
        else:
            wait_reason = "NONE"
            stall = 0.0

        if start > 0.0 and dep_ready <= 0.0 and unit_ready > 0.0:
            wait_reason = "WAIT_UNIT"
            stall = unit_ready

        unit_busy_until[op.unit] = end
        op_end_time[op_id] = end
        entries.append(ScheduleEntry(
            op_id=op_id, start_ns=start, end_ns=end, unit=op.unit,
            wait_reason=wait_reason, stall_ns=stall,
        ))

    total_latency = max(e.end_ns for e in entries) if entries else 0.0
    stall_total = sum(e.stall_ns for e in entries)
    crit_path, _ = dep_graph.critical_path()

    return ScheduleResult(
        entries=entries,
        total_latency_ns=total_latency,
        critical_path=crit_path,
        stall_total_ns=stall_total,
    )
