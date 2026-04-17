#!/usr/bin/env python3
"""Event-driven pipeline scheduler with dual MXU dispatch."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pipeline_ir import PipelineOp
from dependency_analyzer import analyze_dependencies


@dataclass
class MXUPhase:
    phase_type: str  # "weight" | "data"
    start_ns: float
    end_ns: float
    unit_slot: str   # "MXU_W" | "MXU_D"


@dataclass
class ScheduleEntry:
    op_id: str
    start_ns: float
    end_ns: float
    unit: str
    wait_reason: str  # NONE | WAIT_DATA | WAIT_UNIT
    stall_ns: float
    phases: list[MXUPhase] = field(default_factory=list)


@dataclass
class ScheduleResult:
    entries: list[ScheduleEntry]
    total_latency_ns: float
    critical_path: list[str]
    stall_total_ns: float
    fusion_pairs: list[tuple[str, str]] = field(default_factory=list)
    entries_by_id: dict[str, ScheduleEntry] = field(default_factory=dict)

    def __post_init__(self):
        self.entries_by_id = {e.op_id: e for e in self.entries}


# Event types
_ISSUE = 0
_WEIGHT_DONE = 1
_COMPLETE = 2


def _is_dual_mxu(op: PipelineOp) -> bool:
    """MXU op with weight_vprs or data_vprs gets dual-phase dispatch."""
    return op.unit == "MXU" and (bool(op.weight_vprs) or bool(op.data_vprs))


def schedule(ops: list[PipelineOp]) -> ScheduleResult:
    dep_graph = analyze_dependencies(ops)
    op_map = {op.op_id: op for op in ops}

    # Build adjacency from dependency edges.
    successors: dict[str, set[str]] = defaultdict(set)
    predecessors: dict[str, set[str]] = defaultdict(set)
    for e in dep_graph.edges:
        successors[e.from_op].add(e.to_op)
        predecessors[e.to_op].add(e.from_op)

    fusion_set = set(dep_graph.fusion_pairs)

    # Track remaining predecessor count for readiness.
    remaining_deps: dict[str, int] = {}
    for op in ops:
        remaining_deps[op.op_id] = len(predecessors[op.op_id])

    # Unit availability: time when each slot becomes free.
    unit_free: dict[str, float] = {
        "DMA": 0.0, "MXU_W": 0.0, "MXU_D": 0.0, "VPU": 0.0,
    }

    op_end_time: dict[str, float] = {}
    entries: list[ScheduleEntry] = []
    seq = 0  # tie-breaker for heapq

    # Priority queue: (time, seq, event_type, op_id)
    pq: list[tuple[float, int, int, str]] = []

    def push(time: float, event_type: int, op_id: str):
        nonlocal seq
        heapq.heappush(pq, (time, seq, event_type, op_id))
        seq += 1

    # Seed: enqueue ISSUE for all ops with no predecessors.
    for op in ops:
        if remaining_deps[op.op_id] == 0:
            push(0.0, _ISSUE, op.op_id)

    # Per-op tracking for MXU weight phase completion time.
    weight_done_time: dict[str, float] = {}
    # Track when each op became ready (all deps satisfied).
    op_ready_time: dict[str, float] = {}
    # Track issue time for stall calculation.
    op_issue_time: dict[str, float] = {}

    while pq:
        now, _seq, event_type, op_id = heapq.heappop(pq)
        op = op_map[op_id]

        if event_type == _ISSUE:
            op_issue_time[op_id] = now
            # Record when the op was ready to issue (dep_ready time).
            if op_id not in op_ready_time:
                op_ready_time[op_id] = now

            if _is_dual_mxu(op):
                # Dual MXU: need MXU_W first.
                w_ready = unit_free["MXU_W"]
                w_start = max(now, w_ready)
                w_dur = op.latency_ns * 0.1
                w_end = w_start + w_dur
                unit_free["MXU_W"] = w_end
                weight_done_time[op_id] = w_end
                push(w_end, _WEIGHT_DONE, op_id)
            else:
                # Non-MXU or single-phase MXU: need the unit slot.
                slot = "MXU_D" if op.unit == "MXU" else op.unit
                u_ready = unit_free[slot]
                start = max(now, u_ready)
                end = start + op.latency_ns
                unit_free[slot] = end
                op_end_time[op_id] = end

                dep_ready = op_ready_time[op_id]
                wait_reason, stall = _calc_stall(dep_ready, start, op_issue_time[op_id])

                entries.append(ScheduleEntry(
                    op_id=op_id, start_ns=start, end_ns=end,
                    unit=op.unit, wait_reason=wait_reason, stall_ns=stall,
                ))
                push(end, _COMPLETE, op_id)

        elif event_type == _WEIGHT_DONE:
            # Now need MXU_D.
            d_ready = unit_free["MXU_D"]
            d_start = max(now, d_ready)
            d_dur = op.latency_ns * 0.9
            d_end = d_start + d_dur

            w_end = weight_done_time[op_id]
            w_start = w_end - op.latency_ns * 0.1

            unit_free["MXU_D"] = d_end
            op_end_time[op_id] = d_end

            dep_ready = op_ready_time[op_id]
            overall_start = w_start
            wait_reason, stall = _calc_stall(dep_ready, overall_start, op_issue_time[op_id])

            phases = [
                MXUPhase("weight", w_start, w_end, "MXU_W"),
                MXUPhase("data", d_start, d_end, "MXU_D"),
            ]
            entries.append(ScheduleEntry(
                op_id=op_id, start_ns=overall_start, end_ns=d_end,
                unit=op.unit, wait_reason=wait_reason, stall_ns=stall,
                phases=phases,
            ))
            push(d_end, _COMPLETE, op_id)

        elif event_type == _COMPLETE:
            # Release: check successors.
            for succ_id in successors[op_id]:
                remaining_deps[succ_id] -= 1
                if remaining_deps[succ_id] == 0:
                    # Determine ready time.
                    ready = max(op_end_time[p] for p in predecessors[succ_id])
                    op_ready_time[succ_id] = ready

                    # Fusion: consumer starts exactly at producer end.
                    if (op_id, succ_id) in fusion_set:
                        push(ready, _ISSUE, succ_id)
                    else:
                        push(ready, _ISSUE, succ_id)

    total_latency = max(e.end_ns for e in entries) if entries else 0.0
    stall_total = sum(e.stall_ns for e in entries)
    crit_path, _ = dep_graph.critical_path()

    return ScheduleResult(
        entries=entries,
        total_latency_ns=total_latency,
        critical_path=crit_path,
        stall_total_ns=stall_total,
        fusion_pairs=dep_graph.fusion_pairs,
    )


def _calc_stall(dep_ready: float, start: float, issue_time: float) -> tuple[str, float]:
    """Determine wait reason and stall duration."""
    if start <= 0.0:
        return "NONE", 0.0
    if start > dep_ready and dep_ready >= 0.0:
        # Unit wasn't free when deps were ready.
        return "WAIT_UNIT", start - dep_ready
    if dep_ready > issue_time:
        return "WAIT_DATA", dep_ready - issue_time
    return "NONE", 0.0
