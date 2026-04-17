#!/usr/bin/env python3
"""VPR auto-allocator: maps logical VPR IDs to physical 0-31 via graph coloring."""

from __future__ import annotations

import copy
from collections import defaultdict

from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult
from dependency_analyzer import analyze_dependencies


def allocate_vprs(ops: list[PipelineOp], sched: ScheduleResult) -> list[PipelineOp]:
    """Assign physical VPR numbers (0-31) to logical VPRs.

    Uses liveness-based interference graph coloring. When a VPR's last reader
    finishes, its physical register is released for reuse.
    """
    op_map = {op.op_id: op for op in ops}
    entry_map = sched.entries_by_id

    # 1. Collect all logical VPR IDs and their liveness intervals.
    #    Liveness: first_def (start of writer) -> last_use_end (end of last reader).
    vpr_def: dict[int, float] = {}   # logical VPR -> earliest def start time
    vpr_end: dict[int, float] = {}   # logical VPR -> latest use end time

    for op in ops:
        entry = entry_map[op.op_id]
        for v in op.output_vprs:
            if v not in vpr_def or entry.start_ns < vpr_def[v]:
                vpr_def[v] = entry.start_ns
            if v not in vpr_end or entry.end_ns > vpr_end[v]:
                vpr_end[v] = entry.end_ns
        for v in op.input_vprs + op.weight_vprs + op.data_vprs:
            if v not in vpr_end or entry.end_ns > vpr_end[v]:
                vpr_end[v] = entry.end_ns
            # If a VPR is read but never written (shouldn't happen normally),
            # use the read start as def time.
            if v not in vpr_def:
                vpr_def[v] = entry.start_ns

    all_vprs = sorted(vpr_def.keys())
    if not all_vprs:
        return copy.deepcopy(ops)

    # 2. Build interference graph: two logical VPRs interfere if liveness overlaps.
    interferes: dict[int, set[int]] = defaultdict(set)
    for i, va in enumerate(all_vprs):
        for vb in all_vprs[i + 1:]:
            a_start, a_end = vpr_def[va], vpr_end[va]
            b_start, b_end = vpr_def[vb], vpr_end[vb]
            # Overlap: not (a_end <= b_start or b_end <= a_start)
            if not (a_end <= b_start or b_end <= a_start):
                interferes[va].add(vb)
                interferes[vb].add(va)

    # 3. Detect fusion constraints: fusion pairs' shared VPR must map to same physical.
    #    Build union-find groups for VPRs that must share the same physical register.
    dep_graph = analyze_dependencies(ops)
    coalesce: dict[int, int] = {}  # union-find parent

    def find(x: int) -> int:
        while coalesce.get(x, x) != x:
            coalesce[x] = coalesce.get(coalesce[x], coalesce[x])
            x = coalesce[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            coalesce[rb] = ra

    for prod_id, cons_id in dep_graph.fusion_pairs:
        prod = op_map[prod_id]
        cons = op_map[cons_id]
        # The shared VPRs: producer output that consumer reads.
        shared = set(prod.output_vprs) & set(cons.input_vprs)
        for v in shared:
            # These are the same logical VPR already, so they'll naturally
            # get the same physical. But we union them just in case.
            union(v, v)

    # Group VPRs by their union-find root.
    groups: dict[int, list[int]] = defaultdict(list)
    for v in all_vprs:
        groups[find(v)].append(v)

    # 4. Greedy graph coloring.
    color: dict[int, int] = {}

    # Process groups ordered by earliest def time (for determinism and good reuse).
    group_roots = sorted(groups.keys(), key=lambda r: vpr_def[r])

    for root in group_roots:
        group = groups[root]
        if group[0] in color:
            continue

        # Collect all colors used by interfering VPRs of any member.
        used_colors: set[int] = set()
        for v in group:
            for neighbor in interferes[v]:
                nr = find(neighbor)
                if nr in color:
                    used_colors.add(color[nr])

        # Pick smallest available color.
        c = 0
        while c in used_colors:
            c += 1
        if c > 31:
            raise ValueError(
                f"Cannot allocate: need >{31} physical VPRs "
                f"({len(all_vprs)} logical VPRs with too many live simultaneously)"
            )

        for v in group:
            color[find(v)] = c

    # Build final mapping: logical VPR -> physical VPR.
    mapping: dict[int, int] = {}
    for v in all_vprs:
        mapping[v] = color[find(v)]

    # 5. Rewrite ops with physical VPRs.
    result: list[PipelineOp] = []
    for op in ops:
        new_op = copy.deepcopy(op)
        new_op.output_vprs = [mapping[v] for v in op.output_vprs]
        new_op.weight_vprs = [mapping[v] for v in op.weight_vprs]
        new_op.data_vprs = [mapping[v] for v in op.data_vprs]
        # For MXU ops, input_vprs was auto-populated from weight+data in __post_init__.
        # We need to remap from the *original* input_vprs (which may include weight+data).
        new_op.input_vprs = [mapping[v] for v in op.input_vprs]
        result.append(new_op)

    return result
