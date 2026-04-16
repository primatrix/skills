#!/usr/bin/env python3
"""Data dependency analysis with RAW/WAR/WAW hazard detection."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pipeline_ir import PipelineOp


@dataclass
class Dependency:
    from_op: str
    to_op: str
    hazard_type: str   # RAW | WAR | WAW
    resource_type: str  # VPR | VMEM
    resource_id: str   # "VPR[3]" or "q_buf"


@dataclass
class DependencyGraph:
    ops: list[PipelineOp]
    edges: list[Dependency]

    def topo_sort(self) -> list[str]:
        adj: dict[str, list[str]] = defaultdict(list)
        in_deg: dict[str, int] = {op.op_id: 0 for op in self.ops}
        for e in self.edges:
            adj[e.from_op].append(e.to_op)
            in_deg[e.to_op] = in_deg.get(e.to_op, 0) + 1
        q = deque(op_id for op_id, d in in_deg.items() if d == 0)
        order: list[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in adj[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    q.append(m)
        if len(order) != len(self.ops):
            raise ValueError("Cycle detected in dependency graph")
        return order

    def critical_path(self) -> tuple[list[str], float]:
        op_map = {op.op_id: op for op in self.ops}
        order = self.topo_sort()
        dist: dict[str, float] = {op_id: 0.0 for op_id in order}
        pred: dict[str, str | None] = {op_id: None for op_id in order}
        for op_id in order:
            end = dist[op_id] + op_map[op_id].latency_ns
            adj_edges = [e for e in self.edges if e.from_op == op_id]
            for e in adj_edges:
                if end > dist[e.to_op]:
                    dist[e.to_op] = end
                    pred[e.to_op] = op_id
        last = max(order, key=lambda oid: dist[oid] + op_map[oid].latency_ns)
        total = dist[last] + op_map[last].latency_ns
        path: list[str] = []
        cur: str | None = last
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()
        return path, total


def _find_hazards(ops: list[PipelineOp]) -> list[Dependency]:
    edges: list[Dependency] = []
    for i, op_i in enumerate(ops):
        for j in range(i + 1, len(ops)):
            op_j = ops[j]
            for v in op_i.output_vprs:
                if v in op_j.input_vprs:
                    edges.append(Dependency(
                        op_i.op_id, op_j.op_id, "RAW", "VPR", f"VPR[{v}]"))
                if v in op_j.output_vprs:
                    edges.append(Dependency(
                        op_i.op_id, op_j.op_id, "WAW", "VPR", f"VPR[{v}]"))
            for v in op_i.input_vprs:
                if v in op_j.output_vprs:
                    edges.append(Dependency(
                        op_i.op_id, op_j.op_id, "WAR", "VPR", f"VPR[{v}]"))
            for s in op_i.output_vmem:
                if s in op_j.input_vmem:
                    edges.append(Dependency(
                        op_i.op_id, op_j.op_id, "RAW", "VMEM", s))
                if s in op_j.output_vmem:
                    edges.append(Dependency(
                        op_i.op_id, op_j.op_id, "WAW", "VMEM", s))
            for s in op_i.input_vmem:
                if s in op_j.output_vmem:
                    edges.append(Dependency(
                        op_i.op_id, op_j.op_id, "WAR", "VMEM", s))
    return edges


def _deduplicate_edges(edges: list[Dependency]) -> list[Dependency]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Dependency] = []
    for e in edges:
        key = (e.from_op, e.to_op, e.hazard_type)
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def _transitive_reduction(
    ops: list[PipelineOp], edges: list[Dependency]
) -> list[Dependency]:
    op_ids = [op.op_id for op in ops]
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e.from_op].add(e.to_op)
    reachable: dict[str, set[str]] = {}
    for start in op_ids:
        visited: set[str] = set()
        q = deque(adj[start])
        while q:
            n = q.popleft()
            if n in visited:
                continue
            visited.add(n)
            for m in adj[n]:
                if m not in visited:
                    q.append(m)
        reachable[start] = visited
    reduced: list[Dependency] = []
    for e in edges:
        can_reach_indirect = False
        for mid in adj[e.from_op]:
            if mid != e.to_op and e.to_op in reachable[mid]:
                can_reach_indirect = True
                break
        if not can_reach_indirect:
            reduced.append(e)
    return reduced


def analyze_dependencies(ops: list[PipelineOp]) -> DependencyGraph:
    edges = _find_hazards(ops)
    edges = _deduplicate_edges(edges)
    edges = _transitive_reduction(ops, edges)
    return DependencyGraph(ops=ops, edges=edges)
