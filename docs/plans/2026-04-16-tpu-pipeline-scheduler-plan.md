# TPU Pipeline Scheduler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `tpu-pipeline-scheduler` skill to the tpu-perf-model plugin that analyzes explicit register-level pipeline scheduling with RAW/WAR/WAW hazard detection, greedy list scheduling, VPR liveness analysis, and optimal reorder suggestions.

**Architecture:** Six new Python modules under `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/`, sharing only `hw_params.py` from the existing skill. Pipeline: JSON IR → dependency analysis → scheduling → VPR analysis → report generation. CLI entry point ties them together.

**Tech Stack:** Python 3, dataclasses, argparse, json, unittest. No external dependencies.

**Design doc:** `docs/plans/2026-04-16-tpu-pipeline-scheduler-design.md`

**Conventions (from existing codebase):**
- `from __future__ import annotations` on all non-test files
- Mutable `@dataclass` (not frozen) for all IR/result types
- Private functions with `_` prefix, public functions descriptive names
- Tests: `unittest.TestCase`, domain imports inside test method bodies, `self.assertEqual` etc.
- JSON: manual dict construction, `json.dumps(indent=2)`
- Text: accumulate `lines: list[str]`, return `"\n".join(lines)`
- CLI: `argparse`, single `main()`, `if __name__ == "__main__": main()`
- Local imports: bare module names (e.g., `from hw_params import ...`)

**Base path:** `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/`
**Existing hw_params:** `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/hw_params.py`

---

### Task 1: Pipeline IR Data Model

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_ir.py`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_ir.py`

**Step 1: Write the failing tests**

Create `test_pipeline_ir.py`:

```python
#!/usr/bin/env python3
"""Tests for pipeline_ir module."""

import unittest


class TestPipelineOp(unittest.TestCase):
    def test_pipeline_op_fields(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="load_q",
            op_kind="DMA_LOAD",
            input_vprs=[],
            output_vprs=[],
            input_vmem=[],
            output_vmem=["q_buf"],
            latency_ns=200.0,
            unit="DMA",
            label="Load Q tile",
        )
        self.assertEqual(op.op_id, "load_q")
        self.assertEqual(op.op_kind, "DMA_LOAD")
        self.assertEqual(op.output_vmem, ["q_buf"])
        self.assertEqual(op.latency_ns, 200.0)
        self.assertEqual(op.unit, "DMA")

    def test_pipeline_op_default_label(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="x", op_kind="VPU", input_vprs=[0], output_vprs=[1],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        self.assertEqual(op.label, "")

    def test_pipeline_op_all_vprs_property(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="mxu", op_kind="MXU",
            input_vprs=[0, 1, 2, 3], output_vprs=[4, 5],
            input_vmem=[], output_vmem=[],
            latency_ns=500.0, unit="MXU",
        )
        self.assertEqual(op.all_vprs, [0, 1, 2, 3, 4, 5])

    def test_pipeline_op_all_vmem_property(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="dma", op_kind="DMA_LOAD",
            input_vprs=[], output_vprs=[],
            input_vmem=["a"], output_vmem=["b", "c"],
            latency_ns=100.0, unit="DMA",
        )
        self.assertEqual(op.all_vmem, ["a", "b", "c"])


class TestPipelineSpec(unittest.TestCase):
    def test_pipeline_spec_fields(self):
        from pipeline_ir import PipelineOp, PipelineSpec

        ops = [
            PipelineOp(
                op_id="op1", op_kind="DMA_LOAD",
                input_vprs=[], output_vprs=[],
                input_vmem=[], output_vmem=["buf"],
                latency_ns=100.0, unit="DMA",
            ),
        ]
        spec = PipelineSpec(name="test", hw="v7x", ops=ops)
        self.assertEqual(spec.name, "test")
        self.assertEqual(spec.hw, "v7x")
        self.assertEqual(len(spec.ops), 1)

    def test_load_spec_from_dict(self):
        from pipeline_ir import load_spec

        data = {
            "name": "test_kernel",
            "hw": "v7x",
            "ops": [
                {
                    "op_id": "load_q",
                    "op_kind": "DMA_LOAD",
                    "input_vprs": [],
                    "output_vprs": [],
                    "input_vmem": [],
                    "output_vmem": ["q_buf"],
                    "latency_ns": 200,
                    "unit": "DMA",
                    "label": "Load Q",
                },
                {
                    "op_id": "q_to_reg",
                    "op_kind": "VMEM_TO_REG",
                    "input_vprs": [],
                    "output_vprs": [0, 1, 2, 3],
                    "input_vmem": ["q_buf"],
                    "output_vmem": [],
                    "latency_ns": 10,
                    "unit": "VPU",
                },
            ],
        }
        spec = load_spec(data)
        self.assertEqual(spec.name, "test_kernel")
        self.assertEqual(len(spec.ops), 2)
        self.assertEqual(spec.ops[0].op_id, "load_q")
        self.assertEqual(spec.ops[1].output_vprs, [0, 1, 2, 3])
        self.assertEqual(spec.ops[1].label, "")

    def test_load_spec_from_file(self):
        import json
        import os
        import tempfile
        from pipeline_ir import load_spec_from_file

        data = {
            "name": "file_test",
            "hw": "v7x",
            "ops": [
                {
                    "op_id": "op1",
                    "op_kind": "VPU",
                    "input_vprs": [0],
                    "output_vprs": [1],
                    "input_vmem": [],
                    "output_vmem": [],
                    "latency_ns": 50,
                    "unit": "VPU",
                },
            ],
        }
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        try:
            json.dump(data, f)
            f.close()
            spec = load_spec_from_file(f.name)
            self.assertEqual(spec.name, "file_test")
            self.assertEqual(len(spec.ops), 1)
        finally:
            os.unlink(f.name)

    def test_validate_rejects_duplicate_op_ids(self):
        from pipeline_ir import load_spec

        data = {
            "name": "dup",
            "hw": "v7x",
            "ops": [
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [],
                 "output_vprs": [0], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "VPU"},
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [0],
                 "output_vprs": [1], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "VPU"},
            ],
        }
        with self.assertRaises(ValueError):
            load_spec(data)

    def test_validate_rejects_invalid_vpr(self):
        from pipeline_ir import load_spec

        data = {
            "name": "bad_vpr",
            "hw": "v7x",
            "ops": [
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [32],
                 "output_vprs": [], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "VPU"},
            ],
        }
        with self.assertRaises(ValueError):
            load_spec(data)

    def test_validate_rejects_invalid_unit(self):
        from pipeline_ir import load_spec

        data = {
            "name": "bad_unit",
            "hw": "v7x",
            "ops": [
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [],
                 "output_vprs": [0], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "GPU"},
            ],
        }
        with self.assertRaises(ValueError):
            load_spec(data)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_ir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline_ir'`

**Step 3: Write the implementation**

Create `pipeline_ir.py`:

```python
#!/usr/bin/env python3
"""Pipeline IR data model for register-level scheduling analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

_VALID_OP_KINDS = frozenset({
    "DMA_LOAD", "DMA_STORE", "MXU", "VPU", "VMEM_TO_REG", "REG_TO_VMEM",
})
_VALID_UNITS = frozenset({"DMA", "MXU", "VPU"})
_MAX_VPR = 31


@dataclass
class PipelineOp:
    op_id: str
    op_kind: str
    input_vprs: list[int]
    output_vprs: list[int]
    input_vmem: list[str]
    output_vmem: list[str]
    latency_ns: float
    unit: str
    label: str = ""

    @property
    def all_vprs(self) -> list[int]:
        return self.input_vprs + self.output_vprs

    @property
    def all_vmem(self) -> list[str]:
        return self.input_vmem + self.output_vmem


@dataclass
class PipelineSpec:
    name: str
    hw: str
    ops: list[PipelineOp]


def _validate_spec(spec: PipelineSpec) -> None:
    seen_ids: set[str] = set()
    for op in spec.ops:
        if op.op_id in seen_ids:
            raise ValueError(f"Duplicate op_id: {op.op_id}")
        seen_ids.add(op.op_id)
        for v in op.input_vprs + op.output_vprs:
            if v < 0 or v > _MAX_VPR:
                raise ValueError(
                    f"VPR {v} in op {op.op_id} out of range 0-{_MAX_VPR}"
                )
        if op.unit not in _VALID_UNITS:
            raise ValueError(
                f"Invalid unit '{op.unit}' in op {op.op_id}, "
                f"must be one of {sorted(_VALID_UNITS)}"
            )


def _parse_op(d: dict) -> PipelineOp:
    return PipelineOp(
        op_id=d["op_id"],
        op_kind=d["op_kind"],
        input_vprs=d.get("input_vprs", []),
        output_vprs=d.get("output_vprs", []),
        input_vmem=d.get("input_vmem", []),
        output_vmem=d.get("output_vmem", []),
        latency_ns=float(d["latency_ns"]),
        unit=d["unit"],
        label=d.get("label", ""),
    )


def load_spec(data: dict) -> PipelineSpec:
    ops = [_parse_op(o) for o in data["ops"]]
    spec = PipelineSpec(name=data["name"], hw=data.get("hw", "v7x"), ops=ops)
    _validate_spec(spec)
    return spec


def load_spec_from_file(path: str) -> PipelineSpec:
    with open(path) as f:
        return load_spec(json.load(f))
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_ir.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_ir.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_ir.py
git commit -m "feat(tpu-pipeline-scheduler): add Pipeline IR data model with validation"
```

---

### Task 2: Dependency Analyzer

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/dependency_analyzer.py`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_dependency_analyzer.py`

**Step 1: Write the failing tests**

Create `test_dependency_analyzer.py`:

```python
#!/usr/bin/env python3
"""Tests for dependency_analyzer module."""

import unittest


class TestDependencyAnalyzer(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_raw_dependency_on_vpr(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="w", output_vprs=[0]),
            self._make_op(op_id="r", input_vprs=[0]),
        ]
        graph = analyze_dependencies(ops)
        raw_edges = [e for e in graph.edges if e.hazard_type == "RAW"]
        self.assertEqual(len(raw_edges), 1)
        self.assertEqual(raw_edges[0].from_op, "w")
        self.assertEqual(raw_edges[0].to_op, "r")
        self.assertEqual(raw_edges[0].resource_id, "VPR[0]")

    def test_war_dependency_on_vpr(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="r", input_vprs=[0]),
            self._make_op(op_id="w", output_vprs=[0]),
        ]
        graph = analyze_dependencies(ops)
        war_edges = [e for e in graph.edges if e.hazard_type == "WAR"]
        self.assertEqual(len(war_edges), 1)
        self.assertEqual(war_edges[0].from_op, "r")
        self.assertEqual(war_edges[0].to_op, "w")

    def test_waw_dependency_on_vpr(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="w1", output_vprs=[0]),
            self._make_op(op_id="w2", output_vprs=[0]),
        ]
        graph = analyze_dependencies(ops)
        waw_edges = [e for e in graph.edges if e.hazard_type == "WAW"]
        self.assertEqual(len(waw_edges), 1)
        self.assertEqual(waw_edges[0].from_op, "w1")
        self.assertEqual(waw_edges[0].to_op, "w2")

    def test_vmem_raw_dependency(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="store", op_kind="DMA_LOAD", unit="DMA",
                          output_vmem=["buf"]),
            self._make_op(op_id="load", op_kind="VMEM_TO_REG", unit="VPU",
                          input_vmem=["buf"], output_vprs=[0]),
        ]
        graph = analyze_dependencies(ops)
        raw_edges = [e for e in graph.edges
                     if e.hazard_type == "RAW" and e.resource_type == "VMEM"]
        self.assertEqual(len(raw_edges), 1)
        self.assertEqual(raw_edges[0].resource_id, "buf")

    def test_no_dependency_on_different_vprs(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="w", output_vprs=[0]),
            self._make_op(op_id="r", input_vprs=[1]),
        ]
        graph = analyze_dependencies(ops)
        self.assertEqual(len(graph.edges), 0)

    def test_transitive_reduction(self):
        from dependency_analyzer import analyze_dependencies

        # A writes VPR[0], B reads VPR[0] and writes VPR[1],
        # C reads VPR[0] and VPR[1]
        # RAW edges: A->B (VPR[0]), A->C (VPR[0]), B->C (VPR[1])
        # After transitive reduction: A->C via VPR[0] is redundant
        # because A->B->C already covers it
        ops = [
            self._make_op(op_id="A", output_vprs=[0]),
            self._make_op(op_id="B", input_vprs=[0], output_vprs=[1]),
            self._make_op(op_id="C", input_vprs=[0, 1]),
        ]
        graph = analyze_dependencies(ops)
        # A->B (RAW VPR[0]), B->C (RAW VPR[1]) should remain
        # A->C (RAW VPR[0]) should be removed by transitive reduction
        from_to = [(e.from_op, e.to_op) for e in graph.edges]
        self.assertIn(("A", "B"), from_to)
        self.assertIn(("B", "C"), from_to)
        self.assertNotIn(("A", "C"), from_to)

    def test_topo_sort(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="A", output_vprs=[0]),
            self._make_op(op_id="B", input_vprs=[0], output_vprs=[1]),
            self._make_op(op_id="C", input_vprs=[1]),
        ]
        graph = analyze_dependencies(ops)
        order = graph.topo_sort()
        self.assertEqual(order, ["A", "B", "C"])

    def test_critical_path(self):
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="A", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="B", input_vprs=[0], output_vprs=[1],
                          latency_ns=200.0),
            self._make_op(op_id="C", input_vprs=[1], latency_ns=50.0),
        ]
        graph = analyze_dependencies(ops)
        path, length = graph.critical_path()
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(length, 350.0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_dependency_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dependency_analyzer'`

**Step 3: Write the implementation**

Create `dependency_analyzer.py`:

```python
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
        # Find the op with max end time
        last = max(order, key=lambda oid: dist[oid] + op_map[oid].latency_ns)
        total = dist[last] + op_map[last].latency_ns
        # Reconstruct path
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
            # VPR hazards
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
            # VMEM hazards
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
    # Build adjacency for reachability
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e.from_op].add(e.to_op)
    # Compute transitive closure via BFS from each node
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
    # Remove edge u->v if u can reach v through another path
    reduced: list[Dependency] = []
    for e in edges:
        # Check if from_op can reach to_op via a neighbor other than to_op
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
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_dependency_analyzer.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/dependency_analyzer.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_dependency_analyzer.py
git commit -m "feat(tpu-pipeline-scheduler): add dependency analyzer with RAW/WAR/WAW hazard detection"
```

---

### Task 3: Pipeline Scheduler

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_scheduler.py`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_scheduler.py`

**Step 1: Write the failing tests**

Create `test_pipeline_scheduler.py`:

```python
#!/usr/bin/env python3
"""Tests for pipeline_scheduler module."""

import unittest


class TestPipelineScheduler(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_sequential_same_unit(self):
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="a", unit="VPU", latency_ns=100.0),
            self._make_op(op_id="b", unit="VPU", latency_ns=50.0,
                          input_vprs=[]),
        ]
        # No data dependency, but same unit => b starts after a
        result = schedule(ops)
        a_entry = result.entries_by_id["a"]
        b_entry = result.entries_by_id["b"]
        self.assertEqual(a_entry.start_ns, 0.0)
        self.assertEqual(a_entry.end_ns, 100.0)
        self.assertEqual(b_entry.start_ns, 100.0)
        self.assertEqual(b_entry.end_ns, 150.0)

    def test_parallel_different_units(self):
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="dma", unit="DMA", op_kind="DMA_LOAD",
                          latency_ns=200.0, output_vmem=["buf"]),
            self._make_op(op_id="vpu", unit="VPU", latency_ns=100.0,
                          output_vprs=[0]),
        ]
        # No data dependency, different units => parallel
        result = schedule(ops)
        self.assertEqual(result.entries_by_id["dma"].start_ns, 0.0)
        self.assertEqual(result.entries_by_id["vpu"].start_ns, 0.0)

    def test_data_dependency_delays(self):
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="w", unit="VPU", latency_ns=100.0,
                          output_vprs=[0]),
            self._make_op(op_id="r", unit="MXU", latency_ns=50.0,
                          input_vprs=[0]),
        ]
        # RAW dependency on VPR[0]: r must wait for w
        result = schedule(ops)
        self.assertEqual(result.entries_by_id["r"].start_ns, 100.0)
        self.assertEqual(result.entries_by_id["r"].wait_reason, "WAIT_DATA")

    def test_wait_unit_reason(self):
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="a", unit="VPU", latency_ns=100.0),
            self._make_op(op_id="b", unit="VPU", latency_ns=50.0),
        ]
        result = schedule(ops)
        self.assertEqual(result.entries_by_id["b"].wait_reason, "WAIT_UNIT")

    def test_total_latency(self):
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="a", unit="DMA", op_kind="DMA_LOAD",
                          latency_ns=200.0, output_vmem=["buf"]),
            self._make_op(op_id="b", unit="VPU", latency_ns=100.0,
                          input_vmem=["buf"], output_vprs=[0]),
        ]
        result = schedule(ops)
        # a: 0-200 (DMA), b: 200-300 (VPU, waits for a due to VMEM dep)
        self.assertEqual(result.total_latency_ns, 300.0)

    def test_schedule_result_has_critical_path(self):
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="a", unit="VPU", latency_ns=100.0,
                          output_vprs=[0]),
            self._make_op(op_id="b", unit="MXU", latency_ns=50.0,
                          input_vprs=[0]),
        ]
        result = schedule(ops)
        self.assertIn("a", result.critical_path)
        self.assertIn("b", result.critical_path)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline_scheduler'`

**Step 3: Write the implementation**

Create `pipeline_scheduler.py`:

```python
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

    # Build successor map
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
        # Earliest start from data dependencies
        dep_ready = 0.0
        for pred in predecessors[op_id]:
            dep_ready = max(dep_ready, op_end_time[pred])
        # Earliest start from unit availability
        unit_ready = unit_busy_until[op.unit]
        start = max(dep_ready, unit_ready)
        end = start + op.latency_ns

        # Determine wait reason
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

        # For ops that must wait but both are at 0, it's a unit conflict
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
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_scheduler.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_scheduler.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_scheduler.py
git commit -m "feat(tpu-pipeline-scheduler): add greedy list scheduler with stall tracking"
```

---

### Task 4: VPR Liveness Analyzer

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/vpr_analyzer.py`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_vpr_analyzer.py`

**Step 1: Write the failing tests**

Create `test_vpr_analyzer.py`:

```python
#!/usr/bin/env python3
"""Tests for vpr_analyzer module."""

import unittest


class TestVPRAnalyzer(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_single_vpr_liveness(self):
        from pipeline_scheduler import schedule, ScheduleEntry
        from vpr_analyzer import analyze_vpr_liveness

        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        sched = schedule(ops)
        result = analyze_vpr_liveness(ops, sched)
        vpr0 = [lv for lv in result.liveness if lv.vpr_id == 0]
        self.assertEqual(len(vpr0), 1)
        self.assertEqual(vpr0[0].defined_by, "w")
        self.assertEqual(vpr0[0].last_used_by, "r")

    def test_peak_concurrent_vprs(self):
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        ops = [
            self._make_op(op_id="w0", unit="DMA", op_kind="DMA_LOAD",
                          output_vprs=[0], latency_ns=100.0, output_vmem=[]),
            self._make_op(op_id="w1", unit="VPU",
                          output_vprs=[1], latency_ns=100.0),
            self._make_op(op_id="r", unit="MXU", op_kind="MXU",
                          input_vprs=[0, 1], latency_ns=50.0),
        ]
        sched = schedule(ops)
        result = analyze_vpr_liveness(ops, sched)
        self.assertEqual(result.peak_concurrent, 2)

    def test_utilization_ratio(self):
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        sched = schedule(ops)
        result = analyze_vpr_liveness(ops, sched)
        # VPR[0] is live from 100 to 100 (w ends at 100, r starts at 100)
        # utilization = time_live / (total_time * 32)
        self.assertGreaterEqual(result.utilization_ratio, 0.0)
        self.assertLessEqual(result.utilization_ratio, 1.0)

    def test_pressure_warning_when_high(self):
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        # Use many VPRs simultaneously
        out_vprs = list(range(28))  # 28 out of 32 = 87.5%
        ops = [
            self._make_op(op_id="w", output_vprs=out_vprs, latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=out_vprs, latency_ns=50.0),
        ]
        sched = schedule(ops)
        result = analyze_vpr_liveness(ops, sched)
        self.assertGreater(len(result.pressure_warnings), 0)

    def test_no_warning_when_low_pressure(self):
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        sched = schedule(ops)
        result = analyze_vpr_liveness(ops, sched)
        self.assertEqual(len(result.pressure_warnings), 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_vpr_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpr_analyzer'`

**Step 3: Write the implementation**

Create `vpr_analyzer.py`:

```python
#!/usr/bin/env python3
"""VPR liveness analysis and register pressure tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult

_TOTAL_VPRS = 32
_PRESSURE_THRESHOLD = 0.75  # warn when >75% VPRs are live simultaneously


@dataclass
class VPRLiveness:
    vpr_id: int
    defined_by: str
    last_used_by: str
    live_start_ns: float
    live_end_ns: float


@dataclass
class VPROccupancy:
    liveness: list[VPRLiveness]
    peak_concurrent: int
    peak_time_ns: float
    utilization_ratio: float
    pressure_warnings: list[str]


def analyze_vpr_liveness(
    ops: list[PipelineOp], sched: ScheduleResult
) -> VPROccupancy:
    entries = sched.entries_by_id

    # Track define and last-use for each VPR
    # A VPR can be defined multiple times (WAW); track each definition interval
    vpr_defs: dict[int, list[tuple[str, float]]] = {}  # vpr -> [(op_id, end_time)]
    vpr_last_use: dict[int, list[tuple[str, float]]] = {}  # vpr -> [(op_id, start_time)]

    for op in ops:
        entry = entries[op.op_id]
        for v in op.output_vprs:
            vpr_defs.setdefault(v, []).append((op.op_id, entry.end_ns))
        for v in op.input_vprs:
            vpr_last_use.setdefault(v, []).append((op.op_id, entry.start_ns))

    liveness: list[VPRLiveness] = []
    for vpr_id in sorted(set(list(vpr_defs.keys()) + list(vpr_last_use.keys()))):
        defs = vpr_defs.get(vpr_id, [])
        uses = vpr_last_use.get(vpr_id, [])
        if not defs:
            continue
        # For simplicity, take earliest define and latest use
        def_op, def_time = min(defs, key=lambda x: x[1])
        if uses:
            use_op, use_time = max(uses, key=lambda x: x[1])
            live_end = max(def_time, use_time)
        else:
            use_op = def_op
            live_end = def_time
        liveness.append(VPRLiveness(
            vpr_id=vpr_id,
            defined_by=def_op,
            last_used_by=use_op,
            live_start_ns=def_time,
            live_end_ns=live_end,
        ))

    # Compute peak concurrent VPRs via event sweep
    total_time = sched.total_latency_ns
    events: list[tuple[float, int]] = []  # (time, +1/-1)
    for lv in liveness:
        events.append((lv.live_start_ns, +1))
        events.append((lv.live_end_ns, -1))
    events.sort(key=lambda x: (x[0], x[1]))

    peak = 0
    peak_time = 0.0
    current = 0
    for t, delta in events:
        current += delta
        if current > peak:
            peak = current
            peak_time = t

    # Utilization: sum of live durations / (total_time * TOTAL_VPRS)
    total_live = sum(max(0, lv.live_end_ns - lv.live_start_ns) for lv in liveness)
    util = total_live / (total_time * _TOTAL_VPRS) if total_time > 0 else 0.0

    warnings: list[str] = []
    if peak > _TOTAL_VPRS * _PRESSURE_THRESHOLD:
        warnings.append(
            f"VPR pressure critical: {peak}/{_TOTAL_VPRS} VPRs live "
            f"simultaneously at t={peak_time:.0f}ns "
            f"({peak/_TOTAL_VPRS*100:.0f}% utilization)"
        )

    return VPROccupancy(
        liveness=liveness,
        peak_concurrent=peak,
        peak_time_ns=peak_time,
        utilization_ratio=util,
        pressure_warnings=warnings,
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_vpr_analyzer.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/vpr_analyzer.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_vpr_analyzer.py
git commit -m "feat(tpu-pipeline-scheduler): add VPR liveness analyzer with pressure detection"
```

---

### Task 5: Report Generator

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_report.py`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_report.py`

**Step 1: Write the failing tests**

Create `test_pipeline_report.py`:

```python
#!/usr/bin/env python3
"""Tests for pipeline_report module."""

import unittest


class TestDepsReport(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def _make_simple_pipeline(self):
        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        return ops

    def test_deps_text(self):
        from dependency_analyzer import analyze_dependencies
        from pipeline_report import deps_to_text

        ops = self._make_simple_pipeline()
        graph = analyze_dependencies(ops)
        text = deps_to_text(graph)
        self.assertIn("w", text)
        self.assertIn("r", text)
        self.assertIn("RAW", text)
        self.assertIn("VPR[0]", text)

    def test_deps_json(self):
        import json
        from dependency_analyzer import analyze_dependencies
        from pipeline_report import deps_to_json

        ops = self._make_simple_pipeline()
        graph = analyze_dependencies(ops)
        data = json.loads(deps_to_json(graph))
        self.assertIn("edges", data)
        self.assertEqual(len(data["edges"]), 1)
        self.assertEqual(data["edges"][0]["hazard_type"], "RAW")

    def test_deps_mermaid(self):
        from dependency_analyzer import analyze_dependencies
        from pipeline_report import deps_to_mermaid

        ops = self._make_simple_pipeline()
        graph = analyze_dependencies(ops)
        mermaid = deps_to_mermaid(graph)
        self.assertIn("graph", mermaid)
        self.assertIn("w", mermaid)
        self.assertIn("r", mermaid)


class TestGanttReport(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_gantt_text(self):
        from pipeline_scheduler import schedule
        from pipeline_report import gantt_to_text

        ops = [
            self._make_op(op_id="dma", unit="DMA", op_kind="DMA_LOAD",
                          latency_ns=200.0, output_vmem=["buf"]),
            self._make_op(op_id="vpu", unit="VPU", latency_ns=100.0),
        ]
        sched = schedule(ops)
        text = gantt_to_text(sched)
        self.assertIn("DMA", text)
        self.assertIn("VPU", text)
        self.assertIn("dma", text)

    def test_gantt_mermaid(self):
        from pipeline_scheduler import schedule
        from pipeline_report import gantt_to_mermaid

        ops = [
            self._make_op(op_id="dma", unit="DMA", op_kind="DMA_LOAD",
                          latency_ns=200.0, output_vmem=["buf"]),
            self._make_op(op_id="vpu", unit="VPU", latency_ns=100.0),
        ]
        sched = schedule(ops)
        mermaid = gantt_to_mermaid(sched)
        self.assertIn("gantt", mermaid)
        self.assertIn("DMA", mermaid)


class TestVPRReport(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_vpr_heatmap_text(self):
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness
        from pipeline_report import vpr_heatmap_to_text

        ops = [
            self._make_op(op_id="w", output_vprs=[0, 1], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0, 1], latency_ns=50.0),
        ]
        sched = schedule(ops)
        occ = analyze_vpr_liveness(ops, sched)
        text = vpr_heatmap_to_text(occ, sched.total_latency_ns)
        self.assertIn("VPR", text)
        self.assertIn("peak", text.lower())

    def test_vpr_json(self):
        import json
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness
        from pipeline_report import vpr_to_json

        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        sched = schedule(ops)
        occ = analyze_vpr_liveness(ops, sched)
        data = json.loads(vpr_to_json(occ))
        self.assertIn("liveness", data)
        self.assertIn("peak_concurrent", data)


class TestSuggestReport(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_suggest_text(self):
        from pipeline_report import suggest_to_text

        ops = [
            self._make_op(op_id="a", unit="VPU", output_vprs=[0],
                          latency_ns=100.0),
            self._make_op(op_id="b", unit="MXU", op_kind="MXU",
                          input_vprs=[0], latency_ns=200.0),
            self._make_op(op_id="c", unit="DMA", op_kind="DMA_LOAD",
                          latency_ns=50.0, output_vmem=["buf"]),
        ]
        text = suggest_to_text(ops)
        self.assertIn("original", text.lower())


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline_report'`

**Step 3: Write the implementation**

Create `pipeline_report.py`:

```python
#!/usr/bin/env python3
"""Report generation for pipeline scheduling analysis (text/JSON/Mermaid)."""

from __future__ import annotations

import json
from dependency_analyzer import DependencyGraph, analyze_dependencies
from pipeline_scheduler import ScheduleResult, schedule
from vpr_analyzer import VPROccupancy, analyze_vpr_liveness
from pipeline_ir import PipelineOp


# ── Dependency Graph Reports ──

def deps_to_text(graph: DependencyGraph) -> str:
    lines: list[str] = []
    lines.append("=== Data Dependency Graph ===")
    lines.append("")
    lines.append(f"{'From':<15} {'To':<15} {'Hazard':<6} {'Resource':<8} {'ID'}")
    lines.append("-" * 60)
    for e in graph.edges:
        lines.append(
            f"{e.from_op:<15} {e.to_op:<15} {e.hazard_type:<6} "
            f"{e.resource_type:<8} {e.resource_id}"
        )
    lines.append("")
    lines.append(f"Total edges: {len(graph.edges)}")
    return "\n".join(lines)


def deps_to_json(graph: DependencyGraph) -> str:
    payload = {
        "edges": [
            {
                "from_op": e.from_op,
                "to_op": e.to_op,
                "hazard_type": e.hazard_type,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
            }
            for e in graph.edges
        ],
        "total_edges": len(graph.edges),
    }
    return json.dumps(payload, indent=2)


def deps_to_mermaid(graph: DependencyGraph) -> str:
    lines: list[str] = []
    lines.append("graph TD")
    for op in graph.ops:
        label = op.label or op.op_id
        lines.append(f"    {op.op_id}[\"{op.op_id}: {label}\"]")
    for e in graph.edges:
        if e.hazard_type == "RAW":
            arrow = f"-->|RAW {e.resource_id}|"
        elif e.hazard_type == "WAR":
            arrow = f"-.->|WAR {e.resource_id}|"
        else:  # WAW
            arrow = f"==>|WAW {e.resource_id}|"
        lines.append(f"    {e.from_op} {arrow} {e.to_op}")
    return "\n".join(lines)


# ── Pipeline Gantt Reports ──

def gantt_to_text(sched: ScheduleResult) -> str:
    lines: list[str] = []
    lines.append("=== Pipeline Gantt ===")
    lines.append("")
    total = sched.total_latency_ns
    # Group entries by unit
    by_unit: dict[str, list] = {"DMA": [], "MXU": [], "VPU": []}
    for e in sched.entries:
        by_unit.setdefault(e.unit, []).append(e)

    width = 60
    for unit in ["DMA", "MXU", "VPU"]:
        entries = by_unit.get(unit, [])
        if not entries:
            bar = "·" * width
        else:
            bar = list("·" * width)
            for e in entries:
                s = int(e.start_ns / total * width) if total > 0 else 0
                f = max(s + 1, int(e.end_ns / total * width) if total > 0 else 1)
                for i in range(s, min(f, width)):
                    bar[i] = "█"
            bar = "".join(bar)
        lines.append(f"{unit:>4} |{bar}| {total:.0f}ns")

    lines.append("")
    lines.append(f"{'Op':<15} {'Unit':<5} {'Start':>8} {'End':>8} "
                 f"{'Stall':>8} {'Wait'}")
    lines.append("-" * 65)
    for e in sched.entries:
        lines.append(
            f"{e.op_id:<15} {e.unit:<5} {e.start_ns:>8.0f} {e.end_ns:>8.0f} "
            f"{e.stall_ns:>8.0f} {e.wait_reason}"
        )
    lines.append("")
    lines.append(f"Total latency: {total:.0f}ns  "
                 f"Total stall: {sched.stall_total_ns:.0f}ns")
    return "\n".join(lines)


def gantt_to_mermaid(sched: ScheduleResult) -> str:
    lines: list[str] = []
    lines.append("gantt")
    lines.append("    dateFormat X")
    lines.append("    axisFormat %s ns")
    for unit in ["DMA", "MXU", "VPU"]:
        entries = [e for e in sched.entries if e.unit == unit]
        if not entries:
            continue
        lines.append(f"    section {unit}")
        for e in entries:
            crit = "crit, " if e.stall_ns > 0 else ""
            lines.append(
                f"    {e.op_id} :{crit}{int(e.start_ns)}, {int(e.end_ns)}"
            )
    return "\n".join(lines)


# ── VPR Heatmap Reports ──

def vpr_heatmap_to_text(occ: VPROccupancy, total_ns: float) -> str:
    lines: list[str] = []
    lines.append("=== VPR Occupancy Heatmap ===")
    lines.append("")

    # Determine which VPRs are used
    used_vprs = sorted(set(lv.vpr_id for lv in occ.liveness))
    if not used_vprs:
        lines.append("No VPRs used.")
        return "\n".join(lines)

    n_cols = 40
    step = total_ns / n_cols if total_ns > 0 else 1

    # Header
    lines.append(f"{'VPR':>6}  " + "".join(
        f"{int(i * step):>4}" if i % 10 == 0 else "    "
        for i in range(0, n_cols, 1)
    )[:n_cols])

    for vpr_id in used_vprs:
        lv = next(l for l in occ.liveness if l.vpr_id == vpr_id)
        bar: list[str] = []
        for col in range(n_cols):
            t = col * step
            if lv.live_start_ns <= t < lv.live_end_ns:
                bar.append("█")
            else:
                bar.append("·")
        lines.append(f"VPR[{vpr_id:>2}] {''.join(bar)}")

    lines.append("")
    lines.append(f"Peak concurrent VPRs: {occ.peak_concurrent}/32 "
                 f"at t={occ.peak_time_ns:.0f}ns")
    lines.append(f"Utilization ratio: {occ.utilization_ratio:.2%}")
    for w in occ.pressure_warnings:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def vpr_to_json(occ: VPROccupancy) -> str:
    payload = {
        "liveness": [
            {
                "vpr_id": lv.vpr_id,
                "defined_by": lv.defined_by,
                "last_used_by": lv.last_used_by,
                "live_start_ns": lv.live_start_ns,
                "live_end_ns": lv.live_end_ns,
            }
            for lv in occ.liveness
        ],
        "peak_concurrent": occ.peak_concurrent,
        "peak_time_ns": occ.peak_time_ns,
        "utilization_ratio": occ.utilization_ratio,
        "pressure_warnings": occ.pressure_warnings,
    }
    return json.dumps(payload, indent=2)


# ── Optimal Reorder Suggestion ──

def suggest_to_text(ops: list[PipelineOp]) -> str:
    lines: list[str] = []
    lines.append("=== Reorder Suggestion ===")
    lines.append("")

    # Original schedule
    orig_sched = schedule(ops)
    orig_occ = analyze_vpr_liveness(ops, orig_sched)

    lines.append("--- Original Order ---")
    lines.append(f"  Order: {' → '.join(e.op_id for e in orig_sched.entries)}")
    lines.append(f"  Total latency: {orig_sched.total_latency_ns:.0f}ns")
    lines.append(f"  Total stall: {orig_sched.stall_total_ns:.0f}ns")
    lines.append(f"  Peak VPRs: {orig_occ.peak_concurrent}")

    # Try critical-path-first reorder
    dep_graph = analyze_dependencies(ops)
    topo = dep_graph.topo_sort()
    # The greedy scheduler already uses topo order, so the "suggestion" is
    # the same schedule. For a meaningful suggestion, we'd need to try
    # permutations within the topo flexibility. For now, report the analysis.
    lines.append("")
    lines.append("--- Analysis ---")
    lines.append(f"  Critical path: {' → '.join(orig_sched.critical_path)}")
    crit_latency = sum(
        orig_sched.entries_by_id[op_id].end_ns -
        orig_sched.entries_by_id[op_id].start_ns
        for op_id in orig_sched.critical_path
    )
    lines.append(f"  Critical path latency: {crit_latency:.0f}ns")
    parallelism = crit_latency / orig_sched.total_latency_ns if orig_sched.total_latency_ns > 0 else 0
    lines.append(f"  Parallelism efficiency: {parallelism:.2%}")

    return "\n".join(lines)


def suggest_to_json(ops: list[PipelineOp]) -> str:
    orig_sched = schedule(ops)
    orig_occ = analyze_vpr_liveness(ops, orig_sched)

    payload = {
        "original": {
            "order": [e.op_id for e in orig_sched.entries],
            "total_latency_ns": orig_sched.total_latency_ns,
            "stall_total_ns": orig_sched.stall_total_ns,
            "peak_vprs": orig_occ.peak_concurrent,
            "critical_path": orig_sched.critical_path,
        },
    }
    return json.dumps(payload, indent=2)
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_report.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_report.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_report.py
git commit -m "feat(tpu-pipeline-scheduler): add report generator (text/JSON/Mermaid)"
```

---

### Task 6: CLI Entry Point + Example + Integration Test

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_ir_cli.py`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/examples/flash_attention_tile.json`
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_integration.py`

**Step 1: Create the example JSON**

Create `examples/flash_attention_tile.json`:

```json
{
  "name": "flash_attention_tile",
  "hw": "v7x",
  "ops": [
    {
      "op_id": "load_q",
      "op_kind": "DMA_LOAD",
      "input_vprs": [],
      "output_vprs": [],
      "input_vmem": [],
      "output_vmem": ["q_buf"],
      "latency_ns": 200,
      "unit": "DMA",
      "label": "Load Q tile [128,128] from HBM"
    },
    {
      "op_id": "load_k",
      "op_kind": "DMA_LOAD",
      "input_vprs": [],
      "output_vprs": [],
      "input_vmem": [],
      "output_vmem": ["k_buf"],
      "latency_ns": 200,
      "unit": "DMA",
      "label": "Load K tile [128,128] from HBM"
    },
    {
      "op_id": "q_to_reg",
      "op_kind": "VMEM_TO_REG",
      "input_vprs": [],
      "output_vprs": [0, 1, 2, 3],
      "input_vmem": ["q_buf"],
      "output_vmem": [],
      "latency_ns": 10,
      "unit": "VPU",
      "label": "Q buf -> VPR[0:3]"
    },
    {
      "op_id": "k_to_reg",
      "op_kind": "VMEM_TO_REG",
      "input_vprs": [],
      "output_vprs": [4, 5, 6, 7],
      "input_vmem": ["k_buf"],
      "output_vmem": [],
      "latency_ns": 10,
      "unit": "VPU",
      "label": "K buf -> VPR[4:7]"
    },
    {
      "op_id": "mxu_qk",
      "op_kind": "MXU",
      "input_vprs": [0, 1, 2, 3, 4, 5, 6, 7],
      "output_vprs": [8, 9, 10, 11],
      "input_vmem": [],
      "output_vmem": [],
      "latency_ns": 500,
      "unit": "MXU",
      "label": "QK^T matmul -> VPR[8:11]"
    },
    {
      "op_id": "softmax",
      "op_kind": "VPU",
      "input_vprs": [8, 9, 10, 11],
      "output_vprs": [12, 13, 14, 15],
      "input_vmem": [],
      "output_vmem": [],
      "latency_ns": 150,
      "unit": "VPU",
      "label": "softmax(QK^T) -> VPR[12:15]"
    },
    {
      "op_id": "load_v",
      "op_kind": "DMA_LOAD",
      "input_vprs": [],
      "output_vprs": [],
      "input_vmem": [],
      "output_vmem": ["v_buf"],
      "latency_ns": 200,
      "unit": "DMA",
      "label": "Load V tile [128,128] from HBM"
    },
    {
      "op_id": "v_to_reg",
      "op_kind": "VMEM_TO_REG",
      "input_vprs": [],
      "output_vprs": [16, 17, 18, 19],
      "input_vmem": ["v_buf"],
      "output_vmem": [],
      "latency_ns": 10,
      "unit": "VPU",
      "label": "V buf -> VPR[16:19]"
    },
    {
      "op_id": "mxu_sv",
      "op_kind": "MXU",
      "input_vprs": [12, 13, 14, 15, 16, 17, 18, 19],
      "output_vprs": [20, 21, 22, 23],
      "input_vmem": [],
      "output_vmem": [],
      "latency_ns": 500,
      "unit": "MXU",
      "label": "softmax(QK^T) @ V -> VPR[20:23]"
    },
    {
      "op_id": "result_to_vmem",
      "op_kind": "REG_TO_VMEM",
      "input_vprs": [20, 21, 22, 23],
      "output_vprs": [],
      "input_vmem": [],
      "output_vmem": ["out_buf"],
      "latency_ns": 10,
      "unit": "VPU",
      "label": "VPR[20:23] -> out buf"
    },
    {
      "op_id": "store_out",
      "op_kind": "DMA_STORE",
      "input_vprs": [],
      "output_vprs": [],
      "input_vmem": ["out_buf"],
      "output_vmem": [],
      "latency_ns": 200,
      "unit": "DMA",
      "label": "Store output tile to HBM"
    }
  ]
}
```

**Step 2: Write the CLI**

Create `pipeline_ir_cli.py`:

```python
#!/usr/bin/env python3
"""CLI entry point for TPU pipeline scheduling analysis."""

import argparse
import sys

from pipeline_ir import load_spec_from_file
from dependency_analyzer import analyze_dependencies
from pipeline_scheduler import schedule
from vpr_analyzer import analyze_vpr_liveness
from pipeline_report import (
    deps_to_text, deps_to_json, deps_to_mermaid,
    gantt_to_text, gantt_to_mermaid,
    vpr_heatmap_to_text, vpr_to_json,
    suggest_to_text, suggest_to_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="TPU pipeline scheduling analysis with register-level "
                    "dependency and VPR pressure tracking.",
    )
    parser.add_argument(
        "--pipeline", required=True,
        help="Path to pipeline IR JSON file",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--show", default="all",
        help="Sections to show: deps, gantt, vpr, suggest, all "
             "(comma-separated, default: all)",
    )
    parser.add_argument(
        "--mermaid", action="store_true",
        help="Include Mermaid diagram output (text format only)",
    )
    args = parser.parse_args()

    sections = set(args.show.split(","))
    show_all = "all" in sections

    spec = load_spec_from_file(args.pipeline)
    graph = analyze_dependencies(spec.ops)
    sched = schedule(spec.ops)
    occ = analyze_vpr_liveness(spec.ops, sched)

    output_parts: list[str] = []

    if show_all or "deps" in sections:
        if args.format == "json":
            output_parts.append(deps_to_json(graph))
        else:
            output_parts.append(deps_to_text(graph))
            if args.mermaid:
                output_parts.append("")
                output_parts.append(deps_to_mermaid(graph))

    if show_all or "gantt" in sections:
        if args.format == "json":
            pass  # gantt included in schedule JSON
        else:
            output_parts.append(gantt_to_text(sched))
            if args.mermaid:
                output_parts.append("")
                output_parts.append(gantt_to_mermaid(sched))

    if show_all or "vpr" in sections:
        if args.format == "json":
            output_parts.append(vpr_to_json(occ))
        else:
            output_parts.append(vpr_heatmap_to_text(occ, sched.total_latency_ns))

    if show_all or "suggest" in sections:
        if args.format == "json":
            output_parts.append(suggest_to_json(spec.ops))
        else:
            output_parts.append(suggest_to_text(spec.ops))

    print("\n\n".join(output_parts))


if __name__ == "__main__":
    main()
```

**Step 3: Write the integration tests**

Create `test_integration.py`:

```python
#!/usr/bin/env python3
"""Integration tests for tpu-pipeline-scheduler."""

import json
import os
import subprocess
import unittest


class TestPipelineSchedulerE2E(unittest.TestCase):
    def _scripts_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _example_path(self):
        return os.path.join(self._scripts_dir(), "examples",
                            "flash_attention_tile.json")

    def test_full_pipeline_api(self):
        from pipeline_ir import load_spec_from_file
        from dependency_analyzer import analyze_dependencies
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        spec = load_spec_from_file(self._example_path())
        self.assertEqual(spec.name, "flash_attention_tile")
        self.assertEqual(len(spec.ops), 11)

        graph = analyze_dependencies(spec.ops)
        self.assertGreater(len(graph.edges), 0)

        sched = schedule(spec.ops)
        self.assertGreater(sched.total_latency_ns, 0)
        self.assertGreater(len(sched.entries), 0)

        occ = analyze_vpr_liveness(spec.ops, sched)
        self.assertGreater(occ.peak_concurrent, 0)

    def test_cli_text_output(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "text",
                "--show", "all",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Data Dependency Graph", result.stdout)
        self.assertIn("Pipeline Gantt", result.stdout)
        self.assertIn("VPR Occupancy", result.stdout)
        self.assertIn("Reorder Suggestion", result.stdout)

    def test_cli_json_deps(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "json",
                "--show", "deps",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertIn("edges", data)
        self.assertGreater(len(data["edges"]), 0)

    def test_cli_mermaid_output(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "text",
                "--show", "deps,gantt",
                "--mermaid",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("graph TD", result.stdout)
        self.assertIn("gantt", result.stdout)

    def test_cli_single_section(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "text",
                "--show", "vpr",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("VPR", result.stdout)
        self.assertNotIn("Gantt", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

**Step 4: Run all tests**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest -v`
Expected: All tests PASS (unit + integration)

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_ir_cli.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/examples/ \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_integration.py
git commit -m "feat(tpu-pipeline-scheduler): add CLI entry point, example, and integration tests"
```

---

### Task 7: SKILL.md + Plugin Manifest Update

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/SKILL.md`
- Modify: `plugins/tpu-perf-model/.claude-plugin/plugin.json`

**Step 1: Write SKILL.md**

Create `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/SKILL.md`:

```markdown
---
name: tpu-pipeline-scheduler
description: >
  Use when analyzing register-level pipeline scheduling for TPU v7x kernels.
  Trigger when the user asks about instruction-level pipeline analysis,
  VPR register pressure, data hazard detection (RAW/WAR/WAW),
  or optimal instruction ordering for TPU pipelines.
---

# TPU Pipeline Scheduler

Analyze register-level pipeline scheduling for TPU v7x kernels. Given an explicit sequence of instructions with VPR assignments, this skill detects data hazards, schedules across hardware units, analyzes VPR pressure, and suggests optimal ordering.

## When to Use

- Designing optimal instruction interleaving for a Pallas kernel tile
- Analyzing VPR register pressure to determine if a tiling strategy is feasible
- Identifying data dependency bottlenecks (RAW/WAR/WAW hazards)
- Comparing alternative instruction orderings for pipeline efficiency

## Input Format: Pipeline IR

The input is a JSON file describing a sequence of hardware instructions with explicit VPR assignments:

```json
{
  "name": "kernel_tile_name",
  "hw": "v7x",
  "ops": [
    {
      "op_id": "unique_name",
      "op_kind": "DMA_LOAD | DMA_STORE | MXU | VPU | VMEM_TO_REG | REG_TO_VMEM",
      "input_vprs": [0, 1, 2, 3],
      "output_vprs": [4, 5, 6, 7],
      "input_vmem": ["slot_name"],
      "output_vmem": ["slot_name"],
      "latency_ns": 500,
      "unit": "DMA | MXU | VPU",
      "label": "Human-readable description"
    }
  ]
}
```

### Fields

| Field | Description |
|-------|-------------|
| `op_id` | Unique instruction identifier |
| `op_kind` | Instruction type (DMA_LOAD, DMA_STORE, MXU, VPU, VMEM_TO_REG, REG_TO_VMEM) |
| `input_vprs` | VPR numbers read (0-31) |
| `output_vprs` | VPR numbers written (0-31) |
| `input_vmem` | VMEM slot names read |
| `output_vmem` | VMEM slot names written |
| `latency_ns` | Instruction latency in nanoseconds |
| `unit` | Execution unit (DMA, MXU, VPU) |
| `label` | Optional human-readable description |

### TPU v7x Hardware Reference

- 32 VPRs (Vector Pipeline Registers), 4 KiB each
- 3 execution units: DMA, MXU, VPU — each runs one instruction at a time
- Dual MXU at 2307 TFLOPS BF16
- 64 MiB VMEM, 192 GB HBM at 3690 GB/s

## CLI Usage

```bash
# All analyses (text)
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show all

# Dependency graph only (JSON)
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format json --show deps

# Gantt + Mermaid diagrams
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show deps,gantt --mermaid

# VPR pressure only
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show vpr

# Reorder suggestion
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show suggest
```

### CLI Options

| Flag | Values | Description |
|------|--------|-------------|
| `--pipeline` | path | Pipeline IR JSON file (required) |
| `--format` | text, json | Output format (default: text) |
| `--show` | deps, gantt, vpr, suggest, all | Sections to show (comma-separated, default: all) |
| `--mermaid` | flag | Include Mermaid diagrams (text format only) |

## Output Sections

### 1. Data Dependency Graph

Detects three types of data hazards:

| Hazard | Condition | Impact |
|--------|-----------|--------|
| **RAW** (Read-After-Write) | Op B reads VPR[n] that Op A writes | True dependency — B must wait for A |
| **WAR** (Write-After-Read) | Op B writes VPR[n] that Op A reads | Anti-dependency — B can't overwrite before A reads |
| **WAW** (Write-After-Write) | Op B writes VPR[n] that Op A writes | Output dependency — ordering must be preserved |

Same analysis applies to VMEM slots. Transitive reduction is applied to keep the DAG minimal.

Mermaid output uses: solid arrows for RAW, dashed for WAR, dotted for WAW.

### 2. Pipeline Gantt

Shows each hardware unit's timeline with instruction placement and stall markers. Each instruction reports:
- Start/end time in ns
- Wait reason: NONE, WAIT_DATA (blocked on dependency), WAIT_UNIT (unit busy)
- Stall duration

### 3. VPR Occupancy Heatmap

ASCII grid showing which VPRs are live at each time step. Reports:
- Peak concurrent VPR count and when it occurs
- Utilization ratio (average live VPRs / 32)
- Pressure warnings when >75% VPRs are simultaneously live

### 4. Reorder Suggestion

Compares original instruction ordering against analysis:
- Critical path identification and latency
- Parallelism efficiency (critical path / total latency)
- Stall breakdown

## Workflow

1. **Decompose** your kernel tile into Pipeline IR instructions
2. **Assign VPRs** explicitly — this is where the design happens
3. **Run analysis** to identify hazards, stalls, and pressure points
4. **Iterate** on VPR assignments and instruction ordering
5. **Validate** that peak VPR pressure stays within hardware limits (32 VPRs)

## Output Language

Narrative text in Chinese, technical terms (VPR, RAW, WAR, WAW, DMA, MXU, VPU, VMEM, HBM) in English.

## Example

See `scripts/examples/flash_attention_tile.json` for a complete Flash Attention tile decomposition with 11 instructions across DMA/MXU/VPU units using VPR[0:23].
```

**Step 2: Update plugin.json**

Read current `plugin.json`, then add the new skill.

The current `plugin.json` should be read first. Expected content is minimal:
```json
{
  "name": "tpu-perf-model",
  "description": "Theoretical TPU v7x performance modeling via Register/VMEM/HBM data flow simulation",
  "version": "1.0.0"
}
```

Update to:
```json
{
  "name": "tpu-perf-model",
  "description": "Theoretical TPU v7x performance modeling via Register/VMEM/HBM data flow simulation",
  "version": "1.1.0",
  "skills": ["tpu-perf-model", "tpu-pipeline-scheduler"]
}
```

**Step 3: Run full test suite**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/SKILL.md \
       plugins/tpu-perf-model/.claude-plugin/plugin.json
git commit -m "feat(tpu-pipeline-scheduler): add SKILL.md and update plugin manifest"
```

---

### Task 8: Final Verification

**Step 1: Run the complete test suite across both skills**

```bash
cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest -v
cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest -v
```

Expected: All tests pass in both skills.

**Step 2: Run the CLI end-to-end with the example**

```bash
cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts
python pipeline_ir_cli.py --pipeline examples/flash_attention_tile.json --format text --show all --mermaid
```

Expected: Full output with all 4 sections + Mermaid diagrams.

**Step 3: Verify JSON output round-trips**

```bash
python pipeline_ir_cli.py --pipeline examples/flash_attention_tile.json --format json --show deps | python -m json.tool
python pipeline_ir_cli.py --pipeline examples/flash_attention_tile.json --format json --show vpr | python -m json.tool
```

Expected: Valid JSON output for each section.

---

Plan complete and saved to `docs/plans/2026-04-16-tpu-pipeline-scheduler-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
