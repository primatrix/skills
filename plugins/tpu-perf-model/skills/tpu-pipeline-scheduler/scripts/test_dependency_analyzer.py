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

        ops = [
            self._make_op(op_id="A", output_vprs=[0]),
            self._make_op(op_id="B", input_vprs=[0], output_vprs=[1]),
            self._make_op(op_id="C", input_vprs=[0, 1]),
        ]
        graph = analyze_dependencies(ops)
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
