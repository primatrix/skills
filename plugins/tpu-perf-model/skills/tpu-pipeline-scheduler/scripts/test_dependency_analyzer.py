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


class TestFusionDetection(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp

        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            weight_vprs=[], data_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_detects_cross_unit_fusion(self):
        """MXU writes VPR[0:3], VPU reads them immediately — fusion pair."""
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          output_vprs=[0, 1, 2, 3], latency_ns=500.0),
            self._make_op(op_id="vpu", unit="VPU", op_kind="VPU",
                          input_vprs=[0, 1, 2, 3], output_vprs=[4, 5],
                          latency_ns=100.0),
        ]
        graph = analyze_dependencies(ops)
        self.assertIn(("mxu", "vpu"), graph.fusion_pairs)

    def test_no_fusion_same_unit(self):
        """Same unit RAW dependency is NOT fusion."""
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="a", unit="VPU", output_vprs=[0],
                          latency_ns=100.0),
            self._make_op(op_id="b", unit="VPU", input_vprs=[0],
                          latency_ns=50.0),
        ]
        graph = analyze_dependencies(ops)
        self.assertEqual(graph.fusion_pairs, [])

    def test_no_fusion_with_intermediate_op(self):
        """If an intermediate op also reads/writes the VPR, no fusion for
        indirect successor."""
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          output_vprs=[0], latency_ns=500.0),
            self._make_op(op_id="mid", unit="VPU", input_vprs=[0],
                          output_vprs=[1], latency_ns=10.0),
            self._make_op(op_id="vpu2", unit="DMA", op_kind="DMA_STORE",
                          input_vprs=[0], latency_ns=50.0),
        ]
        graph = analyze_dependencies(ops)
        self.assertIn(("mxu", "mid"), graph.fusion_pairs)
        # vpu2 also reads VPR[0] from mxu, but mid reads it first, so
        # mxu->vpu2 is NOT fusion (mid is between them for VPR[0])
        self.assertNotIn(("mxu", "vpu2"), graph.fusion_pairs)


if __name__ == "__main__":
    unittest.main()
