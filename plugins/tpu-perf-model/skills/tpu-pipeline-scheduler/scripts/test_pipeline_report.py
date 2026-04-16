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
        return [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]

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
