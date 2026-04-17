#!/usr/bin/env python3
"""Tests for HTML animation generator."""
import os, tempfile, unittest

class TestPipelineAnimate(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            weight_vprs=[], data_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_generates_html_file(self):
        from pipeline_animate import generate_animation
        from pipeline_scheduler import schedule
        from dependency_analyzer import analyze_dependencies
        ops = [
            self._make_op(op_id="load_q", unit="DMA", op_kind="DMA_LOAD",
                          output_vmem=["q"], latency_ns=200.0),
            self._make_op(op_id="q_reg", unit="VPU", op_kind="VMEM_TO_REG",
                          input_vmem=["q"], output_vprs=[0, 1], latency_ns=10.0),
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          weight_vprs=[0], data_vprs=[1],
                          output_vprs=[2, 3], latency_ns=500.0,
                          pseudocode="S = Q @ K.T"),
            self._make_op(op_id="vpu", unit="VPU", op_kind="VPU",
                          input_vprs=[2, 3], output_vprs=[4, 5],
                          latency_ns=100.0, pseudocode="P = softmax(S)"),
        ]
        sched = schedule(ops)
        graph = analyze_dependencies(ops)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out_path = f.name
        try:
            generate_animation(ops, sched, graph, out_path, title="test")
            self.assertTrue(os.path.exists(out_path))
            with open(out_path) as f:
                html = f.read()
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("svg", html.lower())
            self.assertIn("S = Q @ K.T", html)
            self.assertIn("P = softmax(S)", html)
        finally:
            os.unlink(out_path)

    def test_empty_ops_generates_valid_html(self):
        from pipeline_animate import generate_animation
        from pipeline_scheduler import ScheduleResult
        from dependency_analyzer import DependencyGraph
        sched = ScheduleResult(entries=[], total_latency_ns=0,
                               critical_path=[], stall_total_ns=0,
                               fusion_pairs=[])
        graph = DependencyGraph(ops=[], edges=[], fusion_pairs=[])
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out_path = f.name
        try:
            generate_animation([], sched, graph, out_path)
            with open(out_path) as f:
                html = f.read()
            self.assertIn("<!DOCTYPE html>", html)
        finally:
            os.unlink(out_path)

    def test_html_contains_playback_controls(self):
        from pipeline_animate import generate_animation
        from pipeline_scheduler import schedule
        from dependency_analyzer import analyze_dependencies
        ops = [self._make_op(op_id="a", output_vprs=[0], latency_ns=100.0)]
        sched = schedule(ops)
        graph = analyze_dependencies(ops)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out_path = f.name
        try:
            generate_animation(ops, sched, graph, out_path)
            with open(out_path) as f:
                html = f.read()
            self.assertIn("play", html.lower())
            self.assertIn("speed", html.lower())
        finally:
            os.unlink(out_path)

    def test_html_contains_schedule_data(self):
        from pipeline_animate import generate_animation
        from pipeline_scheduler import schedule
        from dependency_analyzer import analyze_dependencies
        ops = [
            self._make_op(op_id="mxu_qk", unit="MXU", op_kind="MXU",
                          weight_vprs=[0], data_vprs=[1],
                          output_vprs=[2], latency_ns=500.0),
        ]
        sched = schedule(ops)
        graph = analyze_dependencies(ops)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out_path = f.name
        try:
            generate_animation(ops, sched, graph, out_path)
            with open(out_path) as f:
                html = f.read()
            self.assertIn("mxu_qk", html)
            self.assertIn("weight", html.lower())
        finally:
            os.unlink(out_path)

if __name__ == "__main__":
    unittest.main()
