#!/usr/bin/env python3
"""Tests for pipeline_plot module."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestBuildVPRActivity(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_write_interval(self):
        """Op that writes VPR[0] produces a 'write' interval on VPR 0."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0)]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        self.assertIn(0, activity)
        intervals = activity[0]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].unit, "VPU")
        self.assertEqual(intervals[0].access, "write")
        self.assertAlmostEqual(intervals[0].start_ns, 0.0)
        self.assertAlmostEqual(intervals[0].end_ns, 100.0)

    def test_read_interval(self):
        """Op that reads VPR[0] produces a 'read' interval on VPR 0."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        intervals = activity[0]
        writes = [i for i in intervals if i.access == "write"]
        reads = [i for i in intervals if i.access == "read"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].unit, "VPU")

    def test_live_gap_filled(self):
        """VPR that is live between write-end and read-start gets a 'live' interval."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        # w writes VPR[0] on VPU [0,100]. gap occupies VPU [100,300].
        # r reads VPR[0] on VPU [300,350]. Live gap should exist [100,300].
        ops = [
            self._make_op(op_id="w", output_vprs=[0], unit="VPU", latency_ns=100.0),
            self._make_op(op_id="gap", output_vprs=[1], unit="VPU", latency_ns=200.0),
            self._make_op(op_id="r", input_vprs=[0], unit="VPU", latency_ns=50.0),
        ]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        lives = [i for i in activity[0] if i.access == "live"]
        self.assertGreater(len(lives), 0, "Should have a live interval in the gap")

    def test_multiple_vprs(self):
        """Activity dict has entries for all touched VPRs."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="w", output_vprs=[0, 1, 2], latency_ns=100.0),
        ]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        self.assertIn(0, activity)
        self.assertIn(1, activity)
        self.assertIn(2, activity)


class TestPlotVPRTimeline(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_plot_creates_png(self):
        """plot_vpr_timeline produces a valid PNG file."""
        import tempfile
        from pipeline_plot import plot_vpr_timeline
        from pipeline_scheduler import schedule
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="load_q", unit="DMA", op_kind="DMA_LOAD",
                          output_vmem=["q"], latency_ns=200.0),
            self._make_op(op_id="q_reg", unit="VPU", op_kind="VMEM_TO_REG",
                          input_vmem=["q"], output_vprs=[0, 1], latency_ns=10.0),
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          input_vprs=[0, 1], output_vprs=[2, 3], latency_ns=500.0),
            self._make_op(op_id="vpu", unit="VPU", op_kind="VPU",
                          input_vprs=[2, 3], output_vprs=[4, 5], latency_ns=100.0),
        ]
        sched = schedule(ops)
        graph = analyze_dependencies(ops)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name

        try:
            plot_vpr_timeline(ops, sched, graph, out_path, title="test_kernel")
            with open(out_path, "rb") as f:
                header = f.read(8)
            # PNG magic bytes
            self.assertEqual(header[:4], b"\x89PNG")
        finally:
            os.unlink(out_path)

    def test_plot_empty_ops(self):
        """plot_vpr_timeline handles empty ops gracefully."""
        import tempfile
        from pipeline_plot import plot_vpr_timeline
        from pipeline_scheduler import ScheduleResult
        from dependency_analyzer import DependencyGraph

        sched = ScheduleResult(entries=[], total_latency_ns=0,
                               critical_path=[], stall_total_ns=0)
        graph = DependencyGraph(ops=[], edges=[])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name

        try:
            plot_vpr_timeline([], sched, graph, out_path)
            self.assertTrue(os.path.exists(out_path))
        finally:
            os.unlink(out_path)


if __name__ == "__main__":
    unittest.main()
