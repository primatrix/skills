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
        self.assertGreaterEqual(result.utilization_ratio, 0.0)
        self.assertLessEqual(result.utilization_ratio, 1.0)

    def test_pressure_warning_when_high(self):
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        out_vprs = list(range(28))
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
