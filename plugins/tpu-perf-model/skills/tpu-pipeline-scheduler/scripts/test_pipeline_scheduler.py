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
