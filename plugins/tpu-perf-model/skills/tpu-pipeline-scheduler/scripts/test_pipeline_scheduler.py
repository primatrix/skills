#!/usr/bin/env python3
"""Tests for event-driven pipeline scheduler."""
import unittest


class TestEventDrivenScheduler(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            weight_vprs=[], data_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_single_op(self):
        from pipeline_scheduler import schedule
        ops = [self._make_op(op_id="a", latency_ns=100.0)]
        result = schedule(ops)
        self.assertEqual(result.entries_by_id["a"].start_ns, 0.0)
        self.assertEqual(result.entries_by_id["a"].end_ns, 100.0)
        self.assertEqual(result.total_latency_ns, 100.0)

    def test_sequential_same_unit(self):
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="a", unit="VPU", latency_ns=100.0),
            self._make_op(op_id="b", unit="VPU", latency_ns=50.0),
        ]
        result = schedule(ops)
        self.assertEqual(result.entries_by_id["b"].start_ns, 100.0)

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
            self._make_op(op_id="r", unit="MXU", op_kind="MXU",
                          weight_vprs=[0], data_vprs=[],
                          output_vprs=[1], latency_ns=50.0),
        ]
        result = schedule(ops)
        self.assertGreaterEqual(result.entries_by_id["r"].start_ns, 100.0)

    def test_dual_mxu_weight_then_data(self):
        """MXU op should show weight phase before data phase."""
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="mxu_qk", unit="MXU", op_kind="MXU",
                          weight_vprs=[0, 1], data_vprs=[2, 3],
                          output_vprs=[4, 5], latency_ns=500.0),
        ]
        result = schedule(ops)
        entry = result.entries_by_id["mxu_qk"]
        self.assertTrue(hasattr(entry, 'phases'))
        self.assertEqual(len(entry.phases), 2)
        self.assertEqual(entry.phases[0].phase_type, "weight")
        self.assertEqual(entry.phases[1].phase_type, "data")
        self.assertLess(entry.phases[0].start_ns, entry.phases[1].start_ns)

    def test_mxu_pipeline_overlap(self):
        """Two consecutive MXU ops: second weight can overlap with first data."""
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="mxu1", unit="MXU", op_kind="MXU",
                          weight_vprs=[0, 1], data_vprs=[2, 3],
                          output_vprs=[4, 5], latency_ns=500.0),
            self._make_op(op_id="mxu2", unit="MXU", op_kind="MXU",
                          weight_vprs=[6, 7], data_vprs=[8, 9],
                          output_vprs=[10, 11], latency_ns=500.0),
        ]
        result = schedule(ops)
        e1 = result.entries_by_id["mxu1"]
        e2 = result.entries_by_id["mxu2"]
        # mxu2 weight phase can start while mxu1 data phase is running
        self.assertLess(e2.phases[0].start_ns, e1.end_ns)

    def test_fusion_zero_delay(self):
        """Cross-unit fusion pair: consumer starts immediately after producer."""
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          weight_vprs=[0], data_vprs=[1],
                          output_vprs=[2, 3], latency_ns=500.0),
            self._make_op(op_id="vpu", unit="VPU", op_kind="VPU",
                          input_vprs=[2, 3], output_vprs=[4, 5],
                          latency_ns=100.0),
        ]
        result = schedule(ops)
        mxu_end = result.entries_by_id["mxu"].end_ns
        vpu_start = result.entries_by_id["vpu"].start_ns
        self.assertEqual(vpu_start, mxu_end)
        self.assertIn(("mxu", "vpu"), result.fusion_pairs)

    def test_schedule_result_has_critical_path(self):
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="a", unit="VPU", latency_ns=100.0,
                          output_vprs=[0]),
            self._make_op(op_id="b", unit="MXU", op_kind="MXU",
                          weight_vprs=[0], data_vprs=[],
                          output_vprs=[1], latency_ns=50.0),
        ]
        result = schedule(ops)
        self.assertIn("a", result.critical_path)

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


if __name__ == "__main__":
    unittest.main()
