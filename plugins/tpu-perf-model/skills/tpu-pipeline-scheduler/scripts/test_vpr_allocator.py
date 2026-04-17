#!/usr/bin/env python3
"""Tests for VPR auto-allocator."""
import unittest


class TestVPRAllocator(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            weight_vprs=[], data_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_allocates_within_32_vprs(self):
        from vpr_allocator import allocate_vprs
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="w0", output_vprs=[100], latency_ns=100.0),
            self._make_op(op_id="r0", input_vprs=[100], output_vprs=[101], latency_ns=50.0),
        ]
        sched = schedule(ops)
        allocated_ops = allocate_vprs(ops, sched)
        for op in allocated_ops:
            for v in op.output_vprs + op.input_vprs:
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 31)

    def test_reuses_dead_vprs(self):
        """VPR whose last reader is done should be reused."""
        from vpr_allocator import allocate_vprs
        from pipeline_scheduler import schedule
        # a writes VPR 100, b reads 100 and writes 101, c reads 101 and writes 102.
        # VPR 100 dies after b, VPR 101 dies after c.
        # Since all are VPU (sequential): a(0-100), b(100-150), c(150-200).
        # VPR 100 live 0-150, VPR 101 live 100-200, VPR 102 live 150-200.
        # VPR 100 and 102 don't overlap -> can reuse same physical register.
        ops = [
            self._make_op(op_id="a", output_vprs=[100], latency_ns=100.0),
            self._make_op(op_id="b", input_vprs=[100], output_vprs=[101], latency_ns=50.0),
            self._make_op(op_id="c", input_vprs=[101], output_vprs=[102], latency_ns=50.0),
        ]
        sched = schedule(ops)
        allocated = allocate_vprs(ops, sched)
        a_out = allocated[0].output_vprs[0]
        c_out = allocated[2].output_vprs[0]
        # VPR 100 and 102 don't overlap, so can map to same physical VPR
        self.assertEqual(a_out, c_out)

    def test_preserves_liveness_correctness(self):
        """Simultaneously-live VPRs must get different physical registers."""
        from vpr_allocator import allocate_vprs
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="w0", unit="DMA", op_kind="DMA_LOAD",
                          output_vprs=[100], latency_ns=100.0),
            self._make_op(op_id="w1", unit="VPU",
                          output_vprs=[101], latency_ns=100.0),
            self._make_op(op_id="r", unit="MXU", op_kind="MXU",
                          weight_vprs=[100], data_vprs=[101],
                          output_vprs=[102], latency_ns=50.0),
        ]
        sched = schedule(ops)
        allocated = allocate_vprs(ops, sched)
        v100 = allocated[0].output_vprs[0]
        v101 = allocated[1].output_vprs[0]
        self.assertNotEqual(v100, v101)

    def test_fusion_pair_shares_register(self):
        """Fusion pairs must use the same physical VPR for output/input."""
        from vpr_allocator import allocate_vprs
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          weight_vprs=[100], data_vprs=[101],
                          output_vprs=[102], latency_ns=500.0),
            self._make_op(op_id="vpu", unit="VPU", op_kind="VPU",
                          input_vprs=[102], output_vprs=[103],
                          latency_ns=100.0),
        ]
        sched = schedule(ops)
        allocated = allocate_vprs(ops, sched)
        mxu_out = allocated[0].output_vprs[0]
        vpu_in = allocated[1].input_vprs[0]
        self.assertEqual(mxu_out, vpu_in)

    def test_raises_when_exceeds_32(self):
        """Should raise if >32 VPRs needed simultaneously."""
        from vpr_allocator import allocate_vprs
        from pipeline_scheduler import schedule
        ops = [
            self._make_op(op_id="w", output_vprs=list(range(100, 133)), latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=list(range(100, 133)), latency_ns=50.0),
        ]
        sched = schedule(ops)
        with self.assertRaises(ValueError):
            allocate_vprs(ops, sched)


if __name__ == "__main__":
    unittest.main()
