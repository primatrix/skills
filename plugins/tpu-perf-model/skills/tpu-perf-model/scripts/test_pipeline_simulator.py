#!/usr/bin/env python3
"""Tests for pipeline_simulator module."""
import unittest


class TestMicroOpTiming(unittest.TestCase):
    def test_dma_load_time(self):
        from pipeline_simulator import calc_dma_time_ns
        from hw_params import TPU_V7X
        bytes_ = 1 * 1024**2
        t_ns = calc_dma_time_ns(bytes_, TPU_V7X)
        expected_ns = bytes_ / TPU_V7X.hbm_bw_bytes_per_sec * 1e9
        self.assertAlmostEqual(t_ns, expected_ns, places=1)

    def test_mxu_compute_time(self):
        from pipeline_simulator import calc_mxu_time_ns
        from hw_params import TPU_V7X
        flops = 2 * 128 * 128 * 128
        t_ns = calc_mxu_time_ns(flops, TPU_V7X)
        expected_ns = flops / TPU_V7X.mxu_peak_flops * 1e9
        self.assertAlmostEqual(t_ns, expected_ns, places=1)


class TestSingleStepSimulation(unittest.TestCase):
    def test_matmul_step_result(self):
        from pipeline_simulator import simulate_step
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        step = ComputeStep(
            name="matmul", op_type="matmul",
            inputs=[TensorRef("A", (1024, 512), "bf16"), TensorRef("B", (512, 1024), "bf16")],
            outputs=[TensorRef("C", (1024, 1024), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 1024, "N": 1024, "K": 512},
            compute_unit="MXU", fusable_with_prev=False,
        )
        result = simulate_step(step, TPU_V7X)
        self.assertGreater(result.t_compute_ns, 0)
        self.assertGreater(result.t_hbm_ns, 0)
        self.assertIn(result.bottleneck, ("HBM_BW", "COMPUTE"))
        self.assertGreater(result.t_step_ns, 0)

    def test_elementwise_is_memory_bound(self):
        from pipeline_simulator import simulate_step
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        step = ComputeStep(
            name="add", op_type="elementwise",
            inputs=[TensorRef("A", (4096, 4096), "bf16")],
            outputs=[TensorRef("B", (4096, 4096), "bf16")],
            flops_formula="M*N", flops_vars={"M": 4096, "N": 4096},
            compute_unit="VPU", fusable_with_prev=False,
        )
        result = simulate_step(step, TPU_V7X)
        self.assertEqual(result.bottleneck, "HBM_BW")


class TestPipelineSchedule(unittest.TestCase):
    def test_double_buffer_overlaps(self):
        from pipeline_simulator import simulate_step
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        step = ComputeStep(
            name="matmul", op_type="matmul",
            inputs=[TensorRef("A", (4096, 128), "bf16"), TensorRef("B", (128, 4096), "bf16")],
            outputs=[TensorRef("C", (4096, 4096), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 4096, "N": 4096, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        result = simulate_step(step, TPU_V7X)
        naive_time = result.t_hbm_ns + result.t_compute_ns
        self.assertLess(result.t_step_ns, naive_time * 0.99)

    def test_fusion_saves_hbm(self):
        from pipeline_simulator import simulate_steps
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        matmul = ComputeStep(
            name="matmul", op_type="matmul",
            inputs=[TensorRef("A", (1024, 512), "bf16"), TensorRef("B", (512, 1024), "bf16")],
            outputs=[TensorRef("C", (1024, 1024), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 1024, "N": 1024, "K": 512},
            compute_unit="MXU", fusable_with_prev=False,
        )
        scale = ComputeStep(
            name="scale", op_type="elementwise",
            inputs=[TensorRef("C", (1024, 1024), "bf16")],
            outputs=[TensorRef("C_scaled", (1024, 1024), "bf16")],
            flops_formula="M*N", flops_vars={"M": 1024, "N": 1024},
            compute_unit="VPU", fusable_with_prev=True,
        )
        report = simulate_steps([matmul, scale], TPU_V7X)
        self.assertGreater(report.fusion_savings_bytes, 0)


if __name__ == "__main__":
    unittest.main()
