#!/usr/bin/env python3
"""Tests for tiling_optimizer module."""
import unittest

class TestTilingOptimizer(unittest.TestCase):
    def test_matmul_tiling_fits_vmem(self):
        from tiling_optimizer import find_optimal_tiling
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        step = ComputeStep(
            name="matmul", op_type="matmul",
            inputs=[TensorRef("A", (4096, 4096), "bf16"), TensorRef("B", (4096, 4096), "bf16")],
            outputs=[TensorRef("C", (4096, 4096), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 4096, "N": 4096, "K": 4096},
            compute_unit="MXU", fusable_with_prev=False,
        )
        result = find_optimal_tiling(step, TPU_V7X)
        self.assertLessEqual(result.vmem_usage_bytes, TPU_V7X.vmem_capacity_bytes)
        for dim_val in result.block_dims.values():
            self.assertEqual(dim_val % TPU_V7X.alignment, 0)

    def test_tiling_prefers_double_buffer(self):
        from tiling_optimizer import find_optimal_tiling
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        step = ComputeStep(
            name="small_matmul", op_type="matmul",
            inputs=[TensorRef("A", (512, 256), "bf16"), TensorRef("B", (256, 512), "bf16")],
            outputs=[TensorRef("C", (512, 512), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 512, "N": 512, "K": 256},
            compute_unit="MXU", fusable_with_prev=False,
        )
        result = find_optimal_tiling(step, TPU_V7X)
        self.assertTrue(result.double_buffer)

    def test_tiling_report_includes_pipeline_balance(self):
        from tiling_optimizer import find_optimal_tiling_with_analysis
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X
        step = ComputeStep(
            name="matmul", op_type="matmul",
            inputs=[TensorRef("A", (2048, 1024), "bf16"), TensorRef("B", (1024, 2048), "bf16")],
            outputs=[TensorRef("C", (2048, 2048), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 2048, "N": 2048, "K": 1024},
            compute_unit="MXU", fusable_with_prev=False,
        )
        analysis = find_optimal_tiling_with_analysis(step, TPU_V7X)
        self.assertIn("dma_time_per_tile_ns", analysis)
        self.assertIn("compute_time_per_tile_ns", analysis)
        self.assertIn("pipeline_balance_ratio", analysis)

if __name__ == "__main__":
    unittest.main()
