#!/usr/bin/env python3
"""Tests for micro-op graph construction."""
import unittest


class TestMicroOpBuilder(unittest.TestCase):
    def test_matmul_expands_into_dma_reg_mxu_store_pipeline(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul",
            op_type="matmul",
            inputs=[TensorRef("Q", (256, 128), "bf16"), TensorRef("K", (128, 256), "bf16")],
            outputs=[TensorRef("S", (256, 256), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 256, "N": 256, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=4,
            tile_input_bytes=128 * 128 * 2 * 2,
            tile_output_bytes=128 * 128 * 2,
            double_buffer=True,
            vmem_usage_bytes=128 * 128 * 2 * 5,
        )

        graph = build_micro_op_graph_for_step(step, tile)
        op_kinds = [graph.micro_ops[op_id].op_kind for op_id in sorted(graph.micro_ops)]

        self.assertIn("dma_load_hbm_to_vmem", op_kinds)
        self.assertIn("vmem_to_reg", op_kinds)
        self.assertIn("mxu_compute", op_kinds)
        self.assertIn("reg_to_vmem", op_kinds)
        self.assertIn("dma_store_vmem_to_hbm", op_kinds)


if __name__ == "__main__":
    unittest.main()
