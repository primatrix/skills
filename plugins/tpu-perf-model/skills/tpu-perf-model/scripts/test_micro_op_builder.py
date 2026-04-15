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

    def test_fused_elementwise_reuses_previous_fragment(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_pipeline
        from pipeline_simulator import TileConfig

        matmul = ComputeStep(
            name="qk_matmul",
            op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        scale = ComputeStep(
            name="scale_scores",
            op_type="elementwise",
            inputs=[TensorRef("S", (128, 128), "bf16")],
            outputs=[TensorRef("S_scaled", (128, 128), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 128, "N": 128},
            compute_unit="VPU",
            fusable_with_prev=True,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=1,
            tile_input_bytes=128 * 128 * 2 * 2,
            tile_output_bytes=128 * 128 * 2,
            double_buffer=False,
            vmem_usage_bytes=128 * 128 * 2 * 3,
        )

        graph = build_micro_op_graph_for_pipeline(
            [matmul, scale],
            {"qk_matmul": tile, "scale_scores": tile},
        )
        op_kinds = [op.op_kind for op in graph.micro_ops.values()]

        self.assertIn("vpu_compute", op_kinds)
        self.assertEqual(op_kinds.count("dma_load_hbm_to_vmem"), 2)
        self.assertEqual(op_kinds.count("dma_store_vmem_to_hbm"), 1)


if __name__ == "__main__":
    unittest.main()
