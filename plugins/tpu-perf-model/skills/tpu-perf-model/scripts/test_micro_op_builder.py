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
            [tile, tile],
        )
        op_kinds = [op.op_kind for op in graph.micro_ops.values()]

        self.assertIn("vpu_compute", op_kinds)
        self.assertEqual(op_kinds.count("dma_load_hbm_to_vmem"), 2)
        self.assertEqual(op_kinds.count("dma_store_vmem_to_hbm"), 1)

    def test_pipeline_keeps_op_ids_unique_across_steps(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_pipeline
        from pipeline_simulator import TileConfig

        first = ComputeStep(
            name="qk_matmul",
            op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        second = ComputeStep(
            name="sv_matmul",
            op_type="matmul",
            inputs=[TensorRef("S", (128, 128), "bf16"), TensorRef("V", (128, 128), "bf16")],
            outputs=[TensorRef("O", (128, 128), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
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
            [first, second],
            [tile, tile],
        )

        self.assertIn("s0_qk_matmul_load_q_tile0", graph.micro_ops)
        self.assertIn("s1_sv_matmul_load_q_tile0", graph.micro_ops)

    def test_pipeline_supports_duplicate_step_names(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_pipeline
        from pipeline_simulator import TileConfig

        first = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        second = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[TensorRef("S", (128, 128), "bf16"), TensorRef("V", (128, 128), "bf16")],
            outputs=[TensorRef("O", (128, 128), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=1,
            tile_input_bytes=128 * 128 * 2 * 2,
            tile_output_bytes=128 * 128 * 2,
            double_buffer=False,
            vmem_usage_bytes=128 * 128 * 2 * 3,
        )

        graph = build_micro_op_graph_for_pipeline([first, second], [tile, tile])

        self.assertIn("s0_matmul_load_q_tile0", graph.micro_ops)
        self.assertIn("s1_matmul_load_q_tile0", graph.micro_ops)

    def test_fused_step_uses_previous_tile_buffering_for_input_slot(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_pipeline
        from pipeline_simulator import TileConfig

        prev_step = ComputeStep(
            name="producer",
            op_type="matmul",
            inputs=[TensorRef("Q", (256, 128), "bf16"), TensorRef("K", (128, 256), "bf16")],
            outputs=[TensorRef("S", (256, 256), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 256, "N": 256, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        fused_step = ComputeStep(
            name="consumer",
            op_type="elementwise",
            inputs=[TensorRef("S", (256, 256), "bf16")],
            outputs=[TensorRef("S2", (256, 256), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 256, "N": 256},
            compute_unit="VPU",
            fusable_with_prev=True,
        )
        prev_tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=4,
            tile_input_bytes=128 * 128 * 2 * 2,
            tile_output_bytes=128 * 128 * 2,
            double_buffer=False,
            vmem_usage_bytes=128 * 128 * 2 * 3,
        )
        fused_tile = TileConfig(
            block_dims={"dim0": 16384},
            num_tiles=4,
            tile_input_bytes=16384 * 2,
            tile_output_bytes=16384 * 2,
            double_buffer=True,
            vmem_usage_bytes=16384 * 2 * 3,
        )

        graph = build_micro_op_graph_for_pipeline([prev_step, fused_step], [prev_tile, fused_tile])
        move_in = graph.micro_ops["s1_consumer_vmem_to_reg_tile1"]

        self.assertEqual(move_in.required_vmem_slots, ("out_slot0",))

    def test_unfused_dependencies_link_all_inputs(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_pipeline
        from pipeline_simulator import TileConfig

        lhs = ComputeStep(
            name="lhs",
            op_type="elementwise",
            inputs=[TensorRef("A0", (128, 128), "bf16")],
            outputs=[TensorRef("A", (128, 128), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 128, "N": 128},
            compute_unit="VPU",
            fusable_with_prev=False,
        )
        rhs = ComputeStep(
            name="rhs",
            op_type="elementwise",
            inputs=[TensorRef("B0", (128, 128), "bf16")],
            outputs=[TensorRef("B", (128, 128), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 128, "N": 128},
            compute_unit="VPU",
            fusable_with_prev=False,
        )
        matmul = ComputeStep(
            name="combine",
            op_type="matmul",
            inputs=[TensorRef("A", (128, 128), "bf16"), TensorRef("B", (128, 128), "bf16")],
            outputs=[TensorRef("C", (128, 128), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        vpu_tile = TileConfig(
            block_dims={"dim0": 16384},
            num_tiles=1,
            tile_input_bytes=16384 * 2,
            tile_output_bytes=16384 * 2,
            double_buffer=False,
            vmem_usage_bytes=16384 * 2 * 2,
        )
        mm_tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=1,
            tile_input_bytes=128 * 128 * 2 * 2,
            tile_output_bytes=128 * 128 * 2,
            double_buffer=False,
            vmem_usage_bytes=128 * 128 * 2 * 3,
        )

        graph = build_micro_op_graph_for_pipeline([lhs, rhs, matmul], [vpu_tile, vpu_tile, mm_tile])

        self.assertIn("s0_lhs_store_tile0", graph.micro_ops["s2_combine_load_q_tile0"].depends_on)
        self.assertIn("s1_rhs_store_tile0", graph.micro_ops["s2_combine_load_k_tile0"].depends_on)


    def test_matmul_graph_has_mxu_writeback_op(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
            tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
            double_buffer=False, vmem_usage_bytes=128*128*2*3,
        )
        graph = build_micro_op_graph_for_step(step, tile)
        op_kinds = [op.op_kind for op in graph.micro_ops.values()]
        self.assertIn("mxu_writeback", op_kinds)

    def test_matmul_mxu_compute_does_not_hold_acc_reg(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
            tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
            double_buffer=False, vmem_usage_bytes=128*128*2*3,
        )
        graph = build_micro_op_graph_for_step(step, tile)
        mxu_ops = [op for op in graph.micro_ops.values() if op.op_kind == "mxu_compute"]
        for op in mxu_ops:
            self.assertNotIn("acc_reg0", op.required_reg_groups)
            self.assertNotIn("acc_reg1", op.required_reg_groups)

    def test_matmul_fragments_have_vpr_counts(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
            tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
            double_buffer=False, vmem_usage_bytes=128*128*2*3,
        )
        graph = build_micro_op_graph_for_step(step, tile)
        reg_frags = [f for f in graph.fragments.values() if f.home_level == "REG"]
        for frag in reg_frags:
            self.assertGreater(frag.vpr_count, 0, f"{frag.fragment_id} has vpr_count=0")

    def test_matmul_mxu_compute_vpr_count_is_q_plus_k(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
            tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
            double_buffer=False, vmem_usage_bytes=128*128*2*3,
        )
        graph = build_micro_op_graph_for_step(step, tile)
        mxu_ops = [op for op in graph.micro_ops.values() if op.op_kind == "mxu_compute"]
        # Q=8 VPRs + K=8 VPRs = 16
        for op in mxu_ops:
            self.assertEqual(op.required_vpr_count, 16)

    def test_vpu_fragments_have_vpr_counts(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="scale", op_type="elementwise",
            inputs=[TensorRef("S", (128, 128), "bf16")],
            outputs=[TensorRef("S2", (128, 128), "bf16")],
            flops_formula="M*N", flops_vars={"M": 128, "N": 128},
            compute_unit="VPU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"dim0": 16384}, num_tiles=1,
            tile_input_bytes=16384*2, tile_output_bytes=16384*2,
            double_buffer=False, vmem_usage_bytes=16384*2*2,
        )
        graph = build_micro_op_graph_for_step(step, tile)
        reg_frags = [f for f in graph.fragments.values() if f.home_level == "REG"]
        for frag in reg_frags:
            self.assertGreater(frag.vpr_count, 0)

    def test_vpu_compute_vpr_count_is_in_plus_out(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="scale", op_type="elementwise",
            inputs=[TensorRef("S", (128, 128), "bf16")],
            outputs=[TensorRef("S2", (128, 128), "bf16")],
            flops_formula="M*N", flops_vars={"M": 128, "N": 128},
            compute_unit="VPU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"dim0": 16384}, num_tiles=1,
            tile_input_bytes=16384*2, tile_output_bytes=16384*2,
            double_buffer=False, vmem_usage_bytes=16384*2*2,
        )
        graph = build_micro_op_graph_for_step(step, tile)
        vpu_ops = [op for op in graph.micro_ops.values() if op.op_kind == "vpu_compute"]
        # 16384 elements * 2B = 32768 bytes / 4096 = 8 VPRs per tensor
        for op in vpu_ops:
            self.assertEqual(op.required_vpr_count, 16)  # in(8) + out(8)

    def test_fused_vpu_has_vpr_counts(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_pipeline
        from pipeline_simulator import TileConfig

        matmul = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
            outputs=[TensorRef("S", (128, 128), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        scale = ComputeStep(
            name="scale", op_type="elementwise",
            inputs=[TensorRef("S", (128, 128), "bf16")],
            outputs=[TensorRef("S2", (128, 128), "bf16")],
            flops_formula="M*N", flops_vars={"M": 128, "N": 128},
            compute_unit="VPU", fusable_with_prev=True,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
            tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
            double_buffer=False, vmem_usage_bytes=128*128*2*3,
        )
        graph = build_micro_op_graph_for_pipeline([matmul, scale], [tile, tile])
        fused_vpu_ops = [op for op in graph.micro_ops.values()
                         if op.op_kind == "vpu_compute" and op.step_name == "scale"]
        for op in fused_vpu_ops:
            self.assertGreater(op.required_vpr_count, 0)

    def test_calc_vpr_count_bf16_128x128(self):
        from micro_op_builder import _calc_vpr_count
        from hw_params import TPU_V7X
        # 128*128*2 = 32768 bytes / 4096 = 8 VPRs
        self.assertEqual(_calc_vpr_count(128 * 128 * 2, TPU_V7X), 8)

    def test_calc_vpr_count_f32_128x128(self):
        from micro_op_builder import _calc_vpr_count
        from hw_params import TPU_V7X
        # 128*128*4 = 65536 bytes / 4096 = 16 VPRs
        self.assertEqual(_calc_vpr_count(128 * 128 * 4, TPU_V7X), 16)

    def test_calc_vpr_count_rounds_up(self):
        from micro_op_builder import _calc_vpr_count
        from hw_params import TPU_V7X
        # 4097 bytes -> ceil(4097/4096) = 2 VPRs
        self.assertEqual(_calc_vpr_count(4097, TPU_V7X), 2)

    def test_calc_vpr_count_zero_bytes(self):
        from micro_op_builder import _calc_vpr_count
        from hw_params import TPU_V7X
        self.assertEqual(_calc_vpr_count(0, TPU_V7X), 0)


if __name__ == "__main__":
    unittest.main()
