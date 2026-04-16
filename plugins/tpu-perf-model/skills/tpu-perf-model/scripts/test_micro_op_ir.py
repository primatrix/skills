#!/usr/bin/env python3
"""Tests for the micro-op IR."""
import unittest


class TestMicroOpIR(unittest.TestCase):
    def test_micro_op_graph_tracks_roots_and_leaves(self):
        from micro_op_ir import TensorFragment, MicroOp, MicroOpGraph

        q_fragment = TensorFragment(
            fragment_id="q_0_0",
            tensor_name="Q",
            step_name="qk_matmul",
            shape=(128, 128),
            dtype="bf16",
            size_bytes=128 * 128 * 2,
            home_level="HBM",
        )
        load_q = MicroOp(
            op_id="load_q",
            step_name="qk_matmul",
            op_kind="dma_load_hbm_to_vmem",
            depends_on=[],
            input_fragments=[],
            output_fragments=["q_0_0"],
            required_units=("DMA",),
            required_vmem_slots=("slot_q",),
            required_reg_groups=(),
            latency_ns=18.0,
        )
        graph = MicroOpGraph(
            fragments={"q_0_0": q_fragment},
            micro_ops={"load_q": load_q},
        )

        self.assertEqual(graph.root_ops(), ["load_q"])
        self.assertEqual(graph.leaf_ops(), ["load_q"])

    def test_tensor_fragment_has_vpr_count(self):
        from micro_op_ir import TensorFragment
        frag = TensorFragment(
            fragment_id="q_tile0_reg",
            tensor_name="Q",
            step_name="matmul",
            shape=(128, 128),
            dtype="bf16",
            size_bytes=128 * 128 * 2,
            home_level="REG",
            vpr_count=8,
        )
        self.assertEqual(frag.vpr_count, 8)

    def test_tensor_fragment_vpr_count_defaults_to_zero(self):
        from micro_op_ir import TensorFragment
        frag = TensorFragment(
            fragment_id="q_hbm",
            tensor_name="Q",
            step_name="matmul",
            shape=(128, 128),
            dtype="bf16",
            size_bytes=128 * 128 * 2,
            home_level="HBM",
        )
        self.assertEqual(frag.vpr_count, 0)


if __name__ == "__main__":
    unittest.main()
