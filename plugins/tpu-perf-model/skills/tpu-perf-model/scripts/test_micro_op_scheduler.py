#!/usr/bin/env python3
"""Tests for micro-op scheduling."""
import unittest


class TestMicroOpScheduler(unittest.TestCase):
    def test_scheduler_serializes_dma_and_respects_dependencies(self):
        from hw_params import TPU_V7X
        from micro_op_ir import MicroOp, MicroOpGraph
        from micro_op_scheduler import schedule_micro_op_graph

        graph = MicroOpGraph(
            fragments={},
            micro_ops={
                "load_q": MicroOp(
                    op_id="load_q",
                    step_name="qk_matmul",
                    op_kind="dma_load_hbm_to_vmem",
                    depends_on=[],
                    input_fragments=[],
                    output_fragments=["q_vmem"],
                    required_units=("DMA",),
                    required_vmem_slots=("q_slot0",),
                    required_reg_groups=(),
                    latency_ns=10.0,
                ),
                "load_k": MicroOp(
                    op_id="load_k",
                    step_name="qk_matmul",
                    op_kind="dma_load_hbm_to_vmem",
                    depends_on=[],
                    input_fragments=[],
                    output_fragments=["k_vmem"],
                    required_units=("DMA",),
                    required_vmem_slots=("k_slot0",),
                    required_reg_groups=(),
                    latency_ns=10.0,
                ),
                "mxu": MicroOp(
                    op_id="mxu",
                    step_name="qk_matmul",
                    op_kind="mxu_compute",
                    depends_on=["load_q", "load_k"],
                    input_fragments=["q_reg", "k_reg"],
                    output_fragments=["acc"],
                    required_units=("MXU",),
                    required_vmem_slots=(),
                    required_reg_groups=("q_reg0", "k_reg0", "acc_reg0"),
                    latency_ns=20.0,
                ),
            },
        )

        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertEqual(result.op_timings["load_q"].start_ns, 0.0)
        self.assertEqual(result.op_timings["load_k"].start_ns, 10.0)
        self.assertEqual(result.op_timings["mxu"].start_ns, 20.0)


if __name__ == "__main__":
    unittest.main()
