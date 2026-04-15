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

    def test_scheduler_records_wait_vmem_and_wait_reg(self):
        from hw_params import TPU_V7X
        from micro_op_ir import MicroOp, MicroOpGraph
        from micro_op_scheduler import schedule_micro_op_graph

        graph = MicroOpGraph(
            fragments={},
            micro_ops={
                "hold_slot": MicroOp(
                    op_id="hold_slot",
                    step_name="slot_conflict",
                    op_kind="vmem_to_reg",
                    depends_on=[],
                    input_fragments=["x_vmem"],
                    output_fragments=["x_reg"],
                    required_units=(),
                    required_vmem_slots=("slot0",),
                    required_reg_groups=(),
                    latency_ns=5.0,
                ),
                "wait_for_slot": MicroOp(
                    op_id="wait_for_slot",
                    step_name="slot_conflict",
                    op_kind="vmem_to_reg",
                    depends_on=[],
                    input_fragments=["y_vmem"],
                    output_fragments=["y_reg"],
                    required_units=(),
                    required_vmem_slots=("slot0",),
                    required_reg_groups=(),
                    latency_ns=5.0,
                ),
                "hold_reg": MicroOp(
                    op_id="hold_reg",
                    step_name="reg_conflict",
                    op_kind="vpu_compute",
                    depends_on=[],
                    input_fragments=["x_reg"],
                    output_fragments=["z_reg"],
                    required_units=(),
                    required_vmem_slots=(),
                    required_reg_groups=("reg0",),
                    latency_ns=5.0,
                ),
                "wait_for_reg": MicroOp(
                    op_id="wait_for_reg",
                    step_name="reg_conflict",
                    op_kind="vpu_compute",
                    depends_on=[],
                    input_fragments=["y_reg"],
                    output_fragments=["w_reg"],
                    required_units=(),
                    required_vmem_slots=(),
                    required_reg_groups=("reg0",),
                    latency_ns=5.0,
                ),
            },
        )

        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertGreater(result.stall_breakdown["WAIT_VMEM"], 0)
        self.assertGreater(result.stall_breakdown["WAIT_REG"], 0)

    def test_scheduler_extracts_critical_path(self):
        from hw_params import TPU_V7X
        from micro_op_ir import MicroOp, MicroOpGraph
        from micro_op_scheduler import schedule_micro_op_graph

        graph = MicroOpGraph(
            fragments={},
            micro_ops={
                "load_q": MicroOp(
                    op_id="load_q",
                    step_name="critical_path",
                    op_kind="dma_load_hbm_to_vmem",
                    depends_on=[],
                    input_fragments=[],
                    output_fragments=["q_vmem"],
                    required_units=("DMA",),
                    required_vmem_slots=("slot_q",),
                    required_reg_groups=(),
                    latency_ns=5.0,
                ),
                "load_k": MicroOp(
                    op_id="load_k",
                    step_name="critical_path",
                    op_kind="dma_load_hbm_to_vmem",
                    depends_on=[],
                    input_fragments=[],
                    output_fragments=["k_vmem"],
                    required_units=("DMA",),
                    required_vmem_slots=("slot_k",),
                    required_reg_groups=(),
                    latency_ns=5.0,
                ),
                "mxu_main": MicroOp(
                    op_id="mxu_main",
                    step_name="critical_path",
                    op_kind="mxu_compute",
                    depends_on=["load_q", "load_k"],
                    input_fragments=["q_reg", "k_reg"],
                    output_fragments=["acc_reg"],
                    required_units=("MXU",),
                    required_vmem_slots=(),
                    required_reg_groups=("q_reg0", "k_reg0", "acc_reg0"),
                    latency_ns=15.0,
                ),
                "store_out": MicroOp(
                    op_id="store_out",
                    step_name="critical_path",
                    op_kind="dma_store_vmem_to_hbm",
                    depends_on=["mxu_main"],
                    input_fragments=["out_vmem"],
                    output_fragments=["out_hbm"],
                    required_units=("DMA",),
                    required_vmem_slots=("slot_out",),
                    required_reg_groups=(),
                    latency_ns=5.0,
                ),
            },
        )

        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertEqual(result.critical_path[-1], "store_out")
        self.assertIn("mxu_main", result.critical_path)


if __name__ == "__main__":
    unittest.main()
