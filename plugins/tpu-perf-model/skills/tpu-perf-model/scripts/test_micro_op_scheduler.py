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


class TestPeakResources(unittest.TestCase):
    def test_schedule_result_has_peak_fields(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig
        from hw_params import TPU_V7X
        from micro_op_scheduler import schedule_micro_op_graph

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[
                TensorRef(name="Q", shape=(256, 128), dtype="bf16"),
                TensorRef(name="K", shape=(128, 256), dtype="bf16"),
            ],
            outputs=[TensorRef(name="S", shape=(256, 256), dtype="bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 256, "N": 256, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=2, double_buffer=True,
            tile_input_bytes=65536, tile_output_bytes=32768,
            vmem_usage_bytes=196608,
        )
        graph = build_micro_op_graph_for_step(step, tile, step_idx=0)
        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertIsInstance(result.peak_vmem_slots, int)
        self.assertIsInstance(result.peak_reg_groups, int)
        self.assertGreater(result.peak_vmem_slots, 0)
        self.assertGreater(result.peak_reg_groups, 0)

    def test_peak_reg_groups_within_hardware_limit(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig
        from hw_params import TPU_V7X
        from micro_op_scheduler import schedule_micro_op_graph

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[
                TensorRef(name="Q", shape=(256, 128), dtype="bf16"),
                TensorRef(name="K", shape=(128, 256), dtype="bf16"),
            ],
            outputs=[TensorRef(name="S", shape=(256, 256), dtype="bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 256, "N": 256, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=2, double_buffer=True,
            tile_input_bytes=65536, tile_output_bytes=32768,
            vmem_usage_bytes=196608,
        )
        graph = build_micro_op_graph_for_step(step, tile, step_idx=0)
        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertLessEqual(result.peak_reg_groups, TPU_V7X.reg_group_count)

    def test_schedule_result_has_spill_fields(self):
        from hw_params import TPU_V7X
        from micro_op_ir import MicroOp, MicroOpGraph
        from micro_op_scheduler import schedule_micro_op_graph

        graph = MicroOpGraph(
            fragments={},
            micro_ops={
                "load_q": MicroOp(
                    op_id="load_q", step_name="matmul",
                    op_kind="dma_load_hbm_to_vmem", depends_on=[],
                    input_fragments=[], output_fragments=["q_vmem"],
                    required_units=("DMA",), required_vmem_slots=("q_slot",),
                    required_reg_groups=(), latency_ns=10.0,
                ),
            },
        )
        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertEqual(result.spill_count, 0)
        self.assertEqual(result.spill_cost_ns, 0.0)

    def test_schedule_result_has_peak_vpr_count(self):
        from hw_params import TPU_V7X
        from micro_op_ir import MicroOp, MicroOpGraph
        from micro_op_scheduler import schedule_micro_op_graph

        graph = MicroOpGraph(
            fragments={},
            micro_ops={
                "move_q": MicroOp(
                    op_id="move_q", step_name="matmul",
                    op_kind="vmem_to_reg", depends_on=[],
                    input_fragments=["q_vmem"], output_fragments=["q_reg"],
                    required_units=(), required_vmem_slots=("q_slot",),
                    required_reg_groups=("q_reg0",), latency_ns=1.0,
                    required_vpr_count=8,
                ),
            },
        )
        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertEqual(result.peak_vpr_count, 8)


if __name__ == "__main__":
    unittest.main()
