#!/usr/bin/env python3
"""Tests for micro-op reporting."""
import json
import unittest


def _sample_schedule_result():
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
            "mxu": MicroOp(
                op_id="mxu",
                step_name="qk_matmul",
                op_kind="mxu_compute",
                depends_on=["load_q"],
                input_fragments=["q_reg"],
                output_fragments=["acc_reg"],
                required_units=("MXU",),
                required_vmem_slots=(),
                required_reg_groups=("q_reg0", "acc_reg0"),
                latency_ns=20.0,
            ),
        },
    )
    return schedule_micro_op_graph(graph, TPU_V7X)


class TestMicroOpReport(unittest.TestCase):
    def test_micro_report_json_contains_schedule_sections(self):
        from micro_op_report import micro_schedule_to_json

        payload = json.loads(micro_schedule_to_json(_sample_schedule_result(), []))
        self.assertIn("micro_ops", payload)
        self.assertIn("timeline", payload)
        self.assertIn("fragment_residency", payload)
        self.assertIn("critical_path", payload)

    def test_micro_report_text_contains_human_sections(self):
        from micro_op_report import micro_schedule_to_text

        text = micro_schedule_to_text(_sample_schedule_result(), [])
        self.assertIn("Macro Summary", text)
        self.assertIn("Micro-Op Schedule Summary", text)
        self.assertIn("Residency and Occupancy", text)
        self.assertIn("Critical Path and Optimization Hints", text)


def _sample_mermaid_schedule():
    """Build a 2-tile matmul schedule for Mermaid testing."""
    from compute_step import ComputeStep, TensorRef
    from hw_params import TPU_V7X
    from micro_op_builder import build_micro_op_graph_for_step
    from micro_op_scheduler import schedule_micro_op_graph
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul",
        op_type="matmul",
        inputs=[
            TensorRef(name="Q", shape=(256, 128), dtype="bf16"),
            TensorRef(name="K", shape=(128, 256), dtype="bf16"),
        ],
        outputs=[
            TensorRef(name="S", shape=(256, 256), dtype="bf16"),
        ],
        flops_formula="2*M*N*K",
        flops_vars={"M": 256, "N": 256, "K": 128},
        compute_unit="MXU",
        fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128},
        num_tiles=2,
        double_buffer=True,
        tile_input_bytes=65536,
        tile_output_bytes=32768,
        vmem_usage_bytes=196608,
    )
    graph = build_micro_op_graph_for_step(step, tile, step_idx=0)
    schedule = schedule_micro_op_graph(graph, TPU_V7X)
    return schedule, graph


class TestMermaidOutput(unittest.TestCase):
    def test_mermaid_contains_gantt_structure(self):
        from micro_op_report import micro_schedule_to_mermaid

        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("```mermaid", output)
        self.assertIn("gantt", output)
        self.assertIn("dateFormat x", output)
        self.assertIn("section DMA", output)
        self.assertIn("section MXU", output)
        self.assertIn("```\n", output.split("```mermaid")[1])

    def test_mermaid_filters_tiles_by_max(self):
        from micro_op_report import micro_schedule_to_mermaid

        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("tile0", output)
        self.assertNotIn("tile1", output)

    def test_mermaid_shows_ellipsis_when_truncated(self):
        from micro_op_report import micro_schedule_to_mermaid

        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("%%", output)
        self.assertIn("steady-state", output)


if __name__ == "__main__":
    unittest.main()
