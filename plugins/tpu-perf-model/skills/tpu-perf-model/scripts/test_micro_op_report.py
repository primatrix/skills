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


if __name__ == "__main__":
    unittest.main()
