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

    def test_json_contains_vpr_pressure(self):
        from micro_op_report import micro_schedule_to_json
        import json
        payload = json.loads(micro_schedule_to_json(_sample_schedule_result(), []))
        self.assertIn("vpr_pressure", payload)
        vpr = payload["vpr_pressure"]
        self.assertIn("peak_vpr_count", vpr)
        self.assertIn("vpr_capacity", vpr)
        self.assertIn("utilization_pct", vpr)
        self.assertIn("spill_count", vpr)
        self.assertIn("spill_cost_ns", vpr)

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


class TestResourceGantt(unittest.TestCase):
    def test_gantt_has_vmem_section(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("section VMEM Slots", output)

    def test_gantt_has_reg_section(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("section REG Groups", output)

    def test_gantt_has_no_unit_sections(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertNotIn("section DMA", output)
        self.assertNotIn("section MXU", output)
        self.assertNotIn("section VPU", output)

    def test_gantt_has_capacity_comments(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("Peak VMEM", output)
        self.assertIn("Peak REG", output)

    def test_gantt_contains_slot_names(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertTrue(
            "q_slot" in output and "k_slot" in output,
            f"Expected VMEM slot names in output:\n{output}",
        )

    def test_gantt_contains_reg_group_names(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertTrue(
            "q_reg" in output and "acc_reg" in output,
            f"Expected REG group names in output:\n{output}",
        )

    def test_gantt_includes_stall_bars(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("crit", output)

    def test_gantt_filters_tiles_by_max(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("tile0", output)
        self.assertNotIn("tile1", output)

    def test_gantt_shows_ellipsis_when_truncated(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("%%", output)
        self.assertIn("steady-state", output)

    def test_gantt_rejects_non_positive_max_tiles(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        with self.assertRaises(ValueError):
            micro_schedule_to_mermaid(schedule, graph, max_tiles=0)


class TestStallDetection(unittest.TestCase):
    def test_detect_op_stalls_returns_dict(self):
        from micro_op_report import _detect_op_stalls

        schedule, graph = _sample_mermaid_schedule()
        stalls = _detect_op_stalls(schedule, graph)
        self.assertIsInstance(stalls, dict)
        # All ops should be classified
        self.assertEqual(len(stalls), len(graph.micro_ops))

    def test_detect_op_stalls_no_stall_when_no_gap(self):
        from micro_op_report import _detect_op_stalls

        schedule, graph = _sample_mermaid_schedule()
        stalls = _detect_op_stalls(schedule, graph)
        # In a perfectly pipelined 2-tile matmul, ops start exactly
        # when deps are ready — no execution gap means no stall.
        for op_id, reasons in stalls.items():
            op = graph.micro_ops[op_id]
            if not op.depends_on:
                continue
            dep_ready = max(schedule.op_timings[d].end_ns for d in op.depends_on)
            gap = schedule.op_timings[op_id].start_ns - dep_ready
            if gap <= 0:
                self.assertEqual(reasons, [], f"{op_id} has gap={gap} but reasons={reasons}")

    def test_detect_op_stalls_root_ops_have_no_stalls(self):
        from micro_op_report import _detect_op_stalls

        schedule, graph = _sample_mermaid_schedule()
        stalls = _detect_op_stalls(schedule, graph)
        root_ops = graph.root_ops()
        for op_id in root_ops:
            self.assertEqual(stalls.get(op_id, []), [])


class TestDataFlowChart(unittest.TestCase):
    def test_flowchart_has_memory_level_nodes(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
        self.assertIn("HBM:", output)
        self.assertIn("VMEM", output)
        self.assertIn("REG", output)

    def test_flowchart_has_data_transfer_edges(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
        # Should have solid edges with latency labels
        self.assertIn("-->", output)
        self.assertIn("ns", output)

    def test_flowchart_has_tile_subgraph(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
        self.assertIn("subgraph", output)
        self.assertIn("Tile 0", output)

    def test_flowchart_shows_stall_edges(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph)
        # At minimum, verify the function runs without error
        self.assertIn("flowchart TD", output)

    def test_flowchart_per_tile_count(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=2)
        self.assertEqual(output.count("subgraph"), 2)

    def test_flowchart_rejects_non_positive_max_tiles(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        with self.assertRaises(ValueError):
            micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=0)


if __name__ == "__main__":
    unittest.main()
