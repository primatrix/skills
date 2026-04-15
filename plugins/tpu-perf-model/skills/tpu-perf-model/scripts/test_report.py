#!/usr/bin/env python3
"""Tests for report module."""
import json
import unittest

class TestReportJSON(unittest.TestCase):
    def test_pipeline_report_to_json(self):
        from report import pipeline_report_to_json
        from pipeline_simulator import PipelineReport, StepResult
        report = PipelineReport(
            steps=[StepResult(
                name="matmul", op_type="matmul", compute_unit="MXU",
                flops=2_000_000, hbm_bytes=100_000,
                t_hbm_ns=27.1, t_compute_ns=0.87, t_step_ns=27.1,
                bottleneck="HBM_BW", arithmetic_intensity=20.0,
                tile_config=None, fused_with_prev=False, fusion_hbm_savings_bytes=0,
            )],
            total_time_ns=27.1, total_flops=2_000_000, total_hbm_bytes=100_000,
            fusion_savings_bytes=0, overall_arithmetic_intensity=20.0,
            overall_bottleneck="HBM_BW", efficiency_vs_peak=0.032,
        )
        j = pipeline_report_to_json(report)
        data = json.loads(j)
        self.assertIn("steps", data)
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["overall_bottleneck"], "HBM_BW")

class TestReportText(unittest.TestCase):
    def test_pipeline_report_to_text(self):
        from report import pipeline_report_to_text
        from pipeline_simulator import PipelineReport, StepResult
        report = PipelineReport(
            steps=[StepResult(
                name="matmul", op_type="matmul", compute_unit="MXU",
                flops=2_000_000, hbm_bytes=100_000,
                t_hbm_ns=27.1, t_compute_ns=0.87, t_step_ns=27.1,
                bottleneck="HBM_BW", arithmetic_intensity=20.0,
                tile_config=None, fused_with_prev=False, fusion_hbm_savings_bytes=0,
            )],
            total_time_ns=27.1, total_flops=2_000_000, total_hbm_bytes=100_000,
            fusion_savings_bytes=0, overall_arithmetic_intensity=20.0,
            overall_bottleneck="HBM_BW", efficiency_vs_peak=0.032,
        )
        text = pipeline_report_to_text(report)
        self.assertIn("matmul", text)
        self.assertIn("HBM_BW", text)
        self.assertIn("bottleneck", text.lower())

if __name__ == "__main__":
    unittest.main()
