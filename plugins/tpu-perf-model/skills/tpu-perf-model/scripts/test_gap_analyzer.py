#!/usr/bin/env python3
"""Tests for gap_analyzer module."""
import unittest

class TestGapAnalyzer(unittest.TestCase):
    def test_gap_calculation(self):
        from gap_analyzer import compute_gap
        gap = compute_gap(metric="hbm_bytes", theoretical=1_000_000, measured=1_500_000)
        self.assertAlmostEqual(gap.gap_pct, 50.0)
        self.assertIn("excess", gap.diagnosis.lower())

    def test_no_gap(self):
        from gap_analyzer import compute_gap
        gap = compute_gap("mxu_util", theoretical=95.0, measured=94.0)
        self.assertAlmostEqual(gap.gap_pct, -1.05, places=1)

    def test_analyze_eval_result(self):
        from gap_analyzer import analyze_eval_result
        from pipeline_simulator import PipelineReport, StepResult
        theoretical = PipelineReport(
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
        eval_result = {
            "total_time_us": 0.05,
            "metadata": {
                "hw_utilization": {"hbm_bandwidth_bytes": 180_000, "mxu_utilization_pct": 45.0},
                "profile": {"vector_spills": 5, "vector_fills": 3},
            },
        }
        report = analyze_eval_result(theoretical, eval_result)
        self.assertGreater(len(report.gaps), 0)
        self.assertGreater(len(report.top_opportunities), 0)
        self.assertGreater(report.achievable_speedup, 1.0)

if __name__ == "__main__":
    unittest.main()
