#!/usr/bin/env python3
"""Integration test: full pipeline with flash attention example."""
import json
import os
import subprocess
import tempfile
import unittest


class TestFlashAttentionE2E(unittest.TestCase):
    def test_full_pipeline(self):
        from compute_step import load_steps_from_file
        from pipeline_simulator import simulate_steps
        from hw_params import TPU_V7X
        from report import pipeline_report_to_json, pipeline_report_to_text

        example_path = os.path.join(os.path.dirname(__file__), "examples", "flash_attention.json")
        steps = load_steps_from_file(example_path)
        self.assertEqual(len(steps), 4)

        report = simulate_steps(steps, TPU_V7X)
        self.assertEqual(len(report.steps), 4)
        self.assertGreater(report.total_time_ns, 0)
        self.assertGreater(report.total_flops, 0)
        self.assertGreater(report.fusion_savings_bytes, 0)
        self.assertTrue(report.steps[1].fused_with_prev)
        self.assertTrue(report.steps[2].fused_with_prev)

        json_out = pipeline_report_to_json(report)
        data = json.loads(json_out)
        self.assertIn("steps", data)

        text_out = pipeline_report_to_text(report)
        self.assertIn("qk_matmul", text_out)
        self.assertIn("sv_matmul", text_out)

    def test_cli_runs(self):
        scripts_dir = os.path.dirname(__file__)
        example_path = os.path.join(scripts_dir, "examples", "flash_attention.json")
        result = subprocess.run(
            ["python", os.path.join(scripts_dir, "cli.py"),
             "--steps", example_path, "--format", "json"],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertIn("steps", data)

    def test_cli_runs_in_micro_mode(self):
        scripts_dir = os.path.dirname(__file__)
        example_path = os.path.join(scripts_dir, "examples", "flash_attention.json")
        result = subprocess.run(
            [
                "python", os.path.join(scripts_dir, "cli.py"),
                "--steps", example_path,
                "--analysis-level", "micro",
                "--format", "json",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertIn("micro_ops", data)
        self.assertIn("critical_path", data)

    def test_cli_micro_mode_supports_duplicate_step_names(self):
        scripts_dir = os.path.dirname(__file__)
        steps = [
            {
                "name": "matmul",
                "op_type": "matmul",
                "inputs": [
                    {"name": "Q", "shape": [128, 128], "dtype": "bf16"},
                    {"name": "K", "shape": [128, 128], "dtype": "bf16"},
                ],
                "outputs": [{"name": "S", "shape": [128, 128], "dtype": "bf16"}],
                "flops_formula": "2*M*N*K",
                "flops_vars": {"M": 128, "N": 128, "K": 128},
                "compute_unit": "MXU",
                "fusable_with_prev": False,
            },
            {
                "name": "matmul",
                "op_type": "matmul",
                "inputs": [
                    {"name": "S", "shape": [128, 128], "dtype": "bf16"},
                    {"name": "V", "shape": [128, 128], "dtype": "bf16"},
                ],
                "outputs": [{"name": "O", "shape": [128, 128], "dtype": "bf16"}],
                "flops_formula": "2*M*N*K",
                "flops_vars": {"M": 128, "N": 128, "K": 128},
                "compute_unit": "MXU",
                "fusable_with_prev": False,
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(steps, tmp)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [
                    "python", os.path.join(scripts_dir, "cli.py"),
                    "--steps", tmp_path,
                    "--analysis-level", "micro",
                    "--format", "json",
                ],
                capture_output=True, text=True, cwd=scripts_dir,
            )
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            data = json.loads(result.stdout)
            op_ids = {row["op_id"] for row in data["micro_ops"]}
            self.assertIn("s0_matmul_load_q_tile0", op_ids)
            self.assertIn("s1_matmul_load_q_tile0", op_ids)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
