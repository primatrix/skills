#!/usr/bin/env python3
"""Integration tests for tpu-pipeline-scheduler."""

import json
import os
import subprocess
import unittest


class TestPipelineSchedulerE2E(unittest.TestCase):
    def _scripts_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def _example_path(self):
        return os.path.join(self._scripts_dir(), "examples",
                            "flash_attention_tile.json")

    def test_full_pipeline_api(self):
        from pipeline_ir import load_spec_from_file
        from dependency_analyzer import analyze_dependencies
        from pipeline_scheduler import schedule
        from vpr_analyzer import analyze_vpr_liveness

        spec = load_spec_from_file(self._example_path())
        self.assertEqual(spec.name, "flash_attention_tile")
        self.assertEqual(len(spec.ops), 11)

        graph = analyze_dependencies(spec.ops)
        self.assertGreater(len(graph.edges), 0)

        sched = schedule(spec.ops)
        self.assertGreater(sched.total_latency_ns, 0)
        self.assertGreater(len(sched.entries), 0)

        occ = analyze_vpr_liveness(spec.ops, sched)
        self.assertGreater(occ.peak_concurrent, 0)

    def test_cli_text_output(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "text",
                "--show", "all",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("Data Dependency Graph", result.stdout)
        self.assertIn("Pipeline Gantt", result.stdout)
        self.assertIn("VPR Occupancy", result.stdout)
        self.assertIn("Reorder Suggestion", result.stdout)

    def test_cli_json_deps(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "json",
                "--show", "deps",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertIn("edges", data)
        self.assertGreater(len(data["edges"]), 0)

    def test_cli_mermaid_output(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "text",
                "--show", "deps,gantt",
                "--mermaid",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("graph TD", result.stdout)
        self.assertIn("gantt", result.stdout)

    def test_cli_single_section(self):
        scripts_dir = self._scripts_dir()
        result = subprocess.run(
            [
                "python", "pipeline_ir_cli.py",
                "--pipeline", self._example_path(),
                "--format", "text",
                "--show", "vpr",
            ],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("VPR", result.stdout)
        self.assertNotIn("Gantt", result.stdout)


    def test_cli_plot_output(self):
        import tempfile
        scripts_dir = self._scripts_dir()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            result = subprocess.run(
                [
                    "python", "pipeline_ir_cli.py",
                    "--pipeline", self._example_path(),
                    "--plot",
                    "--plot-output", out_path,
                ],
                capture_output=True, text=True, cwd=scripts_dir,
            )
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            with open(out_path, "rb") as f:
                header = f.read(4)
            self.assertEqual(header, b"\x89PNG")
        finally:
            os.unlink(out_path)

    def test_cli_plot_default_name(self):
        """--plot without --plot-output uses <spec_name>_vpr_timeline.png"""
        scripts_dir = self._scripts_dir()
        expected_name = "flash_attention_tile_vpr_timeline.png"
        expected_path = os.path.join(scripts_dir, expected_name)
        try:
            result = subprocess.run(
                [
                    "python", "pipeline_ir_cli.py",
                    "--pipeline", self._example_path(),
                    "--plot",
                ],
                capture_output=True, text=True, cwd=scripts_dir,
            )
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            self.assertTrue(os.path.exists(expected_path),
                            f"Expected {expected_path} to be created")
        finally:
            if os.path.exists(expected_path):
                os.unlink(expected_path)


if __name__ == "__main__":
    unittest.main()
