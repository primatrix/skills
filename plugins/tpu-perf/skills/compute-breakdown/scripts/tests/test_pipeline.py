"""Unit tests for the compute_breakdown.py shared pipeline (Stages 1-3) and CLI."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "compute_breakdown.py"


class TestCLI(unittest.TestCase):
    def test_help_runs(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--mode", r.stdout)
        self.assertIn("summary", r.stdout)
        self.assertIn("by_source", r.stdout)
        self.assertIn("non_compute", r.stdout)
        self.assertIn("roofline", r.stdout)

    def test_no_xplane_returns_absent(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp", "--mode", "summary"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "absent")
        self.assertEqual(doc["reason"], "no_xplane_pb")
        self.assertEqual(doc["mode"], "summary")
        self.assertEqual(doc["profile_dir"], "/tmp")
        self.assertEqual(doc["notes"], [])

    def test_step_and_step_id_mutually_exclusive(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp",
             "--mode", "summary", "--step", "0", "--step-id", "x"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 1)
        self.assertIn("step", r.stderr.lower())


if __name__ == "__main__":
    unittest.main()
