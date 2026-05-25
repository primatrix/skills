"""Smoke tests for the memory_profile.py CLI surface."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "memory_profile.py"


class TestCLI(unittest.TestCase):
    def test_help_runs(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--step", r.stdout)
        self.assertIn("--step-policy", r.stdout)
        self.assertIn("--all-trace", r.stdout)
        self.assertIn("--top", r.stdout)
        self.assertIn("--persistent-threshold-steps", r.stdout)
        self.assertIn("--include-host-pools", r.stdout)
        self.assertIn("--time-samples", r.stdout)

    def test_no_xplane_returns_absent(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "absent")
        self.assertEqual(doc["reason"], "no_xplane_pb")
        self.assertEqual(doc["skill"], "memory-profile")
        self.assertEqual(doc["version"], 1)
        self.assertEqual(doc["inputs"]["profile_dir"], "/tmp")

    def test_step_and_all_trace_mutually_exclusive(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp", "--step", "0", "--all-trace"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not allowed", r.stderr.lower())


if __name__ == "__main__":
    unittest.main()
