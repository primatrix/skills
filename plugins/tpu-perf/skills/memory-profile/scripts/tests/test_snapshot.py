# plugins/tpu-perf/skills/memory-profile/scripts/tests/test_snapshot.py
import json
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
SCRIPT = os.path.join(REPO_ROOT, "plugins", "tpu-perf", "skills", "memory-profile", "scripts", "memory_profile.py")
FIXTURE = "/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128"


def _run(*args):
    proc = subprocess.run(
        [sys.executable, SCRIPT, FIXTURE, *args],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return json.loads(proc.stdout)


@unittest.skipUnless(os.path.isdir(FIXTURE), f"fixture not present at {FIXTURE}")
class TestSnapshot(unittest.TestCase):
    def test_default_peak_step(self):
        out = _run()
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["skill"], "memory-profile")
        self.assertEqual(out["version"], 1)
        self.assertTrue(out["inputs"]["host_plane_present"])
        self.assertEqual(out["step"]["policy"], "peak")
        self.assertGreater(out["peak"]["bytes_total"], 0)
        self.assertEqual(out["peak"]["bytes_by_pool"]["0"], out["peak"]["bytes_total"])
        self.assertEqual(out["pool"]["id"], 0)
        self.assertGreater(out["alive_at_peak"]["n_buffers"], 0)
        self.assertGreater(len(out["rollups"]["by_shape"]), 0)
        self.assertGreater(len(out["rollups"]["by_lifetime_class"]), 0)
        self.assertGreater(len(out["timeline"]["samples"]), 0)
        # Top-K is sorted desc by total_bytes
        sizes = [r["total_bytes"] for r in out["rollups"]["by_shape"] if r.get("total_bytes") is not None]
        # filter out the synthetic 'tail' row if present
        leading = [r for r in out["rollups"]["by_shape"] if r.get("kind") != "tail"]
        leading_sizes = [r["total_bytes"] for r in leading]
        self.assertEqual(leading_sizes, sorted(leading_sizes, reverse=True))

    def test_all_trace(self):
        out = _run("--all-trace")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["step"]["source"], "all_trace")
        self.assertGreater(out["peak"]["bytes_total"], 0)

    def test_step_out_of_range_absent(self):
        out = _run("--step", "9999")
        self.assertEqual(out["status"], "absent")
        self.assertIn("reason", out)


if __name__ == "__main__":
    unittest.main()
