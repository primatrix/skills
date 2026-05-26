# plugins/tpu-perf/skills/memory-profile/scripts/tests/test_invariants.py
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
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _sum_total_bytes(rows):
    return sum(int(r["total_bytes"]) for r in rows)


@unittest.skipUnless(os.path.isdir(FIXTURE), f"fixture not present at {FIXTURE}")
class TestInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = _run()

    def test_I1_topk_plus_tail_equals_alive_total(self):
        out = self.out
        topk_sum = sum(int(b["size_bytes"]) for b in out["alive_at_peak"]["buffers"])
        tail = int(out["alive_at_peak"]["tail"]["total_bytes"])
        self.assertEqual(topk_sum + tail, int(out["alive_at_peak"]["total_bytes"]),
                         "I1: Σ Top-K size_bytes + tail.total_bytes must equal alive_at_peak.total_bytes")

    def test_I2_alive_total_equals_peak_bytes_total(self):
        out = self.out
        self.assertEqual(int(out["alive_at_peak"]["total_bytes"]),
                         int(out["peak"]["bytes_total"]),
                         "I2: alive_at_peak.total_bytes must equal peak.bytes_total")

    def test_I2b_alloc_accounting_drift_within_one_pct(self):
        out = self.out
        drift = float(out["diagnostics"]["alloc_accounting_drift_pct"])
        if drift > 1.0:
            warns = out["diagnostics"]["warnings"]
            self.assertTrue(any("drift" in w.lower() for w in warns),
                            f"I2b: drift={drift}% > 1% must surface a warning; got warnings={warns}")

    def test_I3_by_shape_partition(self):
        rows = self.out["rollups"]["by_shape"]
        self.assertEqual(_sum_total_bytes(rows), int(self.out["alive_at_peak"]["total_bytes"]),
                         "I3: by_shape rows must partition alive_at_peak.total_bytes")

    def test_I4_other_rollups_partition(self):
        out = self.out
        target = int(out["alive_at_peak"]["total_bytes"])
        for key in ("by_tf_op", "by_parent_jit", "by_lifetime_class", "by_dtype"):
            with self.subTest(rollup=key):
                self.assertEqual(_sum_total_bytes(out["rollups"][key]), target,
                                 f"I4: {key} rows must partition alive_at_peak.total_bytes")

    def test_I5_peak_within_pool_reserved(self):
        out = self.out
        self.assertLessEqual(int(out["peak"]["bytes_total"]),
                             int(out["pool"]["bytes_reserved"]),
                             "I5: peak.bytes_total ≤ pool.bytes_reserved")

    def test_I6_timeline_max_at_least_step_peak(self):
        out = self.out
        max_sample = max(int(s["bytes_allocated"]) for s in out["timeline"]["samples"])
        self.assertGreaterEqual(max_sample, int(out["peak"]["bytes_total"]),
                                "I6: max(timeline.samples.bytes_allocated) ≥ peak.bytes_total")

    def test_I7_buffer_alloc_before_peak(self):
        peak_ts = int(self.out["peak"]["ts_ns"])
        for b in self.out["alive_at_peak"]["buffers"]:
            self.assertLessEqual(int(b["alloc_ts_ns"]), peak_ts,
                                 f"I7: buffer addr={b['addr']} alloc_ts > peak_ts")

    def test_I8_no_unmatched_deallocs(self):
        self.assertEqual(int(self.out["diagnostics"]["unmatched_dealloc_count"]), 0,
                         "I8: unmatched_dealloc_count must be 0")

    def test_I9_peak_within_step_window(self):
        out = self.out
        lo, hi = out["step"]["range_ns"]
        self.assertLessEqual(int(lo), int(out["peak"]["ts_ns"]))
        self.assertLessEqual(int(out["peak"]["ts_ns"]), int(hi))


if __name__ == "__main__":
    unittest.main()
