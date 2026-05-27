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
class TestRuntimeInvariants(unittest.TestCase):
    """Runtime-block invariants (R1..R4 in SKILL.md). Run only when fixture has runtime data."""

    @classmethod
    def setUpClass(cls):
        cls.out = _run()
        if cls.out.get("runtime") is None:
            raise unittest.SkipTest(f"runtime block absent: {cls.out.get('runtime_absent_reason')}")
        cls.rt = cls.out["runtime"]
        cls.diag = cls.out["runtime_diagnostics"]

    def test_R1_topk_plus_tail_equals_alive_total(self):
        topk_sum = sum(int(b["size_bytes"]) for b in self.rt["alive_at_peak"]["buffers"])
        tail = int(self.rt["alive_at_peak"]["tail"]["total_bytes"])
        self.assertEqual(topk_sum + tail, int(self.rt["alive_at_peak"]["total_bytes"]),
                         "R1: Σ Top-K size_bytes + tail.total_bytes must equal alive_at_peak.total_bytes")

    def test_R2_alive_total_equals_peak_bytes_total(self):
        self.assertEqual(int(self.rt["alive_at_peak"]["total_bytes"]),
                         int(self.rt["peak"]["bytes_total"]),
                         "R2: alive_at_peak.total_bytes must equal peak.bytes_total")

    def test_R2b_alloc_accounting_drift_within_one_pct(self):
        drift = float(self.diag["alloc_accounting_drift_pct"])
        if drift > 1.0:
            warns = self.diag["warnings"]
            self.assertTrue(any("drift" in w.lower() for w in warns),
                            f"R2b: drift={drift}% > 1% must surface a warning; got warnings={warns}")

    def test_by_shape_partition(self):
        rows = self.rt["rollups"]["by_shape"]
        self.assertEqual(_sum_total_bytes(rows), int(self.rt["alive_at_peak"]["total_bytes"]),
                         "by_shape rows must partition alive_at_peak.total_bytes")

    def test_other_rollups_partition(self):
        target = int(self.rt["alive_at_peak"]["total_bytes"])
        for key in ("by_tf_op", "by_parent_jit", "by_lifetime_class", "by_dtype"):
            with self.subTest(rollup=key):
                self.assertEqual(_sum_total_bytes(self.rt["rollups"][key]), target,
                                 f"{key} rows must partition alive_at_peak.total_bytes")

    def test_R3_peak_within_pool_reserved(self):
        self.assertLessEqual(int(self.rt["peak"]["bytes_total"]),
                             int(self.rt["pool"]["bytes_reserved"]),
                             "R3: peak.bytes_total ≤ pool.bytes_reserved")

    def test_timeline_max_at_least_step_peak(self):
        max_sample = max(int(s["bytes_allocated"]) for s in self.rt["timeline"]["samples"])
        self.assertGreaterEqual(max_sample, int(self.rt["peak"]["bytes_total"]),
                                "max(timeline.samples.bytes_allocated) ≥ peak.bytes_total")

    def test_buffer_alloc_before_peak(self):
        peak_ts = int(self.rt["peak"]["ts_ns"])
        for b in self.rt["alive_at_peak"]["buffers"]:
            self.assertLessEqual(int(b["alloc_ts_ns"]), peak_ts,
                                 f"buffer addr={b['addr']} alloc_ts > peak_ts")

    def test_no_unmatched_deallocs(self):
        self.assertEqual(int(self.diag["unmatched_dealloc_count"]), 0,
                         "unmatched_dealloc_count must be 0")

    def test_R4_peak_within_step_window(self):
        lo, hi = self.rt["step"]["range_ns"]
        self.assertLessEqual(int(lo), int(self.rt["peak"]["ts_ns"]))
        self.assertLessEqual(int(self.rt["peak"]["ts_ns"]), int(hi))


@unittest.skipUnless(os.path.isdir(FIXTURE), f"fixture not present at {FIXTURE}")
class TestHloInvariants(unittest.TestCase):
    """HLO-block invariants (H1..H5 in SKILL.md). Run only when fixture has hlo_proto.pb."""

    @classmethod
    def setUpClass(cls):
        cls.out = _run()
        if cls.out.get("hlo") is None:
            raise unittest.SkipTest(f"hlo block absent: {cls.out.get('hlo_absent_reason')}")
        cls.hlo = cls.out["hlo"]

    def test_H2_entry_peak_within_temp_pool(self):
        self.assertLessEqual(int(self.hlo["schedule_sweep"]["peak_alive_bytes_entry_level"]),
                             int(self.hlo["decomposition"]["temp_pool_bytes"]),
                             "H2: peak_alive_bytes_entry_level ≤ temp_pool_bytes")

    def test_H3_topk_plus_tail_equals_alive_total(self):
        topk_sum = sum(int(b["size_bytes"]) for b in self.hlo["alive_at_peak"]["buffers"])
        tail = int(self.hlo["alive_at_peak"]["tail"]["total_bytes"])
        self.assertEqual(topk_sum + tail, int(self.hlo["alive_at_peak"]["total_bytes"]),
                         "H3: Σ Top-K size_bytes + tail.total_bytes must equal alive_at_peak.total_bytes")

    def test_H4_always_alive_within_temp_pool(self):
        self.assertLessEqual(int(self.hlo["always_alive"]["total_bytes"]),
                             int(self.hlo["decomposition"]["temp_pool_bytes"]),
                             "H4: always_alive.total_bytes ≤ temp_pool_bytes")

    def test_H5_by_opcode_partitions_alive(self):
        # by_opcode partitions alive_at_peak (head + <other> tail row)
        rows = self.hlo["alive_at_peak"]["rollups"]["by_opcode"]
        self.assertEqual(_sum_total_bytes(rows),
                         int(self.hlo["alive_at_peak"]["total_bytes"]),
                         "H5: by_opcode rows must partition alive_at_peak.total_bytes")


if __name__ == "__main__":
    unittest.main()
