"""Cross-mode invariants e2e test (spec §11)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from test_pipeline import (  # noqa: E402
    make_minimal_xspace, add_hlo_event,
)
from test_summary_mode import SCRIPT  # noqa: E402


def _run(profile_dir, *args):
    cmd = [sys.executable, str(SCRIPT), profile_dir] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
    return json.loads(r.stdout)


class TestCrossModeInvariants(unittest.TestCase):
    def setUp(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        # One bf16 fusion (compute), one transpose (data_move),
        # one all-reduce-done (comm; included as comm-stall by default).
        add_hlo_event(xs, em_id=10,
                       hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big",
                       flops=10**12, bytes_accessed=10**8,
                       shape_with_layout="bf16[8]{0}")
        add_hlo_event(xs, em_id=11,
                       hlo_op_text="%t.0 = bf16[8]{0} transpose(bf16[8]{0} %x)",
                       offset_ps=600, duration_ps=5_000_000,
                       hlo_category="data formatting", tf_op="jit/T")
        add_hlo_event(xs, em_id=12,
                       hlo_op_text="ar.0", offset_ps=700,
                       duration_ps=2_000_000,
                       hlo_category="all-reduce-done", tf_op="jit/AR")
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        pathlib.Path(self.tmpdir, "x.xplane.pb").write_bytes(xs.SerializeToString())

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_summary_and_by_source_compute_duration_equal(self):
        d_sum = _run(self.tmpdir, "--mode", "summary")
        d_bs  = _run(self.tmpdir, "--mode", "by_source")
        self.assertEqual(
            d_sum["totals"]["compute_duration_ps"],
            d_bs["totals"]["compute_duration_ps"],
        )

    def test_summary_and_non_compute_data_move_duration_equal_with_no_comm_stalls(self):
        d_sum = _run(self.tmpdir, "--mode", "summary")
        d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
        self.assertEqual(
            d_sum["totals"]["data_move_duration_ps"],
            d_nc["totals"]["data_move_duration_ps"],
        )

    def test_other_duration_equal_across_modes(self):
        d_sum = _run(self.tmpdir, "--mode", "summary")
        d_bs  = _run(self.tmpdir, "--mode", "by_source")
        d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
        self.assertEqual(d_sum["totals"]["other_duration_ps"],
                          d_bs["totals"]["other_duration_ps"])
        self.assertEqual(d_sum["totals"]["other_duration_ps"],
                          d_nc["totals"]["other_duration_ps"])

    def test_n_events_unresolved_equal_across_modes(self):
        d_sum = _run(self.tmpdir, "--mode", "summary")
        d_bs  = _run(self.tmpdir, "--mode", "by_source")
        d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
        self.assertEqual(d_sum["totals"]["n_events_unresolved"],
                          d_bs["totals"]["n_events_unresolved"])
        self.assertEqual(d_sum["totals"]["n_events_unresolved"],
                          d_nc["totals"]["n_events_unresolved"])

    def test_unknown_categories_equal_across_modes(self):
        d_sum = _run(self.tmpdir, "--mode", "summary")
        d_bs  = _run(self.tmpdir, "--mode", "by_source")
        d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
        self.assertEqual(d_sum["totals"]["unknown_categories"],
                          d_bs["totals"]["unknown_categories"])
        self.assertEqual(d_sum["totals"]["unknown_categories"],
                          d_nc["totals"]["unknown_categories"])

    def test_step_window_equal_across_all_four_modes(self):
        windows = [
            _run(self.tmpdir, "--mode", m)["step_window_ps"]
            for m in ("summary", "by_source", "non_compute", "roofline")
        ]
        self.assertEqual(windows[0], windows[1])
        self.assertEqual(windows[0], windows[2])
        self.assertEqual(windows[0], windows[3])

    def test_roofline_step_compute_equals_summary_compute_when_no_skips(self):
        # When all compute events are roofline-eligible, the two values match.
        d_sum = _run(self.tmpdir, "--mode", "summary")
        d_rl  = _run(self.tmpdir, "--mode", "roofline")
        if d_rl["skipped_groups"]["n_no_flops"] == 0 \
            and d_rl["skipped_groups"]["n_no_bytes"] == 0 \
            and d_rl["skipped_groups"]["n_dtype_other"] == 0 \
            and d_rl["skipped_groups"]["n_peak_unknown_for_dtype"] == 0:
            self.assertEqual(
                d_rl["step_summary"]["step_compute_duration_ps"],
                d_sum["totals"]["compute_duration_ps"],
            )


class TestErrorPaths(unittest.TestCase):
    def test_absent_profile_dir_returns_status_absent(self):
        with tempfile.TemporaryDirectory() as empty:
            cmd = [sys.executable, str(SCRIPT), empty, "--mode", "summary"]
            r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "absent")
        self.assertEqual(doc["reason"], "no_xplane_pb")

    def test_step_out_of_range_returns_exit_1(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            cmd = [sys.executable, str(SCRIPT), tmp,
                   "--mode", "summary", "--step", "999"]
            r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("error", r.stderr.lower())


class TestSanityBounds(unittest.TestCase):
    def test_mfu_and_hbm_util_bounded(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        # Realistic-ish: 1 sec actual, well below theoretical roofline.
        add_hlo_event(xs, em_id=10,
                       hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=1_000_000_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big",
                       flops=10**9, bytes_accessed=10**6,
                       shape_with_layout="bf16[8]{0}")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            doc = _run(tmp, "--mode", "roofline")
        for g in doc["groups"]:
            self.assertGreaterEqual(g["mfu"], 0.0)
            self.assertGreaterEqual(g["hbm_util"], 0.0)
            self.assertGreaterEqual(g["shortfall_ps"], 0)
        if doc["groups"]:
            self.assertGreaterEqual(
                doc["step_summary"]["weighted_avg_roofline_util"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
