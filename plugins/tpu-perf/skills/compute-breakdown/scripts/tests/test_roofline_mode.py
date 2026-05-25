"""Mode 4 (roofline) projection."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "_proto"))
sys.path.insert(0, str(HERE))
import compute_breakdown as cb  # noqa: E402
import _peaks  # noqa: E402

from test_pipeline import (  # noqa: E402
    _make_record, make_minimal_xspace, add_hlo_event,
)
from test_summary_mode import SCRIPT  # noqa: E402


def _ctx(**overrides):
    base = {
        "status": "ok",
        "step_id": 1,
        "step_window_ps": [0, 1_000_000],
        "step_duration_ps": 1_000_000,
        "notes": [],
        "pipeline_stats": cb._PipelineStats(),
        "profile_dir": "/x", "device": "/device:TPU:0",
        "xspace_pb_path": "/x/p.xplane.pb",
    }
    base.update(overrides)
    return base


def _peaks_v7x():
    return _peaks.resolve_peaks("v7x")


class TestRooflineEligibility(unittest.TestCase):
    def test_group_with_no_flops_skipped(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=None, bytes_accessed=1024, dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertEqual(doc["groups"], [])
        self.assertEqual(doc["skipped_groups"]["n_no_flops"], 1)

    def test_group_with_zero_flops_skipped(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=0, bytes_accessed=1024, dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertEqual(doc["skipped_groups"]["n_no_flops"], 1)

    def test_group_with_no_bytes_skipped(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=1_000_000, bytes_accessed=None, dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertEqual(doc["skipped_groups"]["n_no_bytes"], 1)

    def test_group_with_dtype_other_skipped(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=1_000_000, bytes_accessed=1024,
                              dtype="other")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertEqual(doc["skipped_groups"]["n_dtype_other"], 1)

    def test_group_with_null_dtype_skipped(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=1_000_000, bytes_accessed=1024, dtype=None)]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertEqual(doc["skipped_groups"]["n_dtype_other"], 1)

    def test_group_with_fp32_no_override_skipped(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=1_000_000, bytes_accessed=1024, dtype="fp32")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertEqual(doc["skipped_groups"]["n_peak_unknown_for_dtype"], 1)

    def test_group_with_fp32_override_eligible(self):
        recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                              flops=1_000_000, bytes_accessed=1024, dtype="fp32")]
        peaks = _peaks.resolve_peaks("v7x", override_tflops_fp32=500.0)
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=peaks)
        self.assertEqual(len(doc["groups"]), 1)


class TestRooflineFormulas(unittest.TestCase):
    def test_bf16_compute_bound_group(self):
        recs = [_make_record(agg_key="A", kind="compute",
                              duration_ps=1_000_000_000,
                              flops=10**15, bytes_accessed=1024,
                              dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        g = doc["groups"][0]
        self.assertEqual(g["bound"], "compute")
        self.assertGreater(g["t_compute_theory_ps"], g["t_hbm_theory_ps"])
        self.assertEqual(g["t_roofline_theory_ps"], g["t_compute_theory_ps"])
        self.assertGreater(g["mfu"], 0)
        self.assertGreater(g["roofline_util"], 0)

    def test_bf16_memory_bound_group(self):
        recs = [_make_record(agg_key="A", kind="compute",
                              duration_ps=1_000_000_000,
                              flops=10**6, bytes_accessed=10**9,
                              dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        g = doc["groups"][0]
        self.assertEqual(g["bound"], "memory")
        self.assertGreater(g["t_hbm_theory_ps"], g["t_compute_theory_ps"])
        self.assertEqual(g["t_roofline_theory_ps"], g["t_hbm_theory_ps"])

    def test_arithmetic_intensity_value(self):
        recs = [_make_record(agg_key="A", kind="compute",
                              duration_ps=1_000_000_000,
                              flops=2048, bytes_accessed=1024,
                              dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        g = doc["groups"][0]
        self.assertAlmostEqual(g["arithmetic_intensity"], 2.0)

    def test_shortfall_nonneg_for_realistic_inputs(self):
        recs = [_make_record(agg_key="A", kind="compute",
                              duration_ps=1_000_000_000,
                              flops=10**9, bytes_accessed=10**6,
                              dtype="bf16")]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        g = doc["groups"][0]
        self.assertGreaterEqual(g["shortfall_ps"], 0)


class TestRooflineDtypeUncertain(unittest.TestCase):
    def test_dtype_uncertain_propagated(self):
        recs = [_make_record(agg_key="A", kind="compute",
                              duration_ps=1_000_000_000,
                              flops=10**9, bytes_accessed=10**6,
                              dtype="bf16", dtype_uncertain=True)]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        self.assertTrue(doc["groups"][0]["dtype_uncertain"])


class TestRooflineStepSummary(unittest.TestCase):
    def test_top_shortfall_top_10_sorted_desc(self):
        recs = [
            _make_record(agg_key=f"K{i}", kind="compute",
                         duration_ps=10**9 * (i + 1),
                         flops=10**6, bytes_accessed=10**6, dtype="bf16")
            for i in range(15)
        ]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        top = doc["step_summary"]["top_shortfall_groups"]
        self.assertEqual(len(top), 10)
        shortfalls = [t["shortfall_ps"] for t in top]
        self.assertEqual(shortfalls, sorted(shortfalls, reverse=True))

    def test_weighted_avg_uses_total_dur_ps_weights(self):
        recs = [
            _make_record(agg_key="A", kind="compute", duration_ps=100,
                         flops=10**6, bytes_accessed=10**6, dtype="bf16"),
            _make_record(agg_key="B", kind="compute", duration_ps=900,
                         flops=10**11, bytes_accessed=10**6, dtype="bf16"),
        ]
        doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
        mfus = {g["agg_key"]: g["mfu"] for g in doc["groups"]}
        weighted = doc["step_summary"]["weighted_avg_mfu"]
        self.assertGreater(weighted, (mfus["A"] + mfus["B"]) / 2)


class TestRooflineEndToEnd(unittest.TestCase):
    def test_emits_valid_json_with_peaks_used(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10,
                       hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big",
                       flops=10**12, bytes_accessed=10**8,
                       shape_with_layout="bf16[8]{0}")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            r = subprocess.run(
                [sys.executable, str(SCRIPT), tmp, "--mode", "roofline"],
                capture_output=True, text=True,
            )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["mode"], "roofline")
        self.assertEqual(doc["chip"], "v7x")
        self.assertEqual(doc["peaks_used"]["peak_tflops_bf16"], 1153.5)
        self.assertEqual(doc["peaks_used"]["peak_hbm_gibps"], 3690.0)
        self.assertEqual(doc["peaks_used"]["source"], "builtin v7x table")
        self.assertGreaterEqual(len(doc["groups"]), 1)

    def test_cli_override_changes_peaks_used_source(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10,
                       hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big",
                       flops=10**12, bytes_accessed=10**8,
                       shape_with_layout="bf16[8]{0}")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            r = subprocess.run(
                [sys.executable, str(SCRIPT), tmp, "--mode", "roofline",
                 "--peak-tflops-bf16", "999.0"],
                capture_output=True, text=True,
            )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["peaks_used"]["peak_tflops_bf16"], 999.0)
        self.assertEqual(doc["peaks_used"]["source"], "cli override")


class TestRooflineCLIWiring(unittest.TestCase):
    def test_chip_default_is_v7x(self):
        ns = cb.build_parser().parse_args(["/x", "--mode", "roofline"])
        self.assertEqual(ns.chip, "v7x")

    def test_peak_overrides_default_none(self):
        ns = cb.build_parser().parse_args(["/x", "--mode", "roofline"])
        self.assertIsNone(ns.peak_tflops_bf16)
        self.assertIsNone(ns.peak_tflops_fp8)
        self.assertIsNone(ns.peak_tflops_fp32)
        self.assertIsNone(ns.peak_tflops_fp16)
        self.assertIsNone(ns.peak_hbm_gibps)

    def test_peak_override_parses_float(self):
        ns = cb.build_parser().parse_args(
            ["/x", "--mode", "roofline",
             "--peak-tflops-bf16", "1500.0",
             "--peak-hbm-gibps", "4000.0"]
        )
        self.assertEqual(ns.peak_tflops_bf16, 1500.0)
        self.assertEqual(ns.peak_hbm_gibps, 4000.0)


if __name__ == "__main__":
    unittest.main()
