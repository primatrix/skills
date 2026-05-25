"""Mode 3 (non_compute) projection."""
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

from test_pipeline import (  # noqa: E402
    _make_record, make_minimal_xspace, add_hlo_event,
)
from test_summary_mode import SCRIPT  # noqa: E402


class TestParseHloOpText(unittest.TestCase):
    def test_dtype_change_only_layout_match(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
            "%c.0 = f32[8,4]{1,0} convert(bf16[8,4]{1,0} %x.1)"
        )
        self.assertEqual(out_dt, "f32")
        self.assertEqual(in_dt, "bf16")
        self.assertEqual(out_lay, "{1,0}")
        self.assertEqual(in_lay, "{1,0}")

    def test_layout_change_only(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
            "%t.0 = bf16[8,4]{0,1} transpose(bf16[8,4]{1,0} %x.1)"
        )
        self.assertEqual(out_dt, "bf16")
        self.assertEqual(in_dt, "bf16")
        self.assertEqual(out_lay, "{0,1}")
        self.assertEqual(in_lay, "{1,0}")

    def test_layout_omitted_returns_none(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
            "%cp.0 = bf16[8,4] copy(bf16[8,4] %x.1)"
        )
        self.assertEqual(out_dt, "bf16")
        self.assertEqual(in_dt, "bf16")
        self.assertIsNone(out_lay)
        self.assertIsNone(in_lay)

    def test_no_match_returns_all_none(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text("")
        self.assertIsNone(out_dt)
        self.assertIsNone(in_dt)
        self.assertIsNone(out_lay)
        self.assertIsNone(in_lay)

    def test_no_match_on_garbage(self):
        out_dt, _, in_dt, _ = cb._parse_hlo_op_text("not an HLO op")
        self.assertIsNone(out_dt)
        self.assertIsNone(in_dt)

    def test_lhs_with_or_without_percent(self):
        out_dt1, *_ = cb._parse_hlo_op_text(
            "%foo = bf16[1]{0} copy(bf16[1]{0} %x)"
        )
        out_dt2, *_ = cb._parse_hlo_op_text(
            "foo.bar = bf16[1]{0} copy(bf16[1]{0} %x)"
        )
        self.assertEqual(out_dt1, "bf16")
        self.assertEqual(out_dt2, "bf16")


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


class TestRunNonComputeMode(unittest.TestCase):
    def test_by_category_aggregates_data_move(self):
        recs = [
            _make_record(agg_key="A", kind="data_move", duration_ps=100,
                         hlo_category="data formatting"),
            _make_record(agg_key="B", kind="data_move", duration_ps=200,
                         hlo_category="data formatting"),
            _make_record(agg_key="C", kind="data_move", duration_ps=50,
                         hlo_category="copy"),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                          include_comm=False,
                                          include_comm_stalls=False)
        rows = {row["hlo_category"]: row for row in doc["by_category"]}
        self.assertEqual(rows["data formatting"]["n_executions"], 2)
        self.assertEqual(rows["data formatting"]["total_dur_ps"], 300)
        self.assertEqual(rows["data formatting"]["min_dur_ps"], 100)
        self.assertEqual(rows["data formatting"]["max_dur_ps"], 200)
        self.assertEqual(rows["data formatting"]["avg_dur_ps"], 150)
        self.assertEqual(rows["copy"]["n_executions"], 1)

    def test_by_source_within_category_per_pair_rows(self):
        recs = [
            _make_record(agg_key="A", kind="data_move", duration_ps=100,
                         hlo_category="data formatting", tf_op="jit/T"),
            _make_record(agg_key="B", kind="data_move", duration_ps=200,
                         hlo_category="data formatting", tf_op="jit/U"),
            _make_record(agg_key="A", kind="data_move", duration_ps=10,
                         hlo_category="data formatting", tf_op="jit/T"),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                          include_comm=False,
                                          include_comm_stalls=False)
        self.assertEqual(len(doc["by_source_within_category"]), 2)
        a_row = next(r for r in doc["by_source_within_category"]
                     if r["agg_key"] == "A")
        self.assertEqual(a_row["n_executions"], 2)
        self.assertEqual(a_row["total_dur_ps"], 110)

    def test_dtype_and_layout_change_from_hlo_op_text(self):
        recs = [
            _make_record(
                agg_key="A", kind="data_move", duration_ps=10,
                hlo_category="data formatting",
                hlo_op="%c.0 = f32[8,4]{1,0} convert(bf16[8,4]{1,0} %x.1)",
                shape_with_layout="f32[8,4]{1,0}",
            ),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                          include_comm=False,
                                          include_comm_stalls=False)
        row = doc["by_source_within_category"][0]
        self.assertTrue(row["dtype_change"])
        self.assertFalse(row["layout_change"])

    def test_layout_change_null_when_layout_omitted(self):
        recs = [
            _make_record(
                agg_key="A", kind="data_move", duration_ps=10,
                hlo_category="copy",
                hlo_op="%cp.0 = bf16[8,4] copy(bf16[8,4] %x.1)",
            ),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                          include_comm=False,
                                          include_comm_stalls=False)
        row = doc["by_source_within_category"][0]
        self.assertFalse(row["dtype_change"])
        self.assertIsNone(row["layout_change"])

    def test_dtype_and_layout_change_null_on_unparseable(self):
        recs = [
            _make_record(
                agg_key="A", kind="data_move", duration_ps=10,
                hlo_category="data formatting",
                hlo_op="not an HLO op",
            ),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                          include_comm=False,
                                          include_comm_stalls=False)
        row = doc["by_source_within_category"][0]
        self.assertIsNone(row["dtype_change"])
        self.assertIsNone(row["layout_change"])

    def test_async_done_included_by_default(self):
        recs = [
            _make_record(agg_key="X", kind="comm", duration_ps=500,
                         hlo_category="all-reduce-done"),
            _make_record(agg_key="A", kind="data_move", duration_ps=10,
                         hlo_category="data formatting"),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(step_duration_ps=10_000),
                                          include_comm=False,
                                          include_comm_stalls=True)
        cats = {row["hlo_category"]: row for row in doc["by_category"]}
        self.assertIn("async-done (comm stall)", cats)
        self.assertEqual(cats["async-done (comm stall)"]["total_dur_ps"], 500)
        self.assertIn(
            "async-done included as comm-stall non-compute time; pass --no-comm-stalls to exclude",
            doc["notes"],
        )
        self.assertAlmostEqual(
            doc["totals"]["non_compute_pct_of_step"], 100.0 * 510 / 10_000
        )

    def test_no_comm_stalls_excludes_async_done(self):
        recs = [
            _make_record(agg_key="X", kind="comm", duration_ps=500,
                         hlo_category="all-reduce-done"),
            _make_record(agg_key="A", kind="data_move", duration_ps=10,
                         hlo_category="data formatting"),
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(step_duration_ps=10_000),
                                          include_comm=False,
                                          include_comm_stalls=False)
        cats = {row["hlo_category"] for row in doc["by_category"]}
        self.assertNotIn("async-done (comm stall)", cats)
        self.assertNotIn(
            "async-done included as comm-stall non-compute time; pass --no-comm-stalls to exclude",
            doc["notes"],
        )

    def test_totals_match_summary_when_no_comm_stalls(self):
        recs = [
            _make_record(agg_key="A", kind="compute",   duration_ps=400),
            _make_record(agg_key="B", kind="data_move", duration_ps=100),
            _make_record(agg_key="C", kind="comm",      duration_ps=300),
        ]
        d_summary = cb._run_summary_mode(recs, ctx=_ctx(),
                                            include_comm=False, top=10)
        d_nc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
        self.assertEqual(
            d_summary["totals"]["data_move_duration_ps"],
            d_nc["totals"]["data_move_duration_ps"],
        )

    def test_shapes_in_out_capped_at_four(self):
        recs = [
            _make_record(
                agg_key="A", kind="data_move", duration_ps=1,
                hlo_category="data formatting",
                hlo_op=f"%t.{i} = bf16[{i},4]{{0,1}} transpose(bf16[4,{i}]{{1,0}} %x)",
            )
            for i in range(5)
        ]
        doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                          include_comm=False,
                                          include_comm_stalls=False)
        row = doc["by_source_within_category"][0]
        self.assertLessEqual(len(row["shapes_in"]), 4)
        self.assertLessEqual(len(row["shapes_out"]), 4)


if __name__ == "__main__":
    unittest.main()
