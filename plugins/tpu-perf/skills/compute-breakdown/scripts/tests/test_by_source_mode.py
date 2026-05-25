"""Mode 2 (by_source) projection."""
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

# Reuse synthetic builders / record factory from the pipeline test module.
from test_pipeline import (  # noqa: E402
    _make_record, make_minimal_xspace, add_hlo_event,
)
from test_summary_mode import SCRIPT  # noqa: E402  (reuse main-script path)


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


class TestRunBySourceMode(unittest.TestCase):
    def test_groups_emitted_in_insertion_order_not_sorted(self):
        # Three groups; durations chosen so a sort would reorder them.
        recs = [
            _make_record(agg_key="A", kind="compute", duration_ps=10),
            _make_record(agg_key="B", kind="compute", duration_ps=100),
            _make_record(agg_key="C", kind="compute", duration_ps=50),
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=False)
        self.assertEqual([g["agg_key"] for g in doc["groups"]], ["A", "B", "C"])
        self.assertEqual(doc["totals"]["n_groups_total"], 3)

    def test_data_move_excluded_by_default(self):
        recs = [
            _make_record(agg_key="C", kind="compute",   duration_ps=10),
            _make_record(agg_key="D", kind="data_move", duration_ps=20),
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=False)
        self.assertEqual([g["agg_key"] for g in doc["groups"]], ["C"])
        # Totals still reflect ALL records (cross-mode invariant).
        self.assertEqual(doc["totals"]["data_move_duration_ps"], 20)
        self.assertEqual(doc["totals"]["compute_duration_ps"], 10)

    def test_data_move_included_when_flag_set(self):
        recs = [
            _make_record(agg_key="C", kind="compute",   duration_ps=10),
            _make_record(agg_key="D", kind="data_move", duration_ps=20),
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=True)
        self.assertEqual({g["agg_key"] for g in doc["groups"]}, {"C", "D"})

    def test_comm_excluded_by_default_and_includable(self):
        recs = [
            _make_record(agg_key="C", kind="compute", duration_ps=10),
            _make_record(agg_key="X", kind="comm",    duration_ps=99),
        ]
        d_off = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
        self.assertEqual([g["agg_key"] for g in d_off["groups"]], ["C"])
        d_on = cb._run_by_source_mode(recs, ctx=_ctx(),
                                        include_comm=True,
                                        include_data_move=False)
        self.assertEqual({g["agg_key"] for g in d_on["groups"]}, {"C", "X"})

    def test_shapes_capped_at_eight_and_flag_set(self):
        # 9 distinct shapes for the same agg_key -> cap at 8, flag true.
        recs = [
            _make_record(agg_key="K", kind="compute", duration_ps=1,
                         shape_with_layout=f"bf16[{i}]{{0}}")
            for i in range(9)
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=False)
        g = doc["groups"][0]
        self.assertEqual(len(g["shapes"]), 8)
        self.assertTrue(g["shapes_truncated"])

    def test_shapes_not_truncated_when_under_cap(self):
        recs = [
            _make_record(agg_key="K", kind="compute", duration_ps=1,
                         shape_with_layout=f"bf16[{i}]{{0}}")
            for i in range(3)
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=False)
        g = doc["groups"][0]
        self.assertEqual(len(g["shapes"]), 3)
        self.assertFalse(g["shapes_truncated"])

    def test_dtypes_histogram_and_uncertain_propagation(self):
        recs = [
            _make_record(agg_key="K", kind="compute", duration_ps=1,
                         dtype="bf16"),
            _make_record(agg_key="K", kind="compute", duration_ps=1,
                         dtype="bf16"),
            _make_record(agg_key="K", kind="compute", duration_ps=1,
                         dtype="fp8", dtype_uncertain=True),
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=False)
        g = doc["groups"][0]
        self.assertEqual(g["dtypes"], {"bf16": 2, "fp8": 1})
        self.assertTrue(g["dtype_uncertain"])

    def test_null_flops_when_no_events_reported(self):
        recs = [
            _make_record(agg_key="K", kind="compute", duration_ps=1,
                         flops=None, bytes_accessed=None,
                         model_flops=None),
        ]
        doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                       include_comm=False,
                                       include_data_move=False)
        g = doc["groups"][0]
        self.assertIsNone(g["flops_sum"])
        self.assertIsNone(g["bytes_accessed_sum"])
        self.assertIsNone(g["model_flops_sum"])

    def test_totals_match_summary_mode(self):
        # Cross-mode invariant: by_source.totals derives from same records
        # without any kind filter on totals computation.
        recs = [
            _make_record(agg_key="A", kind="compute",   duration_ps=400),
            _make_record(agg_key="B", kind="data_move", duration_ps=100),
            _make_record(agg_key="C", kind="comm",      duration_ps=300),
        ]
        ctx = _ctx(step_duration_ps=1000)
        d_summary = cb._run_summary_mode(recs, ctx=ctx,
                                            include_comm=False, top=10)
        d_bysrc = cb._run_by_source_mode(recs, ctx=ctx,
                                            include_comm=False,
                                            include_data_move=False)
        for k in ("compute_duration_ps", "data_move_duration_ps",
                  "comm_duration_ps", "other_duration_ps",
                  "n_events_unresolved", "unknown_categories"):
            self.assertEqual(d_summary["totals"][k], d_bysrc["totals"][k],
                              msg=f"totals[{k}] differs")


if __name__ == "__main__":
    unittest.main()
