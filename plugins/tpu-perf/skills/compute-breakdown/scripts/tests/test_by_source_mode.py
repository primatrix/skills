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


class TestBySourceCLI(unittest.TestCase):
    def test_include_data_move_flag_present(self):
        ns = cb.build_parser().parse_args(
            ["/x", "--mode", "by_source", "--include-data-move"]
        )
        self.assertTrue(ns.include_data_move)

    def test_include_data_move_default_false(self):
        ns = cb.build_parser().parse_args(["/x", "--mode", "by_source"])
        self.assertFalse(ns.include_data_move)


class TestBySourceEndToEnd(unittest.TestCase):
    def test_emits_valid_json_with_groups_block(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big",
                       flops=1_000_000, bytes_accessed=1024,
                       shape_with_layout="bf16[8]{0}")
        add_hlo_event(xs, em_id=11, hlo_op_text="copy.0",
                       offset_ps=600, duration_ps=5_000_000,
                       hlo_category="data formatting")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            r = subprocess.run(
                [sys.executable, str(SCRIPT), tmp, "--mode", "by_source"],
                capture_output=True, text=True,
            )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["mode"], "by_source")
        self.assertEqual(doc["totals"]["n_events_compute"], 1)
        self.assertEqual(doc["totals"]["n_events_data_move"], 1)
        self.assertEqual(len(doc["groups"]), 1)
        self.assertEqual(doc["groups"][0]["tf_op"], "jit/Big")

    def test_include_data_move_adds_data_move_groups(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big")
        add_hlo_event(xs, em_id=11, hlo_op_text="copy.0",
                       offset_ps=600, duration_ps=5_000_000,
                       hlo_category="data formatting", tf_op="jit/Copy")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            r = subprocess.run(
                [sys.executable, str(SCRIPT), tmp, "--mode", "by_source",
                 "--include-data-move"],
                capture_output=True, text=True,
            )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(len(doc["groups"]), 2)


if __name__ == "__main__":
    unittest.main()
