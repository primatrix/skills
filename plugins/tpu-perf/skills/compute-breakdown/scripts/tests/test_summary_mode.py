"""Unit tests for mode 1 (summary) projection logic."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parent
SCRIPT = SCRIPTS_DIR / "compute_breakdown.py"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "_proto"))

import compute_breakdown as cb  # noqa: E402
from test_pipeline import (  # noqa: E402
    _make_record, make_minimal_xspace, add_hlo_event,
)


class TestAggregateByKey(unittest.TestCase):
    def test_groups_by_agg_key(self):
        recs = [
            _make_record(agg_key="A", duration_ps=100),
            _make_record(agg_key="A", duration_ps=200),
            _make_record(agg_key="B", duration_ps=50),
        ]
        out = cb._aggregate_by_key(recs)
        self.assertEqual(set(out.keys()), {"A", "B"})
        self.assertEqual(out["A"].n_executions, 2)
        self.assertEqual(out["A"].total_dur_ps, 300)
        self.assertEqual(out["A"].min_dur_ps, 100)
        self.assertEqual(out["A"].max_dur_ps, 200)
        self.assertEqual(out["B"].n_executions, 1)

    def test_flops_sum_null_safe(self):
        recs = [
            _make_record(agg_key="A", flops=100),
            _make_record(agg_key="A", flops=None),
            _make_record(agg_key="A", flops=200),
        ]
        out = cb._aggregate_by_key(recs)
        self.assertEqual(out["A"].flops_sum, 300)

    def test_flops_sum_all_null_emits_none(self):
        recs = [
            _make_record(agg_key="A", flops=None),
            _make_record(agg_key="A", flops=None),
        ]
        out = cb._aggregate_by_key(recs)
        self.assertIsNone(out["A"].flops_sum)

    def test_hlo_categories_histogram(self):
        recs = [
            _make_record(agg_key="A", hlo_category="loop fusion"),
            _make_record(agg_key="A", hlo_category="loop fusion"),
            _make_record(agg_key="A", hlo_category="convolution fusion"),
        ]
        out = cb._aggregate_by_key(recs)
        self.assertEqual(out["A"].hlo_categories,
                         {"loop fusion": 2, "convolution fusion": 1})

    def test_hlo_op_signature_normalization(self):
        # custom-call buckets by target so 0-cost AllocateBuffer ≠ Pallas kernel
        self.assertEqual(
            cb._hlo_op_signature(
                '%cc.4874 = bf16[..] custom-call(...) custom_call_target="AllocateBuffer"',
                "custom-call"),
            "custom-call:AllocateBuffer")
        self.assertEqual(
            cb._hlo_op_signature(
                '%cc.123 = bf16[..] custom-call(...) custom_call_target="tpu_custom_call"',
                "custom-call"),
            "custom-call:tpu_custom_call")
        # fusion strips trailing .NNN SSA index so different instances collapse
        self.assertEqual(
            cb._hlo_op_signature(
                "%convert_bitcast_fusion.42 = bf16[..] fusion(...)",
                "loop fusion"),
            "convert_bitcast_fusion [loop fusion]")
        self.assertEqual(
            cb._hlo_op_signature(
                "%convert_bitcast_fusion.99 = bf16[..] fusion(...)",
                "loop fusion"),
            "convert_bitcast_fusion [loop fusion]")
        # generic opcode + category
        self.assertEqual(
            cb._hlo_op_signature(
                "%reduce.7 = f32[..] reduce(f32[..] %x), to_apply=...",
                "reduce"),
            "reduce [reduce]")
        # empty / unparseable falls through to a category-only key
        self.assertEqual(cb._hlo_op_signature("", "loop fusion"),
                          "<empty> [loop fusion]")

    def test_hlo_op_breakdown_orders_by_dur_and_separates_targets(self):
        # AllocateBuffer placeholders + Pallas kernel + small fusion in one group;
        # breakdown must order by total_dur_ps and keep AllocateBuffer separate.
        recs = [
            _make_record(
                agg_key="A", duration_ps=75, hlo_category="custom-call",
                hlo_op='%cc.1 = ... custom-call(...) custom_call_target="AllocateBuffer"'),
            _make_record(
                agg_key="A", duration_ps=75, hlo_category="custom-call",
                hlo_op='%cc.2 = ... custom-call(...) custom_call_target="AllocateBuffer"'),
            _make_record(
                agg_key="A", duration_ps=316_880_000_000, hlo_category="custom-call",
                hlo_op='%vmap_jit__pallas.10 = ... custom-call(...) '
                       'custom_call_target="tpu_custom_call"'),
            _make_record(
                agg_key="A", duration_ps=13_930_000_000, hlo_category="loop fusion",
                hlo_op="%convert_bitcast_fusion.5 = bf16[..] fusion(...)"),
        ]
        out = cb._aggregate_by_key(recs)
        rows = out["A"].hlo_op_breakdown(top_n=8)
        sigs = [r["signature"] for r in rows]
        self.assertEqual(sigs[0], "custom-call:tpu_custom_call")
        self.assertEqual(sigs[1], "convert_bitcast_fusion [loop fusion]")
        self.assertIn("custom-call:AllocateBuffer", sigs)
        # Heaviest signature takes ~95% of group time, not "240 events".
        self.assertGreater(rows[0]["pct_of_group"], 90.0)
        self.assertEqual(rows[0]["n_executions"], 1)
        # AllocateBuffer reports its 2 events together, separate from kernel.
        ab = next(r for r in rows if r["signature"] == "custom-call:AllocateBuffer")
        self.assertEqual(ab["n_executions"], 2)
        self.assertEqual(ab["total_dur_ps"], 150)

    def test_example_hlo_op_picks_longest_duration(self):
        # example_hlo_op should track the heaviest single event in the group,
        # not the first one seen — otherwise a 0-cost AllocateBuffer placeholder
        # appearing before a real Pallas kernel would be reported as the
        # "representative" op and mislead users.
        recs = [
            _make_record(agg_key="A", hlo_op="placeholder", duration_ps=10),
            _make_record(agg_key="A", hlo_op="heavy",       duration_ps=1000),
            _make_record(agg_key="A", hlo_op="medium",      duration_ps=100),
        ]
        out = cb._aggregate_by_key(recs)
        self.assertEqual(out["A"].example_hlo_op, "heavy")
        self.assertEqual(out["A"].example_hlo_op_dur_ps, 1000)


class TestComputeTotals(unittest.TestCase):
    def test_per_kind_aggregation(self):
        recs = [
            _make_record(kind="compute",   duration_ps=100),
            _make_record(kind="compute",   duration_ps=200),
            _make_record(kind="data_move", duration_ps=50),
            _make_record(kind="comm",      duration_ps=30),
            _make_record(kind="other",     duration_ps=5,
                         hlo_category="never-seen"),
        ]
        pstats = cb._PipelineStats(while_total_ps=4242,
                                     unknown_categories={"never-seen": 1},
                                     n_events_unresolved=7)
        totals = cb._compute_totals(recs, pstats=pstats, step_duration_ps=10_000)
        self.assertEqual(totals["n_events_total"], 5)
        self.assertEqual(totals["n_events_compute"], 2)
        self.assertEqual(totals["n_events_data_move"], 1)
        self.assertEqual(totals["n_events_comm"], 1)
        self.assertEqual(totals["n_events_other"], 1)
        self.assertEqual(totals["n_events_unresolved"], 7)
        self.assertEqual(totals["compute_duration_ps"], 300)
        self.assertEqual(totals["data_move_duration_ps"], 50)
        self.assertEqual(totals["comm_duration_ps"], 30)
        self.assertEqual(totals["other_duration_ps"], 5)
        self.assertEqual(totals["while_container_duration_ps"], 4242)
        self.assertEqual(totals["non_while_duration_ps_sum"], 300 + 50 + 30 + 5)
        self.assertEqual(totals["unknown_categories"], {"never-seen": 1})
        self.assertAlmostEqual(totals["while_pct_of_step"], 100.0 * 4242 / 10000)


class TestRunSummaryMode(unittest.TestCase):
    def _ctx(self, **overrides):
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

    def test_top_compute_groups_sorted_descending(self):
        recs = [
            _make_record(agg_key="A", kind="compute", duration_ps=10),
            _make_record(agg_key="B", kind="compute", duration_ps=100),
            _make_record(agg_key="C", kind="compute", duration_ps=50),
        ]
        doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=10)
        ranks = [g["agg_key"] for g in doc["top_compute_groups"]]
        self.assertEqual(ranks, ["B", "C", "A"])
        self.assertEqual(doc["top_compute_groups"][0]["rank"], 1)

    def test_top_truncates_to_K(self):
        recs = [_make_record(agg_key=f"K{i}", kind="compute", duration_ps=i + 1)
                for i in range(20)]
        doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=5)
        self.assertEqual(len(doc["top_compute_groups"]), 5)
        self.assertEqual(doc["tail_compute"]["n_groups"], 15)
        # Tail duration = sum of the 15 smallest durations (1..15)
        self.assertEqual(doc["tail_compute"]["dur_ps"], sum(range(1, 16)))

    def test_pct_denominators(self):
        recs = [
            _make_record(agg_key="A", kind="compute",   duration_ps=400),
            _make_record(agg_key="B", kind="data_move", duration_ps=100),
            _make_record(agg_key="C", kind="comm",      duration_ps=300),
        ]
        # comm excluded by default
        doc = cb._run_summary_mode(recs, ctx=self._ctx(step_duration_ps=1000),
                                     include_comm=False, top=10)
        a = doc["top_compute_groups"][0]
        self.assertEqual(a["agg_key"], "A")
        # pct_of_compute denom = compute_duration_ps = 400
        self.assertAlmostEqual(a["pct_of_compute"], 100.0)
        # pct_of_step denom = step_duration_ps = 1000
        self.assertAlmostEqual(a["pct_of_step"], 40.0)

    def test_include_comm_keeps_comm_records_in_totals(self):
        recs = [
            _make_record(agg_key="A", kind="compute", duration_ps=100),
            _make_record(agg_key="B", kind="comm",    duration_ps=200),
        ]
        doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=True, top=10)
        self.assertEqual(doc["totals"]["comm_duration_ps"], 200)

    def test_totals_and_rollup_include_comm_even_when_flag_false(self):
        # Spec §5: `totals` and `by_kind_rollup` always reflect the whole
        # step (cross-kind), regardless of include_comm. The flag only
        # affects which records get ranked into top_compute_groups.
        recs = [
            _make_record(agg_key="A", kind="compute",   duration_ps=100),
            _make_record(agg_key="B", kind="comm",      duration_ps=200),
            _make_record(agg_key="C", kind="data_move", duration_ps=50),
        ]
        doc = cb._run_summary_mode(recs, ctx=self._ctx(),
                                   include_comm=False, top=10)
        self.assertEqual(doc["totals"]["comm_duration_ps"], 200)
        self.assertEqual(doc["totals"]["data_move_duration_ps"], 50)
        rollup = {r["kind"]: r for r in doc["by_kind_rollup"]}
        self.assertEqual(rollup["comm"]["dur_ps"], 200)
        self.assertEqual(rollup["data_move"]["dur_ps"], 50)
        # But comm must NOT appear in top_compute_groups.
        ranked_keys = {g["agg_key"] for g in doc["top_compute_groups"]}
        self.assertNotIn("B", ranked_keys)
        self.assertNotIn("C", ranked_keys)
        self.assertIn("A", ranked_keys)

    def test_agg_key_coverage(self):
        # Coverage counts records that go into the compute ranking, so all
        # records must be kind="compute" for the assertion to hit. The
        # data_move case is exercised separately in mode 3 tests.
        recs = [
            _make_record(agg_key="stack:abc", agg_key_kind="stack",
                         kind="compute", duration_ps=10),
            _make_record(agg_key="stack:abc", agg_key_kind="stack",
                         kind="compute", duration_ps=10),
            _make_record(agg_key="tfop:Foo",  agg_key_kind="tf_op",
                         kind="compute", duration_ps=20),
            _make_record(agg_key="nosrc:pad", agg_key_kind="no_source",
                         kind="compute", duration_ps=5),
        ]
        doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=10)
        self.assertEqual(doc["agg_key_coverage"],
                         {"stack": 2, "tf_op": 1, "no_source": 1})

    def test_top_default_50(self):
        recs = [_make_record(agg_key=f"K{i}", kind="compute", duration_ps=i + 1)
                for i in range(60)]
        doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=50)
        self.assertEqual(len(doc["top_compute_groups"]), 50)


class TestSummaryEndToEnd(unittest.TestCase):
    def test_summary_on_synthetic_xspace_emits_valid_json(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                       offset_ps=100, duration_ps=400_000_000,
                       hlo_category="loop fusion", tf_op="jit/Big",
                       flops=1_000_000, bytes_accessed=1024,
                       shape_with_layout="bf16[8]{0}")
        add_hlo_event(xs, em_id=11, hlo_op_text="small = bf16[2] fusion(...)",
                       offset_ps=600, duration_ps=10_000_000,
                       hlo_category="loop fusion", tf_op="jit/Small",
                       flops=50_000, bytes_accessed=64,
                       shape_with_layout="bf16[2]{0}")
        add_hlo_event(xs, em_id=12, hlo_op_text="copy.0",
                       offset_ps=800, duration_ps=5_000_000,
                       hlo_category="data formatting")
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
            r = subprocess.run(
                [sys.executable, str(SCRIPT), tmp, "--mode", "summary"],
                capture_output=True, text=True,
            )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "ok")
        self.assertEqual(doc["mode"], "summary")
        self.assertEqual(doc["totals"]["n_events_total"], 3)
        self.assertEqual(doc["totals"]["n_events_compute"], 2)
        self.assertEqual(doc["totals"]["n_events_data_move"], 1)
        self.assertGreaterEqual(len(doc["top_compute_groups"]), 1)
        self.assertEqual(doc["top_compute_groups"][0]["agg_key"], "tfop:jit/Big")


if __name__ == "__main__":
    unittest.main()
