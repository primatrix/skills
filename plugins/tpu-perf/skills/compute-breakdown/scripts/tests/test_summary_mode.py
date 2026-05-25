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
    make_minimal_xspace, add_hlo_event,
)


def _make_record(**overrides):
    """Build an EventRecord with sensible defaults; override any field."""
    defaults = dict(
        duration_ps=100, offset_ps=0, step_id=0,
        hlo_category="loop fusion", kind="compute",
        hlo_op="x", tf_op=None, source_stat=None,
        source_stack=None, source_inner=None, source_stack_hash=None,
        agg_key="tfop:x", agg_key_kind="tf_op",
        flops=None, model_flops=None, bytes_accessed=None,
        raw_bytes_accessed=None, shape_with_layout=None,
        dtype=None, dtype_uncertain=False,
        program_id=None, deduplicated_name=None,
    )
    defaults.update(overrides)
    return cb.EventRecord(**defaults)


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

    def test_example_hlo_op_is_first_seen(self):
        recs = [
            _make_record(agg_key="A", hlo_op="first"),
            _make_record(agg_key="A", hlo_op="second"),
        ]
        out = cb._aggregate_by_key(recs)
        self.assertEqual(out["A"].example_hlo_op, "first")


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
