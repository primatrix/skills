"""Unit tests for the compute_breakdown.py shared pipeline (Stages 1-3) and CLI."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "compute_breakdown.py"


class TestCLI(unittest.TestCase):
    def test_help_runs(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--mode", r.stdout)
        self.assertIn("summary", r.stdout)
        self.assertIn("by_source", r.stdout)
        self.assertIn("non_compute", r.stdout)
        self.assertIn("roofline", r.stdout)

    def test_no_xplane_returns_absent(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp", "--mode", "summary"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "absent")
        self.assertEqual(doc["reason"], "no_xplane_pb")
        self.assertEqual(doc["mode"], "summary")
        self.assertEqual(doc["profile_dir"], "/tmp")
        self.assertEqual(doc["notes"], [])

    def test_step_and_step_id_mutually_exclusive(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp",
             "--mode", "summary", "--step", "0", "--step-id", "x"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 1)
        self.assertIn("step", r.stderr.lower())


# ----------------------------------------------------------------------
# Synthetic XSpace builders.
# ----------------------------------------------------------------------
import pathlib
_PROTO_DIR = pathlib.Path(__file__).resolve().parent.parent / "_proto"
sys.path.insert(0, str(_PROTO_DIR))
import xplane_pb2  # noqa: E402


def _add_stat_meta(plane, sm_id: int, name: str) -> None:
    sm = plane.stat_metadata[sm_id]
    sm.id = sm_id
    sm.name = name


def _add_event_meta(plane, em_id: int, name: str, stats: dict | None = None) -> None:
    """stats: {stat_metadata_id: ('str_value'|'int64_value'|'uint64_value'|'double_value', value)}"""
    em = plane.event_metadata[em_id]
    em.id = em_id
    em.name = name
    for sm_id, (vfield, vval) in (stats or {}).items():
        s = em.stats.add()
        s.metadata_id = sm_id
        setattr(s, vfield, vval)


def _add_event(line, em_id: int, offset_ps: int, duration_ps: int,
               per_event_stats: dict | None = None) -> None:
    ev = line.events.add()
    ev.metadata_id = em_id
    ev.offset_ps = offset_ps
    ev.duration_ps = duration_ps
    for sm_id, (vfield, vval) in (per_event_stats or {}).items():
        s = ev.stats.add()
        s.metadata_id = sm_id
        setattr(s, vfield, vval)


# Stat-metadata IDs used across synthetic fixtures (arbitrary but stable).
SM_HLO_CATEGORY = 1
SM_TF_OP = 2
SM_PROGRAM_ID = 3
SM_FLOPS = 4
SM_MODEL_FLOPS = 5
SM_BYTES_ACCESSED = 6
SM_RAW_BYTES_ACCESSED = 7
SM_SHAPE = 8
SM_SOURCE = 9
SM_SOURCE_STACK = 10
SM_FLOW = 11
SM_DEVICE_DURATION_PS = 12
SM_DEDUP_NAME = 13


def make_minimal_xspace(*, device_name: str = "/device:TPU:0",
                         steps: list[tuple[int, int, int]] | None = None) -> "xplane_pb2.XSpace":
    """Build an XSpace with one device plane carrying a `Steps` line and an empty
    `XLA Ops` line. `steps` is [(em_id, offset_ps, duration_ps), ...]."""
    xs = xplane_pb2.XSpace()
    plane = xs.planes.add()
    plane.id = 1
    plane.name = device_name

    _add_stat_meta(plane, SM_HLO_CATEGORY, "hlo_category")
    _add_stat_meta(plane, SM_TF_OP, "tf_op")
    _add_stat_meta(plane, SM_PROGRAM_ID, "program_id")
    _add_stat_meta(plane, SM_FLOPS, "flops")
    _add_stat_meta(plane, SM_MODEL_FLOPS, "model_flops")
    _add_stat_meta(plane, SM_BYTES_ACCESSED, "bytes_accessed")
    _add_stat_meta(plane, SM_RAW_BYTES_ACCESSED, "raw_bytes_accessed")
    _add_stat_meta(plane, SM_SHAPE, "shape_with_layout")
    _add_stat_meta(plane, SM_SOURCE, "source")
    _add_stat_meta(plane, SM_SOURCE_STACK, "source_stack")
    _add_stat_meta(plane, SM_FLOW, "flow")
    _add_stat_meta(plane, SM_DEVICE_DURATION_PS, "device_duration_ps")
    _add_stat_meta(plane, SM_DEDUP_NAME, "deduplicated_name")

    steps_line = plane.lines.add()
    steps_line.id = 100
    steps_line.name = "Steps"
    steps_line.timestamp_ns = 0
    steps_line.duration_ps = 0
    for em_id, off, dur in (steps or []):
        _add_event_meta(plane, em_id, f"step_{em_id}")
        _add_event(steps_line, em_id, off, dur)

    ops_line = plane.lines.add()
    ops_line.id = 101
    ops_line.name = "XLA Ops"
    ops_line.timestamp_ns = 0
    ops_line.duration_ps = 0
    return xs


def add_hlo_event(xs, *, em_id: int, hlo_op_text: str, offset_ps: int,
                   duration_ps: int, hlo_category: str,
                   tf_op: str | None = None,
                   source_stack: str | None = None,
                   source_inner: str | None = None,
                   flops: int | None = None,
                   model_flops: int | None = None,
                   bytes_accessed: int | None = None,
                   raw_bytes_accessed: int | None = None,
                   shape_with_layout: str | None = None,
                   program_id: int | None = None,
                   deduplicated_name: str | None = None) -> None:
    """Add one HLO event on the device plane's XLA Ops line. Stats are
    attached to XEventMetadata.stats (op-level) per profile-anatomy
    schema; per-event stats are not used for HLO ops."""
    plane = xs.planes[0]
    meta_stats: dict = {SM_HLO_CATEGORY: ("str_value", hlo_category)}
    if tf_op is not None:
        meta_stats[SM_TF_OP] = ("str_value", tf_op)
    if source_stack is not None:
        meta_stats[SM_SOURCE_STACK] = ("str_value", source_stack)
    if source_inner is not None:
        meta_stats[SM_SOURCE] = ("str_value", source_inner)
    if flops is not None:
        meta_stats[SM_FLOPS] = ("int64_value", flops)
    if model_flops is not None:
        meta_stats[SM_MODEL_FLOPS] = ("int64_value", model_flops)
    if bytes_accessed is not None:
        meta_stats[SM_BYTES_ACCESSED] = ("int64_value", bytes_accessed)
    if raw_bytes_accessed is not None:
        meta_stats[SM_RAW_BYTES_ACCESSED] = ("int64_value", raw_bytes_accessed)
    if shape_with_layout is not None:
        meta_stats[SM_SHAPE] = ("str_value", shape_with_layout)
    if program_id is not None:
        meta_stats[SM_PROGRAM_ID] = ("int64_value", program_id)
    if deduplicated_name is not None:
        meta_stats[SM_DEDUP_NAME] = ("str_value", deduplicated_name)
    _add_event_meta(plane, em_id, hlo_op_text, meta_stats)
    ops_line = next(l for l in plane.lines if l.name == "XLA Ops")
    _add_event(ops_line, em_id, offset_ps, duration_ps)


class TestSyntheticBuilders(unittest.TestCase):
    def test_minimal_xspace_has_device_plane_and_two_lines(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        self.assertEqual(len(xs.planes), 1)
        plane = xs.planes[0]
        self.assertEqual(plane.name, "/device:TPU:0")
        self.assertEqual({l.name for l in plane.lines}, {"Steps", "XLA Ops"})
        steps_line = next(l for l in plane.lines if l.name == "Steps")
        self.assertEqual(len(steps_line.events), 1)
        self.assertEqual(steps_line.events[0].duration_ps, 1_000_000_000)

    def test_add_hlo_event_attaches_meta_stats_not_per_event(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="fusion.0 = bf16[8,8] fusion(...)",
                      offset_ps=10, duration_ps=500, hlo_category="loop fusion",
                      tf_op="jit/Foo", flops=12345, bytes_accessed=678,
                      shape_with_layout="bf16[8,8]{1,0}",
                      source_stack="/x/y.py:5:1\n/x/z.py:9:2")
        plane = xs.planes[0]
        ops_line = next(l for l in plane.lines if l.name == "XLA Ops")
        self.assertEqual(len(ops_line.events), 1)
        ev = ops_line.events[0]
        self.assertEqual(len(ev.stats), 0, "no per-event stats expected")
        em = plane.event_metadata[ev.metadata_id]
        self.assertGreater(len(em.stats), 0, "op-level stats live on event_metadata")


# ----------------------------------------------------------------------
# EventRecord dataclass + meta-stat extraction.
# ----------------------------------------------------------------------
# _PROTO_DIR (set up in Task 5 Step 1) points at scripts/_proto. Its
# parent is scripts/, where compute_breakdown.py lives — that directory
# MUST be on sys.path before the import below or it raises
# ModuleNotFoundError. Insert it first, then import.
sys.path.insert(0, str(_PROTO_DIR.parent))
import compute_breakdown as cb  # noqa: E402  -- after sys.path insert above


def _make_record(**overrides):
    """Build an EventRecord with sensible defaults; override any field.
    Shared factory used by all test modules."""
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


class TestEventRecord(unittest.TestCase):
    def test_event_record_has_all_spec_fields(self):
        # Fields per spec §4
        expected = {
            "duration_ps", "offset_ps", "step_id",
            "hlo_category", "kind",
            "hlo_op", "tf_op",
            "source_stat", "source_stack", "source_inner", "source_stack_hash",
            "agg_key", "agg_key_kind",
            "flops", "model_flops", "bytes_accessed", "raw_bytes_accessed",
            "shape_with_layout", "dtype", "dtype_uncertain",
            "program_id", "deduplicated_name",
        }
        actual = {f.name for f in cb.EventRecord.__dataclass_fields__.values()}
        self.assertEqual(actual, expected)


class TestExtractMetaStats(unittest.TestCase):
    def test_resolves_stat_names_via_stat_metadata(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="fusion.0",
                      offset_ps=0, duration_ps=100, hlo_category="loop fusion",
                      tf_op="jit/Foo", flops=42, bytes_accessed=99,
                      shape_with_layout="bf16[8]{0}")
        plane = xs.planes[0]
        em = plane.event_metadata[10]
        name_by_id = {smid: sm.name for smid, sm in plane.stat_metadata.items()}
        stats = cb._extract_meta_stats(em, name_by_id)
        self.assertEqual(stats["hlo_category"], "loop fusion")
        self.assertEqual(stats["tf_op"], "jit/Foo")
        self.assertEqual(stats["flops"], 42)
        self.assertEqual(stats["bytes_accessed"], 99)
        self.assertEqual(stats["shape_with_layout"], "bf16[8]{0}")

    def test_returns_empty_dict_for_no_stats(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        plane = xs.planes[0]
        # event_metadata with no stats
        _add_event_meta(plane, 99, "bare-op")
        name_by_id = {smid: sm.name for smid, sm in plane.stat_metadata.items()}
        self.assertEqual(cb._extract_meta_stats(plane.event_metadata[99], name_by_id), {})


class TestClassifyKind(unittest.TestCase):
    def test_compute_categories(self):
        for cat in ["loop fusion", "convolution fusion", "custom fusion",
                    "output fusion", "non-fusion elementwise", "reduce",
                    "reduce-window", "sort", "rng-bit-generator", "custom-call"]:
            self.assertEqual(cb._classify_kind(cat), "compute", cat)

    def test_data_move_categories(self):
        for cat in ["copy-start", "copy-done", "data formatting", "pad",
                    "broadcast", "slice", "dynamic-slice",
                    "dynamic-update-slice", "iota", "convert"]:
            self.assertEqual(cb._classify_kind(cat), "data_move", cat)

    def test_comm_categories(self):
        for cat in ["async-start", "async-done", "all-reduce", "all-gather",
                    "reduce-scatter", "collective-permute"]:
            self.assertEqual(cb._classify_kind(cat), "comm", cat)

    def test_unknown_category_falls_back_to_other(self):
        self.assertEqual(cb._classify_kind("scalar-thing-2031"), "other")
        self.assertEqual(cb._classify_kind(""), "other")


class TestParseDtype(unittest.TestCase):
    def test_known_dtypes(self):
        self.assertEqual(cb._parse_dtype("bf16[8192,4096]{1,0}"), "bf16")
        self.assertEqual(cb._parse_dtype("f8e4m3fn[1024,4096]{1,0}"), "fp8")
        self.assertEqual(cb._parse_dtype("f8e5m2[64]{0}"), "fp8")
        self.assertEqual(cb._parse_dtype("f32[]"), "fp32")
        self.assertEqual(cb._parse_dtype("f16[16,16]{1,0}"), "fp16")

    def test_other_for_unknown_or_unparseable(self):
        self.assertEqual(cb._parse_dtype("s32[8]{0}"), "other")
        self.assertEqual(cb._parse_dtype("s8[8]{0}"), "other")
        self.assertEqual(cb._parse_dtype("pred[]"), "other")
        self.assertEqual(cb._parse_dtype("(bf16[8],bf16[8])"), "other")
        self.assertEqual(cb._parse_dtype("garbage no bracket"), "other")
        self.assertIsNone(cb._parse_dtype(None))


class TestAggKey(unittest.TestCase):
    def test_priority1_stack(self):
        k, kind, h = cb._compute_agg_key(
            source_stack="/x/y.py:5:1\n/x/z.py:9:2",
            tf_op="jit/Foo",
            hlo_category="loop fusion")
        self.assertTrue(k.startswith("stack:"))
        self.assertEqual(kind, "stack")
        self.assertEqual(len(h), 16)
        self.assertEqual(k, f"stack:{h}")

    def test_priority2_tfop(self):
        k, kind, h = cb._compute_agg_key(
            source_stack=None, tf_op="jit/Foo", hlo_category="loop fusion")
        self.assertEqual(k, "tfop:jit/Foo")
        self.assertEqual(kind, "tf_op")
        self.assertIsNone(h)

    def test_priority2_tfop_when_stack_is_empty_string(self):
        # spec §4.1: "source_stack empty" — empty string treated same as None
        k, kind, _ = cb._compute_agg_key(
            source_stack="", tf_op="jit/Bar", hlo_category="reduce")
        self.assertEqual(k, "tfop:jit/Bar")
        self.assertEqual(kind, "tf_op")

    def test_priority3_no_source(self):
        k, kind, h = cb._compute_agg_key(
            source_stack=None, tf_op=None, hlo_category="copy-done")
        self.assertEqual(k, "nosrc:copy-done")
        self.assertEqual(kind, "no_source")
        self.assertIsNone(h)

    def test_priority3_when_tfop_is_empty_string(self):
        k, kind, _ = cb._compute_agg_key(
            source_stack=None, tf_op="", hlo_category="pad")
        self.assertEqual(k, "nosrc:pad")
        self.assertEqual(kind, "no_source")


class TestInnerFrame(unittest.TestCase):
    def test_strips_column_suffix(self):
        self.assertEqual(cb._inner_frame("/a/b.py:5:1\n/a/c.py:9:2"), "/a/c.py:9")

    def test_keeps_file_line_when_no_column(self):
        self.assertEqual(cb._inner_frame("/a/b.py:5\n/a/c.py:9"), "/a/c.py:9")

    def test_skips_trailing_blank_lines(self):
        self.assertEqual(cb._inner_frame("/a/b.py:5:1\n/a/c.py:9:2\n\n"),
                         "/a/c.py:9")

    def test_single_line(self):
        self.assertEqual(cb._inner_frame("/single.py:42:0"), "/single.py:42")

    def test_returns_none_for_none_or_empty(self):
        self.assertIsNone(cb._inner_frame(None))
        self.assertIsNone(cb._inner_frame(""))
        self.assertIsNone(cb._inner_frame("\n\n"))


class TestDtypeUncertain(unittest.TestCase):
    _ALLOWED_CATS = ("convolution fusion", "custom fusion", "output fusion", "custom-call")
    _ALLOWED_DTYPES = ("bf16", "fp32")

    def test_true_only_when_both_conditions_hold(self):
        for cat in self._ALLOWED_CATS:
            for dtype in self._ALLOWED_DTYPES:
                self.assertTrue(cb._is_dtype_uncertain(cat, dtype),
                                msg=f"{cat}/{dtype}")

    def test_false_when_loop_fusion(self):
        for dtype in self._ALLOWED_DTYPES:
            self.assertFalse(cb._is_dtype_uncertain("loop fusion", dtype))

    def test_false_when_non_fusion_elementwise(self):
        self.assertFalse(cb._is_dtype_uncertain("non-fusion elementwise", "bf16"))

    def test_false_when_dtype_fp8(self):
        for cat in self._ALLOWED_CATS:
            self.assertFalse(cb._is_dtype_uncertain(cat, "fp8"))

    def test_false_when_dtype_other_or_none(self):
        self.assertFalse(cb._is_dtype_uncertain("convolution fusion", "other"))
        self.assertFalse(cb._is_dtype_uncertain("convolution fusion", None))


class TestPickStepWindow(unittest.TestCase):
    def _xs_three_steps(self):
        return make_minimal_xspace(steps=[
            (1, 1_000_000, 5_000_000),       # step 0 ends at 6_000_000
            (2, 7_000_000, 5_000_000),       # step 1 ends at 12_000_000
            (3, 13_000_000, 5_000_000),      # step 2 ends at 18_000_000
        ])

    def test_default_picks_middle_step(self):
        xs = self._xs_three_steps()
        plane = xs.planes[0]
        ev, sid, s, e, notes = cb._pick_step_window(plane, step_idx=None, step_id=None)
        self.assertEqual(sid, 1)
        self.assertEqual(s, 7_000_000)
        self.assertEqual(e, 12_000_000)
        self.assertEqual(notes, [])

    def test_step_idx_picks_specific(self):
        xs = self._xs_three_steps()
        ev, sid, s, e, _ = cb._pick_step_window(xs.planes[0], step_idx=0, step_id=None)
        self.assertEqual(sid, 0)
        self.assertEqual(s, 1_000_000)
        self.assertEqual(e, 6_000_000)

    def test_step_idx_out_of_range_raises(self):
        xs = self._xs_three_steps()
        with self.assertRaises(ValueError):
            cb._pick_step_window(xs.planes[0], step_idx=99, step_id=None)
        with self.assertRaises(ValueError):
            cb._pick_step_window(xs.planes[0], step_idx=-1, step_id=None)

    def test_step_id_exact_match(self):
        xs = self._xs_three_steps()
        # Synthetic step names are step_1, step_2, step_3 (em_id mapped)
        ev, sid, s, e, notes = cb._pick_step_window(
            xs.planes[0], step_idx=None, step_id="step_2")
        self.assertEqual(sid, 1)
        self.assertEqual(s, 7_000_000)
        self.assertEqual(notes, [])

    def test_step_id_zero_matches_raises(self):
        xs = self._xs_three_steps()
        with self.assertRaises(ValueError):
            cb._pick_step_window(xs.planes[0], step_idx=None, step_id="nope")

    def test_step_id_multi_match_picks_earliest_with_note(self):
        # Two steps share the same metadata name (we re-use em_id 1)
        xs = make_minimal_xspace(steps=[(1, 5_000_000, 1_000_000),
                                         (1, 1_000_000, 1_000_000),  # earlier
                                         (1, 9_000_000, 1_000_000)])
        ev, sid, s, e, notes = cb._pick_step_window(
            xs.planes[0], step_idx=None, step_id="step_1")
        self.assertEqual(s, 1_000_000)
        self.assertIn("multi-match for step-id; picked first", notes)

    def test_no_steps_line_falls_back_to_full_xla_ops_window(self):
        xs = xplane_pb2.XSpace()
        plane = xs.planes.add()
        plane.name = "/device:TPU:0"
        ops_line = plane.lines.add()
        ops_line.name = "XLA Ops"
        # Two events spanning [10, 30) and [40, 90)
        _add_event_meta(plane, 1, "x")
        _add_event(ops_line, 1, 10, 20)
        _add_event(ops_line, 1, 40, 50)
        ev, sid, s, e, notes = cb._pick_step_window(plane, step_idx=None, step_id=None)
        self.assertIsNone(ev)
        self.assertEqual(sid, -1)
        self.assertEqual(s, 10)
        self.assertEqual(e, 90)
        self.assertIn("no Steps line; falling back to full-plane window", notes)


class TestIterEventRecords(unittest.TestCase):
    def _xs_with_step_and_ops(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        # In window: [0, 1e9)
        add_hlo_event(xs, em_id=10,
                      hlo_op_text="fusion.0 = bf16[8,8] fusion(bf16[8,8] %p)",
                      offset_ps=100, duration_ps=500,
                      hlo_category="loop fusion",
                      tf_op="jit/Foo",
                      flops=1024, bytes_accessed=128,
                      shape_with_layout="bf16[8,8]{1,0}",
                      source_stack="/x/foo.py:5:1\n/x/bar.py:9:2")
        # Out of window (after end)
        add_hlo_event(xs, em_id=11, hlo_op_text="late",
                      offset_ps=2_000_000_000, duration_ps=500,
                      hlo_category="loop fusion")
        # while event in window
        add_hlo_event(xs, em_id=12, hlo_op_text="while.0 = ...",
                      offset_ps=1000, duration_ps=10_000,
                      hlo_category="while")
        # data_move with shape_with_layout
        add_hlo_event(xs, em_id=13, hlo_op_text="copy.0 = bf16[8] copy(bf16[8])",
                      offset_ps=2000, duration_ps=200,
                      hlo_category="data formatting",
                      shape_with_layout="bf16[8]{0}")
        # unknown category
        add_hlo_event(xs, em_id=14, hlo_op_text="weirdo",
                      offset_ps=3000, duration_ps=50,
                      hlo_category="never-seen-category")
        return xs

    def test_yields_records_for_in_window_non_while_events(self):
        xs = self._xs_with_step_and_ops()
        plane = xs.planes[0]
        stats = cb._PipelineStats()
        recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                            step_id=0, stats=stats))
        # 3 records: fusion.0, copy.0, weirdo. (while skipped, late out of window.)
        self.assertEqual(len(recs), 3)
        kinds = sorted(r.kind for r in recs)
        self.assertEqual(kinds, ["compute", "data_move", "other"])

    def test_window_admission_is_start_inclusive_end_exclusive(self):
        xs = make_minimal_xspace(steps=[(1, 0, 100)])
        # Event starts exactly at start_ps -> admit
        add_hlo_event(xs, em_id=10, hlo_op_text="a", offset_ps=0,
                      duration_ps=10, hlo_category="loop fusion")
        # Event starts exactly at end_ps -> exclude
        add_hlo_event(xs, em_id=11, hlo_op_text="b", offset_ps=100,
                      duration_ps=10, hlo_category="loop fusion")
        # Event starts before start_ps -> exclude (even if extends in)
        add_hlo_event(xs, em_id=12, hlo_op_text="c", offset_ps=-50,
                      duration_ps=200, hlo_category="loop fusion")
        plane = xs.planes[0]
        stats = cb._PipelineStats()
        recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=100,
                                            step_id=0, stats=stats))
        ops = sorted(r.hlo_op for r in recs)
        self.assertEqual(ops, ["a"])

    def test_while_skipped_and_accumulated(self):
        xs = self._xs_with_step_and_ops()
        plane = xs.planes[0]
        stats = cb._PipelineStats()
        list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                     step_id=0, stats=stats))
        self.assertEqual(stats.while_total_ps, 10_000)

    def test_unknown_category_counted(self):
        xs = self._xs_with_step_and_ops()
        plane = xs.planes[0]
        stats = cb._PipelineStats()
        list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                     step_id=0, stats=stats))
        self.assertEqual(stats.unknown_categories, {"never-seen-category": 1})

    def test_unresolved_event_metadata_dropped_and_counted(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        plane = xs.planes[0]
        ops = next(l for l in plane.lines if l.name == "XLA Ops")
        # Event references an event_metadata id that doesn't exist
        ev = ops.events.add()
        ev.metadata_id = 9999
        ev.offset_ps = 100
        ev.duration_ps = 50
        stats = cb._PipelineStats()
        recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                            step_id=0, stats=stats))
        self.assertEqual(recs, [])
        self.assertEqual(stats.n_events_unresolved, 1)

    def test_record_field_population(self):
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        add_hlo_event(xs, em_id=10,
                      hlo_op_text="fusion.0 = bf16[8,8] fusion(...)",
                      offset_ps=200, duration_ps=300,
                      hlo_category="loop fusion",
                      tf_op="jit/Foo",
                      flops=2048, bytes_accessed=64,
                      raw_bytes_accessed=72, model_flops=1024,
                      shape_with_layout="bf16[8,8]{1,0}",
                      program_id=5,
                      deduplicated_name="dedup.0",
                      source_inner="/x/foo.py:5",
                      source_stack="/x/foo.py:5:1\n/x/bar.py:9:2")
        plane = xs.planes[0]
        stats = cb._PipelineStats()
        recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                            step_id=42, stats=stats))
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r.duration_ps, 300)
        self.assertEqual(r.offset_ps, 200)
        self.assertEqual(r.step_id, 42)
        self.assertEqual(r.hlo_category, "loop fusion")
        self.assertEqual(r.kind, "compute")
        self.assertEqual(r.tf_op, "jit/Foo")
        self.assertEqual(r.source_stat, "/x/foo.py:5")
        self.assertEqual(r.source_stack, "/x/foo.py:5:1\n/x/bar.py:9:2")
        self.assertEqual(r.source_inner, "/x/bar.py:9")
        self.assertEqual(r.agg_key_kind, "stack")
        self.assertEqual(r.flops, 2048)
        self.assertEqual(r.model_flops, 1024)
        self.assertEqual(r.bytes_accessed, 64)
        self.assertEqual(r.raw_bytes_accessed, 72)
        self.assertEqual(r.shape_with_layout, "bf16[8,8]{1,0}")
        self.assertEqual(r.dtype, "bf16")
        self.assertFalse(r.dtype_uncertain)
        self.assertEqual(r.program_id, 5)
        self.assertEqual(r.deduplicated_name, "dedup.0")


class TestLoadAndNormalize(unittest.TestCase):
    def _write_xspace(self, xs, tmpdir):
        path = pathlib.Path(tmpdir) / "synthetic.xplane.pb"
        path.write_bytes(xs.SerializeToString())
        return path

    def test_loads_picks_step_yields_records(self):
        import tempfile
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000),
                                         (2, 1_000_000_000, 1_000_000_000)])
        add_hlo_event(xs, em_id=10, hlo_op_text="x", offset_ps=500,
                      duration_ps=100, hlo_category="loop fusion",
                      tf_op="jit/Foo", flops=10, bytes_accessed=2,
                      shape_with_layout="bf16[1]{0}")
        add_hlo_event(xs, em_id=11, hlo_op_text="y",
                      offset_ps=1_000_000_500, duration_ps=200,
                      hlo_category="loop fusion")
        with tempfile.TemporaryDirectory() as tmp:
            self._write_xspace(xs, tmp)
            records, ctx = cb._load_and_normalize(
                profile_dir=tmp, device="/device:TPU:0",
                step_idx=None, step_id=None)
        # default = middle step (idx 1) -> window [1e9, 2e9)
        self.assertEqual(ctx["step_id"], 1)
        self.assertEqual(ctx["step_window_ps"], [1_000_000_000, 2_000_000_000])
        self.assertEqual(ctx["step_duration_ps"], 1_000_000_000)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].hlo_op, "y")

    def test_device_not_found_returns_none(self):
        import tempfile
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        with tempfile.TemporaryDirectory() as tmp:
            self._write_xspace(xs, tmp)
            records, ctx = cb._load_and_normalize(
                profile_dir=tmp, device="/device:TPU:99",
                step_idx=None, step_id=None)
        self.assertIsNone(records)
        self.assertEqual(ctx["status"], "absent")
        self.assertEqual(ctx["reason"], "device_not_found")

    def test_no_xla_ops_line_returns_absent(self):
        import tempfile
        xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
        # Remove the XLA Ops line
        plane = xs.planes[0]
        for i, l in enumerate(list(plane.lines)):
            if l.name == "XLA Ops":
                del plane.lines[i]
                break
        with tempfile.TemporaryDirectory() as tmp:
            self._write_xspace(xs, tmp)
            records, ctx = cb._load_and_normalize(
                profile_dir=tmp, device="/device:TPU:0",
                step_idx=None, step_id=None)
        self.assertIsNone(records)
        self.assertEqual(ctx["status"], "absent")
        self.assertEqual(ctx["reason"], "no_xla_ops_line")


if __name__ == "__main__":
    unittest.main()
