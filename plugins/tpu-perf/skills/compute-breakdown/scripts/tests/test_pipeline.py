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


if __name__ == "__main__":
    unittest.main()
