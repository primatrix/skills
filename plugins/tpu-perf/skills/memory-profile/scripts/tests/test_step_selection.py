"""Tests for step window selection (peak/last/first/explicit/all-trace)."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import StepPolicyError, pick_step_window  # noqa: E402

from tests._synthetic import PlaneBuilder, make_xspace  # noqa: E402


def _xspace_with_steps(step_intervals_ns):
    """Build an XSpace with /device:TPU:0 'Steps' line at the given (start, end) ns."""
    xs = make_xspace()
    tpu = PlaneBuilder(xs, "/device:TPU:0")
    line = tpu.add_line("Steps", timestamp_ns=0)
    for i, (start_ns, end_ns) in enumerate(step_intervals_ns):
        # offset_ps = ns * 1000; duration_ps = (end-start) * 1000.
        line.add_event(
            f"step_{i}",
            offset_ps=start_ns * 1000,
            duration_ps=(end_ns - start_ns) * 1000,
        )
    return xs


class TestStepWindow(unittest.TestCase):
    def test_all_trace_yields_unbounded_window(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000)])
        sw = pick_step_window(xs, all_trace=True, policy="peak", explicit=None,
                              peak_ts_ns_hint=500)
        self.assertEqual(sw.source, "all_trace")
        self.assertEqual(sw.range_ns, (0, 2**63 - 1))
        self.assertIsNone(sw.id)

    def test_explicit_step_selects_indexed_event(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=1,
                              peak_ts_ns_hint=None)
        self.assertEqual(sw.id, "step_1")
        self.assertEqual(sw.range_ns, (2_000, 3_000))
        self.assertEqual(sw.source, "steps_line")

    def test_explicit_step_out_of_range_raises(self):
        xs = _xspace_with_steps([(0, 1_000)])
        with self.assertRaises(StepPolicyError):
            pick_step_window(xs, all_trace=False, policy="peak", explicit=5,
                             peak_ts_ns_hint=None)

    def test_policy_last_picks_last_step_event(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000)])
        sw = pick_step_window(xs, all_trace=False, policy="last", explicit=None,
                              peak_ts_ns_hint=None)
        self.assertEqual(sw.id, "step_1")
        self.assertEqual(sw.range_ns, (2_000, 3_000))

    def test_policy_first_picks_first_step_event(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000)])
        sw = pick_step_window(xs, all_trace=False, policy="first", explicit=None,
                              peak_ts_ns_hint=None)
        self.assertEqual(sw.id, "step_0")

    def test_policy_peak_picks_step_containing_hint(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=None,
                              peak_ts_ns_hint=2_500)
        self.assertEqual(sw.id, "step_1")

    def test_policy_peak_falls_back_to_execute_event_when_no_steps_line(self):
        xs = make_xspace()
        host = PlaneBuilder(xs, "/host:CPU")
        line = host.add_line("pjrt_tpu_execute/0", timestamp_ns=0)
        line.add_event("[0] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                       offset_ps=0, duration_ps=10_000_000)  # [0ns, 10_000ns]
        line.add_event("[1] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                       offset_ps=20_000_000, duration_ps=10_000_000)
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=None,
                              peak_ts_ns_hint=22_000)
        self.assertEqual(sw.source, "execute_event")
        self.assertEqual(sw.range_ns, (20_000, 30_000))

    def test_policy_peak_returns_none_when_no_step_data_at_all(self):
        xs = make_xspace()
        PlaneBuilder(xs, "/host:CPU")
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=None,
                              peak_ts_ns_hint=42)
        self.assertIsNone(sw)


if __name__ == "__main__":
    unittest.main()
