"""Tests for the two-pass sweep: timeline samples, global peak, alive snapshot."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import (  # noqa: E402
    AllocEvent, DeallocEvent, HostAllocatorEvents, sweep_first_pass,
    snapshot_at_peak,
)


def _alloc(ts_ns, addr, size, *, shape="bf16[1]", tf_op="op", pool=0,
           bytes_allocated=None, peak=None, fragmentation=0.0,
           parent_chain=None, line="L"):
    return AllocEvent(
        ts_ns=ts_ns, addr=addr, pool_id=pool,
        requested_bytes=size, allocation_bytes=size,
        bytes_allocated=bytes_allocated if bytes_allocated is not None else size,
        peak_bytes_in_use=peak if peak is not None else size,
        fragmentation=fragmentation,
        shape=shape, tf_op=tf_op, data_type="bf16",
        parent_chain=parent_chain or ["[0] Execute (jit_train_step)"],
        line_name=line,
    )


def _dealloc(ts_ns, addr, *, bytes_allocated=0, peak=0, fragmentation=0.0):
    return DeallocEvent(
        ts_ns=ts_ns, addr=addr,
        bytes_allocated=bytes_allocated, peak_bytes_in_use=peak,
        fragmentation=fragmentation, line_name="L",
    )


class TestSweepFirstPass(unittest.TestCase):
    def test_global_peak_is_after_two_allocs_before_dealloc(self):
        events = HostAllocatorEvents(
            allocs=[
                _alloc(100, 0x1, 1000, peak=1000),
                _alloc(200, 0x2, 2000, peak=3000),
            ],
            deallocs=[_dealloc(300, 0x1)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=10)
        self.assertEqual(result.global_peak_ts_ns, 200)
        self.assertEqual(result.global_peak_bytes, 3000)
        self.assertEqual(result.trace_end_live_bytes, 2000)
        self.assertEqual(result.unmatched_dealloc_count, 0)
        self.assertEqual(result.unmatched_alloc_count, 1)  # 0x2 still live

    def test_unmatched_dealloc_counted(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 500)],
            deallocs=[_dealloc(50, 0xFFFF)],  # dealloc with no matching alloc
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=4)
        self.assertEqual(result.unmatched_dealloc_count, 1)

    def test_timeline_has_requested_sample_count(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 1000), _alloc(500, 0x2, 2000, peak=3000)],
            deallocs=[],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=5)
        self.assertEqual(len(result.timeline_samples), 5)
        for s in result.timeline_samples:
            self.assertGreaterEqual(s.bytes_allocated, 0)

    def test_drift_pct_zero_when_allocator_self_consistent(self):
        # If our running sum exactly matches the allocator's bytes_allocated, drift = 0.
        events = HostAllocatorEvents(
            allocs=[
                _alloc(100, 0x1, 1000, bytes_allocated=1000, peak=1000),
                _alloc(200, 0x2, 2000, bytes_allocated=3000, peak=3000),
            ],
            deallocs=[_dealloc(300, 0x1, bytes_allocated=2000, peak=3000)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=4)
        self.assertEqual(result.alloc_accounting_drift_pct, 0.0)


class TestSnapshotAtPeak(unittest.TestCase):
    def _events(self):
        return HostAllocatorEvents(
            allocs=[
                _alloc(100, 0x1, 1000, shape="bf16[A]", tf_op="weight",
                       parent_chain=["[0] Execute (jit_train_step)"]),
                _alloc(200, 0x2, 2000, shape="bf16[B]", tf_op="act",
                       parent_chain=["[0] Execute (jit_train_step)"]),
                _alloc(400, 0x3, 500, shape="bf16[C]", tf_op="tmp",
                       parent_chain=["[0] Execute (jit_train_step)"]),
            ],
            deallocs=[_dealloc(450, 0x3)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )

    def test_alive_at_peak_excludes_yet_to_alloc_and_already_freed(self):
        events = self._events()
        snap = snapshot_at_peak(events, peak_ts_ns=300, step_range_ns=(0, 1_000_000),
                                step_boundaries_ns=[(0, 1_000), (1_000, 2_000)],
                                persistent_threshold_steps=2)
        addrs = {b.addr for b in snap.alive}
        self.assertEqual(addrs, {0x1, 0x2})
        self.assertEqual(snap.alive_total_bytes, 3000)
        self.assertEqual(snap.bytes_total, 3000)

    def test_lifetime_class_persistent_when_crosses_threshold(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 1000, shape="bf16[W]", tf_op="weight")],
            deallocs=[],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        # 5 step boundaries between alloc_ts_ns=100 and trace end at 10_000.
        boundaries = [(0, 1_000), (1_000, 2_000), (2_000, 3_000),
                      (3_000, 4_000), (4_000, 10_000)]
        snap = snapshot_at_peak(events, peak_ts_ns=5_000,
                                step_range_ns=(4_000, 10_000),
                                step_boundaries_ns=boundaries,
                                persistent_threshold_steps=2)
        self.assertEqual(snap.alive[0].lifetime_class, "persistent")
        self.assertGreaterEqual(snap.alive[0].crossed_step_boundaries, 2)

    def test_lifetime_class_transient_when_alloc_and_dealloc_in_same_step(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 500, shape="bf16[T]", tf_op="tmp")],
            deallocs=[_dealloc(200, 0x1)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        snap = snapshot_at_peak(events, peak_ts_ns=150,
                                step_range_ns=(0, 1_000),
                                step_boundaries_ns=[(0, 1_000)],
                                persistent_threshold_steps=2)
        self.assertEqual(snap.alive[0].lifetime_class, "transient")


if __name__ == "__main__":
    unittest.main()
