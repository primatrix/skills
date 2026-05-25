"""Unit tests for the loader: alloc/dealloc extraction and parent-chain walk."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import load_host_allocator_events  # noqa: E402

from tests._synthetic import (  # noqa: E402
    PlaneBuilder, make_alloc_event, make_dealloc_event, make_xspace,
)


class TestLoader(unittest.TestCase):
    def test_returns_absent_when_host_plane_missing(self):
        xs = make_xspace()
        PlaneBuilder(xs, "/device:TPU:0")
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(events)
        self.assertEqual(reason, "host_plane_absent")

    def test_returns_absent_when_no_alloc_events(self):
        xs = make_xspace()
        PlaneBuilder(xs, "/host:CPU")
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(events)
        self.assertEqual(reason, "no_memory_events")

    def test_extracts_alloc_and_dealloc_with_correct_ts(self):
        xs = make_xspace()
        host = PlaneBuilder(xs, "/host:CPU")
        line = host.add_line("pjrt_tpu_execute/0", timestamp_ns=1_000)
        # offset_ps 2_000_000 → 2_000 ns added to line.timestamp_ns 1_000 → ts_ns 3_000.
        make_alloc_event(
            line, offset_ps=2_000_000, addr=0x1000,
            requested=4096, allocation=4096,
            bytes_allocated=4096, peak_bytes_in_use=4096,
            bytes_reserved=10_000, shape="bf16[128]", tf_op="dot",
        )
        make_dealloc_event(
            line, offset_ps=5_000_000, addr=0x1000,
            bytes_allocated=0, peak_bytes_in_use=4096,
            bytes_reserved=10_000,
        )
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(reason)
        self.assertIsNotNone(events)
        self.assertTrue(events.host_plane_present)
        self.assertEqual(len(events.allocs), 1)
        self.assertEqual(len(events.deallocs), 1)
        a = events.allocs[0]
        self.assertEqual(a.ts_ns, 3_000)
        self.assertEqual(a.addr, 0x1000)
        self.assertEqual(a.pool_id, 0)
        self.assertEqual(a.requested_bytes, 4096)
        self.assertEqual(a.shape, "bf16[128]")
        self.assertEqual(a.tf_op, "dot")
        self.assertEqual(a.line_name, "pjrt_tpu_execute/0")
        d = events.deallocs[0]
        self.assertEqual(d.ts_ns, 6_000)
        self.assertEqual(d.addr, 0x1000)
        self.assertEqual(events.pool_capacity, {0: 10_000})

    def test_parent_chain_built_from_time_containment(self):
        xs = make_xspace()
        host = PlaneBuilder(xs, "/host:CPU")
        line = host.add_line("pjrt_tpu_execute/0", timestamp_ns=0)
        # Outer event covers [0ps, 100_000_000ps]; alloc at offset 5_000_000ps.
        line.add_event("[0] Execute (jit_train_step)",
                       offset_ps=0, duration_ps=100_000_000)
        line.add_event("AllocateOutputBuffersWithInputReuse",
                       offset_ps=1_000_000, duration_ps=10_000_000)
        make_alloc_event(
            line, offset_ps=5_000_000, addr=0x2000,
            requested=128, allocation=128,
            bytes_allocated=128, peak_bytes_in_use=128, bytes_reserved=10_000,
        )
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(reason)
        chain = events.allocs[0].parent_chain
        self.assertIn("[0] Execute (jit_train_step)", chain)
        self.assertIn("AllocateOutputBuffersWithInputReuse", chain)
        self.assertEqual(chain[0], "[0] Execute (jit_train_step)")
        self.assertEqual(chain[1], "AllocateOutputBuffersWithInputReuse")


if __name__ == "__main__":
    unittest.main()
