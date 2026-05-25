"""Rollups must partition the alive set with no double-count and no loss."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import AliveBuffer, build_rollups, pick_parent_jit  # noqa: E402


def _ab(addr, size, *, shape, tf_op, dtype, lifetime,
        parent_chain=("[0] Execute (jit_train_step)",)):
    return AliveBuffer(
        addr=addr, pool_id=0, size_bytes=size, alloc_bytes=size,
        shape=shape, tf_op=tf_op, data_type=dtype,
        alloc_ts_ns=0, age_ns_at_peak=0, crossed_step_boundaries=0,
        parent_chain=list(parent_chain),
        lifetime_class=lifetime, deallocated=False,
    )


class TestPickParentJit(unittest.TestCase):
    def test_picks_first_jit_in_chain(self):
        chain = [
            "[3] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
            "AllocateOutputBuffersWithInputReuse",
            "AllocateRawBuffer",
        ]
        self.assertEqual(pick_parent_jit(chain),
                         "[3] CommonPjRtLoadedExecutable::Execute (jit_train_step)")

    def test_falls_back_to_chain_root_when_no_jit(self):
        chain = ["DeferredTpuAllocator::Allocate", "AllocateRawBuffer"]
        self.assertEqual(pick_parent_jit(chain), "DeferredTpuAllocator::Allocate")

    def test_empty_chain_returns_unknown_marker(self):
        self.assertEqual(pick_parent_jit([]), "<no parent>")


class TestBuildRollups(unittest.TestCase):
    def setUp(self):
        self.alive = [
            _ab(0x1, 1000, shape="bf16[A]", tf_op="opA", dtype="bf16",
                lifetime="persistent"),
            _ab(0x2, 2000, shape="bf16[A]", tf_op="opB", dtype="bf16",
                lifetime="transient"),
            _ab(0x3, 500, shape="f32[B]", tf_op="opA", dtype="f32",
                lifetime="unknown"),
        ]

    def test_each_rollup_sums_to_alive_total(self):
        total = sum(b.size_bytes for b in self.alive)
        ru = build_rollups(self.alive, top_k=10, total_bytes=total)
        for key in ("by_lifetime_class", "by_shape", "by_tf_op",
                    "by_parent_jit", "by_dtype"):
            sub_total = sum(row["total_bytes"] for row in ru[key])
            self.assertEqual(sub_total, total, f"{key}: {sub_total} != {total}")

    def test_by_shape_top_k_truncates_with_tail(self):
        many = [_ab(i, (10 - i) * 100, shape=f"bf16[s{i}]", tf_op="x",
                    dtype="bf16", lifetime="persistent") for i in range(8)]
        total = sum(b.size_bytes for b in many)
        ru = build_rollups(many, top_k=3, total_bytes=total)
        shape = ru["by_shape"]
        # 3 head rows + 1 tail row.
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[-1]["key"], "<tail>")
        # Head rows are sorted by total_bytes desc.
        head_bytes = [r["total_bytes"] for r in shape[:3]]
        self.assertEqual(head_bytes, sorted(head_bytes, reverse=True))

    def test_by_lifetime_class_has_no_top_k_truncation(self):
        ru = build_rollups(self.alive, top_k=1, total_bytes=3500)
        keys = {r["key"] for r in ru["by_lifetime_class"]}
        self.assertEqual(keys, {"persistent", "transient", "unknown"})

    def test_lifetime_mix_sums_to_row_total(self):
        ru = build_rollups(self.alive, top_k=10, total_bytes=3500)
        for row in ru["by_shape"] + ru["by_tf_op"] + ru["by_parent_jit"]:
            mix = row["lifetime_mix"]
            self.assertEqual(
                mix["persistent"] + mix["transient"] + mix["unknown"],
                row["total_bytes"],
            )


if __name__ == "__main__":
    unittest.main()
