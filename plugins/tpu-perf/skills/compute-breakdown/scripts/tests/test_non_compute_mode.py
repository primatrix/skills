"""Mode 3 (non_compute) projection."""
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

from test_pipeline import (  # noqa: E402
    _make_record, make_minimal_xspace, add_hlo_event,
)
from test_summary_mode import SCRIPT  # noqa: E402


class TestParseHloOpText(unittest.TestCase):
    def test_dtype_change_only_layout_match(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
            "%c.0 = f32[8,4]{1,0} convert(bf16[8,4]{1,0} %x.1)"
        )
        self.assertEqual(out_dt, "f32")
        self.assertEqual(in_dt, "bf16")
        self.assertEqual(out_lay, "{1,0}")
        self.assertEqual(in_lay, "{1,0}")

    def test_layout_change_only(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
            "%t.0 = bf16[8,4]{0,1} transpose(bf16[8,4]{1,0} %x.1)"
        )
        self.assertEqual(out_dt, "bf16")
        self.assertEqual(in_dt, "bf16")
        self.assertEqual(out_lay, "{0,1}")
        self.assertEqual(in_lay, "{1,0}")

    def test_layout_omitted_returns_none(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
            "%cp.0 = bf16[8,4] copy(bf16[8,4] %x.1)"
        )
        self.assertEqual(out_dt, "bf16")
        self.assertEqual(in_dt, "bf16")
        self.assertIsNone(out_lay)
        self.assertIsNone(in_lay)

    def test_no_match_returns_all_none(self):
        out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text("")
        self.assertIsNone(out_dt)
        self.assertIsNone(in_dt)
        self.assertIsNone(out_lay)
        self.assertIsNone(in_lay)

    def test_no_match_on_garbage(self):
        out_dt, _, in_dt, _ = cb._parse_hlo_op_text("not an HLO op")
        self.assertIsNone(out_dt)
        self.assertIsNone(in_dt)

    def test_lhs_with_or_without_percent(self):
        out_dt1, *_ = cb._parse_hlo_op_text(
            "%foo = bf16[1]{0} copy(bf16[1]{0} %x)"
        )
        out_dt2, *_ = cb._parse_hlo_op_text(
            "foo.bar = bf16[1]{0} copy(bf16[1]{0} %x)"
        )
        self.assertEqual(out_dt1, "bf16")
        self.assertEqual(out_dt2, "bf16")


if __name__ == "__main__":
    unittest.main()
