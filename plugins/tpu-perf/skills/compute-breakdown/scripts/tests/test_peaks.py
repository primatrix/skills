"""v7x peak table and resolver."""
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _peaks  # noqa: E402


class TestBuiltinPeaks(unittest.TestCase):
    def test_v7x_per_device_values(self):
        p = _peaks.BUILTIN_PEAKS["v7x"]
        self.assertEqual(p["peak_tflops_bf16"], 1153.5)
        self.assertEqual(p["peak_tflops_fp8"], 2307.0)
        self.assertEqual(p["peak_hbm_gibps"], 3690.0)
        self.assertIsNone(p["peak_tflops_fp32"])
        self.assertIsNone(p["peak_tflops_fp16"])


class TestResolvePeaks(unittest.TestCase):
    def test_no_overrides_returns_builtin_with_source_tag(self):
        p = _peaks.resolve_peaks("v7x")
        self.assertEqual(p["peak_tflops_bf16"], 1153.5)
        self.assertEqual(p["source"], "builtin v7x table")
        self.assertEqual(p["unit"], "GiB/s (base-1024) per device")
        self.assertIn("bf16", p["ridge_points"])
        self.assertIn("fp8", p["ridge_points"])
        self.assertNotIn("fp32", p["ridge_points"])

    def test_ridge_point_formula(self):
        p = _peaks.resolve_peaks("v7x")
        expected = (1153.5 * 1e12) / (3690.0 * (1024 ** 3))
        self.assertAlmostEqual(p["ridge_points"]["bf16"], expected, places=2)

    def test_overrides_set_source_to_cli_override(self):
        p = _peaks.resolve_peaks("v7x", override_tflops_bf16=2000.0)
        self.assertEqual(p["peak_tflops_bf16"], 2000.0)
        self.assertEqual(p["source"], "cli override")
        self.assertEqual(p["peak_tflops_fp8"], 2307.0)

    def test_override_fills_null_dtype(self):
        p = _peaks.resolve_peaks("v7x", override_tflops_fp32=500.0)
        self.assertEqual(p["peak_tflops_fp32"], 500.0)
        self.assertIn("fp32", p["ridge_points"])
        self.assertEqual(p["source"], "cli override")

    def test_unknown_chip_raises(self):
        with self.assertRaises(KeyError):
            _peaks.resolve_peaks("unknown-chip-x")


if __name__ == "__main__":
    unittest.main()
