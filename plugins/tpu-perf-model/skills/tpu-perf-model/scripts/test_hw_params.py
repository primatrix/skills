#!/usr/bin/env python3
"""Tests for hw_params module."""
import unittest


class TestHWParams(unittest.TestCase):
    def test_v7x_hbm_capacity_bytes(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.hbm_capacity_bytes, 192 * 1024**3)

    def test_v7x_vmem_capacity_bytes(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.vmem_capacity_bytes, 64 * 1024**2)

    def test_v7x_vpr_count(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.vpr_count, 32)

    def test_v7x_vpr_size_bytes(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.vpr_size_bytes, 8 * 128 * 4)

    def test_v7x_hbm_bandwidth(self):
        from hw_params import TPU_V7X
        self.assertAlmostEqual(TPU_V7X.hbm_bw_bytes_per_sec, 3690e9)

    def test_v7x_mxu_peak_flops(self):
        from hw_params import TPU_V7X
        self.assertAlmostEqual(TPU_V7X.mxu_peak_flops, 2307e12)

    def test_dtype_bytes(self):
        from hw_params import dtype_bytes
        self.assertEqual(dtype_bytes("bf16"), 2)
        self.assertEqual(dtype_bytes("f32"), 4)
        self.assertEqual(dtype_bytes("int8"), 1)

    def test_alignment(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.alignment, 128)


if __name__ == "__main__":
    unittest.main()
