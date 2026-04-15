#!/usr/bin/env python3
"""Tests for compute_step module."""
import json
import unittest


class TestTensorRef(unittest.TestCase):
    def test_size_bytes_bf16(self):
        from compute_step import TensorRef
        t = TensorRef(name="A", shape=(4096, 128), dtype="bf16")
        self.assertEqual(t.size_bytes, 4096 * 128 * 2)

    def test_size_bytes_f32(self):
        from compute_step import TensorRef
        t = TensorRef(name="B", shape=(128, 4096), dtype="f32")
        self.assertEqual(t.size_bytes, 128 * 4096 * 4)

    def test_numel(self):
        from compute_step import TensorRef
        t = TensorRef(name="C", shape=(32, 64, 128), dtype="bf16")
        self.assertEqual(t.numel, 32 * 64 * 128)


class TestComputeStep(unittest.TestCase):
    def test_eval_flops_matmul(self):
        from compute_step import ComputeStep, TensorRef
        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[TensorRef("Q", (4096, 128), "bf16"), TensorRef("K", (128, 4096), "bf16")],
            outputs=[TensorRef("S", (4096, 4096), "bf16")],
            flops_formula="2*M*N*K", flops_vars={"M": 4096, "N": 4096, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        self.assertEqual(step.eval_flops(), 2 * 4096 * 4096 * 128)

    def test_total_input_bytes(self):
        from compute_step import ComputeStep, TensorRef
        step = ComputeStep(
            name="scale", op_type="elementwise",
            inputs=[TensorRef("S", (4096, 4096), "bf16")],
            outputs=[TensorRef("S_scaled", (4096, 4096), "bf16")],
            flops_formula="M*N", flops_vars={"M": 4096, "N": 4096},
            compute_unit="VPU", fusable_with_prev=True,
        )
        self.assertEqual(step.total_input_bytes, 4096 * 4096 * 2)
        self.assertEqual(step.total_output_bytes, 4096 * 4096 * 2)

    def test_from_json(self):
        from compute_step import ComputeStep
        data = {
            "name": "add", "op_type": "elementwise",
            "inputs": [{"name": "A", "shape": [1024], "dtype": "bf16"}],
            "outputs": [{"name": "B", "shape": [1024], "dtype": "bf16"}],
            "flops_formula": "N", "flops_vars": {"N": 1024},
            "compute_unit": "VPU", "fusable_with_prev": False,
        }
        step = ComputeStep.from_dict(data)
        self.assertEqual(step.name, "add")
        self.assertEqual(step.eval_flops(), 1024)

    def test_load_steps_from_json_string(self):
        from compute_step import load_steps
        json_str = json.dumps([{
            "name": "op1", "op_type": "elementwise",
            "inputs": [{"name": "X", "shape": [256], "dtype": "bf16"}],
            "outputs": [{"name": "Y", "shape": [256], "dtype": "bf16"}],
            "flops_formula": "N", "flops_vars": {"N": 256},
            "compute_unit": "VPU", "fusable_with_prev": False,
        }])
        steps = load_steps(json_str)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].name, "op1")


if __name__ == "__main__":
    unittest.main()
