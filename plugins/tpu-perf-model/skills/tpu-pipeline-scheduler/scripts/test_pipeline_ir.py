#!/usr/bin/env python3
"""Tests for pipeline_ir module."""

import unittest


class TestPipelineOp(unittest.TestCase):
    def test_pipeline_op_fields(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="load_q",
            op_kind="DMA_LOAD",
            input_vprs=[],
            output_vprs=[],
            input_vmem=[],
            output_vmem=["q_buf"],
            latency_ns=200.0,
            unit="DMA",
            label="Load Q tile",
        )
        self.assertEqual(op.op_id, "load_q")
        self.assertEqual(op.op_kind, "DMA_LOAD")
        self.assertEqual(op.output_vmem, ["q_buf"])
        self.assertEqual(op.latency_ns, 200.0)
        self.assertEqual(op.unit, "DMA")

    def test_pipeline_op_default_label(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="x", op_kind="VPU", input_vprs=[0], output_vprs=[1],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        self.assertEqual(op.label, "")

    def test_pipeline_op_all_vprs_property(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="mxu", op_kind="MXU",
            input_vprs=[0, 1, 2, 3], output_vprs=[4, 5],
            input_vmem=[], output_vmem=[],
            latency_ns=500.0, unit="MXU",
        )
        self.assertEqual(op.all_vprs, [0, 1, 2, 3, 4, 5])

    def test_pipeline_op_all_vmem_property(self):
        from pipeline_ir import PipelineOp

        op = PipelineOp(
            op_id="dma", op_kind="DMA_LOAD",
            input_vprs=[], output_vprs=[],
            input_vmem=["a"], output_vmem=["b", "c"],
            latency_ns=100.0, unit="DMA",
        )
        self.assertEqual(op.all_vmem, ["a", "b", "c"])


class TestPipelineSpec(unittest.TestCase):
    def test_pipeline_spec_fields(self):
        from pipeline_ir import PipelineOp, PipelineSpec

        ops = [
            PipelineOp(
                op_id="op1", op_kind="DMA_LOAD",
                input_vprs=[], output_vprs=[],
                input_vmem=[], output_vmem=["buf"],
                latency_ns=100.0, unit="DMA",
            ),
        ]
        spec = PipelineSpec(name="test", hw="v7x", ops=ops)
        self.assertEqual(spec.name, "test")
        self.assertEqual(spec.hw, "v7x")
        self.assertEqual(len(spec.ops), 1)

    def test_load_spec_from_dict(self):
        from pipeline_ir import load_spec

        data = {
            "name": "test_kernel",
            "hw": "v7x",
            "ops": [
                {
                    "op_id": "load_q",
                    "op_kind": "DMA_LOAD",
                    "input_vprs": [],
                    "output_vprs": [],
                    "input_vmem": [],
                    "output_vmem": ["q_buf"],
                    "latency_ns": 200,
                    "unit": "DMA",
                    "label": "Load Q",
                },
                {
                    "op_id": "q_to_reg",
                    "op_kind": "VMEM_TO_REG",
                    "input_vprs": [],
                    "output_vprs": [0, 1, 2, 3],
                    "input_vmem": ["q_buf"],
                    "output_vmem": [],
                    "latency_ns": 10,
                    "unit": "VPU",
                },
            ],
        }
        spec = load_spec(data)
        self.assertEqual(spec.name, "test_kernel")
        self.assertEqual(len(spec.ops), 2)
        self.assertEqual(spec.ops[0].op_id, "load_q")
        self.assertEqual(spec.ops[1].output_vprs, [0, 1, 2, 3])
        self.assertEqual(spec.ops[1].label, "")

    def test_load_spec_from_file(self):
        import json
        import os
        import tempfile
        from pipeline_ir import load_spec_from_file

        data = {
            "name": "file_test",
            "hw": "v7x",
            "ops": [
                {
                    "op_id": "op1",
                    "op_kind": "VPU",
                    "input_vprs": [0],
                    "output_vprs": [1],
                    "input_vmem": [],
                    "output_vmem": [],
                    "latency_ns": 50,
                    "unit": "VPU",
                },
            ],
        }
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        try:
            json.dump(data, f)
            f.close()
            spec = load_spec_from_file(f.name)
            self.assertEqual(spec.name, "file_test")
            self.assertEqual(len(spec.ops), 1)
        finally:
            os.unlink(f.name)

    def test_validate_rejects_duplicate_op_ids(self):
        from pipeline_ir import load_spec

        data = {
            "name": "dup",
            "hw": "v7x",
            "ops": [
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [],
                 "output_vprs": [0], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "VPU"},
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [0],
                 "output_vprs": [1], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "VPU"},
            ],
        }
        with self.assertRaises(ValueError):
            load_spec(data)

    def test_validate_rejects_invalid_vpr(self):
        from pipeline_ir import load_spec

        data = {
            "name": "bad_vpr",
            "hw": "v7x",
            "ops": [
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [32],
                 "output_vprs": [], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "VPU"},
            ],
        }
        with self.assertRaises(ValueError):
            load_spec(data)

    def test_validate_rejects_invalid_unit(self):
        from pipeline_ir import load_spec

        data = {
            "name": "bad_unit",
            "hw": "v7x",
            "ops": [
                {"op_id": "a", "op_kind": "VPU", "input_vprs": [],
                 "output_vprs": [0], "input_vmem": [], "output_vmem": [],
                 "latency_ns": 10, "unit": "GPU"},
            ],
        }
        with self.assertRaises(ValueError):
            load_spec(data)


if __name__ == "__main__":
    unittest.main()
