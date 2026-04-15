#!/usr/bin/env python3
"""ComputeStep and TensorRef data structures for TPU performance modeling."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from hw_params import dtype_bytes


@dataclass
class TensorRef:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        return self.numel * dtype_bytes(self.dtype)

    @staticmethod
    def from_dict(d: dict) -> TensorRef:
        return TensorRef(name=d["name"], shape=tuple(d["shape"]), dtype=d["dtype"])

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass
class ComputeStep:
    name: str
    op_type: str
    inputs: list[TensorRef]
    outputs: list[TensorRef]
    flops_formula: str
    flops_vars: dict[str, int]
    compute_unit: str
    fusable_with_prev: bool

    def eval_flops(self) -> int:
        allowed = {k: v for k, v in self.flops_vars.items()}
        allowed.update({"math": math, "ceil": math.ceil, "log2": math.log2})
        return int(eval(self.flops_formula, {"__builtins__": {}}, allowed))

    @property
    def total_input_bytes(self) -> int:
        return sum(t.size_bytes for t in self.inputs)

    @property
    def total_output_bytes(self) -> int:
        return sum(t.size_bytes for t in self.outputs)

    @property
    def total_io_bytes(self) -> int:
        return self.total_input_bytes + self.total_output_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.eval_flops() / self.total_io_bytes

    @staticmethod
    def from_dict(d: dict) -> ComputeStep:
        return ComputeStep(
            name=d["name"], op_type=d["op_type"],
            inputs=[TensorRef.from_dict(t) for t in d["inputs"]],
            outputs=[TensorRef.from_dict(t) for t in d["outputs"]],
            flops_formula=d["flops_formula"], flops_vars=d["flops_vars"],
            compute_unit=d["compute_unit"], fusable_with_prev=d.get("fusable_with_prev", False),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "op_type": self.op_type,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "flops_formula": self.flops_formula, "flops_vars": self.flops_vars,
            "compute_unit": self.compute_unit, "fusable_with_prev": self.fusable_with_prev,
        }


def load_steps(json_str: str) -> list[ComputeStep]:
    data = json.loads(json_str)
    return [ComputeStep.from_dict(d) for d in data]


def load_steps_from_file(path: str) -> list[ComputeStep]:
    with open(path) as f:
        return load_steps(f.read())
