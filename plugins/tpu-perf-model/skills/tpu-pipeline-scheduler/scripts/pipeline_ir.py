#!/usr/bin/env python3
"""Pipeline IR data model for register-level scheduling analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

_VALID_OP_KINDS = frozenset({
    "DMA_LOAD", "DMA_STORE", "MXU", "VPU", "VMEM_TO_REG", "REG_TO_VMEM",
})
_VALID_UNITS = frozenset({"DMA", "MXU", "VPU"})
_MAX_VPR = 31


@dataclass
class PipelineOp:
    op_id: str
    op_kind: str
    input_vprs: list[int]
    output_vprs: list[int]
    input_vmem: list[str]
    output_vmem: list[str]
    latency_ns: float
    unit: str
    label: str = ""
    weight_vprs: list[int] = field(default_factory=list)
    data_vprs: list[int] = field(default_factory=list)
    pseudocode: str = ""

    def __post_init__(self):
        if self.op_kind == "MXU" and not self.input_vprs and (self.weight_vprs or self.data_vprs):
            self.input_vprs = self.weight_vprs + self.data_vprs

    @property
    def all_vprs(self) -> list[int]:
        return self.input_vprs + self.output_vprs

    @property
    def all_vmem(self) -> list[str]:
        return self.input_vmem + self.output_vmem


@dataclass
class PipelineSpec:
    name: str
    hw: str
    ops: list[PipelineOp]


def _validate_spec(spec: PipelineSpec) -> None:
    seen_ids: set[str] = set()
    for op in spec.ops:
        if op.op_id in seen_ids:
            raise ValueError(f"Duplicate op_id: {op.op_id}")
        seen_ids.add(op.op_id)
        for v in op.input_vprs + op.output_vprs + op.weight_vprs + op.data_vprs:
            if v < 0 or v > _MAX_VPR:
                raise ValueError(
                    f"VPR {v} in op {op.op_id} out of range 0-{_MAX_VPR}"
                )
        if op.unit not in _VALID_UNITS:
            raise ValueError(
                f"Invalid unit '{op.unit}' in op {op.op_id}, "
                f"must be one of {sorted(_VALID_UNITS)}"
            )


def _parse_op(d: dict) -> PipelineOp:
    return PipelineOp(
        op_id=d["op_id"],
        op_kind=d["op_kind"],
        input_vprs=d.get("input_vprs", []),
        output_vprs=d.get("output_vprs", []),
        input_vmem=d.get("input_vmem", []),
        output_vmem=d.get("output_vmem", []),
        latency_ns=float(d["latency_ns"]),
        unit=d["unit"],
        label=d.get("label", ""),
        weight_vprs=d.get("weight_vprs", []),
        data_vprs=d.get("data_vprs", []),
        pseudocode=d.get("pseudocode", ""),
    )


def load_spec(data: dict) -> PipelineSpec:
    ops = [_parse_op(o) for o in data["ops"]]
    spec = PipelineSpec(name=data["name"], hw=data.get("hw", "v7x"), ops=ops)
    _validate_spec(spec)
    return spec


def load_spec_from_file(path: str) -> PipelineSpec:
    with open(path) as f:
        return load_spec(json.load(f))
