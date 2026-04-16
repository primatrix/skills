#!/usr/bin/env python3
"""Core data structures for fragment-level micro-op analysis."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TensorFragment:
    fragment_id: str
    tensor_name: str
    step_name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    home_level: str
    producer_op: str | None = None
    consumer_ops: tuple[str, ...] = ()
    vpr_count: int = 0


@dataclass
class MicroOp:
    op_id: str
    step_name: str
    op_kind: str
    depends_on: list[str]
    input_fragments: list[str]
    output_fragments: list[str]
    required_units: tuple[str, ...]
    required_vmem_slots: tuple[str, ...]
    required_reg_groups: tuple[str, ...]
    latency_ns: float
    required_vpr_count: int = 0


@dataclass
class MicroOpGraph:
    fragments: dict[str, TensorFragment]
    micro_ops: dict[str, MicroOp]

    def root_ops(self) -> list[str]:
        return sorted(op_id for op_id, op in self.micro_ops.items() if not op.depends_on)

    def leaf_ops(self) -> list[str]:
        parents = {dep for op in self.micro_ops.values() for dep in op.depends_on}
        return sorted(op_id for op_id in self.micro_ops if op_id not in parents)
