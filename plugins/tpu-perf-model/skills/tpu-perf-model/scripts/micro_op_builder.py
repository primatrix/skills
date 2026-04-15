#!/usr/bin/env python3
"""Expand tiled compute steps into fragment-level micro-op graphs."""
from __future__ import annotations

from compute_step import ComputeStep
from micro_op_ir import MicroOp, MicroOpGraph, TensorFragment
from pipeline_simulator import TileConfig


def _tile_suffix(tile_idx: int) -> str:
    return f"tile{tile_idx}"


def _buffer_slot(prefix: str, tile_idx: int, double_buffer: bool) -> str:
    slot_idx = tile_idx % 2 if double_buffer else 0
    return f"{prefix}_slot{slot_idx}"


def _reg_group(prefix: str, tile_idx: int, double_buffer: bool) -> str:
    group_idx = tile_idx % 2 if double_buffer else 0
    return f"{prefix}_reg{group_idx}"


def _add_fragment(
    fragments: dict[str, TensorFragment],
    fragment_id: str,
    tensor_name: str,
    step_name: str,
    shape: tuple[int, ...],
    dtype: str,
    size_bytes: int,
    home_level: str,
    producer_op: str | None = None,
) -> None:
    fragments[fragment_id] = TensorFragment(
        fragment_id=fragment_id,
        tensor_name=tensor_name,
        step_name=step_name,
        shape=shape,
        dtype=dtype,
        size_bytes=size_bytes,
        home_level=home_level,
        producer_op=producer_op,
    )


def _add_micro_op(
    micro_ops: dict[str, MicroOp],
    op_id: str,
    step_name: str,
    op_kind: str,
    depends_on: list[str],
    input_fragments: list[str],
    output_fragments: list[str],
    required_units: tuple[str, ...],
    required_vmem_slots: tuple[str, ...],
    required_reg_groups: tuple[str, ...],
    latency_ns: float = 1.0,
) -> None:
    micro_ops[op_id] = MicroOp(
        op_id=op_id,
        step_name=step_name,
        op_kind=op_kind,
        depends_on=depends_on,
        input_fragments=input_fragments,
        output_fragments=output_fragments,
        required_units=required_units,
        required_vmem_slots=required_vmem_slots,
        required_reg_groups=required_reg_groups,
        latency_ns=latency_ns,
    )


def build_micro_op_graph_for_step(step: ComputeStep, tile: TileConfig) -> MicroOpGraph:
    """Expand one tiled step into a fragment-level micro-op graph."""
    if step.op_type != "matmul":
        raise NotImplementedError("build_micro_op_graph_for_step currently supports matmul only")

    fragments: dict[str, TensorFragment] = {}
    micro_ops: dict[str, MicroOp] = {}

    bm = tile.block_dims.get("M", 1)
    bn = tile.block_dims.get("N", 1)
    bk = tile.block_dims.get("K", 1)
    dtype = step.inputs[0].dtype
    bytes_per_input = bm * bk * 2
    bytes_per_weight = bk * bn * 2
    bytes_per_output = bm * bn * 2

    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)

        q_hbm = f"{step.inputs[0].name}_{suffix}_hbm"
        q_vmem = f"{step.inputs[0].name}_{suffix}_vmem"
        q_reg = f"{step.inputs[0].name}_{suffix}_reg"

        k_hbm = f"{step.inputs[1].name}_{suffix}_hbm"
        k_vmem = f"{step.inputs[1].name}_{suffix}_vmem"
        k_reg = f"{step.inputs[1].name}_{suffix}_reg"

        acc_reg = f"{step.outputs[0].name}_{suffix}_acc"
        out_vmem = f"{step.outputs[0].name}_{suffix}_vmem"
        out_hbm = f"{step.outputs[0].name}_{suffix}_hbm"

        _add_fragment(fragments, q_hbm, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "HBM")
        _add_fragment(fragments, q_vmem, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "VMEM")
        _add_fragment(fragments, q_reg, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "REG")

        _add_fragment(fragments, k_hbm, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "HBM")
        _add_fragment(fragments, k_vmem, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "VMEM")
        _add_fragment(fragments, k_reg, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "REG")

        _add_fragment(fragments, acc_reg, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "REG")
        _add_fragment(fragments, out_vmem, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "VMEM")
        _add_fragment(fragments, out_hbm, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "HBM")

        q_slot = _buffer_slot("q", tile_idx, tile.double_buffer)
        k_slot = _buffer_slot("k", tile_idx, tile.double_buffer)
        out_slot = _buffer_slot("out", tile_idx, tile.double_buffer)
        q_reg_group = _reg_group("q", tile_idx, tile.double_buffer)
        k_reg_group = _reg_group("k", tile_idx, tile.double_buffer)
        acc_reg_group = _reg_group("acc", tile_idx, tile.double_buffer)

        load_q = f"load_q_{suffix}"
        load_k = f"load_k_{suffix}"
        move_q = f"vmem_to_reg_q_{suffix}"
        move_k = f"vmem_to_reg_k_{suffix}"
        mxu = f"mxu_{suffix}"
        spill = f"reg_to_vmem_{suffix}"
        store = f"store_{suffix}"

        _add_micro_op(
            micro_ops,
            load_q,
            step.name,
            "dma_load_hbm_to_vmem",
            [],
            [q_hbm],
            [q_vmem],
            ("DMA",),
            (q_slot,),
            (),
        )
        _add_micro_op(
            micro_ops,
            load_k,
            step.name,
            "dma_load_hbm_to_vmem",
            [],
            [k_hbm],
            [k_vmem],
            ("DMA",),
            (k_slot,),
            (),
        )
        _add_micro_op(
            micro_ops,
            move_q,
            step.name,
            "vmem_to_reg",
            [load_q],
            [q_vmem],
            [q_reg],
            (),
            (q_slot,),
            (q_reg_group,),
        )
        _add_micro_op(
            micro_ops,
            move_k,
            step.name,
            "vmem_to_reg",
            [load_k],
            [k_vmem],
            [k_reg],
            (),
            (k_slot,),
            (k_reg_group,),
        )
        _add_micro_op(
            micro_ops,
            mxu,
            step.name,
            "mxu_compute",
            [move_q, move_k],
            [q_reg, k_reg],
            [acc_reg],
            ("MXU",),
            (),
            (q_reg_group, k_reg_group, acc_reg_group),
        )
        _add_micro_op(
            micro_ops,
            spill,
            step.name,
            "reg_to_vmem",
            [mxu],
            [acc_reg],
            [out_vmem],
            (),
            (out_slot,),
            (acc_reg_group,),
        )
        _add_micro_op(
            micro_ops,
            store,
            step.name,
            "dma_store_vmem_to_hbm",
            [spill],
            [out_vmem],
            [out_hbm],
            ("DMA",),
            (out_slot,),
            (),
        )

    return MicroOpGraph(fragments=fragments, micro_ops=micro_ops)
