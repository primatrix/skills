#!/usr/bin/env python3
"""Expand tiled compute steps into fragment-level micro-op graphs."""
from __future__ import annotations

import math

from compute_step import ComputeStep
from hw_params import TPUParams, TPU_V7X, dtype_bytes
from micro_op_ir import MicroOp, MicroOpGraph, TensorFragment
from pipeline_simulator import TileConfig


def _calc_vpr_count(size_bytes: int, hw: TPUParams) -> int:
    if size_bytes == 0:
        return 0
    return math.ceil(size_bytes / hw.vpr_size_bytes)


def _tile_suffix(tile_idx: int) -> str:
    return f"tile{tile_idx}"


def _step_token(step: ComputeStep, step_idx: int | None) -> str:
    if step_idx is None:
        return step.name
    return f"s{step_idx}_{step.name}"


def _fragment_id(step_token: str, tensor_name: str, suffix: str, level: str) -> str:
    return f"{step_token}_{tensor_name}_{suffix}_{level.lower()}"


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
    vpr_count: int = 0,
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
        vpr_count=vpr_count,
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
    required_vpr_count: int = 0,
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
        required_vpr_count=required_vpr_count,
    )


def build_micro_op_graph_for_step(
    step: ComputeStep,
    tile: TileConfig,
    step_idx: int | None = None,
) -> MicroOpGraph:
    """Expand one tiled step into a fragment-level micro-op graph."""
    if step.op_type == "matmul":
        return _build_matmul_graph(step, tile, step_idx)
    return _build_vpu_graph(step, tile, step_idx)


def _build_matmul_graph(step: ComputeStep, tile: TileConfig, step_idx: int | None) -> MicroOpGraph:
    """Expand one tiled matmul step into a fragment-level micro-op graph."""

    fragments: dict[str, TensorFragment] = {}
    micro_ops: dict[str, MicroOp] = {}
    step_token = _step_token(step, step_idx)
    hw = TPU_V7X

    bm = tile.block_dims.get("M", 1)
    bn = tile.block_dims.get("N", 1)
    bk = tile.block_dims.get("K", 1)
    dtype = step.inputs[0].dtype
    dtype_b = dtype_bytes(dtype)
    bytes_per_input = bm * bk * dtype_b
    bytes_per_weight = bk * bn * dtype_b
    bytes_per_output = bm * bn * dtype_b

    q_vpr = _calc_vpr_count(bytes_per_input, hw)
    k_vpr = _calc_vpr_count(bytes_per_weight, hw)
    result_vpr = _calc_vpr_count(bytes_per_output, hw)

    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)

        q_hbm = _fragment_id(step_token, step.inputs[0].name, suffix, "hbm")
        q_vmem = _fragment_id(step_token, step.inputs[0].name, suffix, "vmem")
        q_reg = _fragment_id(step_token, step.inputs[0].name, suffix, "reg")

        k_hbm = _fragment_id(step_token, step.inputs[1].name, suffix, "hbm")
        k_vmem = _fragment_id(step_token, step.inputs[1].name, suffix, "vmem")
        k_reg = _fragment_id(step_token, step.inputs[1].name, suffix, "reg")

        result_reg = _fragment_id(step_token, step.outputs[0].name, suffix, "reg")
        out_vmem = _fragment_id(step_token, step.outputs[0].name, suffix, "vmem")
        out_hbm = _fragment_id(step_token, step.outputs[0].name, suffix, "hbm")

        _add_fragment(fragments, q_hbm, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "HBM")
        _add_fragment(fragments, q_vmem, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "VMEM")
        _add_fragment(fragments, q_reg, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "REG", vpr_count=q_vpr)

        _add_fragment(fragments, k_hbm, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "HBM")
        _add_fragment(fragments, k_vmem, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "VMEM")
        _add_fragment(fragments, k_reg, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "REG", vpr_count=k_vpr)

        _add_fragment(fragments, result_reg, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "REG", vpr_count=result_vpr)
        _add_fragment(fragments, out_vmem, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "VMEM")
        _add_fragment(fragments, out_hbm, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "HBM")

        q_slot = _buffer_slot("q", tile_idx, tile.double_buffer)
        k_slot = _buffer_slot("k", tile_idx, tile.double_buffer)
        out_slot = _buffer_slot("out", tile_idx, tile.double_buffer)
        q_reg_group = _reg_group("q", tile_idx, tile.double_buffer)
        k_reg_group = _reg_group("k", tile_idx, tile.double_buffer)
        acc_reg_group = _reg_group("acc", tile_idx, tile.double_buffer)

        load_q = f"{step_token}_load_q_{suffix}"
        load_k = f"{step_token}_load_k_{suffix}"
        move_q = f"{step_token}_vmem_to_reg_q_{suffix}"
        move_k = f"{step_token}_vmem_to_reg_k_{suffix}"
        mxu = f"{step_token}_mxu_{suffix}"
        writeback = f"{step_token}_mxu_writeback_{suffix}"
        spill = f"{step_token}_reg_to_vmem_{suffix}"
        store = f"{step_token}_store_{suffix}"

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
            [],
            ("MXU",),
            (),
            (q_reg_group, k_reg_group),
            required_vpr_count=q_vpr + k_vpr,
        )
        _add_micro_op(
            micro_ops,
            writeback,
            step.name,
            "mxu_writeback",
            [mxu],
            [],
            [result_reg],
            (),
            (),
            (acc_reg_group,),
            required_vpr_count=result_vpr,
        )
        _add_micro_op(
            micro_ops,
            spill,
            step.name,
            "reg_to_vmem",
            [writeback],
            [result_reg],
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


def _build_vpu_graph(step: ComputeStep, tile: TileConfig, step_idx: int | None) -> MicroOpGraph:
    """Expand a vector-style step into load, VPU compute, and store micro-ops."""
    fragments: dict[str, TensorFragment] = {}
    micro_ops: dict[str, MicroOp] = {}
    step_token = _step_token(step, step_idx)

    dtype = step.inputs[0].dtype
    dtype_b = dtype_bytes(dtype)
    total_numel = step.inputs[0].numel
    tile_numel = max(total_numel // max(tile.num_tiles, 1), 1)
    bytes_per_fragment = tile_numel * dtype_b

    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)
        input_hbm = _fragment_id(step_token, step.inputs[0].name, suffix, "hbm")
        input_vmem = _fragment_id(step_token, step.inputs[0].name, suffix, "vmem")
        input_reg = _fragment_id(step_token, step.inputs[0].name, suffix, "reg")
        output_reg = _fragment_id(step_token, step.outputs[0].name, suffix, "reg")
        output_vmem = _fragment_id(step_token, step.outputs[0].name, suffix, "vmem")
        output_hbm = _fragment_id(step_token, step.outputs[0].name, suffix, "hbm")

        _add_fragment(fragments, input_hbm, step.inputs[0].name, step.name, (tile_numel,), dtype, bytes_per_fragment, "HBM")
        _add_fragment(fragments, input_vmem, step.inputs[0].name, step.name, (tile_numel,), dtype, bytes_per_fragment, "VMEM")
        _add_fragment(fragments, input_reg, step.inputs[0].name, step.name, (tile_numel,), dtype, bytes_per_fragment, "REG")
        _add_fragment(fragments, output_reg, step.outputs[0].name, step.name, (tile_numel,), step.outputs[0].dtype, bytes_per_fragment, "REG")
        _add_fragment(fragments, output_vmem, step.outputs[0].name, step.name, (tile_numel,), step.outputs[0].dtype, bytes_per_fragment, "VMEM")
        _add_fragment(fragments, output_hbm, step.outputs[0].name, step.name, (tile_numel,), step.outputs[0].dtype, bytes_per_fragment, "HBM")

        in_slot = _buffer_slot("in", tile_idx, tile.double_buffer)
        out_slot = _buffer_slot("out", tile_idx, tile.double_buffer)
        in_reg_group = _reg_group("vin", tile_idx, tile.double_buffer)
        out_reg_group = _reg_group("vout", tile_idx, tile.double_buffer)

        load = f"{step_token}_load_{suffix}"
        move_in = f"{step_token}_vmem_to_reg_{suffix}"
        vpu = f"{step_token}_vpu_{suffix}"
        spill = f"{step_token}_reg_to_vmem_{suffix}"
        store = f"{step_token}_store_{suffix}"

        _add_micro_op(micro_ops, load, step.name, "dma_load_hbm_to_vmem", [], [input_hbm], [input_vmem], ("DMA",), (in_slot,), ())
        _add_micro_op(micro_ops, move_in, step.name, "vmem_to_reg", [load], [input_vmem], [input_reg], (), (in_slot,), (in_reg_group,))
        _add_micro_op(micro_ops, vpu, step.name, "vpu_compute", [move_in], [input_reg], [output_reg], ("VPU",), (), (in_reg_group, out_reg_group))
        _add_micro_op(micro_ops, spill, step.name, "reg_to_vmem", [vpu], [output_reg], [output_vmem], (), (out_slot,), (out_reg_group,))
        _add_micro_op(micro_ops, store, step.name, "dma_store_vmem_to_hbm", [spill], [output_vmem], [output_hbm], ("DMA",), (out_slot,), ())

    return MicroOpGraph(fragments=fragments, micro_ops=micro_ops)


def _merge_graphs(target: MicroOpGraph, source: MicroOpGraph) -> None:
    target.fragments.update(source.fragments)
    target.micro_ops.update(source.micro_ops)


def _remove_trailing_store(
    graph: MicroOpGraph,
    prev_step: ComputeStep,
    prev_step_idx: int,
    tile_idx: int,
) -> str:
    prev_step_token = _step_token(prev_step, prev_step_idx)
    suffix = _tile_suffix(tile_idx)
    out_hbm = _fragment_id(prev_step_token, prev_step.outputs[0].name, suffix, "hbm")
    store_ids = [
        op_id for op_id, op in graph.micro_ops.items()
        if op.op_kind == "dma_store_vmem_to_hbm" and out_hbm in op.output_fragments
    ]
    for store_id in store_ids:
        del graph.micro_ops[store_id]
    graph.fragments.pop(out_hbm, None)
    return _fragment_id(prev_step_token, prev_step.outputs[0].name, suffix, "vmem")


def _append_fused_vpu_step(
    graph: MicroOpGraph,
    prev_step: ComputeStep,
    prev_step_idx: int,
    step: ComputeStep,
    tile: TileConfig,
    prev_tile: TileConfig,
    step_idx: int,
) -> None:
    numel = max(step.outputs[0].numel // max(tile.num_tiles, 1), 1)
    bytes_per_fragment = max(step.outputs[0].size_bytes // max(tile.num_tiles, 1), 1)
    step_token = _step_token(step, step_idx)

    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)
        input_vmem = _remove_trailing_store(graph, prev_step, prev_step_idx, tile_idx)
        input_reg = _fragment_id(step_token, step.inputs[0].name, suffix, "reg")
        output_reg = _fragment_id(step_token, step.outputs[0].name, suffix, "reg")
        output_vmem = _fragment_id(step_token, step.outputs[0].name, suffix, "vmem")
        output_hbm = _fragment_id(step_token, step.outputs[0].name, suffix, "hbm")

        _add_fragment(
            graph.fragments,
            input_reg,
            step.inputs[0].name,
            step.name,
            (numel,),
            step.inputs[0].dtype,
            bytes_per_fragment,
            "REG",
        )
        _add_fragment(
            graph.fragments,
            output_reg,
            step.outputs[0].name,
            step.name,
            (numel,),
            step.outputs[0].dtype,
            bytes_per_fragment,
            "REG",
        )
        _add_fragment(
            graph.fragments,
            output_vmem,
            step.outputs[0].name,
            step.name,
            (numel,),
            step.outputs[0].dtype,
            bytes_per_fragment,
            "VMEM",
        )
        _add_fragment(
            graph.fragments,
            output_hbm,
            step.outputs[0].name,
            step.name,
            (numel,),
            step.outputs[0].dtype,
            bytes_per_fragment,
            "HBM",
        )

        input_slot = _buffer_slot("out", tile_idx, prev_tile.double_buffer)
        output_slot = _buffer_slot("fused_out", tile_idx, tile.double_buffer)
        input_reg_group = _reg_group("fused_in", tile_idx, tile.double_buffer)
        output_reg_group = _reg_group("fused_out", tile_idx, tile.double_buffer)

        move_in = f"{step_token}_vmem_to_reg_{suffix}"
        vpu = f"{step_token}_vpu_{suffix}"
        spill = f"{step_token}_reg_to_vmem_{suffix}"
        store = f"{step_token}_store_{suffix}"

        producer_candidates = [
            op_id for op_id, op in graph.micro_ops.items()
            if input_vmem in op.output_fragments
        ]
        depends_on = sorted(producer_candidates)

        _add_micro_op(
            graph.micro_ops,
            move_in,
            step.name,
            "vmem_to_reg",
            depends_on,
            [input_vmem],
            [input_reg],
            (),
            (input_slot,),
            (input_reg_group,),
        )
        _add_micro_op(
            graph.micro_ops,
            vpu,
            step.name,
            "vpu_compute",
            [move_in],
            [input_reg],
            [output_reg],
            ("VPU",),
            (),
            (input_reg_group, output_reg_group),
        )
        _add_micro_op(
            graph.micro_ops,
            spill,
            step.name,
            "reg_to_vmem",
            [vpu],
            [output_reg],
            [output_vmem],
            (),
            (output_slot,),
            (output_reg_group,),
        )
        _add_micro_op(
            graph.micro_ops,
            store,
            step.name,
            "dma_store_vmem_to_hbm",
            [spill],
            [output_vmem],
            [output_hbm],
            ("DMA",),
            (output_slot,),
            (),
        )


def _latest_hbm_producer(graph: MicroOpGraph, tensor_name: str, tile_idx: int) -> str | None:
    suffix = _tile_suffix(tile_idx)
    candidates = [
        op_id
        for op_id, op in graph.micro_ops.items()
        if any(
            graph.fragments[fragment_id].tensor_name == tensor_name
            and fragment_id.endswith(f"_{suffix}_hbm")
            for fragment_id in op.output_fragments
            if fragment_id in graph.fragments
        )
    ]
    if not candidates:
        return None
    return candidates[-1]


def _link_unfused_dependencies(
    graph: MicroOpGraph,
    step_graph: MicroOpGraph,
    step: ComputeStep,
    tile: TileConfig,
) -> None:
    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)
        for input_ref in step.inputs:
            producer_id = _latest_hbm_producer(graph, input_ref.name, tile_idx)
            if producer_id is None:
                continue
            load_candidates = [
                op for op in step_graph.micro_ops.values()
                if op.op_kind == "dma_load_hbm_to_vmem"
                and any(
                    step_graph.fragments[fragment_id].tensor_name == input_ref.name
                    and fragment_id.endswith(f"_{suffix}_hbm")
                    for fragment_id in op.input_fragments
                    if fragment_id in step_graph.fragments
                )
            ]
            for op in load_candidates:
                if producer_id not in op.depends_on:
                    op.depends_on.append(producer_id)


def build_micro_op_graph_for_pipeline(
    steps: list[ComputeStep],
    tile_configs: list[TileConfig],
) -> MicroOpGraph:
    """Expand a step pipeline into one micro-op graph."""
    if not steps:
        return MicroOpGraph(fragments={}, micro_ops={})

    graph = build_micro_op_graph_for_step(steps[0], tile_configs[0], step_idx=0)
    prev_step = steps[0]
    prev_idx = 0

    for step_idx, step in enumerate(steps[1:], 1):
        tile = tile_configs[step_idx]
        if step.fusable_with_prev and step.compute_unit == "VPU":
            _append_fused_vpu_step(
                graph,
                prev_step,
                prev_idx,
                step,
                tile,
                tile_configs[step_idx - 1],
                step_idx,
            )
        else:
            step_graph = build_micro_op_graph_for_step(step, tile, step_idx=step_idx)
            _link_unfused_dependencies(graph, step_graph, step, tile)
            _merge_graphs(graph, step_graph)
        prev_step = step
        prev_idx = step_idx

    return graph
