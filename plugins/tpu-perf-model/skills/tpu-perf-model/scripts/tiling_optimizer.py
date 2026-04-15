#!/usr/bin/env python3
"""Tiling optimizer: finds optimal block shape that balances DMA and compute time."""
from __future__ import annotations

import math
from dataclasses import dataclass

from compute_step import ComputeStep
from hw_params import TPUParams, dtype_bytes
from pipeline_simulator import TileConfig, calc_dma_time_ns, calc_mxu_time_ns, calc_vpu_time_ns


def _candidate_dims(max_val: int, alignment: int) -> list[int]:
    """Generate multiples of alignment up to max_val."""
    candidates = []
    d = alignment
    while d <= max_val:
        candidates.append(d)
        d += alignment
    return candidates


def _matmul_tile_vmem(bm: int, bn: int, bk: int, dtype_b: int, double_buffer: bool) -> int:
    """Compute VMEM usage for a matmul tile."""
    tile_a = bm * bk * dtype_b
    tile_b = bk * bn * dtype_b
    tile_c = bm * bn * dtype_b
    tile_in = tile_a + tile_b
    multiplier = 2 if double_buffer else 1
    return multiplier * tile_in + tile_c


def _matmul_pipeline_time(
    bm: int, bn: int, bk: int, M: int, N: int, K: int,
    dtype_b: int, hw: TPUParams, double_buffer: bool,
) -> float:
    """Compute total pipeline time for a matmul with given tiling.

    When tiling only along M (bn==N, bk==K), B is loaded once and reused
    across M tiles, so per-tile DMA only includes A tile + C tile.
    """
    tile_a = bm * bk * dtype_b
    tile_b = bk * bn * dtype_b
    tile_c = bm * bn * dtype_b

    tile_flops = 2 * bm * bn * bk

    num_m = math.ceil(M / bm)
    num_n = math.ceil(N / bn)
    num_k = math.ceil(K / bk)
    num_tiles = num_m * num_n * num_k

    # When bn==N and bk==K, B is loaded once and stays in VMEM
    if bn == N and bk == K and num_m > 1:
        t_b_load = calc_dma_time_ns(tile_b, hw)
        per_tile_dma = tile_a + tile_c
        t_dma_tile = calc_dma_time_ns(per_tile_dma, hw)
        t_compute_tile = calc_mxu_time_ns(tile_flops, hw)
        if double_buffer and num_tiles > 1:
            return (t_b_load + t_dma_tile
                    + (num_tiles - 1) * max(t_dma_tile, t_compute_tile)
                    + t_compute_tile)
        else:
            return t_b_load + num_tiles * (t_dma_tile + t_compute_tile)
    else:
        tile_io_bytes = tile_a + tile_b + tile_c
        t_dma_tile = calc_dma_time_ns(tile_io_bytes, hw)
        t_compute_tile = calc_mxu_time_ns(tile_flops, hw)
        if double_buffer and num_tiles > 1:
            return t_dma_tile + (num_tiles - 1) * max(t_dma_tile, t_compute_tile) + t_compute_tile
        else:
            return num_tiles * (t_dma_tile + t_compute_tile)


def find_optimal_tiling(step: ComputeStep, hw: TPUParams) -> TileConfig:
    """Find tiling that minimizes pipeline time for a compute step."""
    if step.op_type == "matmul":
        return _find_matmul_tiling(step, hw)
    else:
        return _find_elementwise_tiling(step, hw)


def _find_matmul_tiling(step: ComputeStep, hw: TPUParams) -> TileConfig:
    """Exhaustive search over (bm, bn, bk) for matmul."""
    M = step.flops_vars.get("M", 1)
    N = step.flops_vars.get("N", 1)
    K = step.flops_vars.get("K", 1)
    dtype_b = dtype_bytes(step.inputs[0].dtype)

    max_bm = min(M, 2048)
    max_bn = min(N, 2048)
    max_bk = min(K, 1024)

    bm_candidates = _candidate_dims(max_bm, hw.alignment)
    bn_candidates = _candidate_dims(max_bn, hw.alignment)
    bk_candidates = _candidate_dims(max_bk, hw.alignment)

    best_time = float("inf")
    best_config = None

    for bm in bm_candidates:
        for bn in bn_candidates:
            for bk in bk_candidates:
                # Try double buffer first
                vmem_db = _matmul_tile_vmem(bm, bn, bk, dtype_b, double_buffer=True)
                num_m = math.ceil(M / bm)
                num_n = math.ceil(N / bn)
                num_k = math.ceil(K / bk)
                num_tiles = num_m * num_n * num_k

                if vmem_db <= hw.vmem_capacity_bytes and num_tiles > 1:
                    t = _matmul_pipeline_time(bm, bn, bk, M, N, K, dtype_b, hw, True)
                    if t < best_time:
                        tile_a = bm * bk * dtype_b
                        tile_b = bk * bn * dtype_b
                        best_time = t
                        best_config = TileConfig(
                            block_dims={"M": bm, "N": bn, "K": bk},
                            num_tiles=num_tiles,
                            tile_input_bytes=tile_a + tile_b,
                            tile_output_bytes=bm * bn * dtype_b,
                            double_buffer=True,
                            vmem_usage_bytes=vmem_db,
                        )
                else:
                    # Try without double buffer
                    vmem_sb = _matmul_tile_vmem(bm, bn, bk, dtype_b, double_buffer=False)
                    if vmem_sb <= hw.vmem_capacity_bytes:
                        t = _matmul_pipeline_time(bm, bn, bk, M, N, K, dtype_b, hw, False)
                        if t < best_time:
                            tile_a = bm * bk * dtype_b
                            tile_b = bk * bn * dtype_b
                            best_time = t
                            best_config = TileConfig(
                                block_dims={"M": bm, "N": bn, "K": bk},
                                num_tiles=num_tiles,
                                tile_input_bytes=tile_a + tile_b,
                                tile_output_bytes=bm * bn * dtype_b,
                                double_buffer=False,
                                vmem_usage_bytes=vmem_sb,
                            )

    if best_config is None:
        # Fallback: single tile covering everything
        tile_in = step.total_input_bytes
        tile_out = step.total_output_bytes
        best_config = TileConfig(
            block_dims={"M": M, "N": N, "K": K},
            num_tiles=1,
            tile_input_bytes=tile_in,
            tile_output_bytes=tile_out,
            double_buffer=False,
            vmem_usage_bytes=tile_in + tile_out,
        )

    return best_config


def _find_elementwise_tiling(step: ComputeStep, hw: TPUParams) -> TileConfig:
    """Simple 1D tiling for elementwise ops."""
    numel = step.inputs[0].numel
    dtype_b = dtype_bytes(step.inputs[0].dtype)
    n_in = len(step.inputs)
    n_out = len(step.outputs)

    # Find largest tile that fits in VMEM with double buffering
    best_tile = hw.alignment
    for tile_size in _candidate_dims(min(numel, 2048 * hw.alignment), hw.alignment):
        tile_in = tile_size * dtype_b * n_in
        tile_out = tile_size * dtype_b * n_out
        vmem = 2 * tile_in + tile_out
        if vmem <= hw.vmem_capacity_bytes:
            best_tile = tile_size

    tile_in = best_tile * dtype_b * n_in
    tile_out = best_tile * dtype_b * n_out
    num_tiles = math.ceil(numel / best_tile)
    double_buffer = num_tiles > 1

    return TileConfig(
        block_dims={"dim0": best_tile},
        num_tiles=num_tiles,
        tile_input_bytes=tile_in,
        tile_output_bytes=tile_out,
        double_buffer=double_buffer,
        vmem_usage_bytes=(2 * tile_in + tile_out) if double_buffer else (tile_in + tile_out),
    )


def find_optimal_tiling_with_analysis(step: ComputeStep, hw: TPUParams) -> dict:
    """Find optimal tiling and return analysis with pipeline balance info."""
    tile_config = find_optimal_tiling(step, hw)

    t_dma_tile = calc_dma_time_ns(tile_config.tile_input_bytes + tile_config.tile_output_bytes, hw)

    if step.compute_unit == "MXU":
        if step.op_type == "matmul":
            bm = tile_config.block_dims.get("M", 1)
            bn = tile_config.block_dims.get("N", 1)
            bk = tile_config.block_dims.get("K", 1)
            tile_flops = 2 * bm * bn * bk
        else:
            tile_flops = step.eval_flops() // max(tile_config.num_tiles, 1)
        t_compute_tile = calc_mxu_time_ns(tile_flops, hw)
    else:
        tile_flops = step.eval_flops() // max(tile_config.num_tiles, 1)
        t_compute_tile = calc_vpu_time_ns(tile_flops, hw)

    ratio = t_dma_tile / t_compute_tile if t_compute_tile > 0 else float("inf")
    bottleneck = "COMPUTE" if t_compute_tile > t_dma_tile else "HBM_BW"

    return {
        "tile_config": tile_config,
        "dma_time_per_tile_ns": t_dma_tile,
        "compute_time_per_tile_ns": t_compute_tile,
        "pipeline_balance_ratio": ratio,
        "bottleneck_per_tile": bottleneck,
    }
