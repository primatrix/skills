#!/usr/bin/env python3
"""Instruction-level pipeline simulator for TPU performance modeling.

Takes a list of ComputeStep and produces timing analysis with:
- Per-step HBM transfer time, compute time, bottleneck identification
- Double-buffering pipeline overlap modeling
- Fusion analysis (saves HBM round-trips for fused ops)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from compute_step import ComputeStep
from hw_params import TPUParams


def calc_dma_time_ns(bytes_: int, hw: TPUParams) -> float:
    """HBM transfer time in nanoseconds."""
    return bytes_ / hw.hbm_bw_bytes_per_sec * 1e9


def calc_mxu_time_ns(flops: int, hw: TPUParams) -> float:
    """MXU compute time in nanoseconds."""
    return flops / hw.mxu_peak_flops * 1e9


def calc_vpu_time_ns(flops: int, hw: TPUParams) -> float:
    """VPU compute time in nanoseconds (rough: ~1/100 of MXU peak)."""
    vpu_flops = hw.mxu_peak_flops / 100
    return flops / vpu_flops * 1e9


@dataclass
class TileConfig:
    """Tiling configuration for a compute step."""
    block_dims: dict[str, int]
    num_tiles: int
    tile_input_bytes: int
    tile_output_bytes: int
    double_buffer: bool
    vmem_usage_bytes: int


@dataclass
class StepResult:
    """Result of simulating one compute step."""
    name: str
    op_type: str
    compute_unit: str
    flops: int
    hbm_bytes: int
    t_hbm_ns: float
    t_compute_ns: float
    t_step_ns: float
    bottleneck: str  # "HBM_BW" or "COMPUTE"
    arithmetic_intensity: float
    tile_config: TileConfig
    fused_with_prev: bool = False
    fusion_hbm_savings_bytes: int = 0


@dataclass
class PipelineReport:
    """Aggregate report for a sequence of compute steps."""
    steps: list[StepResult]
    total_time_ns: float
    total_flops: int
    total_hbm_bytes: int
    fusion_savings_bytes: int
    overall_arithmetic_intensity: float
    overall_bottleneck: str
    efficiency_vs_peak: float


def _find_tile_config(step: ComputeStep, hw: TPUParams) -> TileConfig:
    """Find a tile configuration that fits in VMEM, enabling double buffering if possible."""
    vmem = hw.vmem_capacity_bytes
    total_input = step.total_input_bytes
    total_output = step.total_output_bytes

    if step.op_type == "matmul":
        # Try tiling along M dimension for matmul
        M = step.flops_vars.get("M", 1)
        N = step.flops_vars.get("N", 1)
        K = step.flops_vars.get("K", 1)
        dtype_b = step.inputs[0].size_bytes // (step.inputs[0].numel or 1)

        # Find smallest tile along M that fits in VMEM with double buffering
        # and produces multiple tiles (needed for overlap).
        # First pass: find all valid tile sizes, prefer ones with num_tiles > 1.
        best = None
        tile_m = M
        while tile_m >= 1:
            tile_in = (tile_m * K + K * N) * dtype_b
            tile_out = (tile_m * N) * dtype_b
            db_vmem = 2 * tile_in + tile_out
            if db_vmem <= vmem:
                num_tiles = math.ceil(M / tile_m)
                if num_tiles > 1:
                    return TileConfig(
                        block_dims={"M": tile_m, "N": N, "K": K},
                        num_tiles=num_tiles,
                        tile_input_bytes=tile_in,
                        tile_output_bytes=tile_out,
                        double_buffer=True,
                        vmem_usage_bytes=db_vmem,
                    )
                elif best is None:
                    best = (tile_m, tile_in, tile_out, db_vmem)
            tile_m //= 2

        # Only 1 tile fits — no double buffering possible
        if best:
            tile_m, tile_in, tile_out, db_vmem = best
            return TileConfig(
                block_dims={"M": tile_m, "N": N, "K": K},
                num_tiles=1,
                tile_input_bytes=tile_in,
                tile_output_bytes=tile_out,
                double_buffer=False,
                vmem_usage_bytes=tile_in + tile_out,
            )

        # Fallback: single tile, no double buffer
        return TileConfig(
            block_dims={"M": 1, "N": N, "K": K},
            num_tiles=M,
            tile_input_bytes=(K + K * N) * dtype_b,
            tile_output_bytes=N * dtype_b,
            double_buffer=M > 1,
            vmem_usage_bytes=(2 * (K + K * N) + N) * dtype_b,
        )

    else:
        # Elementwise / other: tile along first dim
        shape = step.inputs[0].shape
        numel = step.inputs[0].numel
        dtype_b = step.inputs[0].size_bytes // numel if numel else 1

        tile_elems = numel
        first_dim = shape[0] if shape else 1
        tile_first = first_dim

        while tile_first > 1:
            tile_elems = (numel // first_dim) * tile_first
            tile_in = tile_elems * dtype_b * len(step.inputs)
            tile_out = tile_elems * dtype_b * len(step.outputs)
            db_vmem = 2 * tile_in + tile_out
            if db_vmem <= vmem:
                num_tiles = math.ceil(first_dim / tile_first)
                double_buffer = num_tiles > 1
                usage = db_vmem if double_buffer else (tile_in + tile_out)
                return TileConfig(
                    block_dims={"dim0": tile_first},
                    num_tiles=num_tiles,
                    tile_input_bytes=tile_in,
                    tile_output_bytes=tile_out,
                    double_buffer=double_buffer,
                    vmem_usage_bytes=usage,
                )
            tile_first //= 2

        tile_in = total_input
        tile_out = total_output
        return TileConfig(
            block_dims={"dim0": 1},
            num_tiles=first_dim,
            tile_input_bytes=tile_in // first_dim,
            tile_output_bytes=tile_out // first_dim,
            double_buffer=first_dim > 1,
            vmem_usage_bytes=2 * (tile_in // first_dim) + tile_out // first_dim,
        )


def simulate_step(
    step: ComputeStep,
    hw: TPUParams,
    fused_with_prev: bool = False,
    prev_output_bytes: int = 0,
) -> StepResult:
    """Simulate a single compute step and return timing results."""
    flops = step.eval_flops()
    hbm_bytes = step.total_io_bytes

    # Fusion: skip re-reading inputs that came from previous step's output
    fusion_savings = 0
    if fused_with_prev and prev_output_bytes > 0:
        fusion_savings = min(prev_output_bytes, step.total_input_bytes)
        # Save both the read of fused input and the write from prev step
        fusion_savings += min(prev_output_bytes, step.total_input_bytes)
        hbm_bytes -= fusion_savings

    t_hbm_ns = calc_dma_time_ns(hbm_bytes, hw)

    if step.compute_unit == "MXU":
        t_compute_ns = calc_mxu_time_ns(flops, hw)
    else:
        t_compute_ns = calc_vpu_time_ns(flops, hw)

    tile_config = _find_tile_config(step, hw)

    # Pipeline timing with double buffering
    if tile_config.double_buffer and tile_config.num_tiles > 1:
        n = tile_config.num_tiles
        t_dma_tile = t_hbm_ns / n
        t_compute_tile = t_compute_ns / n
        # First tile: load only. Last tile: compute only. Middle tiles: overlap.
        t_step_ns = t_dma_tile + (n - 1) * max(t_dma_tile, t_compute_tile) + t_compute_tile
    else:
        t_step_ns = t_hbm_ns + t_compute_ns

    bottleneck = "COMPUTE" if t_compute_ns > t_hbm_ns else "HBM_BW"
    ai = flops / hbm_bytes if hbm_bytes > 0 else float("inf")

    return StepResult(
        name=step.name,
        op_type=step.op_type,
        compute_unit=step.compute_unit,
        flops=flops,
        hbm_bytes=hbm_bytes,
        t_hbm_ns=t_hbm_ns,
        t_compute_ns=t_compute_ns,
        t_step_ns=t_step_ns,
        bottleneck=bottleneck,
        arithmetic_intensity=ai,
        tile_config=tile_config,
        fused_with_prev=fused_with_prev,
        fusion_hbm_savings_bytes=fusion_savings,
    )


def simulate_steps(steps: list[ComputeStep], hw: TPUParams) -> PipelineReport:
    """Simulate a sequence of compute steps with fusion analysis."""
    results: list[StepResult] = []
    total_fusion_savings = 0

    for i, step in enumerate(steps):
        fused = step.fusable_with_prev and i > 0
        prev_out_bytes = steps[i - 1].total_output_bytes if i > 0 else 0

        result = simulate_step(step, hw, fused_with_prev=fused, prev_output_bytes=prev_out_bytes)
        total_fusion_savings += result.fusion_hbm_savings_bytes
        results.append(result)

    total_time = sum(r.t_step_ns for r in results)
    total_flops = sum(r.flops for r in results)
    total_hbm = sum(r.hbm_bytes for r in results)
    overall_ai = total_flops / total_hbm if total_hbm > 0 else float("inf")
    overall_bottleneck = "COMPUTE" if overall_ai > hw.ridge_point else "HBM_BW"
    peak_time_ns = total_flops / hw.mxu_peak_flops * 1e9
    efficiency = peak_time_ns / total_time if total_time > 0 else 0.0

    return PipelineReport(
        steps=results,
        total_time_ns=total_time,
        total_flops=total_flops,
        total_hbm_bytes=total_hbm,
        fusion_savings_bytes=total_fusion_savings,
        overall_arithmetic_intensity=overall_ai,
        overall_bottleneck=overall_bottleneck,
        efficiency_vs_peak=efficiency,
    )
