#!/usr/bin/env python3
"""Instruction-level pipeline simulator for TPU performance modeling.

Takes a list of ComputeStep and produces timing analysis with:
- Per-step HBM transfer time, compute time, bottleneck identification
- Double-buffering pipeline overlap modeling
- Fusion analysis (saves HBM round-trips for fused ops)
"""
from __future__ import annotations

from dataclasses import dataclass

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
        # Fusion saves one HBM write from the previous op and one HBM read for this op.
        fusion_savings = 2 * min(prev_output_bytes, step.total_input_bytes)
        hbm_bytes -= fusion_savings
        hbm_bytes = max(0, hbm_bytes)

    t_hbm_ns = calc_dma_time_ns(hbm_bytes, hw)

    if step.compute_unit == "MXU":
        t_compute_ns = calc_mxu_time_ns(flops, hw)
    else:
        t_compute_ns = calc_vpu_time_ns(flops, hw)

    from tiling_optimizer import find_optimal_tiling
    tile_config = find_optimal_tiling(step, hw)

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
