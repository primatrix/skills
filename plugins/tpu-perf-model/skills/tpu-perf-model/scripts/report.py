#!/usr/bin/env python3
"""Report formatter: JSON and text output for pipeline and comparison reports."""
from __future__ import annotations

import json

from pipeline_simulator import PipelineReport, StepResult
from gap_analyzer import ComparisonReport


def _format_bytes(b: int) -> str:
    """Format byte count in human-readable form."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GiB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MiB"
    if b >= 1024:
        return f"{b / 1024:.2f} KiB"
    return f"{b} B"


def _format_ns(ns: float) -> str:
    """Format nanoseconds in human-readable form."""
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.2f} ns"


def _step_to_dict(s: StepResult) -> dict:
    """Convert a StepResult to a JSON-serializable dict."""
    d = {
        "name": s.name,
        "op_type": s.op_type,
        "compute_unit": s.compute_unit,
        "flops": s.flops,
        "hbm_bytes": s.hbm_bytes,
        "t_hbm_ns": s.t_hbm_ns,
        "t_compute_ns": s.t_compute_ns,
        "t_step_ns": s.t_step_ns,
        "bottleneck": s.bottleneck,
        "arithmetic_intensity": s.arithmetic_intensity,
        "fused_with_prev": s.fused_with_prev,
        "fusion_hbm_savings_bytes": s.fusion_hbm_savings_bytes,
    }
    if s.tile_config is not None:
        d["tile_config"] = {
            "block_dims": s.tile_config.block_dims,
            "num_tiles": s.tile_config.num_tiles,
            "double_buffer": s.tile_config.double_buffer,
            "vmem_usage_bytes": s.tile_config.vmem_usage_bytes,
        }
    return d


def pipeline_report_to_json(report: PipelineReport) -> str:
    """Convert a PipelineReport to a JSON string."""
    data = {
        "steps": [_step_to_dict(s) for s in report.steps],
        "summary": {
            "total_time_ns": report.total_time_ns,
            "total_flops": report.total_flops,
            "total_hbm_bytes": report.total_hbm_bytes,
            "fusion_savings_bytes": report.fusion_savings_bytes,
            "overall_arithmetic_intensity": report.overall_arithmetic_intensity,
            "overall_bottleneck": report.overall_bottleneck,
            "efficiency_vs_peak": report.efficiency_vs_peak,
        },
    }
    return json.dumps(data, indent=2)


def pipeline_report_to_text(report: PipelineReport) -> str:
    """Convert a PipelineReport to human-readable text."""
    lines = ["=== Pipeline Performance Report ===", ""]

    for s in report.steps:
        lines.append(f"Step: {s.name} ({s.op_type}, {s.compute_unit})")
        lines.append(f"  FLOPs: {s.flops:,}  |  HBM: {_format_bytes(s.hbm_bytes)}")
        lines.append(f"  t_hbm: {_format_ns(s.t_hbm_ns)}  |  t_compute: {_format_ns(s.t_compute_ns)}  |  t_step: {_format_ns(s.t_step_ns)}")
        lines.append(f"  Bottleneck: {s.bottleneck}  |  AI: {s.arithmetic_intensity:.2f}")
        if s.fused_with_prev:
            lines.append(f"  Fused with prev (saved {_format_bytes(s.fusion_hbm_savings_bytes)})")
        lines.append("")

    lines.append("--- Summary ---")
    lines.append(f"Total time: {_format_ns(report.total_time_ns)}")
    lines.append(f"Total FLOPs: {report.total_flops:,}")
    lines.append(f"Total HBM: {_format_bytes(report.total_hbm_bytes)}")
    if report.fusion_savings_bytes > 0:
        lines.append(f"Fusion savings: {_format_bytes(report.fusion_savings_bytes)}")
    lines.append(f"Overall AI: {report.overall_arithmetic_intensity:.2f}")
    lines.append(f"Overall bottleneck: {report.overall_bottleneck}")
    lines.append(f"Efficiency vs peak: {report.efficiency_vs_peak:.3f}")

    return "\n".join(lines)


def comparison_report_to_text(report: ComparisonReport) -> str:
    """Convert a ComparisonReport to human-readable text."""
    lines = ["=== Gap Analysis Report ===", ""]

    for gap in report.gaps:
        sign = "+" if gap.gap_pct > 0 else ""
        lines.append(f"{gap.metric}: theoretical={gap.theoretical:.2f}, measured={gap.measured:.2f} ({sign}{gap.gap_pct:.1f}%)")
        lines.append(f"  {gap.diagnosis}")
        lines.append("")

    if report.top_opportunities:
        lines.append("--- Top Opportunities ---")
        for i, opp in enumerate(report.top_opportunities, 1):
            lines.append(f"  {i}. {opp}")
        lines.append("")

    lines.append(f"Achievable speedup: {report.achievable_speedup:.2f}x")
    lines.append(f"Theoretical time: {_format_ns(report.theoretical_time_ns)}")
    lines.append(f"Measured time: {_format_ns(report.measured_time_ns)}")

    return "\n".join(lines)
