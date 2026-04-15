#!/usr/bin/env python3
"""Gap analyzer: compares theoretical model output vs measured eval_result.json data."""
from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline_simulator import PipelineReport


@dataclass
class GapEntry:
    """A single gap between theoretical and measured values."""
    metric: str
    theoretical: float
    measured: float
    gap_pct: float
    diagnosis: str


@dataclass
class ComparisonReport:
    """Full comparison report between theoretical model and measured data."""
    gaps: list[GapEntry]
    top_opportunities: list[str]
    achievable_speedup: float
    theoretical_time_ns: float
    measured_time_ns: float


_DIAGNOSES = {
    "hbm_bytes": {
        "excess": "Excess HBM traffic — check for unfused ops or redundant loads",
        "under": "Measured HBM traffic below theoretical — possible caching or fusion benefit",
    },
    "total_time": {
        "excess": "Measured time exceeds theoretical — likely pipeline stalls or overhead",
        "under": "Measured time below theoretical — model may overestimate",
    },
    "mxu_util": {
        "excess": "MXU utilization exceeds theoretical estimate",
        "under": "MXU utilization below theoretical — check tile shapes and padding",
    },
}


def _diagnose(metric: str, gap_pct: float) -> str:
    """Generate a diagnostic message for a gap."""
    kind = "excess" if gap_pct > 0 else "under"
    templates = _DIAGNOSES.get(metric, {})
    if kind in templates:
        return templates[kind]
    if gap_pct > 0:
        return f"Excess {metric}: measured {gap_pct:.1f}% above theoretical"
    return f"{metric} within theoretical expectations"


def compute_gap(metric: str, theoretical: float, measured: float) -> GapEntry:
    """Compute the gap between theoretical and measured values.

    gap_pct = (measured - theoretical) / theoretical * 100
    Positive means measured is worse (higher bytes, higher time).
    For utilization metrics, positive means measured exceeds theoretical.
    """
    if theoretical == 0:
        gap_pct = 0.0 if measured == 0 else float("inf")
    else:
        gap_pct = (measured - theoretical) / abs(theoretical) * 100.0
    diagnosis = _diagnose(metric, gap_pct)
    return GapEntry(
        metric=metric,
        theoretical=theoretical,
        measured=measured,
        gap_pct=gap_pct,
        diagnosis=diagnosis,
    )


def analyze_eval_result(theoretical: PipelineReport, eval_result: dict) -> ComparisonReport:
    """Compare theoretical pipeline report against measured eval_result data."""
    gaps: list[GapEntry] = []
    opportunities: list[str] = []

    # Time comparison
    measured_time_ns = eval_result.get("total_time_us", 0) * 1000.0
    if measured_time_ns > 0:
        time_gap = compute_gap("total_time", theoretical.total_time_ns, measured_time_ns)
        gaps.append(time_gap)
        if time_gap.gap_pct > 10:
            opportunities.append(
                f"Reduce execution time gap ({time_gap.gap_pct:.0f}% excess): {time_gap.diagnosis}"
            )

    metadata = eval_result.get("metadata", {})
    hw_util = metadata.get("hw_utilization", {})

    # HBM bytes comparison
    measured_hbm = hw_util.get("hbm_bandwidth_bytes")
    if measured_hbm is not None:
        hbm_gap = compute_gap("hbm_bytes", theoretical.total_hbm_bytes, measured_hbm)
        gaps.append(hbm_gap)
        if hbm_gap.gap_pct > 10:
            opportunities.append(
                f"Reduce HBM traffic ({hbm_gap.gap_pct:.0f}% excess): {hbm_gap.diagnosis}"
            )

    # MXU utilization comparison
    measured_mxu = hw_util.get("mxu_utilization_pct")
    if measured_mxu is not None:
        theoretical_mxu = theoretical.efficiency_vs_peak * 100.0
        mxu_gap = compute_gap("mxu_util", theoretical_mxu, measured_mxu)
        gaps.append(mxu_gap)
        if mxu_gap.gap_pct < -10:
            opportunities.append(
                f"Improve MXU utilization ({abs(mxu_gap.gap_pct):.0f}% below theoretical): {mxu_gap.diagnosis}"
            )

    # Vector spills/fills
    profile = metadata.get("profile", {})
    spills = profile.get("vector_spills", 0)
    fills = profile.get("vector_fills", 0)
    if spills > 0 or fills > 0:
        opportunities.append(
            f"Eliminate vector register spills ({spills}) and fills ({fills}) — reduce VMEM pressure"
        )

    # Ensure at least one opportunity if there are gaps
    if gaps and not opportunities:
        worst = max(gaps, key=lambda g: abs(g.gap_pct))
        opportunities.append(f"Investigate {worst.metric} gap ({worst.gap_pct:.1f}%)")

    # Achievable speedup: ratio of measured to theoretical time
    if measured_time_ns > 0 and theoretical.total_time_ns > 0:
        achievable_speedup = measured_time_ns / theoretical.total_time_ns
    else:
        achievable_speedup = 1.0

    return ComparisonReport(
        gaps=gaps,
        top_opportunities=opportunities,
        achievable_speedup=achievable_speedup,
        theoretical_time_ns=theoretical.total_time_ns,
        measured_time_ns=measured_time_ns,
    )


def load_eval_result(path: str) -> dict:
    """Load eval_result.json from disk."""
    with open(path) as f:
        return json.load(f)
