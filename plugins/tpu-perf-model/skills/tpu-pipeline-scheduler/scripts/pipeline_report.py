#!/usr/bin/env python3
"""Report generation for pipeline scheduling analysis (text/JSON/Mermaid)."""

from __future__ import annotations

import json
from dependency_analyzer import DependencyGraph, analyze_dependencies
from pipeline_scheduler import ScheduleResult, schedule
from vpr_analyzer import VPROccupancy, analyze_vpr_liveness
from pipeline_ir import PipelineOp


def deps_to_text(graph: DependencyGraph) -> str:
    lines: list[str] = []
    lines.append("=== Data Dependency Graph ===")
    lines.append("")
    lines.append(f"{'From':<15} {'To':<15} {'Hazard':<6} {'Resource':<8} {'ID'}")
    lines.append("-" * 60)
    for e in graph.edges:
        lines.append(
            f"{e.from_op:<15} {e.to_op:<15} {e.hazard_type:<6} "
            f"{e.resource_type:<8} {e.resource_id}"
        )
    lines.append("")
    lines.append(f"Total edges: {len(graph.edges)}")
    return "\n".join(lines)


def deps_to_json(graph: DependencyGraph) -> str:
    payload = {
        "edges": [
            {
                "from_op": e.from_op,
                "to_op": e.to_op,
                "hazard_type": e.hazard_type,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
            }
            for e in graph.edges
        ],
        "total_edges": len(graph.edges),
    }
    return json.dumps(payload, indent=2)


def deps_to_mermaid(graph: DependencyGraph) -> str:
    lines: list[str] = []
    lines.append("graph TD")
    for op in graph.ops:
        label = op.label or op.op_id
        lines.append(f"    {op.op_id}[\"{op.op_id}: {label}\"]")
    for e in graph.edges:
        if e.hazard_type == "RAW":
            arrow = f"-->|RAW {e.resource_id}|"
        elif e.hazard_type == "WAR":
            arrow = f"-.->|WAR {e.resource_id}|"
        else:
            arrow = f"==>|WAW {e.resource_id}|"
        lines.append(f"    {e.from_op} {arrow} {e.to_op}")
    return "\n".join(lines)


def _build_unit_spans(sched: ScheduleResult) -> dict[str, list[tuple[float, float, str]]]:
    """Build time spans per unit row: DMA, MXU_W, MXU_D, VPU.

    For MXU entries with phases, split into MXU_W and MXU_D rows.
    For MXU entries without phases, place entire span on MXU_W.
    Returns {unit_row: [(start, end, op_id), ...]}.
    """
    spans: dict[str, list[tuple[float, float, str]]] = {
        "DMA": [], "MXU_W": [], "MXU_D": [], "VPU": [],
    }
    for e in sched.entries:
        if e.unit == "MXU" and e.phases:
            for ph in e.phases:
                spans[ph.unit_slot].append((ph.start_ns, ph.end_ns, e.op_id))
        elif e.unit == "MXU":
            spans["MXU_W"].append((e.start_ns, e.end_ns, e.op_id))
        else:
            spans.setdefault(e.unit, []).append((e.start_ns, e.end_ns, e.op_id))
    return spans


def gantt_to_text(sched: ScheduleResult) -> str:
    lines: list[str] = []
    lines.append("=== Pipeline Gantt ===")
    lines.append("")
    total = sched.total_latency_ns
    spans = _build_unit_spans(sched)

    width = 60
    for unit in ["DMA", "MXU_W", "MXU_D", "VPU"]:
        row_spans = spans.get(unit, [])
        if not row_spans:
            bar = "·" * width
        else:
            bar = list("·" * width)
            for s_ns, e_ns, _ in row_spans:
                s = int(s_ns / total * width) if total > 0 else 0
                f = max(s + 1, int(e_ns / total * width) if total > 0 else 1)
                for i in range(s, min(f, width)):
                    bar[i] = "█"
            bar = "".join(bar)
        lines.append(f"{unit:>5} |{bar}| {total:.0f}ns")

    lines.append("")
    lines.append(f"{'Op':<15} {'Unit':<5} {'Start':>8} {'End':>8} "
                 f"{'Stall':>8} {'Wait'}")
    lines.append("-" * 65)
    for e in sched.entries:
        if e.unit == "MXU" and e.phases:
            for ph in e.phases:
                lines.append(
                    f"{e.op_id:<15} {ph.unit_slot:<5} {ph.start_ns:>8.0f} "
                    f"{ph.end_ns:>8.0f} {e.stall_ns:>8.0f} {e.wait_reason}"
                )
        else:
            lines.append(
                f"{e.op_id:<15} {e.unit:<5} {e.start_ns:>8.0f} {e.end_ns:>8.0f} "
                f"{e.stall_ns:>8.0f} {e.wait_reason}"
            )
    lines.append("")
    lines.append(f"Total latency: {total:.0f}ns  "
                 f"Total stall: {sched.stall_total_ns:.0f}ns")
    return "\n".join(lines)


def gantt_to_mermaid(sched: ScheduleResult) -> str:
    lines: list[str] = []
    lines.append("gantt")
    lines.append("    dateFormat X")
    lines.append("    axisFormat %s ns")
    spans = _build_unit_spans(sched)
    for unit in ["DMA", "MXU_W", "MXU_D", "VPU"]:
        row_spans = spans.get(unit, [])
        if not row_spans:
            continue
        lines.append(f"    section {unit}")
        # Look up stall info from original entries
        entry_map = {e.op_id: e for e in sched.entries}
        for s_ns, e_ns, op_id in row_spans:
            entry = entry_map.get(op_id)
            crit = "crit, " if entry and entry.stall_ns > 0 else ""
            lines.append(
                f"    {op_id} :{crit}{int(s_ns)}, {int(e_ns)}"
            )
    return "\n".join(lines)


def vpr_heatmap_to_text(occ: VPROccupancy, total_ns: float) -> str:
    lines: list[str] = []
    lines.append("=== VPR Occupancy Heatmap ===")
    lines.append("")

    used_vprs = sorted(set(lv.vpr_id for lv in occ.liveness))
    if not used_vprs:
        lines.append("No VPRs used.")
        return "\n".join(lines)

    n_cols = 40
    step = total_ns / n_cols if total_ns > 0 else 1

    lines.append(f"{'VPR':>6}  " + "".join(
        f"{int(i * step):>4}" if i % 10 == 0 else "    "
        for i in range(0, n_cols, 1)
    )[:n_cols])

    for vpr_id in used_vprs:
        lv = next(l for l in occ.liveness if l.vpr_id == vpr_id)
        bar: list[str] = []
        for col in range(n_cols):
            t = col * step
            if lv.live_start_ns <= t < lv.live_end_ns:
                bar.append("█")
            else:
                bar.append("·")
        lines.append(f"VPR[{vpr_id:>2}] {''.join(bar)}")

    lines.append("")
    lines.append(f"Peak concurrent VPRs: {occ.peak_concurrent}/32 "
                 f"at t={occ.peak_time_ns:.0f}ns")
    lines.append(f"Utilization ratio: {occ.utilization_ratio:.2%}")
    for w in occ.pressure_warnings:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def vpr_to_json(occ: VPROccupancy) -> str:
    payload = {
        "liveness": [
            {
                "vpr_id": lv.vpr_id,
                "defined_by": lv.defined_by,
                "last_used_by": lv.last_used_by,
                "live_start_ns": lv.live_start_ns,
                "live_end_ns": lv.live_end_ns,
            }
            for lv in occ.liveness
        ],
        "peak_concurrent": occ.peak_concurrent,
        "peak_time_ns": occ.peak_time_ns,
        "utilization_ratio": occ.utilization_ratio,
        "pressure_warnings": occ.pressure_warnings,
    }
    return json.dumps(payload, indent=2)


def suggest_to_text(ops: list[PipelineOp]) -> str:
    lines: list[str] = []
    lines.append("=== Reorder Suggestion ===")
    lines.append("")

    orig_sched = schedule(ops)
    orig_occ = analyze_vpr_liveness(ops, orig_sched)

    lines.append("--- Original Order ---")
    lines.append(f"  Order: {' → '.join(e.op_id for e in orig_sched.entries)}")
    lines.append(f"  Total latency: {orig_sched.total_latency_ns:.0f}ns")
    lines.append(f"  Total stall: {orig_sched.stall_total_ns:.0f}ns")
    lines.append(f"  Peak VPRs: {orig_occ.peak_concurrent}")

    dep_graph = analyze_dependencies(ops)
    lines.append("")
    lines.append("--- Analysis ---")
    lines.append(f"  Critical path: {' → '.join(orig_sched.critical_path)}")
    crit_latency = sum(
        orig_sched.entries_by_id[op_id].end_ns -
        orig_sched.entries_by_id[op_id].start_ns
        for op_id in orig_sched.critical_path
    )
    lines.append(f"  Critical path latency: {crit_latency:.0f}ns")
    parallelism = crit_latency / orig_sched.total_latency_ns if orig_sched.total_latency_ns > 0 else 0
    lines.append(f"  Parallelism efficiency: {parallelism:.2%}")

    return "\n".join(lines)


def suggest_to_json(ops: list[PipelineOp]) -> str:
    orig_sched = schedule(ops)
    orig_occ = analyze_vpr_liveness(ops, orig_sched)

    payload = {
        "original": {
            "order": [e.op_id for e in orig_sched.entries],
            "total_latency_ns": orig_sched.total_latency_ns,
            "stall_total_ns": orig_sched.stall_total_ns,
            "peak_vprs": orig_occ.peak_concurrent,
            "critical_path": orig_sched.critical_path,
        },
    }
    return json.dumps(payload, indent=2)
