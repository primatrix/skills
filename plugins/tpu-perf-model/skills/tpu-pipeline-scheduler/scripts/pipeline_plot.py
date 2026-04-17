#!/usr/bin/env python3
"""VPR timeline plot — data layer for register activity visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult
from dependency_analyzer import DependencyGraph
from vpr_analyzer import analyze_vpr_liveness


@dataclass
class VPRInterval:
    """A time interval where a VPR is in a specific state."""
    vpr_id: int
    start_ns: float
    end_ns: float
    unit: str        # DMA | MXU | VPU
    access: str      # write | read | live
    op_id: str


def build_vpr_activity(
    ops: list[PipelineOp], sched: ScheduleResult
) -> dict[int, list[VPRInterval]]:
    """Build per-VPR activity intervals from schedule results.

    For each VPR, produces intervals tagged with (unit, access_type):
    - "write": the op's time window when it writes this VPR (output_vprs)
    - "read":  the op's time window when it reads this VPR (input_vprs)
    - "live":  gaps between write-end and last-read-end where VPR holds data
    """
    entries = sched.entries_by_id
    raw: dict[int, list[VPRInterval]] = {}

    for op in ops:
        entry = entries[op.op_id]
        for v in op.output_vprs:
            raw.setdefault(v, []).append(VPRInterval(
                vpr_id=v, start_ns=entry.start_ns, end_ns=entry.end_ns,
                unit=op.unit, access="write", op_id=op.op_id,
            ))
        for v in op.input_vprs:
            raw.setdefault(v, []).append(VPRInterval(
                vpr_id=v, start_ns=entry.start_ns, end_ns=entry.end_ns,
                unit=op.unit, access="read", op_id=op.op_id,
            ))

    # Fill "live" gaps: VPR holds data between active intervals
    occ = analyze_vpr_liveness(ops, sched)
    lv_by_id = {lv.vpr_id: lv for lv in occ.liveness}
    result: dict[int, list[VPRInterval]] = {}
    for vpr_id, intervals in raw.items():
        intervals.sort(key=lambda i: i.start_ns)
        lv = lv_by_id.get(vpr_id)
        if not lv:
            result[vpr_id] = intervals
            continue

        filled: list[VPRInterval] = []
        # Merge overlapping active intervals to find gaps
        active_spans = sorted((i.start_ns, i.end_ns) for i in intervals)
        merged: list[tuple[float, float]] = []
        for s, e in active_spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Unit for live gaps comes from the first writer
        def_unit = next((i.unit for i in intervals if i.access == "write"), intervals[0].unit)

        # Insert live intervals in gaps within liveness range
        prev_end = lv.live_start_ns
        for ms, me in merged:
            if ms > prev_end and ms > lv.live_start_ns:
                gap_start = max(prev_end, lv.live_start_ns)
                if gap_start < ms:
                    filled.append(VPRInterval(
                        vpr_id=vpr_id, start_ns=gap_start, end_ns=ms,
                        unit=def_unit, access="live", op_id="",
                    ))
            prev_end = me
        # Trailing live gap
        if prev_end < lv.live_end_ns:
            filled.append(VPRInterval(
                vpr_id=vpr_id, start_ns=prev_end, end_ns=lv.live_end_ns,
                unit=def_unit, access="live", op_id="",
            ))

        filled.extend(intervals)
        filled.sort(key=lambda i: (i.start_ns, {"write": 0, "read": 1, "live": 2}[i.access]))
        result[vpr_id] = filled

    return result


# --- Color scheme ---
_COLORS = {
    ("DMA", "write"): "#1a5276",
    ("DMA", "read"):  "#5dade2",
    ("DMA", "live"):  "#d4e6f1",
    ("MXU", "write"): "#922b21",
    ("MXU", "read"):  "#e74c3c",
    ("MXU", "live"):  "#f5b7b1",
    ("VPU", "write"): "#196f3d",
    ("VPU", "read"):  "#27ae60",
    ("VPU", "live"):  "#d5f5e3",
}

_UNIT_COLORS = {
    "DMA": "#2980b9", "MXU": "#c0392b", "VPU": "#27ae60",
    "MXU_W": "#c0392b", "MXU_D": "#e74c3c",
}

_HAZARD_STYLES = {
    "RAW": {"linestyle": "-",  "color": "#333333"},
    "WAR": {"linestyle": "--", "color": "#888888"},
    "WAW": {"linestyle": ":",  "color": "#aaaaaa"},
}


def plot_vpr_timeline(
    ops: list[PipelineOp],
    sched: ScheduleResult,
    graph: DependencyGraph,
    output_path: str,
    title: str = "",
) -> None:
    """Render VPR timeline heatmap with Gantt strips and dependency arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    import matplotlib.patches as mpatches

    if not ops or sched.total_latency_ns == 0:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, "No operations to plot", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    activity = build_vpr_activity(ops, sched)
    used_vprs = sorted(activity.keys())
    n_vprs = len(used_vprs)
    vpr_to_row = {v: i for i, v in enumerate(used_vprs)}
    total_ns = sched.total_latency_ns

    # --- Figure layout: Gantt (top, small) + Heatmap (main) ---
    fig_height = max(4, 1.5 + n_vprs * 0.4)
    fig, (ax_gantt, ax_heat) = plt.subplots(
        2, 1, figsize=(14, fig_height),
        gridspec_kw={"height_ratios": [1, max(3, n_vprs * 0.35)]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # === Gantt strips ===
    unit_order = ["DMA", "MXU_W", "MXU_D", "VPU"]
    for u_idx, unit in enumerate(unit_order):
        for entry in sched.entries:
            if entry.unit == "MXU" and entry.phases:
                for ph in entry.phases:
                    if ph.unit_slot == unit:
                        ax_gantt.barh(
                            u_idx, ph.end_ns - ph.start_ns, left=ph.start_ns,
                            height=0.7, color=_UNIT_COLORS[unit], alpha=0.85,
                            edgecolor="white", linewidth=0.5,
                        )
                        width = ph.end_ns - ph.start_ns
                        if width / total_ns > 0.08:
                            ax_gantt.text(
                                ph.start_ns + width / 2, u_idx,
                                entry.op_id, ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold",
                            )
            elif entry.unit == "MXU" and unit == "MXU_W":
                ax_gantt.barh(
                    u_idx, entry.end_ns - entry.start_ns, left=entry.start_ns,
                    height=0.7, color=_UNIT_COLORS[unit], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                width = entry.end_ns - entry.start_ns
                if width / total_ns > 0.08:
                    ax_gantt.text(
                        entry.start_ns + width / 2, u_idx,
                        entry.op_id, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )
            elif entry.unit == unit:
                ax_gantt.barh(
                    u_idx, entry.end_ns - entry.start_ns, left=entry.start_ns,
                    height=0.7, color=_UNIT_COLORS[unit], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                width = entry.end_ns - entry.start_ns
                if width / total_ns > 0.08:
                    ax_gantt.text(
                        entry.start_ns + width / 2, u_idx,
                        entry.op_id, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )

    ax_gantt.set_yticks(range(len(unit_order)))
    ax_gantt.set_yticklabels(unit_order, fontsize=9, fontweight="bold")
    ax_gantt.set_ylim(-0.5, len(unit_order) - 0.5)
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlim(0, total_ns)
    ax_gantt.tick_params(axis="x", labelbottom=False)
    ax_gantt.set_ylabel("Unit", fontsize=9)
    ax_gantt.spines["top"].set_visible(False)
    ax_gantt.spines["right"].set_visible(False)

    # === VPR Heatmap ===
    for vpr_id, intervals in activity.items():
        row = vpr_to_row[vpr_id]
        for iv in intervals:
            color = _COLORS.get((iv.unit, iv.access), "#eeeeee")
            width = iv.end_ns - iv.start_ns
            if width <= 0:
                continue
            rect = Rectangle(
                (iv.start_ns, row - 0.4), width, 0.8,
                facecolor=color, edgecolor="white", linewidth=0.3,
            )
            ax_heat.add_patch(rect)

    ax_heat.set_yticks(range(n_vprs))
    ax_heat.set_yticklabels([f"VPR[{v}]" for v in used_vprs], fontsize=8,
                            fontfamily="monospace")
    ax_heat.set_ylim(-0.5, n_vprs - 0.5)
    ax_heat.invert_yaxis()
    ax_heat.set_xlim(0, total_ns)
    ax_heat.set_xlabel("Time (ns)", fontsize=10)
    ax_heat.set_ylabel("VPR Register", fontsize=9)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)

    # === Dependency arrows (VPR-type only) ===
    entries_by_id = sched.entries_by_id
    for edge in graph.edges:
        if edge.resource_type != "VPR":
            continue
        from_entry = entries_by_id.get(edge.from_op)
        to_entry = entries_by_id.get(edge.to_op)
        if not from_entry or not to_entry:
            continue

        # Find the VPR that creates this dependency
        vpr_num = int(edge.resource_id.replace("VPR[", "").replace("]", ""))
        if vpr_num not in vpr_to_row:
            continue

        # Arrow from end of producer to start of consumer
        from_x = from_entry.end_ns
        to_x = to_entry.start_ns
        from_row = vpr_to_row[vpr_num]
        to_row = from_row

        style = _HAZARD_STYLES.get(edge.hazard_type, _HAZARD_STYLES["RAW"])

        arrow = FancyArrowPatch(
            (from_x, from_row - 0.45), (to_x, to_row - 0.45),
            connectionstyle="arc3,rad=-0.2",
            arrowstyle="->,head_width=3,head_length=3",
            linewidth=1.0, **style,
        )
        ax_heat.add_patch(arrow)

    # === Legend ===
    legend_handles = []
    for unit in ["DMA", "MXU", "VPU"]:
        for access, label in [("write", "Write"), ("read", "Read"), ("live", "Live")]:
            c = _COLORS[(unit, access)]
            legend_handles.append(mpatches.Patch(color=c, label=f"{unit} {label}"))
    # Hazard arrow legend
    for hz, style in _HAZARD_STYLES.items():
        legend_handles.append(plt.Line2D(
            [0], [0], color=style["color"], linestyle=style["linestyle"],
            linewidth=1.5, label=f"{hz}",
        ))

    ax_heat.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=True,
        ncol=1, title="Legend", title_fontsize=8,
    )

    # === Title ===
    occ = analyze_vpr_liveness(ops, sched)
    title_str = title or "VPR Timeline"
    fig.suptitle(
        f"{title_str}  |  {total_ns:.0f}ns  |  "
        f"Peak VPR: {occ.peak_concurrent}/32  |  "
        f"Stall: {sched.stall_total_ns:.0f}ns",
        fontsize=11, fontweight="bold", y=0.98,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
