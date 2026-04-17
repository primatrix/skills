#!/usr/bin/env python3
"""CLI entry point for TPU pipeline scheduling analysis."""

import argparse
import sys

from pipeline_ir import load_spec_from_file
from dependency_analyzer import analyze_dependencies
from pipeline_scheduler import schedule
from vpr_analyzer import analyze_vpr_liveness
from pipeline_report import (
    deps_to_text, deps_to_json, deps_to_mermaid,
    gantt_to_text, gantt_to_mermaid,
    vpr_heatmap_to_text, vpr_to_json,
    suggest_to_text, suggest_to_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="TPU pipeline scheduling analysis with register-level "
                    "dependency and VPR pressure tracking.",
    )
    parser.add_argument(
        "--pipeline", required=True,
        help="Path to pipeline IR JSON file",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--show", default="all",
        help="Sections to show: deps, gantt, vpr, suggest, all "
             "(comma-separated, default: all)",
    )
    parser.add_argument(
        "--mermaid", action="store_true",
        help="Include Mermaid diagram output (text format only)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate VPR timeline plot as PNG image",
    )
    parser.add_argument(
        "--plot-output",
        help="Output path for plot (default: <spec_name>_vpr_timeline.png)",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Generate HTML animation of the pipeline schedule",
    )
    parser.add_argument(
        "--animate-output",
        help="Output path for animation (default: <spec_name>_pipeline.html)",
    )
    args = parser.parse_args()

    sections = set(args.show.split(","))
    show_all = "all" in sections

    spec = load_spec_from_file(args.pipeline)
    graph = analyze_dependencies(spec.ops)
    sched = schedule(spec.ops)
    occ = analyze_vpr_liveness(spec.ops, sched)

    output_parts: list[str] = []

    if show_all or "deps" in sections:
        if args.format == "json":
            output_parts.append(deps_to_json(graph))
        else:
            output_parts.append(deps_to_text(graph))
            if args.mermaid:
                output_parts.append("")
                output_parts.append(deps_to_mermaid(graph))

    if show_all or "gantt" in sections:
        if args.format == "json":
            pass
        else:
            output_parts.append(gantt_to_text(sched))
            if args.mermaid:
                output_parts.append("")
                output_parts.append(gantt_to_mermaid(sched))

    if show_all or "vpr" in sections:
        if args.format == "json":
            output_parts.append(vpr_to_json(occ))
        else:
            output_parts.append(vpr_heatmap_to_text(occ, sched.total_latency_ns))

    if show_all or "suggest" in sections:
        if args.format == "json":
            output_parts.append(suggest_to_json(spec.ops))
        else:
            output_parts.append(suggest_to_text(spec.ops))

    if args.plot:
        from pipeline_plot import plot_vpr_timeline
        plot_path = args.plot_output or f"{spec.name}_vpr_timeline.png"
        plot_vpr_timeline(spec.ops, sched, graph, plot_path, title=spec.name)
        print(f"Plot saved to: {plot_path}")

    if args.animate:
        from pipeline_animate import generate_animation
        anim_path = args.animate_output or f"{spec.name}_pipeline.html"
        generate_animation(spec.ops, sched, graph, anim_path, title=spec.name)
        print(f"Animation saved to: {anim_path}")

    print("\n\n".join(output_parts))


if __name__ == "__main__":
    main()
