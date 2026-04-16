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

    print("\n\n".join(output_parts))


if __name__ == "__main__":
    main()
