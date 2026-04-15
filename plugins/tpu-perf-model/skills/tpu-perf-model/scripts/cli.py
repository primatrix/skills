#!/usr/bin/env python3
"""CLI entry point for TPU performance model.

Usage:
    python cli.py --steps steps.json [--eval eval_result.json] [--format text|json]
"""
import argparse
import sys

from compute_step import load_steps, load_steps_from_file
from hw_params import TPU_V7X
from micro_op_builder import build_micro_op_graph_for_pipeline
from micro_op_report import micro_schedule_to_json, micro_schedule_to_text
from micro_op_scheduler import schedule_micro_op_graph
from pipeline_simulator import simulate_steps
from tiling_optimizer import find_optimal_tiling, find_optimal_tiling_with_analysis
from gap_analyzer import analyze_eval_result, load_eval_result
from report import _step_to_dict, pipeline_report_to_json, pipeline_report_to_text
from report import comparison_report_to_text


def main():
    parser = argparse.ArgumentParser(description="TPU v7x Performance Model")
    parser.add_argument("--steps", required=True, help="Path to ComputeSteps JSON file")
    parser.add_argument("--eval", help="Path to eval_result.json for gap analysis")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--analysis-level", choices=["step", "micro"], default="step", help="Simulation depth")
    parser.add_argument("--show-timeline", action="store_true", help="Show micro-op timeline details")
    parser.add_argument("--show-residency", action="store_true", help="Show micro-op residency details")
    parser.add_argument("--show-critical-path", action="store_true", help="Show micro-op critical path details")
    parser.add_argument("--tiling", action="store_true", help="Show detailed tiling analysis")
    args = parser.parse_args()

    steps = load_steps_from_file(args.steps)
    report = simulate_steps(steps, TPU_V7X)
    step_results = [_step_to_dict(step_result) for step_result in report.steps]

    if args.analysis_level == "micro":
        tile_configs = [find_optimal_tiling(step, TPU_V7X) for step in steps]
        graph = build_micro_op_graph_for_pipeline(steps, tile_configs)
        schedule = schedule_micro_op_graph(graph, TPU_V7X)
        if args.format == "json":
            print(micro_schedule_to_json(schedule, step_results))
        else:
            print(micro_schedule_to_text(schedule, step_results))
    else:
        if args.format == "json":
            print(pipeline_report_to_json(report))
        else:
            print(pipeline_report_to_text(report))

    if args.tiling:
        print("\n" + "=" * 70)
        print("Detailed Tiling Analysis")
        print("=" * 70)
        for step in steps:
            analysis = find_optimal_tiling_with_analysis(step, TPU_V7X)
            tc = analysis["tile_config"]
            print(f"\n  {step.name}:")
            print(f"    Optimal tile:           {tc.block_dims}")
            print(f"    DMA time/tile:          {analysis['dma_time_per_tile_ns']:.2f} ns")
            print(f"    Compute time/tile:      {analysis['compute_time_per_tile_ns']:.2f} ns")
            print(f"    Pipeline balance ratio: {analysis['pipeline_balance_ratio']:.2f} (1.0 = perfect)")
            print(f"    Per-tile bottleneck:    {analysis['bottleneck_per_tile']}")

    if args.eval:
        eval_data = load_eval_result(args.eval)
        comparison = analyze_eval_result(report, eval_data)
        print()
        print(comparison_report_to_text(comparison))


if __name__ == "__main__":
    main()
