#!/usr/bin/env python3
"""CLI entry point for TPU performance model.

Usage:
    python cli.py --steps steps.json [--eval eval_result.json] [--format text|json]
"""
import argparse
import json
import sys

from compute_step import load_steps, load_steps_from_file
from hw_params import TPU_V7X
from pipeline_simulator import simulate_steps
from tiling_optimizer import find_optimal_tiling_with_analysis
from gap_analyzer import analyze_eval_result, load_eval_result
from report import pipeline_report_to_json, pipeline_report_to_text
from report import comparison_report_to_text


def main():
    parser = argparse.ArgumentParser(description="TPU v7x Performance Model")
    parser.add_argument("--steps", required=True, help="Path to ComputeSteps JSON file")
    parser.add_argument("--eval", help="Path to eval_result.json for gap analysis")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--tiling", action="store_true", help="Show detailed tiling analysis")
    args = parser.parse_args()

    steps = load_steps_from_file(args.steps)
    report = simulate_steps(steps, TPU_V7X)

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
