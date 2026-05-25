"""
Entry point for tpu-perf:compute-breakdown.

Four --mode subcommands share a load -> step-pick -> event-iterate ->
normalize pipeline (Stages 1-3); only Stage 4 (final projection)
differs per mode. Output is exactly one top-level JSON object on
stdout. See spec
docs/superpowers/specs/2026-05-24-tpu-perf-compute-breakdown-design.md.
"""
import argparse
import json
import pathlib
import sys

# Make the vendored protobuf module importable regardless of cwd. Stage 1
# helpers (added in chunk 2) call xplane_pb2.XSpace().ParseFromString(...).
_PROTO_DIR = pathlib.Path(__file__).parent / "_proto"
sys.path.insert(0, str(_PROTO_DIR))
import xplane_pb2  # noqa: E402  (after sys.path insert, by design)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compute_breakdown.py",
        description="Compute-efficiency analysis of a TPU pretraining profile.",
    )
    p.add_argument("profile_dir", help="Path to a profile directory containing *.xplane.pb")
    p.add_argument("--mode", required=True,
                   choices=["summary", "by_source", "non_compute", "roofline"])
    p.add_argument("--device", default="/device:TPU:0",
                   help="XPlane name to analyze (default: /device:TPU:0)")
    p.add_argument("--step", type=int, default=None,
                   help="0-indexed step to analyze (default: middle step)")
    p.add_argument("--step-id", default=None,
                   help="Exact match against Step XEventMetadata.name")
    p.add_argument("--include-comm", action="store_true",
                   help="Include kind=comm events in the analysis")

    # Mode 1
    p.add_argument("--top", type=int, default=50,
                   help="(summary) top K compute groups to emit (default: 50)")
    # Mode 2
    p.add_argument("--include-data-move", action="store_true",
                   help="(by_source) also emit kind=data_move groups")
    # Mode 3
    p.add_argument("--no-comm-stalls", action="store_true",
                   help="(non_compute) exclude async-done from non-compute table")
    # Mode 4
    p.add_argument("--chip", default="v7x", choices=["v7x"],
                   help="(roofline) chip generation; only v7x supported today")
    p.add_argument("--peak-tflops-bf16", type=float, default=None)
    p.add_argument("--peak-tflops-fp8", type=float, default=None)
    p.add_argument("--peak-tflops-fp32", type=float, default=None)
    p.add_argument("--peak-tflops-fp16", type=float, default=None)
    p.add_argument("--peak-hbm-gibps", type=float, default=None)
    return p


def _emit(doc: dict) -> None:
    json.dump(doc, sys.stdout)
    sys.stdout.write("\n")


def _absent(reason: str, mode: str, profile_dir: str) -> dict:
    return {"status": "absent", "reason": reason, "mode": mode,
            "profile_dir": profile_dir, "notes": []}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.step is not None and args.step_id is not None:
        print("error: cannot pass both --step and --step-id", file=sys.stderr)
        return 1

    profile_dir = pathlib.Path(args.profile_dir)
    pbs = sorted(profile_dir.glob("*.xplane.pb")) if profile_dir.is_dir() else []
    if not pbs:
        _emit(_absent("no_xplane_pb", args.mode, args.profile_dir))
        return 0

    # Stages 1-4 not yet implemented; placeholder for chunk 2+.
    _emit(_absent("not_implemented", args.mode, args.profile_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
