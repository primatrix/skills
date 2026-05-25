"""
Entry point for tpu-perf:memory-profile.

Locates HBM peak occupancy within a chosen step (or across the full
trace) and emits one JSON object on stdout describing the peak moment
plus every buffer alive at that moment, rollups, timeline samples, and
diagnostics. Single-mode skill — no --mode flag. See spec
docs/superpowers/specs/2026-05-25-tpu-perf-memory-profile-design.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


SKILL_NAME = "memory-profile"
SCHEMA_VERSION = 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_profile.py",
        description=(
            "Locate HBM peak occupancy within a step and list every buffer "
            "alive at that moment."
        ),
    )
    p.add_argument("profile_dir", help="Profile directory containing *.xplane.pb")

    step_group = p.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step", type=int, default=None,
        help="Explicit step index on /device:TPU:0 'Steps' line (0-based).",
    )
    step_group.add_argument(
        "--all-trace", action="store_true",
        help="Disable step scoping; analyse the full trace window.",
    )

    p.add_argument(
        "--step-policy", choices=("peak", "last", "first"), default="peak",
        help="Default step picker when --step is not given (default: peak).",
    )
    p.add_argument("--top", type=int, default=30,
                   help="Top-K applied to alive_at_peak.buffers and rollup tables.")
    p.add_argument("--persistent-threshold-steps", type=int, default=2,
                   help="Min crossed step boundaries for lifetime_class=persistent.")
    p.add_argument("--include-host-pools", action="store_true",
                   help="Include allocator pools other than HBM (id != 0).")
    p.add_argument("--time-samples", type=int, default=200,
                   help="Number of equally-spaced timeline samples.")
    return p


def _emit_absent(profile_dir: str, reason: str, **extra) -> dict:
    return {
        "status": "absent",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "reason": reason,
        "inputs": {"profile_dir": profile_dir, **extra},
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile_dir = args.profile_dir
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        json.dump(_emit_absent(profile_dir, "no_xplane_pb"), sys.stdout)
        sys.stdout.write("\n")
        return 0
    # Wired up in Task 6.
    json.dump(_emit_absent(profile_dir, "not_implemented"), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
