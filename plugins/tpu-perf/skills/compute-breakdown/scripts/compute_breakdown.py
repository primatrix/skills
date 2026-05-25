"""
Entry point for tpu-perf:compute-breakdown.

Four --mode subcommands share a load -> step-pick -> event-iterate ->
normalize pipeline (Stages 1-3); only Stage 4 (final projection)
differs per mode. Output is exactly one top-level JSON object on
stdout. See spec
docs/superpowers/specs/2026-05-24-tpu-perf-compute-breakdown-design.md.
"""
import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys

# Make the vendored protobuf module importable regardless of cwd. Stage 1
# helpers (added in chunk 2) call xplane_pb2.XSpace().ParseFromString(...).
_PROTO_DIR = pathlib.Path(__file__).parent / "_proto"
sys.path.insert(0, str(_PROTO_DIR))
import xplane_pb2  # noqa: E402  (after sys.path insert, by design)


# ----------------------------------------------------------------------
# Stage 3 per-event normalized record. Field schema per spec §4.
# ----------------------------------------------------------------------
@dataclasses.dataclass
class EventRecord:
    duration_ps: int
    offset_ps: int
    step_id: int
    hlo_category: str
    kind: str                     # 'compute' | 'data_move' | 'comm' | 'other'
    hlo_op: str
    tf_op: str | None
    source_stat: str | None
    source_stack: str | None
    source_inner: str | None
    source_stack_hash: str | None
    agg_key: str
    agg_key_kind: str             # 'stack' | 'tf_op' | 'no_source'
    flops: int | None
    model_flops: int | None
    bytes_accessed: int | None
    raw_bytes_accessed: int | None
    shape_with_layout: str | None
    dtype: str | None
    dtype_uncertain: bool
    program_id: int | None
    deduplicated_name: str | None


def _extract_meta_stats(event_metadata, stat_name_by_id: dict) -> dict:
    """Resolve the stats list on an XEventMetadata into {name: value}.
    Values use the discriminated `oneof value` (six variants) per
    profile-anatomy schema."""
    out: dict = {}
    for s in event_metadata.stats:
        name = stat_name_by_id.get(s.metadata_id)
        if not name:
            continue
        vf = s.WhichOneof("value")
        if vf is None:
            continue
        out[name] = getattr(s, vf)
    return out


_COMPUTE_CATS = frozenset({
    "loop fusion", "convolution fusion", "custom fusion", "output fusion",
    "non-fusion elementwise", "reduce", "reduce-window",
    "sort", "rng-bit-generator", "custom-call",
})
_DATA_MOVE_CATS = frozenset({
    "copy-start", "copy-done", "data formatting", "pad", "broadcast",
    "slice", "dynamic-slice", "dynamic-update-slice", "iota", "convert",
})
_COMM_CATS = frozenset({
    "async-start", "async-done", "all-reduce", "all-gather",
    "reduce-scatter", "collective-permute",
})

_DTYPE_PREFIX_RE = re.compile(r"^([a-z][a-z0-9]*)\[")
_DTYPE_MAP = {
    "bf16": "bf16",
    "f8e4m3fn": "fp8", "f8e5m2": "fp8",
    "f32": "fp32",
    "f16": "fp16",
}


def _classify_kind(hlo_category: str) -> str:
    if hlo_category in _COMPUTE_CATS:
        return "compute"
    if hlo_category in _DATA_MOVE_CATS:
        return "data_move"
    if hlo_category in _COMM_CATS:
        return "comm"
    return "other"


def _parse_dtype(shape_with_layout: str | None) -> str | None:
    if shape_with_layout is None:
        return None
    m = _DTYPE_PREFIX_RE.match(shape_with_layout)
    if not m:
        return "other"
    return _DTYPE_MAP.get(m.group(1), "other")


def _compute_agg_key(*, source_stack: str | None, tf_op: str | None,
                      hlo_category: str) -> tuple[str, str, str | None]:
    """Returns (agg_key, agg_key_kind, source_stack_hash | None).
    Three-tier fallback per spec §4.1."""
    if source_stack:
        h = hashlib.sha1(source_stack.encode("utf-8")).hexdigest()[:16]
        return f"stack:{h}", "stack", h
    if tf_op:
        return f"tfop:{tf_op}", "tf_op", None
    return f"nosrc:{hlo_category}", "no_source", None


def _inner_frame(source_stack: str | None) -> str | None:
    """Innermost frame of `source_stack`: last non-empty line, stripped
    to `file:line` (drop trailing `:<col>` suffix). Spec §4 record schema."""
    if not source_stack:
        return None
    lines = [ln for ln in source_stack.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    # Strip trailing :<col> if present (last colon-separated token).
    # Heuristic: file:line:col -> file:line; file:line -> file:line.
    parts = last.rsplit(":", 2)
    if len(parts) == 3:
        return f"{parts[0]}:{parts[1]}"
    return last


_UNCERTAIN_CATS = frozenset({
    "convolution fusion", "custom fusion", "output fusion", "custom-call",
})
_UNCERTAIN_DTYPES = frozenset({"bf16", "fp32"})


def _is_dtype_uncertain(hlo_category: str, dtype: str | None) -> bool:
    """Spec §4.3: True iff category ∈ {fusion family that wraps mixed-precision
    compute} AND dtype ∈ {bf16, fp32}."""
    return hlo_category in _UNCERTAIN_CATS and dtype in _UNCERTAIN_DTYPES


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
