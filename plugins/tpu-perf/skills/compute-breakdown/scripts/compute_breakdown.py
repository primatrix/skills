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
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import xplane_pb2  # noqa: E402  (after sys.path insert, by design)
import _peaks  # noqa: E402


# Per-group cap on number of unique HLO-op signatures retained for the
# `hlo_op_breakdown` field. Above this many distinct signatures, the rest
# are folded into one overflow bucket so output size stays bounded.
_HLO_OP_BREAKDOWN_CAP = 64

# Number of slowest signature rows surfaced per group in the JSON
# output's `hlo_op_breakdown`. The full retained dict (up to
# _HLO_OP_BREAKDOWN_CAP) is sorted by total_dur_ps desc and the top
# _HLO_OP_BREAKDOWN_OUTPUT_N rows are emitted; the rest is folded into
# the overflow bucket.
_HLO_OP_BREAKDOWN_OUTPUT_N = 8


# Regex precompiled for HLO-op-text signature extraction. We can't bucket by
# raw HLO text — each metadata_id has a unique text (varying SSA numbers,
# operand names) so 690 events produce 500+ distinct strings, which defeats
# the rollup. Instead we normalize: for custom-calls, use the call target;
# for other ops, use the op opcode (the token before the first `[`).
_HLO_OPCODE_RE = re.compile(r'^\s*%?[\w.]+\s*=\s*[^=]*?\b([a-z][a-z0-9_-]*)\s*\(')
_CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]+)"')
# kLoop fusions get further sub-categorized by the leading-fusion-name token
# (e.g. `convert_bitcast_fusion`, `pad_bitcast_fusion`) so the breakdown can
# distinguish dtype convert from pad-only from generic fusion work.
_FUSION_NAME_RE = re.compile(r'^\s*%([a-z_][a-z0-9_]*)\b')


def _hlo_op_signature(hlo_text: str, hlo_category: str) -> str:
    """Bucket key for hlo_op_breakdown rollup. Designed so that semantically
    similar ops collapse to the same key, but distinct *kinds* of work stay
    separate.

    - custom-call → "custom-call:<target>" (e.g. "custom-call:tpu_custom_call",
      "custom-call:AllocateBuffer"). Crucial: AllocateBuffer at 75ps and
      tpu_custom_call at 2.6ms must NOT collapse into one bucket.
    - loop/input fusion → "<fusion-name-prefix> [<category>]"
      (e.g. "convert_bitcast_fusion [loop fusion]"). Strips the trailing
      `.NNN` SSA index so different instances of the same logical fusion
      collapse.
    - everything else → the HLO opcode (`dot`, `reduce`, `slice`, ...)
      qualified by category for readability ("dot [convolution]").

    For Pallas TPU custom-calls a richer signature would also include the
    inner kernel name (e.g. `kda_intra_chunk_bwd_subchunk`), but those names
    are SSA-level identifiers in the IR text and not always exposed in the
    metadata's `name`. Callers wanting that detail should look at
    `example_hlo_op` on the heaviest bucket.
    """
    if not hlo_text:
        return f"<empty> [{hlo_category}]"
    if hlo_category == "custom-call":
        m = _CUSTOM_CALL_TARGET_RE.search(hlo_text)
        if m:
            return f"custom-call:{m.group(1)}"
        return "custom-call:<no-target>"
    if "fusion" in hlo_category:
        m = _FUSION_NAME_RE.match(hlo_text)
        if m:
            # Strip trailing .NNN SSA index from the fusion name token.
            name = re.sub(r'\.\d+$', '', m.group(1))
            return f"{name} [{hlo_category}]"
        return f"<unparsed-fusion> [{hlo_category}]"
    m = _HLO_OPCODE_RE.match(hlo_text)
    if m:
        return f"{m.group(1)} [{hlo_category}]"
    return f"<unparsed> [{hlo_category}]"


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


HLO_OP_RE = re.compile(
    r'^\s*%?[\w.]+\s*=\s*'
    r'([a-z][a-z0-9]*)\['
    r'([^\]]*)\]'
    r'(\{[^}]*\})?'
    r'\s+\w[-\w]*\s*\('
    r'\s*([a-z][a-z0-9]*)\['
    r'([^\]]*)\]'
    r'(\{[^}]*\})?'
)


def _parse_hlo_op_text(text: str) -> tuple:
    """Extract (out_dtype, out_layout, in_dtype, in_layout) from an HLO IR
    string. Inspects only the first operand. Returns all-None on no match."""
    if not text:
        return (None, None, None, None)
    m = HLO_OP_RE.match(text)
    if not m:
        return (None, None, None, None)
    return (m.group(1), m.group(3), m.group(4), m.group(6))


def _parse_hlo_op_text_full(text: str) -> tuple:
    """Like `_parse_hlo_op_text` but also returns the out/in shape strings.
    Returns (out_dt, out_shape, out_lay, in_dt, in_shape, in_lay) or
    a six-None tuple on no match."""
    if not text:
        return (None, None, None, None, None, None)
    m = HLO_OP_RE.match(text)
    if not m:
        return (None, None, None, None, None, None)
    return (m.group(1), m.group(2), m.group(3),
            m.group(4), m.group(5), m.group(6))


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
ASYNC_DONE_CATEGORIES = frozenset({
    "async-done",
    "all-reduce-done",
    "all-gather-done",
    "reduce-scatter-done",
    "collective-permute-done",
    "send-done",
    "recv-done",
})

_COMM_CATS = frozenset({
    "async-start", "async-done", "all-reduce", "all-gather",
    "reduce-scatter", "collective-permute",
    "send", "recv",
}) | ASYNC_DONE_CATEGORIES

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


def _pick_step_window(plane, *, step_idx: int | None, step_id: str | None):
    """Spec §4.5. Returns (step_event_or_None, step_index_int, start_ps,
    end_ps, notes_list)."""
    steps_line = next((l for l in plane.lines if l.name == "Steps"), None)
    notes: list[str] = []
    if steps_line is None or len(steps_line.events) == 0:
        # Fallback: full XLA Ops window.
        ops = next((l for l in plane.lines if l.name == "XLA Ops"), None)
        if ops is None or len(ops.events) == 0:
            return None, -1, 0, 0, ["no Steps line; falling back to full-plane window",
                                     "XLA Ops line empty or missing"]
        starts = [ev.offset_ps for ev in ops.events]
        ends = [ev.offset_ps + ev.duration_ps for ev in ops.events]
        notes.append("no Steps line; falling back to full-plane window")
        return None, -1, min(starts), max(ends), notes

    sorted_steps = sorted(steps_line.events, key=lambda e: e.offset_ps)

    if step_id is not None:
        em_map = plane.event_metadata
        matches = [(i, e) for i, e in enumerate(sorted_steps)
                   if em_map.get(e.metadata_id) is not None
                   and em_map[e.metadata_id].name == step_id]
        if not matches:
            raise ValueError(f"--step-id {step_id!r} matched zero Step events")
        if len(matches) > 1:
            notes.append("multi-match for step-id; picked first")
        idx, ev = matches[0]
    elif step_idx is not None:
        if step_idx < 0 or step_idx >= len(sorted_steps):
            raise ValueError(
                f"--step {step_idx} out of range [0, {len(sorted_steps)})")
        idx, ev = step_idx, sorted_steps[step_idx]
    else:
        # Auto-pick: busiest step by XLA Ops event count. Falls back to middle
        # when the XLA Ops line is missing/empty or no step window has any
        # events (e.g. all steps are warmup/compile or events landed outside
        # step windows). The middle-step heuristic alone is brittle: profiles
        # with long warmup tails can place the middle index inside an idle
        # window that has near-zero compute events, leading to empty/garbage
        # output. Counting events per step is O(n_events * log n_steps) and
        # robustly picks the actual training step.
        ops_line = next((l for l in plane.lines if l.name == "XLA Ops"), None)
        idx = None
        if ops_line is not None and len(ops_line.events) > 0:
            import bisect
            starts = [s.offset_ps for s in sorted_steps]
            ends = [s.offset_ps + s.duration_ps for s in sorted_steps]
            counts = [0] * len(sorted_steps)
            for ev2 in ops_line.events:
                i = bisect.bisect_right(starts, ev2.offset_ps) - 1
                if 0 <= i < len(sorted_steps) and ev2.offset_ps < ends[i]:
                    counts[i] += 1
            best = max(range(len(counts)), key=lambda i: counts[i])
            if counts[best] > 0:
                idx = best
                notes.append(
                    f"auto-picked busiest step (idx={idx}, "
                    f"n_xla_ops_events={counts[idx]} of {sum(counts)} total "
                    f"across {len(sorted_steps)} steps)")
        if idx is None:
            idx = len(sorted_steps) // 2
            notes.append(
                f"auto-picked middle step (idx={idx}); "
                "no XLA Ops events to score steps by")
        ev = sorted_steps[idx]

    return ev, idx, ev.offset_ps, ev.offset_ps + ev.duration_ps, notes


@dataclasses.dataclass
class _PipelineStats:
    """Sidechannel from _iter_event_records to the caller."""
    n_events_unresolved: int = 0
    while_total_ps: int = 0
    unknown_categories: dict = dataclasses.field(default_factory=dict)


def _iter_event_records(plane, *, start_ps: int, end_ps: int, step_id: int,
                         stats: _PipelineStats):
    """Yield one EventRecord per admitted XLA-Ops event in [start_ps, end_ps).
    Mutates `stats` for events that don't yield a record (unresolved, while,
    unknown category)."""
    ops_line = next((l for l in plane.lines if l.name == "XLA Ops"), None)
    if ops_line is None:
        return
    stat_name_by_id = {smid: sm.name for smid, sm in plane.stat_metadata.items()}

    for ev in ops_line.events:
        if not (start_ps <= ev.offset_ps < end_ps):
            continue
        em = plane.event_metadata.get(ev.metadata_id)
        if em is None:
            stats.n_events_unresolved += 1
            continue
        mstats = _extract_meta_stats(em, stat_name_by_id)
        hlo_cat = mstats.get("hlo_category", "")
        if hlo_cat == "while":
            stats.while_total_ps += ev.duration_ps
            continue

        kind = _classify_kind(hlo_cat)
        if kind == "other":
            # Track unrecognized categories so the spec maintainer can update §4.4.
            stats.unknown_categories[hlo_cat] = stats.unknown_categories.get(hlo_cat, 0) + 1

        source_stack = mstats.get("source_stack")
        tf_op = mstats.get("tf_op")
        shape = mstats.get("shape_with_layout")
        dtype = _parse_dtype(shape) if shape else None
        agg_key, agg_kind, stack_hash = _compute_agg_key(
            source_stack=source_stack, tf_op=tf_op, hlo_category=hlo_cat)

        yield EventRecord(
            duration_ps=ev.duration_ps,
            offset_ps=ev.offset_ps,
            step_id=step_id,
            hlo_category=hlo_cat,
            kind=kind,
            hlo_op=em.name,
            tf_op=tf_op,
            source_stat=mstats.get("source"),
            source_stack=source_stack,
            source_inner=_inner_frame(source_stack),
            source_stack_hash=stack_hash,
            agg_key=agg_key,
            agg_key_kind=agg_kind,
            flops=mstats.get("flops"),
            model_flops=mstats.get("model_flops"),
            bytes_accessed=mstats.get("bytes_accessed"),
            raw_bytes_accessed=mstats.get("raw_bytes_accessed"),
            shape_with_layout=shape,
            dtype=dtype,
            dtype_uncertain=_is_dtype_uncertain(hlo_cat, dtype),
            program_id=mstats.get("program_id"),
            deduplicated_name=mstats.get("deduplicated_name"),
        )


def _load_and_normalize(*, profile_dir: str, device: str,
                          step_idx: int | None, step_id: str | None):
    """Stages 1+2+3. Returns (records, ctx).

    `records` is a list[EventRecord] when status is ok; None when absent.
    `ctx` always carries `status` ('ok' | 'absent'); on absent it also has
    `reason` and `notes`. On ok it carries: step_id, step_window_ps,
    step_duration_ps, notes (list), pipeline_stats (_PipelineStats),
    profile_dir (str), device (str), xspace_pb_path (str).
    """
    pdir = pathlib.Path(profile_dir)
    pbs = sorted(pdir.glob("*.xplane.pb")) if pdir.is_dir() else []
    if not pbs:
        return None, {"status": "absent", "reason": "no_xplane_pb", "notes": []}

    xs = xplane_pb2.XSpace()
    with open(pbs[0], "rb") as f:
        xs.ParseFromString(f.read())

    plane = next((p for p in xs.planes if p.name == device), None)
    if plane is None:
        have = [p.name for p in xs.planes]
        return None, {"status": "absent", "reason": "device_not_found",
                      "notes": [f"have: {have}"]}

    if not any(l.name == "XLA Ops" for l in plane.lines):
        have = [l.name for l in plane.lines]
        return None, {"status": "absent", "reason": "no_xla_ops_line",
                      "notes": [f"have: {have}"]}

    step_event, sid, s_ps, e_ps, notes = _pick_step_window(
        plane, step_idx=step_idx, step_id=step_id)
    pstats = _PipelineStats()
    records = list(_iter_event_records(
        plane, start_ps=s_ps, end_ps=e_ps, step_id=sid, stats=pstats))

    return records, {
        "status": "ok",
        "step_id": sid,
        "step_window_ps": [s_ps, e_ps],
        "step_duration_ps": e_ps - s_ps,
        "notes": notes,
        "pipeline_stats": pstats,
        "profile_dir": str(pdir),
        "device": device,
        "xspace_pb_path": str(pbs[0]),
    }


@dataclasses.dataclass
class _GroupAgg:
    agg_key: str
    agg_key_kind: str
    source_inner: str | None
    source_stack: str | None
    tf_op: str | None
    kind: str                            # canonical kind of the group
    n_executions: int = 0
    total_dur_ps: int = 0
    min_dur_ps: int = 0
    max_dur_ps: int = 0
    _flops_sum: int = 0
    _flops_seen: int = 0                 # how many records contributed flops
    _bytes_sum: int = 0
    _bytes_seen: int = 0
    _model_flops_sum: int = 0
    _model_flops_seen: int = 0
    hlo_categories: dict = dataclasses.field(default_factory=dict)
    shapes: list = dataclasses.field(default_factory=list)
    shapes_truncated: bool = False
    dtypes: dict = dataclasses.field(default_factory=dict)
    dtype_uncertain: bool = False        # OR of all member records
    first_dtype: str | None = None       # dtype of the FIRST record in
                                          # the group; never overwritten.
                                          # Used by mode 4 (roofline) per
                                          # spec §8.2.
    example_hlo_op: str | None = None     # HLO text of the SLOWEST single
                                           # event in the group. Replaced
                                           # whenever we see a longer dur.
                                           # Avoids the AllocateBuffer trap
                                           # where the first record was a
                                           # zero-cost placeholder.
    example_hlo_op_dur_ps: int = 0        # dur of the example_hlo_op event,
                                           # so callers can sanity-check
                                           # that the example is meaningful.
    # Per-HLO-text rollup so callers can see time distribution within a group
    # without trusting a single example. Key: hlo_op text. Value: [total_dur_ps,
    # n_executions, hlo_category]. Capped at _HLO_OP_BREAKDOWN_CAP unique texts;
    # overflow folded into a single "(other-N-texts)" bucket.
    _hlo_op_dur: dict = dataclasses.field(default_factory=dict)
    _hlo_op_overflow_dur: int = 0
    _hlo_op_overflow_n: int = 0
    _hlo_op_overflow_count: int = 0

    @property
    def avg_dur_ps(self) -> float:
        return self.total_dur_ps / self.n_executions if self.n_executions else 0.0

    @property
    def flops_sum(self) -> int | None:
        return self._flops_sum if self._flops_seen > 0 else None

    @property
    def bytes_accessed_sum(self) -> int | None:
        return self._bytes_sum if self._bytes_seen > 0 else None

    @property
    def model_flops_sum(self) -> int | None:
        return self._model_flops_sum if self._model_flops_seen > 0 else None

    def hlo_op_breakdown(self, top_n: int = _HLO_OP_BREAKDOWN_OUTPUT_N) -> list:
        """Top-N normalized-signature rows sorted by total_dur_ps desc.

        Each row: {signature, hlo_category, total_dur_ps, n_executions,
        pct_of_group, example_hlo_op, example_hlo_op_dur_ps}. Allows callers
        to verify time distribution within a group without trusting a single
        first-record example. Below-cutoff rows + cap overflow are folded
        into one (other-N-buckets) row.

        Signature normalization (see _hlo_op_signature):
          - custom-call:<target>          (so AllocateBuffer ≠ tpu_custom_call)
          - <fusion-name-prefix> [...]    (so convert_bitcast ≠ pad_bitcast)
          - <opcode> [<category>]         (everything else)
        """
        items = sorted(self._hlo_op_dur.items(), key=lambda kv: -kv[1][0])
        rows = []
        kept = items[:top_n]
        rest = items[top_n:]
        rest_dur = sum(v[0] for _, v in rest) + self._hlo_op_overflow_dur
        rest_n = sum(v[1] for _, v in rest) + self._hlo_op_overflow_n
        rest_buckets = len(rest) + (
            1 if self._hlo_op_overflow_count > 0 else 0)
        denom = self.total_dur_ps if self.total_dur_ps else 1
        for sig, (dur, n, cat, ex_text, ex_dur) in kept:
            rows.append({
                "signature": sig,
                "hlo_category": cat,
                "total_dur_ps": dur,
                "n_executions": n,
                "pct_of_group": round(100.0 * dur / denom, 3),
                "example_hlo_op": ex_text,
                "example_hlo_op_dur_ps": ex_dur,
            })
        if rest_buckets > 0 or rest_dur > 0:
            rows.append({
                "signature": f"(other-{rest_buckets}-signatures)",
                "hlo_category": None,
                "total_dur_ps": rest_dur,
                "n_executions": rest_n,
                "pct_of_group": round(100.0 * rest_dur / denom, 3),
                "example_hlo_op": None,
                "example_hlo_op_dur_ps": None,
            })
        return rows


def _aggregate_by_key(records: list[EventRecord],
                        *, dedupe_shapes_cap: int = 8) -> dict:
    groups: dict[str, _GroupAgg] = {}
    for r in records:
        g = groups.get(r.agg_key)
        if g is None:
            g = _GroupAgg(
                agg_key=r.agg_key, agg_key_kind=r.agg_key_kind,
                source_inner=r.source_inner, source_stack=r.source_stack,
                tf_op=r.tf_op, kind=r.kind,
                example_hlo_op=r.hlo_op,
                example_hlo_op_dur_ps=r.duration_ps,
                min_dur_ps=r.duration_ps, max_dur_ps=r.duration_ps,
                first_dtype=r.dtype,    # spec §8.2: first record wins.
            )
            groups[r.agg_key] = g
        g.n_executions += 1
        g.total_dur_ps += r.duration_ps
        if r.duration_ps < g.min_dur_ps:
            g.min_dur_ps = r.duration_ps
        if r.duration_ps > g.max_dur_ps:
            g.max_dur_ps = r.duration_ps
        # Replace example_hlo_op whenever a longer single-event dur shows up.
        # First-record-wins is dangerous when groups mix zero-cost placeholders
        # (e.g. AllocateBuffer custom-calls at 75ps) with real heavy ops
        # (e.g. tpu_custom_call Pallas kernels at 2.6ms) — the longest event
        # is a far better representative than the first.
        if r.duration_ps > g.example_hlo_op_dur_ps:
            g.example_hlo_op = r.hlo_op
            g.example_hlo_op_dur_ps = r.duration_ps
        # Per-signature rollup. Bucketing by raw HLO text doesn't collapse
        # related ops (each metadata_id has unique SSA numbers), so we
        # normalize via _hlo_op_signature: custom-calls keyed by target,
        # fusions by fusion-name-prefix, everything else by opcode.
        # Cell layout: [total_dur_ps, n_executions, hlo_category,
        #               example_hlo_text, example_dur_ps].
        sig = _hlo_op_signature(r.hlo_op, r.hlo_category)
        cell = g._hlo_op_dur.get(sig)
        if cell is not None:
            cell[0] += r.duration_ps
            cell[1] += 1
            if r.duration_ps > cell[4]:
                cell[3] = r.hlo_op
                cell[4] = r.duration_ps
        elif len(g._hlo_op_dur) < _HLO_OP_BREAKDOWN_CAP:
            g._hlo_op_dur[sig] = [r.duration_ps, 1, r.hlo_category,
                                   r.hlo_op, r.duration_ps]
        else:
            g._hlo_op_overflow_dur += r.duration_ps
            g._hlo_op_overflow_n += 1
            g._hlo_op_overflow_count += 1
        if r.flops is not None:
            g._flops_sum += r.flops
            g._flops_seen += 1
        if r.bytes_accessed is not None:
            g._bytes_sum += r.bytes_accessed
            g._bytes_seen += 1
        if r.model_flops is not None:
            g._model_flops_sum += r.model_flops
            g._model_flops_seen += 1
        g.hlo_categories[r.hlo_category] = g.hlo_categories.get(r.hlo_category, 0) + 1
        if r.shape_with_layout and r.shape_with_layout not in g.shapes:
            if len(g.shapes) < dedupe_shapes_cap:
                g.shapes.append(r.shape_with_layout)
            else:
                g.shapes_truncated = True
        if r.dtype:
            g.dtypes[r.dtype] = g.dtypes.get(r.dtype, 0) + 1
        if r.dtype_uncertain:
            g.dtype_uncertain = True
    return groups


def _compute_totals(records: list[EventRecord], *, pstats: _PipelineStats,
                     step_duration_ps: int) -> dict:
    """Spec §5 totals block: per-kind sums, counts, while accounting,
    unknown_categories, unresolved counter."""
    n_by_kind = {"compute": 0, "data_move": 0, "comm": 0, "other": 0}
    d_by_kind = {"compute": 0, "data_move": 0, "comm": 0, "other": 0}
    for r in records:
        n_by_kind[r.kind] += 1
        d_by_kind[r.kind] += r.duration_ps
    non_while_sum = sum(d_by_kind.values())
    while_pct = (100.0 * pstats.while_total_ps / step_duration_ps
                 if step_duration_ps > 0 else 0.0)
    return {
        "n_events_total":         len(records),
        "n_events_compute":       n_by_kind["compute"],
        "n_events_data_move":     n_by_kind["data_move"],
        "n_events_comm":          n_by_kind["comm"],
        "n_events_other":         n_by_kind["other"],
        "n_events_unresolved":    pstats.n_events_unresolved,
        "compute_duration_ps":    d_by_kind["compute"],
        "data_move_duration_ps":  d_by_kind["data_move"],
        "comm_duration_ps":       d_by_kind["comm"],
        "other_duration_ps":      d_by_kind["other"],
        "while_container_duration_ps": pstats.while_total_ps,
        "non_while_duration_ps_sum":   non_while_sum,
        "while_pct_of_step":      round(while_pct, 3),
        "unknown_categories":     dict(pstats.unknown_categories),
    }


def _run_summary_mode(records: list[EventRecord], *, ctx: dict,
                        include_comm: bool, top: int) -> dict:
    pstats = ctx["pipeline_stats"]
    step_dur = ctx["step_duration_ps"]
    # Spec §5: `totals` and `by_kind_rollup` are cross-kind summaries of
    # the *whole* step — they reflect the actual breakdown including
    # comm/data_move regardless of the include_comm flag. The flag only
    # controls which records get ranked into `top_compute_groups`.
    # `include_comm` historically allowed comm into the ranking too;
    # we keep that for parity with --include-data-move (mode 2).
    totals = _compute_totals(records, pstats=pstats, step_duration_ps=step_dur)
    if include_comm:
        rankable = [r for r in records if r.kind in ("compute", "comm")]
    else:
        rankable = [r for r in records if r.kind == "compute"]
    compute_records = rankable
    groups = _aggregate_by_key(compute_records)
    ordered = sorted(groups.values(), key=lambda g: -g.total_dur_ps)

    compute_dur = totals["compute_duration_ps"] or 1
    step_dur_safe = step_dur or 1

    def _g_to_dict(g: _GroupAgg, rank: int) -> dict:
        return {
            "rank": rank,
            "agg_key":      g.agg_key,
            "agg_key_kind": g.agg_key_kind,
            "source_inner": g.source_inner,
            "tf_op":        g.tf_op,
            "source_stack": g.source_stack,
            "n_executions": g.n_executions,
            "total_dur_ps": g.total_dur_ps,
            "min_dur_ps":   g.min_dur_ps,
            "max_dur_ps":   g.max_dur_ps,
            "avg_dur_ps":   round(g.avg_dur_ps, 3),
            "pct_of_compute": round(100.0 * g.total_dur_ps / compute_dur, 3),
            "pct_of_step":    round(100.0 * g.total_dur_ps / step_dur_safe, 3),
            "hlo_categories": dict(g.hlo_categories),
            "flops_sum":          g.flops_sum,
            "bytes_accessed_sum": g.bytes_accessed_sum,
            "example_hlo_op":     g.example_hlo_op,
            "example_hlo_op_dur_ps": g.example_hlo_op_dur_ps,
            "hlo_op_breakdown":   g.hlo_op_breakdown(),
        }

    top_list = [_g_to_dict(g, i + 1) for i, g in enumerate(ordered[:top])]
    tail = ordered[top:]
    tail_dur = sum(g.total_dur_ps for g in tail)

    coverage = {"stack": 0, "tf_op": 0, "no_source": 0}
    for r in compute_records:
        coverage[r.agg_key_kind] = coverage.get(r.agg_key_kind, 0) + 1

    by_kind_rollup: dict = {}
    for kind in ("compute", "data_move", "comm"):
        n = totals[f"n_events_{kind}"]
        d = totals[f"{kind}_duration_ps"]
        by_kind_rollup[kind] = {
            "n": n, "dur_ps": d,
            "pct_of_step": round(100.0 * d / step_dur_safe, 3),
        }

    return {
        "status": "ok",
        "mode": "summary",
        "profile_dir": ctx["profile_dir"],
        "device": ctx["device"],
        "step_id": ctx["step_id"],
        "step_window_ps": ctx["step_window_ps"],
        "step_duration_ps": step_dur,
        "notes": list(ctx["notes"]),
        "totals": totals,
        "agg_key_coverage": coverage,
        "top_compute_groups": top_list,
        "tail_compute": {"n_groups_omitted": len(tail), "dur_ps": tail_dur},
        "by_kind_rollup": by_kind_rollup,
    }


def _run_by_source_mode(records: list[EventRecord], *, ctx: dict,
                          include_comm: bool,
                          include_data_move: bool) -> dict:
    """Mode 2: full per-agg_key table for client-side scope filtering."""
    step_dur = ctx["step_duration_ps"]
    pstats = ctx["pipeline_stats"]
    totals = _compute_totals(records, pstats=pstats, step_duration_ps=step_dur)
    kept_kinds = {"compute"}
    if include_data_move:
        kept_kinds.add("data_move")
    if include_comm:
        kept_kinds.add("comm")
    visible = [r for r in records if r.kind in kept_kinds]
    groups = _aggregate_by_key(visible, dedupe_shapes_cap=8)
    group_rows = []
    for g in groups.values():
        n = g.n_executions
        group_rows.append({
            "agg_key": g.agg_key,
            "agg_key_kind": g.agg_key_kind,
            "source_inner": g.source_inner,
            "source_stack": g.source_stack,
            "tf_op": g.tf_op,
            "kind": g.kind,
            "hlo_categories": dict(g.hlo_categories),
            "n_executions": n,
            "total_dur_ps": g.total_dur_ps,
            "min_dur_ps": g.min_dur_ps,
            "max_dur_ps": g.max_dur_ps,
            "avg_dur_ps": g.total_dur_ps // n if n else 0,
            "flops_sum": g.flops_sum,
            "model_flops_sum": g.model_flops_sum,
            "bytes_accessed_sum": g.bytes_accessed_sum,
            "shapes": list(g.shapes),
            "shapes_truncated": g.shapes_truncated,
            "dtypes": dict(g.dtypes),
            "dtype_uncertain": g.dtype_uncertain,
            "example_hlo_op": g.example_hlo_op,
            "example_hlo_op_dur_ps": g.example_hlo_op_dur_ps,
            "hlo_op_breakdown": g.hlo_op_breakdown(),
        })
    totals_out = dict(totals)
    totals_out["n_groups_total"] = len(group_rows)
    return {
        "status": "ok",
        "mode": "by_source",
        "profile_dir": ctx["profile_dir"],
        "device": ctx["device"],
        "step_id": ctx["step_id"],
        "step_window_ps": ctx["step_window_ps"],
        "step_duration_ps": step_dur,
        "notes": list(ctx["notes"]),
        "totals": totals_out,
        "groups": group_rows,
    }


_COMM_STALL_CATEGORY = "async-done (comm stall)"
_COMM_STALL_NOTE = (
    "async-done included as comm-stall non-compute time; "
    "pass --no-comm-stalls to exclude"
)


def _run_non_compute_mode(records: list[EventRecord], *, ctx: dict,
                            include_comm: bool,
                            include_comm_stalls: bool) -> dict:
    """Mode 3: padding/cast/copy/transpose audit. Two layers:
    by_category and by_source_within_category."""
    step_dur = ctx["step_duration_ps"]
    pstats = ctx["pipeline_stats"]
    totals = _compute_totals(records, pstats=pstats, step_duration_ps=step_dur)

    visible: list[EventRecord] = []
    for r in records:
        if r.kind == "data_move":
            visible.append(r)
        elif (include_comm_stalls and r.kind == "comm"
              and r.hlo_category in ASYNC_DONE_CATEGORIES):
            visible.append(dataclasses.replace(
                r, hlo_category=_COMM_STALL_CATEGORY
            ))
        elif include_comm and r.kind == "comm":
            visible.append(r)

    cat_acc: dict[str, dict] = {}
    for r in visible:
        c = cat_acc.get(r.hlo_category)
        if c is None:
            c = {
                "hlo_category": r.hlo_category,
                "n_executions": 0, "total_dur_ps": 0,
                "min_dur_ps": r.duration_ps, "max_dur_ps": r.duration_ps,
                "agg_keys": set(),
                "agg_key_coverage": {"stack": 0, "tf_op": 0, "no_source": 0},
            }
            cat_acc[r.hlo_category] = c
        c["n_executions"] += 1
        c["total_dur_ps"] += r.duration_ps
        if r.duration_ps < c["min_dur_ps"]:
            c["min_dur_ps"] = r.duration_ps
        if r.duration_ps > c["max_dur_ps"]:
            c["max_dur_ps"] = r.duration_ps
        c["agg_keys"].add(r.agg_key)
        c["agg_key_coverage"][r.agg_key_kind] = (
            c["agg_key_coverage"].get(r.agg_key_kind, 0) + 1
        )
    by_category = []
    for c in cat_acc.values():
        n = c["n_executions"]
        by_category.append({
            "hlo_category":     c["hlo_category"],
            "n_executions":     n,
            "total_dur_ps":     c["total_dur_ps"],
            "min_dur_ps":       c["min_dur_ps"],
            "max_dur_ps":       c["max_dur_ps"],
            "avg_dur_ps":       c["total_dur_ps"] // n if n else 0,
            "n_groups":         len(c["agg_keys"]),
            "agg_key_coverage": c["agg_key_coverage"],
        })

    pair_acc: dict = {}
    for r in visible:
        key = (r.hlo_category, r.agg_key)
        p = pair_acc.get(key)
        if p is None:
            p = {
                "hlo_category":  r.hlo_category,
                "agg_key":       r.agg_key,
                "agg_key_kind":  r.agg_key_kind,
                "source_inner":  r.source_inner,
                "source_stack":  r.source_stack,
                "tf_op":         r.tf_op,
                "n_executions": 0, "total_dur_ps": 0,
                "min_dur_ps":   r.duration_ps, "max_dur_ps": r.duration_ps,
                "shapes_in":  [], "shapes_out": [],
                "example_hlo_op": r.hlo_op,
                "_dtype_change_seen": False,
                "_dtype_change_value": False,
                "_layout_change_seen": False,
                "_layout_change_value": False,
                "_layout_change_null": False,
            }
            pair_acc[key] = p
        p["n_executions"] += 1
        p["total_dur_ps"] += r.duration_ps
        if r.duration_ps < p["min_dur_ps"]:
            p["min_dur_ps"] = r.duration_ps
        if r.duration_ps > p["max_dur_ps"]:
            p["max_dur_ps"] = r.duration_ps
        (out_dt, out_shape, out_lay,
         in_dt, in_shape, in_lay) = _parse_hlo_op_text_full(r.hlo_op)
        if out_dt is not None and in_dt is not None:
            p["_dtype_change_seen"] = True
            if out_dt != in_dt:
                p["_dtype_change_value"] = True
            if out_shape is not None and len(p["shapes_out"]) < 4:
                s = f"{out_dt}[{out_shape}]" + (out_lay if out_lay else "")
                if s not in p["shapes_out"]:
                    p["shapes_out"].append(s)
            if in_shape is not None and len(p["shapes_in"]) < 4:
                s = f"{in_dt}[{in_shape}]" + (in_lay if in_lay else "")
                if s not in p["shapes_in"]:
                    p["shapes_in"].append(s)
            if out_lay is None or in_lay is None:
                p["_layout_change_null"] = True
            else:
                p["_layout_change_seen"] = True
                if out_lay != in_lay:
                    p["_layout_change_value"] = True

    by_source_within_category = []
    for p in pair_acc.values():
        n = p["n_executions"]
        dtype_change = (p["_dtype_change_value"]
                          if p["_dtype_change_seen"] else None)
        if p["_layout_change_seen"]:
            layout_change = p["_layout_change_value"]
        elif p["_layout_change_null"]:
            layout_change = None
        else:
            layout_change = None
        by_source_within_category.append({
            "hlo_category":   p["hlo_category"],
            "agg_key":        p["agg_key"],
            "agg_key_kind":   p["agg_key_kind"],
            "source_inner":   p["source_inner"],
            "source_stack":   p["source_stack"],
            "tf_op":          p["tf_op"],
            "n_executions":   n,
            "total_dur_ps":   p["total_dur_ps"],
            "min_dur_ps":     p["min_dur_ps"],
            "max_dur_ps":     p["max_dur_ps"],
            "avg_dur_ps":     p["total_dur_ps"] // n if n else 0,
            "shapes_in":      p["shapes_in"] or None,
            "shapes_out":     p["shapes_out"] or None,
            "dtype_change":   dtype_change,
            "layout_change":  layout_change,
            "example_hlo_op": p["example_hlo_op"],
        })

    non_compute_dur = sum(p["total_dur_ps"] for p in pair_acc.values())
    compute_dur = totals["compute_duration_ps"]
    totals_out = dict(totals)
    totals_out["non_compute_pct_of_step"] = (
        round(100.0 * non_compute_dur / step_dur, 3) if step_dur > 0 else 0.0
    )
    totals_out["non_compute_pct_of_compute"] = (
        round(100.0 * non_compute_dur / compute_dur, 3) if compute_dur > 0 else 0.0
    )

    notes = list(ctx["notes"])
    if include_comm_stalls and any(
        r.kind == "comm" and r.hlo_category in ASYNC_DONE_CATEGORIES
        for r in records
    ):
        notes.append(_COMM_STALL_NOTE)

    return {
        "status":           "ok",
        "mode":             "non_compute",
        "profile_dir":      ctx["profile_dir"],
        "device":           ctx["device"],
        "step_id":          ctx["step_id"],
        "step_window_ps":   ctx["step_window_ps"],
        "step_duration_ps": step_dur,
        "notes":            notes,
        "totals":           totals_out,
        "by_category":      by_category,
        "by_source_within_category": by_source_within_category,
    }


def _run_roofline_mode(records: list[EventRecord], *, ctx: dict,
                          peaks: dict) -> dict:
    """Mode 4: dtype-aware roofline analysis on the resolved peaks table."""
    step_dur = ctx["step_duration_ps"]

    compute_recs = [r for r in records if r.kind == "compute"]
    groups = _aggregate_by_key(compute_recs, dedupe_shapes_cap=8)

    eligible_rows = []
    skipped = {
        "n_no_flops":               0,
        "n_no_bytes":               0,
        "n_dtype_other":            0,
        "n_peak_unknown_for_dtype": 0,
        "total_dur_ps_skipped":     0,
    }

    for g in groups.values():
        dt = g.first_dtype

        if g.flops_sum is None or g.flops_sum <= 0:
            skipped["n_no_flops"] += 1
            skipped["total_dur_ps_skipped"] += g.total_dur_ps
            continue
        if g.bytes_accessed_sum is None or g.bytes_accessed_sum <= 0:
            skipped["n_no_bytes"] += 1
            skipped["total_dur_ps_skipped"] += g.total_dur_ps
            continue
        if dt not in ("bf16", "fp8", "fp16", "fp32"):
            skipped["n_dtype_other"] += 1
            skipped["total_dur_ps_skipped"] += g.total_dur_ps
            continue
        peak_tflops = peaks.get(f"peak_tflops_{dt}")
        peak_hbm = peaks.get("peak_hbm_gibps")
        if peak_tflops is None or peak_hbm is None:
            skipped["n_peak_unknown_for_dtype"] += 1
            skipped["total_dur_ps_skipped"] += g.total_dur_ps
            continue

        t_compute_seconds = g.flops_sum / (peak_tflops * 1e12)
        t_compute_theory_ps = t_compute_seconds * 1e12
        t_hbm_seconds = g.bytes_accessed_sum / (peak_hbm * (1024 ** 3))
        t_hbm_theory_ps = t_hbm_seconds * 1e12
        t_roofline_theory_ps = max(t_compute_theory_ps, t_hbm_theory_ps)

        arithmetic_intensity = g.flops_sum / g.bytes_accessed_sum
        ridge_point = (peak_tflops * 1e12) / (peak_hbm * (1024 ** 3))
        bound = "compute" if arithmetic_intensity >= ridge_point else "memory"

        mfu = t_compute_theory_ps / g.total_dur_ps if g.total_dur_ps > 0 else 0.0
        hbm_util = t_hbm_theory_ps / g.total_dur_ps if g.total_dur_ps > 0 else 0.0
        roofline_util = (t_roofline_theory_ps / g.total_dur_ps
                            if g.total_dur_ps > 0 else 0.0)
        shortfall_ps = g.total_dur_ps - t_roofline_theory_ps
        shortfall_pct = (1 - roofline_util) * 100

        eligible_rows.append({
            "agg_key":              g.agg_key,
            "agg_key_kind":         g.agg_key_kind,
            "source_inner":         g.source_inner,
            "tf_op":                g.tf_op,
            "hlo_categories":       dict(g.hlo_categories),
            "n_executions":         g.n_executions,
            "total_dur_ps":         g.total_dur_ps,
            "flops_sum":            g.flops_sum,
            "bytes_accessed_sum":   g.bytes_accessed_sum,
            "dtype":                dt,
            "dtype_uncertain":      g.dtype_uncertain,
            "arithmetic_intensity": round(arithmetic_intensity, 4),
            "ridge_point":          round(ridge_point, 2),
            "bound":                bound,
            "t_compute_theory_ps":  int(t_compute_theory_ps),
            "t_hbm_theory_ps":      int(t_hbm_theory_ps),
            "t_roofline_theory_ps": int(t_roofline_theory_ps),
            "mfu":                  round(mfu, 4),
            "hbm_util":             round(hbm_util, 4),
            "roofline_util":        round(roofline_util, 4),
            "shortfall_ps":         int(shortfall_ps),
            "shortfall_pct":        round(shortfall_pct, 2),
        })

    total_eligible_dur = sum(r["total_dur_ps"] for r in eligible_rows)

    def _weighted(field: str) -> float:
        if total_eligible_dur <= 0:
            return 0.0
        return sum(r[field] * r["total_dur_ps"] for r in eligible_rows) / total_eligible_dur

    top_shortfall = sorted(eligible_rows,
                              key=lambda r: r["shortfall_ps"], reverse=True)[:10]
    top_shortfall_short = [
        {
            "agg_key":      r["agg_key"],
            "source_inner": r["source_inner"],
            "tf_op":        r["tf_op"],
            "total_dur_ps": r["total_dur_ps"],
            "shortfall_ps": r["shortfall_ps"],
            "bound":        r["bound"],
        }
        for r in top_shortfall
    ]

    # Coverage transparency: weighted_avg_mfu is computed only over the
    # rooflined (eligible) groups. The weighted_avg_* numbers are
    # representative of the *full* compute only when rooflined_pct_of_compute
    # is high. When much compute is skipped (no FLOPs metadata, dtype=other,
    # no peak for dtype), the averages become a partial picture and the
    # caveat must be surfaced. See SKILL.md "Roofline coverage".
    step_compute_total = sum(r.duration_ps for r in records if r.kind == "compute")
    rooflined_pct = (round(100.0 * total_eligible_dur / step_compute_total, 2)
                     if step_compute_total > 0 else 0.0)
    skipped_pct = (round(100.0 * skipped["total_dur_ps_skipped"] / step_compute_total, 2)
                   if step_compute_total > 0 else 0.0)

    step_summary = {
        "step_compute_duration_ps":   total_eligible_dur,
        # Renamed-for-clarity aliases — emit alongside the legacy field.
        "rooflined_dur_ps":           total_eligible_dur,
        "step_compute_dur_ps_total":  step_compute_total,
        "rooflined_pct_of_compute":   rooflined_pct,
        "skipped_pct_of_compute":     skipped_pct,
        "weighted_avg_mfu":           round(_weighted("mfu"), 4),
        "weighted_avg_hbm_util":      round(_weighted("hbm_util"), 4),
        "weighted_avg_roofline_util": round(_weighted("roofline_util"), 4),
        "step_shortfall_ps":          int(sum(r["shortfall_ps"] for r in eligible_rows)),
        "top_shortfall_groups":       top_shortfall_short,
    }

    return {
        "status":           "ok",
        "mode":             "roofline",
        "profile_dir":      ctx["profile_dir"],
        "device":           ctx["device"],
        "step_id":          ctx["step_id"],
        "step_window_ps":   ctx["step_window_ps"],
        "step_duration_ps": step_dur,
        "notes":            list(ctx["notes"]),
        "chip":             ctx.get("chip", "v7x"),
        "peaks_used":       peaks,
        "step_summary":     step_summary,
        "groups":           eligible_rows,
        "skipped_groups":   skipped,
    }


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
    p.add_argument("--no-comm-stalls", dest="include_comm_stalls",
                   action="store_false", default=True,
                   help="(non_compute) exclude async-done events; default is to include them as 'async-done (comm stall)'")
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

    try:
        records, ctx = _load_and_normalize(
            profile_dir=args.profile_dir,
            device=args.device,
            step_idx=args.step,
            step_id=args.step_id,
        )
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1

    if records is None:
        # Absent path. ctx already carries status/reason/notes.
        out = {
            "status": ctx["status"], "reason": ctx["reason"],
            "mode": args.mode, "profile_dir": args.profile_dir,
            "notes": ctx.get("notes", []),
        }
        _emit(out)
        return 0

    if args.mode == "summary":
        _emit(_run_summary_mode(records, ctx=ctx,
                                  include_comm=args.include_comm,
                                  top=args.top))
        return 0

    if args.mode == "by_source":
        _emit(_run_by_source_mode(records, ctx=ctx,
                                     include_comm=args.include_comm,
                                     include_data_move=args.include_data_move))
        return 0

    if args.mode == "non_compute":
        _emit(_run_non_compute_mode(records, ctx=ctx,
                                       include_comm=args.include_comm,
                                       include_comm_stalls=args.include_comm_stalls))
        return 0

    if args.mode == "roofline":
        peaks = _peaks.resolve_peaks(
            args.chip,
            override_tflops_bf16=args.peak_tflops_bf16,
            override_tflops_fp8=args.peak_tflops_fp8,
            override_tflops_fp32=args.peak_tflops_fp32,
            override_tflops_fp16=args.peak_tflops_fp16,
            override_hbm_gibps=args.peak_hbm_gibps,
        )
        ctx_with_chip = dict(ctx)
        ctx_with_chip["chip"] = args.chip
        _emit(_run_roofline_mode(records, ctx=ctx_with_chip, peaks=peaks))
        return 0

    # Other modes wired up in later chunks.
    _emit({"status": "absent", "reason": "not_implemented",
           "mode": args.mode, "profile_dir": args.profile_dir,
           "notes": []})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
