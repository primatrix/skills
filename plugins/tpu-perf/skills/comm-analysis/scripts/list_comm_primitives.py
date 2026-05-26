"""
List every communication primitive in a TPU profile, with rich attributes.

Usage:
    python3 list_comm_primitives.py <profile_dir> [--by kind|source|op] [--limit N]
                                    [--sort-by stall|wall|not_cov]
                                    [--include-copies] [--json out.json]

Reads the device planes (TC, SC0, SC1) of *.xplane.pb. Pairs Async XLA Ops
events by 'flow' stat. Adds sync collectives from XLA Ops. HLO join (axis,
group_size, channel_id) is optional — happens automatically if a
*.hlo_proto.pb is present.

Per-row `not_cov_ps`: time NOT overlapped by compute on the SAME core kind
(TC comm vs TC compute, SC comm vs SC compute — separate timelines because
TC and SC don't compete for resources). This is the only authoritative
critical-path metric on unpaired-dominated captures, where stall_ps/hidden_ps
collapse to sentinel values (stall ≈ wall, hidden = 0).

Sort key behavior:
  - stall:    Σ stall_ps           — valid when async pairing is intact
  - wall:     Σ wall_ps            — engine busy time (parallel-friendly)
  - not_cov:  Σ not_cov_ps         — true critical-path exposure
  - default:  ADAPTIVE — uses `stall` when unpaired ratio ≤ 50%; auto-
              switches to `not_cov` and emits a [warn] header otherwise.
              The skill's SKILL.md §8 corollary explains why.

Output: a header line, then a per-row table for the chosen aggregation view.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc


# ---------------------------------------------------------------------------
# Replica-group extraction (walks all four proto locations)
# ---------------------------------------------------------------------------

def _replica_ids(instr) -> list[int]:
    """Extract replica_ids of the first replica group.

    Walks the four proto locations in priority order, matching how XLA
    emits collectives across versions / lowering paths:

      1. `mesh_axes_replica_group_list` (field 93, Shardy / SDY) — synthesize
         the first group from the mesh axes the collective spans.
      2. `collective_device_list.replica_groups` (field 87, modern enumerated)
      3. `iota_collective_device_list` (field 92, Iota expansion)
      4. `replica_groups` (field 49, legacy, usually empty in modern HLO)

    Returns [] if no group info is present anywhere — this can happen for
    `call` wrappers around real collectives; resolve those via
    `cc.resolve_collective_via_call` before calling `_replica_ids`.
    """
    mesh = getattr(instr, "mesh_axes_replica_group_list", None)
    if mesh is not None and mesh.HasField("mesh") and mesh.mesh.axes and mesh.axes:
        return _replica_ids_from_mesh_axes(mesh)
    cdl = getattr(instr, "collective_device_list", None)
    if cdl is not None and cdl.replica_groups:
        return list(cdl.replica_groups[0].replica_ids)
    iota = getattr(instr, "iota_collective_device_list", None)
    if iota is not None and iota.num_replica_groups > 0 and iota.num_devices_per_group > 0:
        return list(range(iota.num_devices_per_group))
    if instr.replica_groups:
        return list(instr.replica_groups[0].replica_ids)
    return []


def _replica_ids_from_mesh_axes(mesh) -> list[int]:
    """Synthesize the first replica group from a MeshAxesReplicaGroupListProto.

    The collective contracts over the mesh axes listed in `mesh.axes` (each
    pointing at a `mesh.mesh.axes[idx]` of size `S_idx`). The first replica
    group is the set of devices that share the same coordinates on all OTHER
    mesh axes — i.e. the cartesian product of the contracted axes anchored
    at the origin of every other axis.

    `mesh.mesh.device_ids` is a row-major flattening of the full mesh; we
    return the first |group_size|-many devices that belong to the same group
    as device 0. If `device_ids` is unset, fall back to row-major default
    (id = sum of coord_i * stride_i).
    """
    sizes = [a.size for a in mesh.mesh.axes]
    contracted_idx = {ar.mesh_axis_index for ar in mesh.axes}
    # Group size = product of sizes of contracted axes (assuming no SubAxis).
    gs = 1
    for ai in contracted_idx:
        if 0 <= ai < len(sizes):
            gs *= sizes[ai]
    # Row-major strides
    n = len(sizes)
    strides = [1] * n
    for k in range(n - 2, -1, -1):
        strides[k] = strides[k + 1] * sizes[k + 1]

    # Walk the contracted axes (varying), keeping non-contracted at 0.
    contracted_ordered = sorted(contracted_idx)
    contracted_sizes = [sizes[ai] for ai in contracted_ordered]

    def _coords_iter():
        # Yield every combination of the contracted axes' coordinates.
        idx = [0] * len(contracted_ordered)
        while True:
            yield tuple(idx)
            # increment
            for k in range(len(idx) - 1, -1, -1):
                idx[k] += 1
                if idx[k] < contracted_sizes[k]:
                    break
                idx[k] = 0
            else:
                return

    device_ids = list(mesh.mesh.device_ids)
    has_device_ids = len(device_ids) > 0
    out: list[int] = []
    for combo in _coords_iter():
        coord = [0] * n
        for j, ai in enumerate(contracted_ordered):
            coord[ai] = combo[j]
        flat = sum(c * s for c, s in zip(coord, strides))
        if has_device_ids and 0 <= flat < len(device_ids):
            out.append(device_ids[flat])
        else:
            out.append(flat)
        if len(out) >= gs:
            break
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_KIND_BY_HLO_CATEGORY = {
    "all-reduce": "AllReduce",
    "all-gather": "AllGather",
    "reduce-scatter": "ReduceScatter",
    "all-to-all": "AllToAll",
    "collective-permute": "CollectivePermute",
    "send": "P2P",
    "recv": "P2P",
    "copy-start": "Copy",
    "copy-done": "Copy",
}

_OP_REGEX = re.compile(
    r"^(all-reduce|all-gather|reduce-scatter|all-to-all|"
    r"collective-permute|send|recv|copy)\b"
)


def classify(hlo_op: str | None, hlo_category: str | None) -> str:
    if hlo_category and hlo_category in _KIND_BY_HLO_CATEGORY:
        return _KIND_BY_HLO_CATEGORY[hlo_category]
    if hlo_op:
        m = _OP_REGEX.match(hlo_op)
        if m:
            return _KIND_BY_HLO_CATEGORY.get(m.group(1), "Unknown")
    return "Unknown"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _resolve_instr(hlo_instrs, by_comp_id, op_name):
    """Look up an HLO instruction by canonical name, then follow any `call`
    wrapper to the real collective. Returns the instruction to read replica
    info from, or None.
    """
    if not hlo_instrs:
        return None
    instr = hlo_instrs.get(op_name)
    if instr is None:
        return None
    if instr.opcode == "call" and by_comp_id is not None:
        resolved = cc.resolve_collective_via_call(instr, by_comp_id)
        if resolved is not None:
            return resolved
    return instr


def _row_from_async_pair(plane, start_ev, done_ev, hlo_instrs, by_comp_id,
                         merged_compute_by_core):
    """start_ev may be None (unpaired). done_ev is always set."""
    s_done = cc.event_stats(plane, done_ev)
    md_done = cc.event_metadata_stats(plane, done_ev)

    hlo_op_raw = s_done.get("hlo_op") or cc.event_name(plane, done_ev)
    op_name = cc.canonical_op_name(str(hlo_op_raw))
    instr = _resolve_instr(hlo_instrs, by_comp_id, op_name)

    if start_ev is not None:
        start_ps = start_ev.offset_ps
        end_ps = done_ev.offset_ps + done_ev.duration_ps
        wall_ps = end_ps - start_ps
        unpaired = False
    else:
        start_ps = done_ev.offset_ps
        end_ps = done_ev.offset_ps + done_ev.duration_ps
        wall_ps = done_ev.duration_ps
        unpaired = True
    stall_ps = s_done.get("device_duration_ps") or done_ev.duration_ps
    hidden_ps = max(0, int(wall_ps) - int(stall_ps))

    # Per-op critical-path exposure: time NOT covered by compute on the SAME
    # core kind. TC comm vs TC compute, SC comm vs SC compute — they don't
    # compete with each other.
    core = cc.core_kind(plane)
    merged = merged_compute_by_core.get(core, [])
    not_cov_ps = cc.not_covered_by_compute(start_ps, end_ps, merged)

    return _build_row(
        plane=plane, ev=done_ev, op_name=op_name, mode="async",
        wall_ps=int(wall_ps), stall_ps=int(stall_ps), hidden_ps=hidden_ps,
        not_cov_ps=int(not_cov_ps),
        ev_stats=s_done, md_stats=md_done, instr=instr,
        unpaired=unpaired, flow=s_done.get("flow"),
    )


def _row_from_sync(plane, ev, hlo_instrs, by_comp_id, merged_compute_by_core):
    s = cc.event_stats(plane, ev)
    md = cc.event_metadata_stats(plane, ev)
    hlo_op_raw = s.get("hlo_op") or cc.event_name(plane, ev)
    op_name = cc.canonical_op_name(str(hlo_op_raw))
    instr = _resolve_instr(hlo_instrs, by_comp_id, op_name)

    wall_ps = ev.duration_ps
    stall_ps = ev.duration_ps   # sync collectives are always exposed
    hidden_ps = 0

    # Sync collectives sit on the XLA Ops line of THIS core; their interval
    # is excluded from the merged compute set, so not_cov should equal wall
    # (modulo the rare case where another core's compute spans this window —
    # but cross-core overlap is meaningless for resource contention). We
    # still call the helper for consistency; it returns wall_ps when there's
    # no same-core compute overlap.
    core = cc.core_kind(plane)
    merged = merged_compute_by_core.get(core, [])
    start_ps = ev.offset_ps
    end_ps = ev.offset_ps + ev.duration_ps
    not_cov_ps = cc.not_covered_by_compute(start_ps, end_ps, merged)

    return _build_row(
        plane=plane, ev=ev, op_name=op_name, mode="sync",
        wall_ps=int(wall_ps), stall_ps=int(stall_ps), hidden_ps=hidden_ps,
        not_cov_ps=int(not_cov_ps),
        ev_stats=s, md_stats=md, instr=instr, unpaired=False, flow=None,
    )


def _build_row(*, plane, ev, op_name, mode, wall_ps, stall_ps, hidden_ps,
               not_cov_ps, ev_stats, md_stats, instr, unpaired, flow):
    hlo_op = ev_stats.get("hlo_op") or cc.event_name(plane, ev)
    kind = classify(hlo_op, md_stats.get("hlo_category"))
    bytes_ = md_stats.get("bytes_accessed") or md_stats.get("raw_bytes_accessed") or 0

    # Source: prefer XEventMetadata.stats.source / source_stack; fall back to HLO.
    source = md_stats.get("source") or md_stats.get("source_stack")
    if not source and instr is not None and instr.metadata.source_file:
        source = f"{instr.metadata.source_file}:{instr.metadata.source_line}"

    # Axis & group_size from HLO replica_groups (mesh-spec join is in axis_bandwidth.py).
    axis = "—"
    group_size = 0
    channel_id = None
    mesh_axis_indices: tuple[int, ...] = ()
    mesh_axis_sizes: tuple[int, ...] = ()
    if instr is not None:
        ids = _replica_ids(instr)
        group_size = len(ids)
        mesh = getattr(instr, "mesh_axes_replica_group_list", None)
        if mesh is not None and mesh.HasField("mesh") and mesh.mesh.axes and mesh.axes:
            # Modern Shardy form. The mesh is per-collective and Shardy may
            # rename / reorder axes between instructions, so a raw `axis_2`
            # from one collective is NOT the same logical axis as `axis_2`
            # from another. The reliable invariant is the (name, size) pair
            # of each contracted axis.
            ax_meta = list(mesh.mesh.axes)  # [(name, size), ...]
            all_idx = sorted({ar.mesh_axis_index for ar in mesh.axes})
            # Drop size-1 axes from the label and from the stored indices —
            # they're degenerate (a 1-way contraction is a no-op) and they
            # add visual noise to labels like axis_0=1024+axis_1=1.
            idx = tuple(
                i for i in all_idx
                if 0 <= i < len(ax_meta) and ax_meta[i].size > 1
            )
            if not idx and all_idx:
                # All contracted axes are size-1: this is effectively a
                # local op (group_size=1). Keep the original (non-empty)
                # tuple so debugging info is preserved.
                idx = tuple(all_idx)
            mesh_axis_indices = idx
            mesh_axis_sizes = tuple(
                ax_meta[i].size if 0 <= i < len(ax_meta) else 0 for i in idx
            )
            # First-pass label. Logical-name join happens in axis_bandwidth.py.
            parts = []
            for i in idx:
                if 0 <= i < len(ax_meta):
                    parts.append(f"{ax_meta[i].name}={ax_meta[i].size}")
                else:
                    parts.append(f"axis_{i}=?")
            axis = "+".join(parts) if parts else "—"
        elif group_size > 1:
            # Replica info came from the legacy collective_device_list /
            # iota / replica_groups path, not from a Shardy mesh. We have
            # no axis names — record group size so the user sees something
            # useful. axis_bandwidth.py can still do topology-coord
            # attribution if a mesh-spec is provided.
            axis = f"{group_size}-way"
        # channel_id is an int64 scalar in this vendored proto (not a message).
        if hasattr(instr, "channel_id") and instr.channel_id:
            channel_id = int(instr.channel_id)

    return {
        "op_name": op_name,
        "kind": kind,
        "mode": mode,
        "core": cc.core_kind(plane),
        "axis": axis,
        "group_size": group_size,
        "bidir": "?",   # filled later when axis_bandwidth or post-processing runs
        "bytes": int(bytes_) if bytes_ else 0,
        "wall_ps": wall_ps,
        "stall_ps": stall_ps,
        "hidden_ps": hidden_ps,
        "not_cov_ps": not_cov_ps,
        "source": source or "",
        "flow": int(flow) if flow is not None else None,
        "program_id": md_stats.get("program_id"),
        "channel_id": channel_id,
        "unpaired": unpaired,
        "mesh_axis_indices": mesh_axis_indices,
        "mesh_axis_sizes": mesh_axis_sizes,
    }


# ---------------------------------------------------------------------------
# Public entry point used by other scripts
# ---------------------------------------------------------------------------

def build_rows(profile_dir, *, include_copies=False) -> list[dict[str, Any]]:
    xs = cc.load_xspace(profile_dir)
    if xs is None:
        return []
    hlo_module = cc.load_hlo_module(profile_dir)
    hlo_instrs = cc.hlo_instructions(hlo_module) if hlo_module else {}
    by_comp_id = cc.computations_by_id(hlo_module) if hlo_module else {}

    # Per-core merged compute timeline. Used to compute per-row not_cov_ps —
    # the time a comm op was running while NO same-core compute was running.
    # TC and SC are kept separate because they don't compete for resources.
    merged_compute = cc.merged_compute_by_core(xs)

    rows: list[dict[str, Any]] = []
    for plane in cc.iter_device_planes(xs):
        async_ln = cc.async_xla_line(plane)
        if async_ln is not None:
            for s, d in cc.pair_async_events(plane, async_ln):
                rows.append(_row_from_async_pair(
                    plane, s, d, hlo_instrs, by_comp_id, merged_compute))
        xla_ln = cc.xla_ops_line(plane)
        if xla_ln is not None:
            for ev in xla_ln.events:
                s = cc.event_stats(plane, ev)
                md = cc.event_metadata_stats(plane, ev)
                hlo_op = s.get("hlo_op") or cc.event_name(plane, ev)
                kind = classify(hlo_op, md.get("hlo_category"))
                if kind in {"AllReduce", "AllGather", "ReduceScatter",
                            "AllToAll", "CollectivePermute", "P2P"}:
                    rows.append(_row_from_sync(
                        plane, ev, hlo_instrs, by_comp_id, merged_compute))
                # XLA Ops Copy events are uncommon and not collected here.

    if not include_copies:
        rows = [r for r in rows if r["kind"] != "Copy"]
    return rows


def unpaired_ratio(rows: list[dict[str, Any]]) -> float:
    """Fraction of async rows that are flow-singletons (unpaired).

    When this exceeds 50%, per-row stall_ps and hidden_ps are sentinel
    (stall ≈ wall, hidden = 0) and any sort/aggregation that uses them
    becomes meaningless. Callers should switch to not_cov_ps in that case.
    Sync rows are excluded from the denominator (they're never paired).
    """
    async_rows = [r for r in rows if r["mode"] == "async"]
    if not async_rows:
        return 0.0
    return sum(1 for r in async_rows if r["unpaired"]) / len(async_rows)


# ---------------------------------------------------------------------------
# Aggregation views
# ---------------------------------------------------------------------------

_SORT_FIELD = {
    "stall": "sum_stall_ps",
    "wall": "sum_wall_ps",
    "not_cov": "sum_not_cov_ps",
}


def _agg_by_kind(rows, sort_by="stall"):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["kind"], r["axis"], r["core"])].append(r)
    out = []
    for (kind, axis, core), grp in buckets.items():
        walls = sorted(r["wall_ps"] for r in grp)
        stalls = sorted(r["stall_ps"] for r in grp)
        out.append({
            "kind": kind, "axis": axis, "core": core,
            "count": len(grp),
            "sum_wall_ps": sum(walls),
            "sum_stall_ps": sum(stalls),
            "sum_not_cov_ps": sum(r["not_cov_ps"] for r in grp),
            "p50_stall_ps": stalls[len(stalls)//2],
            "p99_stall_ps": stalls[max(0, int(len(stalls)*0.99) - 1)],
        })
    out.sort(key=lambda r: -r[_SORT_FIELD[sort_by]])
    return out


def _agg_by_source(rows, sort_by="stall"):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["source"] or "(unknown)"].append(r)
    out = []
    for src, grp in buckets.items():
        kinds = defaultdict(int)
        for r in grp:
            kinds[r["kind"]] += 1
        dom_kind = max(kinds.items(), key=lambda kv: kv[1])[0]
        out.append({
            "source": src,
            "count": len(grp),
            "sum_wall_ps": sum(r["wall_ps"] for r in grp),
            "sum_stall_ps": sum(r["stall_ps"] for r in grp),
            "sum_not_cov_ps": sum(r["not_cov_ps"] for r in grp),
            "dom_kind": dom_kind,
        })
    out.sort(key=lambda r: -r[_SORT_FIELD[sort_by]])
    return out


def _agg_by_op(rows, sort_by="stall"):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["op_name"]].append(r)
    out = []
    for op, grp in buckets.items():
        out.append({
            "op_name": op,
            "kind": grp[0]["kind"], "axis": grp[0]["axis"], "core": grp[0]["core"],
            "count": len(grp),
            "sum_wall_ps": sum(r["wall_ps"] for r in grp),
            "sum_stall_ps": sum(r["stall_ps"] for r in grp),
            "sum_not_cov_ps": sum(r["not_cov_ps"] for r in grp),
        })
    out.sort(key=lambda r: -r[_SORT_FIELD[sort_by]])
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_us(ps): return f"{ps/1e6:.3f}" if ps else "0.000"


def _print_by_kind(agg, limit):
    print(f"{'kind':<20}{'axis':<24}{'core':<6}{'count':>7}"
          f"{'Σwall(us)':>13}{'Σstall(us)':>14}{'ΣNOT_cov(us)':>15}"
          f"{'p50_stall(us)':>16}{'p99_stall(us)':>16}")
    for row in agg[:limit]:
        print(f"{row['kind']:<20}{row['axis'][:23]:<24}{row['core']:<6}"
              f"{row['count']:>7}{_fmt_us(row['sum_wall_ps']):>13}"
              f"{_fmt_us(row['sum_stall_ps']):>14}"
              f"{_fmt_us(row['sum_not_cov_ps']):>15}"
              f"{_fmt_us(row['p50_stall_ps']):>16}"
              f"{_fmt_us(row['p99_stall_ps']):>16}")


def _print_by_source(agg, limit):
    print(f"{'source':<60}{'count':>7}{'Σwall(us)':>13}{'Σstall(us)':>14}"
          f"{'ΣNOT_cov(us)':>15}{'dom_kind':>20}")
    for row in agg[:limit]:
        print(f"{row['source'][:58]:<60}{row['count']:>7}"
              f"{_fmt_us(row['sum_wall_ps']):>13}{_fmt_us(row['sum_stall_ps']):>14}"
              f"{_fmt_us(row['sum_not_cov_ps']):>15}"
              f"{row['dom_kind']:>20}")


def _print_by_op(agg, limit):
    print(f"{'op_name':<50}{'kind':<18}{'axis':<24}{'core':<6}"
          f"{'count':>7}{'Σwall(us)':>13}{'Σstall(us)':>14}{'ΣNOT_cov(us)':>15}")
    for row in agg[:limit]:
        print(f"{row['op_name'][:48]:<50}{row['kind']:<18}{row['axis'][:23]:<24}"
              f"{row['core']:<6}{row['count']:>7}"
              f"{_fmt_us(row['sum_wall_ps']):>13}{_fmt_us(row['sum_stall_ps']):>14}"
              f"{_fmt_us(row['sum_not_cov_ps']):>15}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--by", choices=["kind", "source", "op"], default="kind")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument(
        "--sort-by", choices=["stall", "wall", "not_cov", "auto"],
        default="auto",
        help=("Sort key for aggregations. 'auto' (default) = stall when "
              "unpaired ratio ≤ 50%, otherwise not_cov."),
    )
    ap.add_argument("--include-copies", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = build_rows(args.profile_dir, include_copies=args.include_copies)
    if not rows:
        print(f"[absent] no usable comm events in {args.profile_dir}")
        return

    r = unpaired_ratio(rows)

    # Adaptive sort: when more than half of async events are flow-singletons,
    # stall_ps is a degenerate sentinel and any sort using it lies. Switch
    # to not_cov (per-op time NOT covered by same-core compute) and warn.
    if args.sort_by == "auto":
        sort_by = "not_cov" if r > 0.5 else "stall"
    else:
        sort_by = args.sort_by

    n_unpaired = sum(1 for r_ in rows if r_["unpaired"])
    print(f"comm primitives: {len(rows)} rows  (mode async/sync mix; "
          f"unpaired={n_unpaired}, unpaired_ratio={r:.0%})")
    print(f"sort_by={sort_by}" + (
        "  [auto: capture is unpaired-dominated; stall is sentinel — using NOT_cov]"
        if args.sort_by == "auto" and sort_by == "not_cov" else ""
    ))
    if r > 0.5 and sort_by == "stall":
        print(f"  [warn] {r:.0%} of async events are flow-singletons; stall_ps "
              f"is sentinel (≈ wall_ps). Consider --sort-by not_cov for true "
              f"critical-path exposure.")

    if args.by == "kind":
        _print_by_kind(_agg_by_kind(rows, sort_by=sort_by), args.limit)
    elif args.by == "source":
        _print_by_source(_agg_by_source(rows, sort_by=sort_by), args.limit)
    else:
        _print_by_op(_agg_by_op(rows, sort_by=sort_by), args.limit)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"rows": rows,
                       "unpaired_ratio": r,
                       "sort_by": sort_by,
                       "agg": {"by_kind": _agg_by_kind(rows, sort_by=sort_by),
                               "by_source": _agg_by_source(rows, sort_by=sort_by),
                               "by_op": _agg_by_op(rows, sort_by=sort_by)}},
                      f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
