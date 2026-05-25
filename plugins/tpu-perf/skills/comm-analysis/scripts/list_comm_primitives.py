"""
List every communication primitive in a TPU profile, with rich attributes.

Usage:
    python3 list_comm_primitives.py <profile_dir> [--by kind|source|op] [--limit N]
                                    [--include-copies] [--json out.json]

Reads the device planes (TC, SC0, SC1) of *.xplane.pb. Pairs Async XLA Ops
events by 'flow' stat. Adds sync collectives from XLA Ops. HLO join (axis,
group_size, channel_id) is optional — happens automatically if a
*.hlo_proto.pb is present.

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
# Replica-group extraction (walks all three proto locations)
# ---------------------------------------------------------------------------

def _replica_ids(instr) -> list[int]:
    """Extract replica_ids of the first replica group, walking the three
    proto locations. Modern HLO empties the legacy `replica_groups` field
    (49) and stores replica IDs in `collective_device_list.replica_groups`
    (87) or `iota_collective_device_list` (92, an Iota expansion).
    Returns [] if no group info is present.
    """
    if instr.replica_groups:
        return list(instr.replica_groups[0].replica_ids)
    cdl = getattr(instr, "collective_device_list", None)
    if cdl is not None and cdl.replica_groups:
        return list(cdl.replica_groups[0].replica_ids)
    iota = getattr(instr, "iota_collective_device_list", None)
    if iota is not None and iota.num_replica_groups > 0 and iota.num_devices_per_group > 0:
        # Synthesize the first group: ids 0..num_devices_per_group-1
        return list(range(iota.num_devices_per_group))
    return []


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

def _row_from_async_pair(plane, start_ev, done_ev, hlo_instrs):
    """start_ev may be None (unpaired). done_ev is always set."""
    s_done = cc.event_stats(plane, done_ev)
    md_done = cc.event_metadata_stats(plane, done_ev)

    hlo_op_raw = s_done.get("hlo_op") or cc.event_name(plane, done_ev)
    op_name = cc.canonical_op_name(str(hlo_op_raw))
    instr = hlo_instrs.get(op_name) if hlo_instrs else None

    if start_ev is not None:
        wall_ps = (done_ev.offset_ps + done_ev.duration_ps) - start_ev.offset_ps
        unpaired = False
    else:
        wall_ps = done_ev.duration_ps
        unpaired = True
    stall_ps = s_done.get("device_duration_ps") or done_ev.duration_ps
    hidden_ps = max(0, int(wall_ps) - int(stall_ps))

    return _build_row(
        plane=plane, ev=done_ev, op_name=op_name, mode="async",
        wall_ps=int(wall_ps), stall_ps=int(stall_ps), hidden_ps=hidden_ps,
        ev_stats=s_done, md_stats=md_done, instr=instr,
        unpaired=unpaired, flow=s_done.get("flow"),
    )


def _row_from_sync(plane, ev, hlo_instrs):
    s = cc.event_stats(plane, ev)
    md = cc.event_metadata_stats(plane, ev)
    hlo_op_raw = s.get("hlo_op") or cc.event_name(plane, ev)
    op_name = cc.canonical_op_name(str(hlo_op_raw))
    instr = hlo_instrs.get(op_name) if hlo_instrs else None

    wall_ps = ev.duration_ps
    stall_ps = ev.duration_ps   # sync collectives are always exposed
    hidden_ps = 0

    return _build_row(
        plane=plane, ev=ev, op_name=op_name, mode="sync",
        wall_ps=int(wall_ps), stall_ps=int(stall_ps), hidden_ps=hidden_ps,
        ev_stats=s, md_stats=md, instr=instr, unpaired=False, flow=None,
    )


def _build_row(*, plane, ev, op_name, mode, wall_ps, stall_ps, hidden_ps,
               ev_stats, md_stats, instr, unpaired, flow):
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
    if instr is not None:
        ids = _replica_ids(instr)
        group_size = len(ids)
        # axis stays "—" here; full attribution happens in axis_bandwidth.
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
        "source": source or "",
        "flow": int(flow) if flow is not None else None,
        "program_id": md_stats.get("program_id"),
        "channel_id": channel_id,
        "unpaired": unpaired,
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

    rows: list[dict[str, Any]] = []
    for plane in cc.iter_device_planes(xs):
        async_ln = cc.async_xla_line(plane)
        if async_ln is not None:
            for s, d in cc.pair_async_events(plane, async_ln):
                rows.append(_row_from_async_pair(plane, s, d, hlo_instrs))
        xla_ln = cc.xla_ops_line(plane)
        if xla_ln is not None:
            for ev in xla_ln.events:
                s = cc.event_stats(plane, ev)
                md = cc.event_metadata_stats(plane, ev)
                hlo_op = s.get("hlo_op") or cc.event_name(plane, ev)
                kind = classify(hlo_op, md.get("hlo_category"))
                if kind in {"AllReduce", "AllGather", "ReduceScatter",
                            "AllToAll", "CollectivePermute", "P2P"}:
                    rows.append(_row_from_sync(plane, ev, hlo_instrs))
                # XLA Ops Copy events are uncommon and not collected here.

    if not include_copies:
        rows = [r for r in rows if r["kind"] != "Copy"]
    return rows


# ---------------------------------------------------------------------------
# Aggregation views
# ---------------------------------------------------------------------------

def _agg_by_kind(rows):
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
            "p50_stall_ps": stalls[len(stalls)//2],
            "p99_stall_ps": stalls[max(0, int(len(stalls)*0.99) - 1)],
        })
    out.sort(key=lambda r: -r["sum_stall_ps"])
    return out


def _agg_by_source(rows):
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
            "dom_kind": dom_kind,
        })
    out.sort(key=lambda r: -r["sum_stall_ps"])
    return out


def _agg_by_op(rows):
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
        })
    out.sort(key=lambda r: -r["sum_stall_ps"])
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_us(ps): return f"{ps/1e6:.3f}" if ps else "0.000"


def _print_by_kind(agg, limit):
    print(f"{'kind':<20}{'axis':<10}{'core':<6}{'count':>7}"
          f"{'Σwall(us)':>13}{'Σstall(us)':>14}{'p50_stall(us)':>16}{'p99_stall(us)':>16}")
    for row in agg[:limit]:
        print(f"{row['kind']:<20}{row['axis']:<10}{row['core']:<6}"
              f"{row['count']:>7}{_fmt_us(row['sum_wall_ps']):>13}"
              f"{_fmt_us(row['sum_stall_ps']):>14}"
              f"{_fmt_us(row['p50_stall_ps']):>16}"
              f"{_fmt_us(row['p99_stall_ps']):>16}")


def _print_by_source(agg, limit):
    print(f"{'source':<60}{'count':>7}{'Σwall(us)':>13}{'Σstall(us)':>14}{'dom_kind':>20}")
    for row in agg[:limit]:
        print(f"{row['source'][:58]:<60}{row['count']:>7}"
              f"{_fmt_us(row['sum_wall_ps']):>13}{_fmt_us(row['sum_stall_ps']):>14}"
              f"{row['dom_kind']:>20}")


def _print_by_op(agg, limit):
    print(f"{'op_name':<50}{'kind':<18}{'axis':<10}{'core':<6}"
          f"{'count':>7}{'Σwall(us)':>13}{'Σstall(us)':>14}")
    for row in agg[:limit]:
        print(f"{row['op_name'][:48]:<50}{row['kind']:<18}{row['axis']:<10}"
              f"{row['core']:<6}{row['count']:>7}"
              f"{_fmt_us(row['sum_wall_ps']):>13}{_fmt_us(row['sum_stall_ps']):>14}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--by", choices=["kind", "source", "op"], default="kind")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--include-copies", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = build_rows(args.profile_dir, include_copies=args.include_copies)
    if not rows:
        print(f"[absent] no usable comm events in {args.profile_dir}")
        return

    print(f"comm primitives: {len(rows)} rows  (mode async/sync mix; "
          f"unpaired={sum(1 for r in rows if r['unpaired'])})")

    if args.by == "kind":
        _print_by_kind(_agg_by_kind(rows), args.limit)
    elif args.by == "source":
        _print_by_source(_agg_by_source(rows), args.limit)
    else:
        _print_by_op(_agg_by_op(rows), args.limit)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"rows": rows,
                       "agg": {"by_kind": _agg_by_kind(rows),
                               "by_source": _agg_by_source(rows),
                               "by_op": _agg_by_op(rows)}}, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
