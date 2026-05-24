"""
Per-axis bandwidth utilization for TPU comm primitives.

Joins per-event rows from list_comm_primitives.build_rows() with HLO
replica_groups and an optional mesh-spec YAML to attribute each collective
to a physical or logical mesh axis. Computes NCCL-style bus BW and
utilization vs peak ICI link BW.

Peak BW resolution order:
  1. xprof XStat (peak_ici_* / peak_link_*) via cc.peak_ici_link_gbps_from_xprof(xs)
  2. mesh_spec.peak_link_gbps
  3. --peak-ici-link-gbps flag
  4. None  ⇒ utilization column dropped, [warn] printed.

When `bidir=yes`, the displayed `util%` doubles the single-direction bus_BW
because both ICI directions carry traffic simultaneously.

---------------------------------------------------------------------------
KNOWN LIMITATION — cloned-wrapper join failure
---------------------------------------------------------------------------
On some captures, xprof events reference op names like `all-reduce.3008.cloned.1`,
which exist in HLO as opcode=`call` wrappers around the real collective rather
than the collective itself. The wrapper has no replica info, so axis stays "—"
and group_size stays 0 for those rows. Substring-matching back to the real
collective is too risky (could attribute to the wrong group), so we count the
unattributed rows and emit a single `[warn]` summary line instead.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc
import list_comm_primitives as lcp
from list_comm_primitives import _replica_ids


# ---------------------------------------------------------------------------
# Mesh-spec parsing
# ---------------------------------------------------------------------------

def load_mesh_spec(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        print("[warn] pyyaml not installed; --mesh-spec ignored", file=sys.stderr)
        return {}
    return yaml.safe_load(pathlib.Path(path).read_text()) or {}


# ---------------------------------------------------------------------------
# Axis attribution: replica_ids -> varying physical dims
# ---------------------------------------------------------------------------

def _coords(replica_id: int, topology: tuple[int, int, int]) -> tuple[int, int, int]:
    X, Y, Z = topology
    x = replica_id // (Y * Z)
    y = (replica_id // Z) % Y
    z = replica_id % Z
    return (x, y, z)


def attribute_axis(replica_ids: list[int],
                   topology: tuple[int, int, int],
                   logical_axes: dict | None) -> tuple[str, int]:
    """
    Returns (axis_label, group_size).

    axis_label is one of: "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ", or a
    logical name from the mesh spec if matched. Falls back to
    "stride-N group" if topology is unknown.
    """
    if not replica_ids:
        return ("—", 0)
    if topology == (0, 0, 0):
        return (f"stride-{len(replica_ids)} group", len(replica_ids))

    coords = [_coords(r, topology) for r in replica_ids]
    varies = []
    for dim_idx, dim_name in enumerate("XYZ"):
        vals = {c[dim_idx] for c in coords}
        if len(vals) > 1:
            varies.append(dim_name)
    physical = "".join(varies) if varies else "—"

    if logical_axes:
        for logical_name, info in logical_axes.items():
            dims = set(info.get("dims") or [])
            if dims and dims == set(varies):
                return (logical_name, len(replica_ids))

    return (physical, len(replica_ids))


# ---------------------------------------------------------------------------
# Bidirectional dual-issue heuristic
# ---------------------------------------------------------------------------

def _shape_key(shape) -> bytes:
    return shape.SerializeToString() if shape else b""


def _sharding_key(instr) -> bytes:
    # `sharding` is a message type → HasField works (returns False on default).
    return instr.sharding.SerializeToString() if instr.HasField("sharding") else b""


def _replica_groups_key(instr) -> bytes:
    """Hash the (first) replica group of `instr`. We use _replica_ids so we
    pick up modern HLO's collective_device_list / iota_collective_device_list,
    not just the legacy field-49 replica_groups (which is empty)."""
    h = hashlib.blake2b(digest_size=16)
    ids = _replica_ids(instr)
    h.update(b"|" + b",".join(str(i).encode() for i in ids))
    return h.digest()


def bidir_clusters(hlo_module) -> dict[str, bool]:
    """
    Returns {canonical_op_name: bidir}. bidir=True if the cluster of
    instructions sharing (opcode, shape, replica_groups, sharding) has
    >=2 distinct channel_ids.
    """
    if hlo_module is None:
        return {}
    by_cluster: dict[tuple, list[tuple[str, int | None]]] = collections.defaultdict(list)
    for c in hlo_module.computations:
        for i in c.instructions:
            if i.opcode not in {"all-reduce", "all-gather", "reduce-scatter",
                                "all-to-all", "collective-permute", "send", "recv"}:
                continue
            cluster_key = (i.opcode,
                           _shape_key(i.shape),
                           _replica_groups_key(i),
                           _sharding_key(i))
            # channel_id is int64 scalar (no HasField); 0 means "unset".
            ch = int(i.channel_id) if i.channel_id else None
            by_cluster[cluster_key].append((cc.canonical_op_name(i.name), ch))

    out: dict[str, bool] = {}
    for members in by_cluster.values():
        chs = {m[1] for m in members if m[1] is not None}
        bidir = len(chs) >= 2
        for op_name, _ in members:
            out[op_name] = bidir
    return out


# ---------------------------------------------------------------------------
# Bus BW formulas
# ---------------------------------------------------------------------------

def bus_bw_gbps(kind: str, group_size: int, bytes_: int, time_ps: int) -> Optional[float]:
    if not (group_size > 0 and bytes_ > 0 and time_ps > 0):
        return None
    secs = time_ps / 1e12
    factor = {
        "AllReduce":      lambda N: 2.0 * (N - 1) / N,
        "AllGather":      lambda N: (N - 1) / N,
        "ReduceScatter":  lambda N: (N - 1) / N,
        "AllToAll":       lambda N: (N - 1) / N,
        "CollectivePermute": lambda N: 1.0,
        "P2P":            lambda N: 1.0,
    }.get(kind)
    if factor is None:
        return None
    return factor(group_size) * (bytes_ / 1e9) / secs


# ---------------------------------------------------------------------------
# Peak BW resolution
# ---------------------------------------------------------------------------

def resolve_peak_link_gbps(profile_dir, mesh_spec, cli_flag) -> tuple[Optional[float], str]:
    """Returns (peak_gbps, source_label).

    Order: xprof XStat → mesh-spec → CLI flag → None.
    (op_stats has no ICI entry — see _comm_common banner.)
    """
    xs = cc.load_xspace(profile_dir)
    if xs is not None:
        v = cc.peak_ici_link_gbps_from_xprof(xs)
        if v is not None:
            return (v, "xprof")
    if mesh_spec.get("peak_link_gbps"):
        return (float(mesh_spec["peak_link_gbps"]), "mesh-spec")
    if cli_flag is not None:
        return (float(cli_flag), "cli flag")
    return (None, "unknown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_COLLECTIVE_KINDS = {"AllReduce", "AllGather", "ReduceScatter",
                     "AllToAll", "CollectivePermute", "P2P"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--mesh-spec", default=None)
    ap.add_argument("--peak-ici-link-gbps", type=float, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = lcp.build_rows(args.profile_dir, include_copies=False)
    if not rows:
        print(f"[absent] no usable comm events in {args.profile_dir}")
        return

    mesh_spec = load_mesh_spec(args.mesh_spec)
    topology = tuple(mesh_spec.get("topology") or (0, 0, 0))
    logical_axes = mesh_spec.get("axes") or {}
    links_per_axis = int(mesh_spec.get("links_per_axis", 2))

    hlo_module = cc.load_hlo_module(args.profile_dir)
    hlo_instrs = cc.hlo_instructions(hlo_module) if hlo_module else {}
    bidir_map = bidir_clusters(hlo_module)

    # Annotate rows in place: axis (with mesh-spec join) and bidir.
    unattributed_cloned = 0   # HLO entry exists but has no replica info
    unattributed_no_hlo = 0   # No HLO entry at all
    for r in rows:
        instr = hlo_instrs.get(r["op_name"])
        if instr is not None:
            replica_ids = _replica_ids(instr)
            if replica_ids:
                axis, gs = attribute_axis(replica_ids, topology, logical_axes)
                r["axis"] = axis
                if r["group_size"] == 0:
                    r["group_size"] = gs
            elif r["kind"] in _COLLECTIVE_KINDS:
                # KNOWN LIMITATION: cloned-wrapper join failure.
                # The HLO entry exists but has no replica info (likely a
                # `call` wrapper around the actual collective).
                unattributed_cloned += 1
        elif r["kind"] in _COLLECTIVE_KINDS:
            # No HLO entry at all for this collective op_name.
            unattributed_no_hlo += 1
        r["bidir"] = "yes" if bidir_map.get(r["op_name"]) else "no"
        r["bus_bw_gbps"] = bus_bw_gbps(r["kind"], r["group_size"],
                                       r["bytes"], r["wall_ps"])
        r["effective_bus_bw_gbps"] = (r["bus_bw_gbps"] * 2.0
                                      if r["bus_bw_gbps"] and r["bidir"] == "yes"
                                      else r["bus_bw_gbps"])

    peak_link, peak_src = resolve_peak_link_gbps(args.profile_dir, mesh_spec,
                                                  args.peak_ici_link_gbps)
    peak_axis_gbps = peak_link * links_per_axis if peak_link is not None else None
    print(f"peak ICI link: "
          f"{f'{peak_link:.1f} GB/s' if peak_link else '?'}  ({peak_src})  "
          f"links_per_axis={links_per_axis}  "
          f"⇒ peak_axis={f'{peak_axis_gbps:.1f}' if peak_axis_gbps else '?'} GB/s")
    if peak_link is None:
        print("[warn] peak ICI BW unknown — utilization omitted")

    # Per-axis aggregate
    by_axis = collections.defaultdict(list)
    for r in rows:
        by_axis[(r["axis"], r["core"])].append(r)

    print(f"\n{'axis':<14}{'core':<6}{'count':>7}"
          f"{'Σbytes(MB)':>14}{'Σwall(us)':>13}{'bus_BW(GB/s)':>16}"
          f"{'util%':>8}")
    for (axis, core), grp in sorted(by_axis.items(),
                                    key=lambda kv: -sum(r['wall_ps'] for r in kv[1]))[:args.limit]:
        sb = sum(r["bytes"] for r in grp)
        sw = sum(r["wall_ps"] for r in grp)
        bw = None
        # Dominant kind in this bucket, used for the formula.
        if sw > 0 and sb > 0:
            kinds = collections.Counter(r["kind"] for r in grp)
            dom_kind = kinds.most_common(1)[0][0]
            dom_gs = max((r["group_size"] for r in grp), default=0)
            bw = bus_bw_gbps(dom_kind, dom_gs, sb, sw)
        util = (bw / peak_axis_gbps * 100.0) if (bw and peak_axis_gbps) else None
        print(f"{axis:<14}{core:<6}{len(grp):>7}"
              f"{sb/1e6:>14.2f}{sw/1e6:>13.3f}"
              f"{(f'{bw:.2f}' if bw else '—'):>16}"
              f"{(f'{util:.1f}' if util is not None else '—'):>8}")

    # Top-N per-collective table
    print(f"\nTop-{args.limit} per-collective by Σstall:")
    print(f"{'op_name':<48}{'kind':<16}{'axis':<10}{'core':<6}"
          f"{'bidir':<6}{'wall(us)':>11}{'stall(us)':>11}{'bus_BW(GB/s)':>14}{'util%':>7}")
    for r in sorted(rows, key=lambda r: -r["stall_ps"])[:args.limit]:
        bw = r.get("bus_bw_gbps")
        eff_bw = r.get("effective_bus_bw_gbps")
        util = (eff_bw / peak_axis_gbps * 100.0) if (eff_bw and peak_axis_gbps) else None
        print(f"{r['op_name'][:46]:<48}{r['kind']:<16}{r['axis']:<10}{r['core']:<6}"
              f"{r['bidir']:<6}{r['wall_ps']/1e6:>11.3f}{r['stall_ps']/1e6:>11.3f}"
              f"{(f'{bw:.2f}' if bw else '—'):>14}"
              f"{(f'{util:.1f}' if util is not None else '—'):>7}")

    if unattributed_cloned:
        print(f"\n[warn] {unattributed_cloned} collective rows could not be "
              f"axis-attributed (cloned-wrapper join failure — see banner)")
    if unattributed_no_hlo:
        print(f"\n[warn] {unattributed_no_hlo} collective rows have no HLO "
              f"counterpart (HLO module missing or op renamed)")

    if args.json_out:
        out = {"peak_link_gbps": peak_link, "peak_src": peak_src,
               "links_per_axis": links_per_axis, "rows": rows}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
