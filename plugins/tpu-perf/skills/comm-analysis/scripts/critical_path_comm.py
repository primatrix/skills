"""
Per-op critical-path attribution for comm primitives in a TPU profile.

Usage:
    python3 critical_path_comm.py <profile_dir> [--top-n 20] [--json out.json]
                                  [--core TC|SC0|SC1|all]

Why this script exists:
  On unpaired-dominated captures (every Async XLA Op is a flow-singleton),
  per-row stall_ps and hidden_ps are SENTINEL values: stall ≈ wall, hidden = 0.
  Any "Top-N exposed contributors" table sorted by stall in that regime is
  measuring engine-busy time, NOT critical-path exposure.

  The only authoritative per-op metric is `NOT_cov_by_compute`: time the
  comm op was running while NO same-core compute was running. That's what
  shows up on the critical path.

  TC and SC are kept STRICTLY separate. TC comm primitives are checked
  against TC compute intervals; SC comm primitives against SC compute. They
  do not compete for resources, so mixing them inflates "overlap" on both
  sides.

What this script reports (per requested core):

  Table 1 — Top-N comm ops by Σ NOT_cov_by_compute
    Which ops contribute most to the critical path. Use this to pick
    optimization targets. Each row also shows wall, stall, cov_pct.

  Table 2 — Top-N coverage gaps (1 − cov%)
    Worst-overlapped ops by RATIO. Useful for finding small ops that are
    fully exposed (wall ≈ NOT_cov), even if their absolute contribution is
    moderate. Filtered to ops with wall ≥ 1% of the largest wall to avoid
    surfacing trivial events.

  Table 3 — Top-N source lines by Σ NOT_cov
    Lines (file:line) ranked by the sum of NOT_cov across all collectives
    they emit. This is the table to act on when deciding where to insert
    `with_sharding_constraint` / split a bundled op / restructure a layer.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc
import list_comm_primitives as lcp


def _summarize_for_core(rows: list[dict], core: str) -> dict:
    """Compute the three top-N tables for a single core kind."""
    core_rows = [r for r in rows if r["core"] == core]
    if not core_rows:
        return {"core": core, "absent": True}

    n_async = sum(1 for r in core_rows if r["mode"] == "async")
    n_unpaired = sum(1 for r in core_rows if r["unpaired"])
    unp_ratio = (n_unpaired / n_async) if n_async else 0.0

    # Concurrency: Σwall / union_wall over async comm intervals on this core.
    # > 1.2 ⇒ multiple ICI links ran in parallel and per-row stall is per-link
    # engine-busy time, NOT exposed time. NOT_cov is still authoritative.
    intervals = [
        (int(r["start_ps"]), int(r["start_ps"]) + int(r["wall_ps"]))
        for r in core_rows if r["mode"] == "async" and "start_ps" in r
    ]
    concurrency, _, _ = cc.comm_concurrency(intervals) if intervals else (1.0, 0, 0)

    # ---- Table 1: per op_name, Σ not_cov ----
    by_op = defaultdict(list)
    for r in core_rows:
        by_op[r["op_name"]].append(r)
    top_by_not_cov = []
    for op, grp in by_op.items():
        sum_wall = sum(r["wall_ps"] for r in grp)
        sum_stall = sum(r["stall_ps"] for r in grp)
        sum_not_cov = sum(r["not_cov_ps"] for r in grp)
        cov_pct = (1 - sum_not_cov / sum_wall) * 100 if sum_wall else 0.0
        top_by_not_cov.append({
            "op_name": op, "kind": grp[0]["kind"], "axis": grp[0]["axis"],
            "count": len(grp),
            "sum_wall_ps": sum_wall, "sum_stall_ps": sum_stall,
            "sum_not_cov_ps": sum_not_cov, "cov_pct": cov_pct,
        })
    top_by_not_cov.sort(key=lambda r: -r["sum_not_cov_ps"])

    # ---- Table 2: coverage gap (worst-overlapped). Filter trivial ops. ----
    if top_by_not_cov:
        max_wall = max(r["sum_wall_ps"] for r in top_by_not_cov)
        threshold = max(1, max_wall // 100)  # 1% of largest wall
    else:
        threshold = 0
    coverage_gap = [r for r in top_by_not_cov if r["sum_wall_ps"] >= threshold]
    coverage_gap = sorted(coverage_gap, key=lambda r: -(100 - r["cov_pct"]))

    # ---- Table 3: per source line, Σ not_cov ----
    by_src = defaultdict(list)
    for r in core_rows:
        by_src[r["source"] or "(unknown)"].append(r)
    top_by_source = []
    for src, grp in by_src.items():
        sum_wall = sum(r["wall_ps"] for r in grp)
        sum_not_cov = sum(r["not_cov_ps"] for r in grp)
        cov_pct = (1 - sum_not_cov / sum_wall) * 100 if sum_wall else 0.0
        kinds = defaultdict(int)
        for r in grp:
            kinds[r["kind"]] += 1
        dom_kind = max(kinds.items(), key=lambda kv: kv[1])[0]
        top_by_source.append({
            "source": src, "count": len(grp), "dom_kind": dom_kind,
            "sum_wall_ps": sum_wall, "sum_not_cov_ps": sum_not_cov,
            "cov_pct": cov_pct,
        })
    top_by_source.sort(key=lambda r: -r["sum_not_cov_ps"])

    return {
        "core": core,
        "absent": False,
        "n_rows": len(core_rows),
        "n_async": n_async,
        "n_unpaired": n_unpaired,
        "unpaired_ratio": unp_ratio,
        "comm_concurrency": concurrency,
        "top_by_not_cov": top_by_not_cov,
        "coverage_gap": coverage_gap,
        "top_by_source": top_by_source,
    }


def _us(ps: int) -> str:
    return f"{ps/1e6:.3f}"


def _print_summary(s: dict, *, top_n: int):
    if s["absent"]:
        print(f"\n=== {s['core']}: no comm primitives in this capture ===")
        return

    print(f"\n=== Critical path: {s['core']} "
          f"({s['n_rows']} comm rows, {s['n_async']} async, "
          f"unpaired_ratio={s['unpaired_ratio']:.0%}, "
          f"comm_concurrency={s.get('comm_concurrency', 1.0):.2f}) ===")
    if s["unpaired_ratio"] > 0.5:
        print(f"  [info] capture is unpaired-dominated. NOT_cov is the only "
              f"authoritative per-op exposure metric here.")
    elif s.get("comm_concurrency", 1.0) > 1.2:
        print(f"  [info] comm concurrency = {s['comm_concurrency']:.2f} "
              f"(multiple ICI links in parallel). Σstall is non-additive vs "
              f"wall-clock; NOT_cov is the authoritative critical-path metric.")

    # ---- Table 1 ----
    print(f"\n  -- Table 1: Top-{top_n} ops by Σ NOT_cov_by_compute "
          f"(critical-path contribution) --")
    print(f"  {'op_name':<46}{'kind':<14}{'axis':<22}{'count':>7}"
          f"{'Σwall(us)':>14}{'ΣNOT_cov(us)':>14}{'cov%':>7}")
    for r in s["top_by_not_cov"][:top_n]:
        print(f"  {r['op_name'][:44]:<46}{r['kind']:<14}{r['axis'][:21]:<22}"
              f"{r['count']:>7}{_us(r['sum_wall_ps']):>14}"
              f"{_us(r['sum_not_cov_ps']):>14}{r['cov_pct']:>6.1f}%")

    # ---- Table 2 ----
    print(f"\n  -- Table 2: Top-{top_n} coverage gaps (1−cov%, "
          f"filtered to wall ≥ 1% of max) --")
    print(f"  {'op_name':<46}{'kind':<14}{'count':>7}{'Σwall(us)':>14}"
          f"{'ΣNOT_cov(us)':>14}{'cov%':>7}")
    for r in s["coverage_gap"][:top_n]:
        print(f"  {r['op_name'][:44]:<46}{r['kind']:<14}"
              f"{r['count']:>7}{_us(r['sum_wall_ps']):>14}"
              f"{_us(r['sum_not_cov_ps']):>14}{r['cov_pct']:>6.1f}%")

    # ---- Table 3 ----
    print(f"\n  -- Table 3: Top-{top_n} source lines by Σ NOT_cov --")
    print(f"  {'source':<60}{'dom_kind':<14}{'count':>7}"
          f"{'Σwall(us)':>14}{'ΣNOT_cov(us)':>14}{'cov%':>7}")
    for r in s["top_by_source"][:top_n]:
        print(f"  {r['source'][:58]:<60}{r['dom_kind']:<14}"
              f"{r['count']:>7}{_us(r['sum_wall_ps']):>14}"
              f"{_us(r['sum_not_cov_ps']):>14}{r['cov_pct']:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--core", choices=["TC", "SC0", "SC1", "all"], default="all",
                    help="Restrict report to a single core kind. "
                         "Default 'all' prints TC, then any SC core that "
                         "actually carries comm primitives.")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = lcp.build_rows(args.profile_dir)
    if not rows:
        print(f"[absent] no usable comm events in {args.profile_dir}")
        return

    if args.core == "all":
        cores_present = sorted({r["core"] for r in rows},
                               key=lambda c: (c != "TC", c))
    else:
        cores_present = [args.core]

    summaries = []
    for core in cores_present:
        s = _summarize_for_core(rows, core)
        summaries.append(s)
        _print_summary(s, top_n=args.top_n)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summaries": summaries}, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
