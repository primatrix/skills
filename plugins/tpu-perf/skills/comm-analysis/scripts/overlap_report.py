"""
Compute/comm overlap report for a TPU profile.

Per step on the device plane, computes:
  - compute_busy_ps   = ∪(compute intervals)
  - comm_inflight_ps  = ∪(comm intervals)
  - overlapped_ps     = ∪(compute ∩ comm)
  - exposed_comm_ps   = comm_inflight - overlapped
  - overlap_ratio     = overlapped / comm_inflight

Sweep-line union math; intervals are clipped to the step window.

Compute intervals come from the "XLA Ops" line, EXCLUDING:
  1. comm categories (collectives — already accounted for in comm)
  2. wrapper / control-flow categories (`while`, `call`, `conditional`,
     `async-start/done`, `copy-start/done`, `*-done`).

The wrapper exclusion is non-obvious but critical: in MaxText / `jax.lax`
captures, the outer `while_loop` appears on XLA Ops as a single event whose
duration spans the entire body (often = the whole step). `async-done`
events likewise carry the full async wall as their `duration_ps`. Counting
either as "compute" makes compute_busy ≈ step_time, and every comm interval
falls inside it ⇒ fake 100% overlap, ~0us exposed. See `_WRAPPER_HLO_CATEGORIES`.

Sanity-check: the sweep-derived exposed_comm vs Σ done.device_duration_ps
within the step (the metadata-reported exposed time). Mismatch above 5%
prints `[warn] step N`. Sweep is authoritative.

After the table an `[info]` line reports which wrapper categories were
excluded and how much time they covered, so a reader can sanity-check the
fix on their own capture.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc


_COMM_HLO_CATEGORIES = {
    "all-reduce", "all-gather", "reduce-scatter",
    "all-to-all", "collective-permute", "send", "recv",
}

# Categories that appear on the "XLA Ops" line but do NOT represent real TC
# compute. Excluded from compute_intervals so they don't fake-overlap comm.
#
# - async-start / async-done: collective wrappers; async-done's duration is the
#   full async wall (mirror of Async XLA Ops). Counting it as compute would
#   make every async collective look fully hidden.
# - while / call / conditional: control-flow CONTAINERS whose duration spans
#   their entire body. In MaxText captures the outer `while_loop` is one event
#   covering the whole training step (~hundreds of seconds), which by itself
#   makes compute_busy ≈ step_time and forces overlap_ratio → 1.0.
# - copy-start / copy-done / send-done / recv-done /
#   collective-permute-start / collective-permute-done: DMA / collective
#   completion wrappers, not compute.
_WRAPPER_HLO_CATEGORIES = {
    "while", "call", "conditional",
    "async-start", "async-done",
    "copy-start", "copy-done",
    "send-done", "recv-done",
    "collective-permute-start", "collective-permute-done",
}

_NON_COMPUTE_HLO_CATEGORIES = _COMM_HLO_CATEGORIES | _WRAPPER_HLO_CATEGORIES


# ---------------------------------------------------------------------------
# Sweep-line interval union / intersection
# ---------------------------------------------------------------------------

def _clip(intervals: Iterable[tuple[int, int]],
          window: tuple[int, int]) -> list[tuple[int, int]]:
    lo, hi = window
    out = []
    for a, b in intervals:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2:
            out.append((a2, b2))
    return out


def union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += cur_b - cur_a
    return total


def intersection_intervals(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return intervals that are in BOTH a's union and b's union."""
    if not a or not b:
        return []
    a = sorted(a); b = sorted(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


# ---------------------------------------------------------------------------
# Per-plane interval extraction
# ---------------------------------------------------------------------------

def _compute_intervals(plane) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """
    Returns (intervals, dropped_by_category).

    `dropped_by_category` is the per-category Σduration of XLA Ops events
    that were NOT counted as compute (comm + wrappers). Reported alongside
    the overlap table as a sanity check — if a wrapper category dominates,
    the user can see what was excluded.
    """
    ln = cc.xla_ops_line(plane)
    if ln is None:
        return [], {}
    out: list[tuple[int, int]] = []
    dropped: dict[str, int] = {}
    for ev in ln.events:
        md = cc.event_metadata_stats(plane, ev)
        cat = md.get("hlo_category")
        if cat in _NON_COMPUTE_HLO_CATEGORIES:
            dropped[cat] = dropped.get(cat, 0) + ev.duration_ps
            continue
        out.append((ev.offset_ps, ev.offset_ps + ev.duration_ps))
    return out, dropped


def _comm_intervals(plane) -> list[tuple[int, int]]:
    """
    Returns the list of communication intervals on this plane.

    Combines async collective pairs (start..done) and sync collectives
    identified via the XLA-ops line's hlo_category metadata.
    """
    intervals: list[tuple[int, int]] = []

    async_ln = cc.async_xla_line(plane)
    if async_ln is not None:
        for s, d in cc.pair_async_events(plane, async_ln):
            if s is not None:
                intervals.append((s.offset_ps, d.offset_ps + d.duration_ps))
            else:
                # Unpaired: treat the done-event's window as the interval.
                intervals.append((d.offset_ps, d.offset_ps + d.duration_ps))

    xla_ln = cc.xla_ops_line(plane)
    if xla_ln is not None:
        for ev in xla_ln.events:
            md = cc.event_metadata_stats(plane, ev)
            if md.get("hlo_category") in _COMM_HLO_CATEGORIES:
                intervals.append((ev.offset_ps, ev.offset_ps + ev.duration_ps))

    return intervals


# ---------------------------------------------------------------------------
# Per-step report
# ---------------------------------------------------------------------------

def _step_windows(plane) -> list[tuple[int, tuple[int, int]]]:
    """Returns [(step_id_or_index, (lo_ps, hi_ps)), ...]."""
    ln = cc.steps_line(plane)
    if ln is not None and ln.events:
        return [(i, (ev.offset_ps, ev.offset_ps + ev.duration_ps))
                for i, ev in enumerate(ln.events)]
    # Fallback: synthesize one global window covering all events on this plane.
    lo = sys.maxsize; hi = 0
    for line in plane.lines:
        for ev in line.events:
            lo = min(lo, ev.offset_ps)
            hi = max(hi, ev.offset_ps + ev.duration_ps)
    if hi == 0:
        return []
    return [(-1, (lo, hi))]


def report_for_plane(plane, *, warn_eps: float = 0.05) -> dict:
    compute_all, dropped_by_cat = _compute_intervals(plane)
    comm_all = _comm_intervals(plane)

    rows = []
    totals = {"compute_busy_ps": 0, "comm_inflight_ps": 0,
              "overlapped_ps": 0, "exposed_comm_ps": 0,
              "step_total_ps": 0}
    warns = []
    ln = cc.async_xla_line(plane)
    async_pairs = list(cc.pair_async_events(plane, ln)) if ln is not None else []
    # Pre-compute per-pair (done_offset, exposed_ps) for window clipping below.
    pair_meta = [(d.offset_ps, int(cc.event_stats(plane, d).get("device_duration_ps") or d.duration_ps))
                 for _, d in async_pairs]

    for step_id, window in _step_windows(plane):
        comp = _clip(compute_all, window)
        comm = _clip(comm_all, window)
        meta_exposed_in_step = sum(exposed for off, exposed in pair_meta
                                   if window[0] <= off < window[1])

        compute_busy = union_length(comp)
        comm_inflight = union_length(comm)
        overlapped = union_length(intersection_intervals(comp, comm))
        exposed_comm = max(0, comm_inflight - overlapped)
        ratio = (overlapped / comm_inflight) if comm_inflight else float("nan")

        if (comm_inflight > 0 and meta_exposed_in_step > 0
                and abs(exposed_comm - meta_exposed_in_step) / max(meta_exposed_in_step, 1)
                > warn_eps):
            warns.append((step_id, exposed_comm, meta_exposed_in_step))

        rows.append({
            "step": step_id, "step_ps": window[1] - window[0],
            "compute_busy_ps": compute_busy,
            "comm_inflight_ps": comm_inflight,
            "overlapped_ps": overlapped,
            "exposed_comm_ps": exposed_comm,
            "overlap_ratio": ratio,
        })
        for k in totals:
            if k == "step_total_ps":
                totals[k] += window[1] - window[0]
            elif k in rows[-1]:
                totals[k] += rows[-1][k]

    return {"plane": plane.name, "core": cc.core_kind(plane),
            "rows": rows, "totals": totals, "warns": warns,
            "dropped_by_category": dropped_by_cat}


# ---------------------------------------------------------------------------
# Top-N contributors per plane (works for TC and SC)
# ---------------------------------------------------------------------------

def top_contributors_for_plane(
    plane,
    merged_compute: list[tuple[int, int]],
    *,
    limit: int,
) -> tuple[list[dict], int, int]:
    """Per-op contributors on this plane, with NOT_cov computed against the
    SAME-core merged compute timeline (TC comm vs TC compute, SC comm vs SC
    compute — never cross-core).

    Returns (rows, n_async, n_unpaired). The caller decides how to label /
    sort the table based on n_unpaired/n_async.

    Each row has: op_name, hlo_category, wall_ps, stall_ps, hidden_ps,
    not_cov_ps, hidden_ratio.
    """
    out: list[dict] = []
    n_async = 0
    n_unpaired = 0
    async_ln = cc.async_xla_line(plane)
    if async_ln is not None:
        for s, d in cc.pair_async_events(plane, async_ln):
            n_async += 1
            ds = cc.event_stats(plane, d)
            md = cc.event_metadata_stats(plane, d)
            stall = int(ds.get("device_duration_ps") or d.duration_ps)
            if s is not None:
                start_ps = s.offset_ps
                end_ps = d.offset_ps + d.duration_ps
                wall = end_ps - start_ps
            else:
                n_unpaired += 1
                start_ps = d.offset_ps
                end_ps = d.offset_ps + d.duration_ps
                wall = d.duration_ps
            hidden = max(0, wall - stall)
            not_cov = cc.not_covered_by_compute(start_ps, end_ps, merged_compute)
            out.append({
                "op_name": cc.canonical_op_name(str(ds.get("hlo_op") or cc.event_name(plane, d))),
                "hlo_category": md.get("hlo_category"),
                "wall_ps": int(wall), "stall_ps": stall, "hidden_ps": hidden,
                "not_cov_ps": int(not_cov),
                "hidden_ratio": hidden / wall if wall else 0.0,
            })
    return out, n_async, n_unpaired


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_step_table(report: dict, label: str):
    def _fmt_ratio(r: float) -> str:
        return f"{r:.2f}" if r == r else "—"

    rows = report["rows"]; t = report["totals"]
    print(f"\n=== {label} ({report['plane']}) ===")
    print(f"{'step':>6}{'step(us)':>12}{'compute(us)':>14}"
          f"{'comm(us)':>12}{'overlap(us)':>14}{'exposed(us)':>14}{'ratio':>8}")
    for r in rows:
        print(f"{r['step']:>6}{r['step_ps']/1e6:>12.3f}"
              f"{r['compute_busy_ps']/1e6:>14.3f}"
              f"{r['comm_inflight_ps']/1e6:>12.3f}"
              f"{r['overlapped_ps']/1e6:>14.3f}"
              f"{r['exposed_comm_ps']/1e6:>14.3f}"
              f"{_fmt_ratio(r['overlap_ratio']):>8}")
    total_ratio = (t['overlapped_ps'] / t['comm_inflight_ps']) if t['comm_inflight_ps'] else float('nan')
    print(f"{'TOTAL':>6}{t['step_total_ps']/1e6:>12.3f}"
          f"{t['compute_busy_ps']/1e6:>14.3f}"
          f"{t['comm_inflight_ps']/1e6:>12.3f}"
          f"{t['overlapped_ps']/1e6:>14.3f}"
          f"{t['exposed_comm_ps']/1e6:>14.3f}"
          f"{_fmt_ratio(total_ratio):>8}")
    for step_id, sweep, meta in report["warns"]:
        ratio_hint = ""
        if meta > 0 and sweep / max(meta, 1) < 0.1:
            ratio_hint = "  (unpaired-dominated capture: meta double-counts overlapped time)"
        print(f"  [warn] step {step_id}: sweep_exposed={sweep/1e6:.3f}us  "
              f"meta_exposed={meta/1e6:.3f}us  Δ>5%; sweep authoritative{ratio_hint}")

    dropped = report.get("dropped_by_category", {})
    if dropped:
        # Show only wrapper categories — comm exclusions are obvious.
        wrapper_drop = {c: d for c, d in dropped.items() if c in _WRAPPER_HLO_CATEGORIES}
        if wrapper_drop:
            parts = ", ".join(f"{c}={d/1e6:.0f}us"
                              for c, d in sorted(wrapper_drop.items(),
                                                 key=lambda kv: -kv[1]))
            print(f"  [info] excluded from compute (wrapper/container ops): {parts}")


def _print_top_table(plane, merged_compute, *, core_label: str, limit: int):
    """Print the top-N per-op contributors for a single plane.

    Adaptive labeling and sort. Two distinct sentinel regimes — both make
    `stall_ps` unreliable as a critical-path metric:

      1. unpaired-dominated (`unpaired_ratio > 0.5`): flow-singleton fallback
         set stall = wall, hidden = 0 for most rows.
      2. concurrent-comm (`comm_concurrency > 1.2`): multiple ICI links ran
         in parallel. Per-row stall_ps is per-link engine-busy time, not
         exposed time; Σstall is non-additive vs wall-clock.

    In either regime we switch the title to "comm engine busy contributors
    (NOT exposed)", suppress the hidden% column (sentinel/misleading), and
    sort by NOT_cov_ps (sweep-derived, authoritative). The NOT_cov column
    is ALWAYS shown.
    """
    rows, n_async, n_unpaired = top_contributors_for_plane(
        plane, merged_compute, limit=limit)
    if not rows:
        return

    ratio = n_unpaired / n_async if n_async else 0.0
    # Concurrency from the same async pairs.
    intervals = []
    async_ln = cc.async_xla_line(plane)
    if async_ln is not None:
        for s, d in cc.pair_async_events(plane, async_ln):
            if s is not None:
                intervals.append((s.offset_ps, d.offset_ps + d.duration_ps))
            else:
                intervals.append((d.offset_ps, d.offset_ps + d.duration_ps))
    concurrency, _, _ = cc.comm_concurrency(intervals)

    degenerate = ratio > 0.5 or concurrency > 1.2

    if degenerate:
        if ratio > 0.5:
            why = f"capture is unpaired-dominated, unpaired_ratio={ratio:.0%}"
        else:
            why = f"comm concurrency = {concurrency:.2f} (parallel ICI links)"
        title = (f"\nTop-{limit} {core_label} comm engine busy contributors "
                 f"(NOT 'exposed' — {why}):")
        rows.sort(key=lambda r: -r["not_cov_ps"])
        print(title)
        print(f"  [warn] hidden% column suppressed (stall is sentinel in this "
              f"regime). Sort key = NOT_cov_by_compute, computed by sweep "
              f"against {core_label} compute intervals on the same core.")
        print(f"{'op_name':<48}{'hlo_category':<20}{'wall(us)':>11}"
              f"{'stall(us)':>11}{'NOT_cov(us)':>13}")
        for r in rows[:limit]:
            print(f"{r['op_name'][:46]:<48}"
                  f"{(r['hlo_category'] or '?')[:18]:<20}"
                  f"{r['wall_ps']/1e6:>11.3f}"
                  f"{r['stall_ps']/1e6:>11.3f}"
                  f"{r['not_cov_ps']/1e6:>13.3f}")
    else:
        title = f"\nTop-{limit} {core_label} exposed-comm contributors:"
        rows.sort(key=lambda r: -r["stall_ps"])
        print(title)
        print(f"{'op_name':<48}{'hlo_category':<20}{'wall(us)':>11}"
              f"{'stall(us)':>11}{'hidden':>9}{'NOT_cov(us)':>13}")
        for r in rows[:limit]:
            print(f"{r['op_name'][:46]:<48}"
                  f"{(r['hlo_category'] or '?')[:18]:<20}"
                  f"{r['wall_ps']/1e6:>11.3f}"
                  f"{r['stall_ps']/1e6:>11.3f}"
                  f"{r['hidden_ratio']*100:>8.1f}%"
                  f"{r['not_cov_ps']/1e6:>13.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    xs = cc.load_xspace(args.profile_dir)
    if xs is None:
        print(f"[absent] no *.xplane.pb in {args.profile_dir}")
        return

    planes = [p for p in cc.iter_device_planes(xs) if "CUSTOM" not in p.name]
    planes_by_name = {p.name: p for p in planes}
    tc_reports = []
    sc_reports = []
    for plane in planes:
        rep = report_for_plane(plane)
        if cc.core_kind(plane) == "TC":
            tc_reports.append(rep)
        else:
            sc_reports.append(rep)

    if not tc_reports:
        print("[absent] no /device:TPU:N (TensorCore) plane")
        return

    for rep in tc_reports:
        if rep["rows"] and rep["rows"][0]["step"] == -1:
            print("[fallback] no Steps line; using global window")
        _print_step_table(rep, "TC compute/comm overlap")

    if sc_reports:
        for rep in sc_reports:
            _print_step_table(rep, f"{rep['core']} comm (separate; doesn't compete with TC compute)")
    else:
        print("\n(no SparseCore planes present in this capture)")

    # Per-core merged compute: TC comm rows sweep against TC compute,
    # SC comm rows sweep against SC compute. Never cross.
    merged_compute = cc.merged_compute_by_core(xs)

    # Top-N contributors per plane. TC first; SC tables follow only when SC
    # actually carries comm primitives (typical SparseCore captures do).
    for rep in tc_reports:
        plane = planes_by_name[rep["plane"]]
        _print_top_table(plane, merged_compute.get("TC", []),
                         core_label="TC", limit=args.limit)
    for rep in sc_reports:
        plane = planes_by_name[rep["plane"]]
        ck = cc.core_kind(plane)
        _print_top_table(plane, merged_compute.get(ck, []),
                         core_label=ck, limit=args.limit)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tc": tc_reports, "sc": sc_reports}, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
