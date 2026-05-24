"""
Compute/comm overlap report for a TPU profile.

Per step on the device plane, computes:
  - compute_busy_ps   = ∪(compute intervals)
  - comm_inflight_ps  = ∪(comm intervals)
  - overlapped_ps     = ∪(compute ∩ comm)
  - exposed_comm_ps   = comm_inflight - overlapped
  - overlap_ratio     = overlapped / comm_inflight

Sweep-line union math; intervals are clipped to the step window.

Sanity-check: the sweep-derived exposed_comm vs Σ done.device_duration_ps
within the step (the metadata-reported exposed time). Mismatch above 5%
prints `[warn] step N`. Sweep is authoritative.
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

def _compute_intervals(plane) -> list[tuple[int, int]]:
    ln = cc.xla_ops_line(plane)
    if ln is None:
        return []
    out = []
    for ev in ln.events:
        md = cc.event_metadata_stats(plane, ev)
        cat = md.get("hlo_category")
        if cat in _COMM_HLO_CATEGORIES:
            continue
        out.append((ev.offset_ps, ev.offset_ps + ev.duration_ps))
    return out


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
    compute_all = _compute_intervals(plane)
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
            "rows": rows, "totals": totals, "warns": warns}


# ---------------------------------------------------------------------------
# Top-N exposed contributors (across all steps, TC plane only)
# ---------------------------------------------------------------------------

def top_exposed_per_collective(plane, *, limit: int) -> list[dict]:
    out = []
    async_ln = cc.async_xla_line(plane)
    if async_ln is not None:
        # Unpaired events represent exposed-only slices (no start event to span the
        # overlapped window), so hidden_ratio collapses to 0% — that's accurate, not noise.
        for s, d in cc.pair_async_events(plane, async_ln):
            ds = cc.event_stats(plane, d)
            md = cc.event_metadata_stats(plane, d)
            stall = int(ds.get("device_duration_ps") or d.duration_ps)
            wall = (d.offset_ps + d.duration_ps - s.offset_ps) if s is not None else d.duration_ps
            hidden = max(0, wall - stall)
            out.append({
                "op_name": cc.canonical_op_name(str(ds.get("hlo_op") or cc.event_name(plane, d))),
                "hlo_category": md.get("hlo_category"),
                "wall_ps": int(wall), "stall_ps": stall, "hidden_ps": hidden,
                "hidden_ratio": hidden / wall if wall else 0.0,
            })
    out.sort(key=lambda r: -r["stall_ps"])
    return out[:limit]


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

    print(f"\nTop-{args.limit} TC exposed-comm contributors:")
    print(f"{'op_name':<48}{'hlo_category':<20}{'wall(us)':>11}{'stall(us)':>11}{'hidden':>9}")
    for rep in tc_reports:
        plane = planes_by_name[rep["plane"]]
        top = top_exposed_per_collective(plane, limit=args.limit)
        for r in top:
            print(f"{r['op_name'][:46]:<48}"
                  f"{(r['hlo_category'] or '?')[:18]:<20}"
                  f"{r['wall_ps']/1e6:>11.3f}"
                  f"{r['stall_ps']/1e6:>11.3f}"
                  f"{r['hidden_ratio']*100:>8.1f}%")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tc": tc_reports, "sc": sc_reports}, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
