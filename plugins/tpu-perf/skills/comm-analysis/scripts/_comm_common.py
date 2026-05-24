"""
Shared helpers for the comm-analysis skill.

Reads:
- xplane.pb (reuses xplane_pb2 from profile-anatomy)
- hlo_proto.pb (uses local _proto/hlo_pb2)
- op_stats.pb  (uses local _proto/op_stats_pb2)

No __main__; this is a library module.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Iterator, Optional

# Reuse profile-anatomy's xplane_pb2 — explicit dependency.
_HERE = pathlib.Path(__file__).resolve().parent
_PROFILE_ANATOMY_PROTO = (
    _HERE.parent.parent / "profile-anatomy" / "scripts" / "_proto"
)
sys.path.insert(0, str(_PROFILE_ANATOMY_PROTO))
sys.path.insert(0, str(_HERE / "_proto"))

import xplane_pb2          # noqa: E402  (from profile-anatomy)
import hlo_pb2             # noqa: E402  (from this skill's _proto)
import op_stats_pb2        # noqa: E402  (from this skill's _proto)


# ---------------------------------------------------------------------------
# XSpace loading
# ---------------------------------------------------------------------------

def load_xspace(profile_dir: str | pathlib.Path) -> Optional[xplane_pb2.XSpace]:
    """Load the first *.xplane.pb in profile_dir. Returns None if absent."""
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        return None
    xs = xplane_pb2.XSpace()
    xs.ParseFromString(pbs[0].read_bytes())
    return xs


def iter_device_planes(xs: xplane_pb2.XSpace) -> Iterator[xplane_pb2.XPlane]:
    """Yield every plane whose name starts with '/device:' — TC and SC."""
    for p in xs.planes:
        if p.name.startswith("/device:"):
            yield p


def core_kind(plane: xplane_pb2.XPlane) -> str:
    """Map a device plane name to TC / SC0 / SC1."""
    name = plane.name
    if "SparseCore 0" in name:
        return "SC0"
    if "SparseCore 1" in name:
        return "SC1"
    return "TC"


# ---------------------------------------------------------------------------
# Stat resolvers
# ---------------------------------------------------------------------------

def stat_name_by_id(plane: xplane_pb2.XPlane) -> dict[int, str]:
    return {smid: sm.name for smid, sm in plane.stat_metadata.items()}


def _xstat_value(stat):
    """Unwrap the 6-variant XStat oneof. Returns None if value field unset."""
    vf = stat.WhichOneof("value")
    return getattr(stat, vf) if vf else None


def event_stats(plane: xplane_pb2.XPlane, ev: xplane_pb2.XEvent) -> dict[str, object]:
    """Resolve XEvent.stats (per-execution counters) into a name -> value dict."""
    names = stat_name_by_id(plane)
    return {names[s.metadata_id]: _xstat_value(s)
            for s in ev.stats if s.metadata_id in names}


def event_metadata_stats(
    plane: xplane_pb2.XPlane, ev: xplane_pb2.XEvent
) -> dict[str, object]:
    """Resolve XEventMetadata.stats (op-level facts: hlo_category, flops, …)."""
    em = plane.event_metadata.get(ev.metadata_id)
    if em is None:
        return {}
    names = stat_name_by_id(plane)
    return {names[s.metadata_id]: _xstat_value(s)
            for s in em.stats if s.metadata_id in names}


def event_name(plane: xplane_pb2.XPlane, ev: xplane_pb2.XEvent) -> str:
    em = plane.event_metadata.get(ev.metadata_id)
    return em.name if em is not None else "?"


# ---------------------------------------------------------------------------
# Async pairing
# ---------------------------------------------------------------------------

def async_xla_line(plane: xplane_pb2.XPlane) -> Optional[xplane_pb2.XLine]:
    return next((ln for ln in plane.lines if ln.name == "Async XLA Ops"), None)


def xla_ops_line(plane: xplane_pb2.XPlane) -> Optional[xplane_pb2.XLine]:
    return next((ln for ln in plane.lines if ln.name == "XLA Ops"), None)


def steps_line(plane: xplane_pb2.XPlane) -> Optional[xplane_pb2.XLine]:
    return next((ln for ln in plane.lines if ln.name == "Steps"), None)


def pair_async_events(
    plane: xplane_pb2.XPlane, line: xplane_pb2.XLine
) -> list[tuple[Optional[xplane_pb2.XEvent], xplane_pb2.XEvent]]:
    """
    Group events on `line` by their 'flow' XStat. For each flow group:
      - 1 event:  yield (None, ev) — caller treats as fully exposed.
      - 2 events: yield (start, done) sorted by offset_ps.
      - >=3:      yield (start_min, done_max); intermediate events are
                   silently dropped. This case is vanishingly rare in
                   practice; if a caller needs to detect it, build the
                   flow grouping directly rather than via this helper.
    Events with no 'flow' stat are returned as (None, ev).
    """
    by_flow: dict[int, list[xplane_pb2.XEvent]] = {}
    unpaired: list[xplane_pb2.XEvent] = []
    for ev in line.events:
        stats = event_stats(plane, ev)
        flow = stats.get("flow")
        if flow is None:
            unpaired.append(ev)
            continue
        by_flow.setdefault(flow, []).append(ev)

    pairs: list[tuple[Optional[xplane_pb2.XEvent], xplane_pb2.XEvent]] = []
    for flow, evs in by_flow.items():
        evs_sorted = sorted(evs, key=lambda e: e.offset_ps)
        if len(evs_sorted) == 1:
            pairs.append((None, evs_sorted[0]))
        else:
            # First as start, last as done.
            pairs.append((evs_sorted[0], evs_sorted[-1]))
    for ev in unpaired:
        pairs.append((None, ev))
    return pairs
