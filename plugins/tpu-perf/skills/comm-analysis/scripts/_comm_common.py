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


# ---------------------------------------------------------------------------
# HLO module loading and joining
# ---------------------------------------------------------------------------
#
# Important: *.hlo_proto.pb files in xprof captures are serialized
# `xla.HloProto` (with `hlo_module = 1`), NOT bare `HloModuleProto`. We MUST
# parse via HloProto and then read `.hlo_module`. (Verified on the fixture
# during plan task 2.)

import re

_CALL_SUFFIX = re.compile(r"\.(call-start|call-done|start|done)$")


def canonical_op_name(name: str) -> str:
    """Strip async-pairing suffixes so start/done events join to one HLO instr."""
    return _CALL_SUFFIX.sub("", name)


def _hlo_program_id(module: hlo_pb2.HloModuleProto) -> int:
    """HloModuleProto.id is the program_id used in xprof XStats."""
    return module.id


def load_hlo_module(
    profile_dir: str | pathlib.Path,
    *,
    prefer_program_id: int | None = None,
) -> Optional[hlo_pb2.HloModuleProto]:
    """
    Pick the most relevant *.hlo_proto.pb in profile_dir.

    *.hlo_proto.pb files are serialized xla.HloProto (with hlo_module = 1),
    not bare HloModuleProto — parse via HloProto and return .hlo_module.

    Selection order:
      1. If `prefer_program_id` is given, return the module with matching id.
      2. Largest file size.
      3. Lexicographic name as final tie-break.
    """
    pbs = sorted(pathlib.Path(profile_dir).glob("*.hlo_proto.pb"))
    if not pbs:
        return None

    parsed: list[tuple[pathlib.Path, hlo_pb2.HloModuleProto]] = []
    for pb in pbs:
        hp = hlo_pb2.HloProto()
        try:
            hp.ParseFromString(pb.read_bytes())
        except Exception:
            continue
        parsed.append((pb, hp.hlo_module))
    if not parsed:
        return None

    if prefer_program_id is not None:
        for _, m in parsed:
            if _hlo_program_id(m) == prefer_program_id:
                return m
    parsed.sort(key=lambda t: (-t[0].stat().st_size, t[0].name))
    return parsed[0][1]


def hlo_instructions(
    module: hlo_pb2.HloModuleProto,
) -> dict[str, hlo_pb2.HloInstructionProto]:
    """Flatten every (computation, instruction) into a {canonical_name: instr} map.

    Note: the legacy `instruction.replica_groups` (field 49) is empty in
    modern HLO. Callers extracting replica info should walk
    `instruction.collective_device_list.replica_groups` or
    `instruction.iota_collective_device_list` instead.
    """
    out: dict[str, hlo_pb2.HloInstructionProto] = {}
    for c in module.computations:
        for i in c.instructions:
            out[canonical_op_name(i.name)] = i
    return out


# ---------------------------------------------------------------------------
# op_stats.pb — memory roofline only (interconnect peaks live in xprof XStats)
# ---------------------------------------------------------------------------
#
# OpStats.perf_env.peak_bws_giga_bytes_per_second (field 5, repeated double)
# is indexed by upstream MemBwType: HBM_RW, SRAM_RD, SRAM_WR, CMEM_RD,
# CMEM_WR, VMEM_RD, VMEM_WR. There is NO ICI / interconnect entry — do NOT
# add a `peak_ici_link_gbps_from_op_stats` helper; ICI peak BW must come
# from xprof XStats via peak_ici_link_gbps_from_xprof().
#
# (The plan's original draft used `peak_bw_giga_bytes_per_second_list` and
# `ICI_PEAK_INDEX = 7`; both are wrong per upstream proto, fixed here.)

def load_op_stats(profile_dir: str | pathlib.Path) -> Optional[op_stats_pb2.OpStats]:
    pbs = sorted(pathlib.Path(profile_dir).glob("*op_stats.pb"))
    if not pbs:
        return None
    o = op_stats_pb2.OpStats()
    try:
        o.ParseFromString(pbs[0].read_bytes())
    except Exception:
        return None
    return o


# MemBwType index of HBM_RW within peak_bws_giga_bytes_per_second.
HBM_PEAK_INDEX = 0


def peak_hbm_gbps_from_op_stats(o: op_stats_pb2.OpStats) -> Optional[float]:
    """Read peak HBM bandwidth (GiB/s) from OpStats.

    Prefers the MemBwType-indexed list (modern); falls back to the legacy
    scalar `peak_hbm_bw_giga_bytes_per_second` if the list is unset.
    """
    arr = list(o.perf_env.peak_bws_giga_bytes_per_second)
    if len(arr) > HBM_PEAK_INDEX and arr[HBM_PEAK_INDEX] > 0:
        return float(arr[HBM_PEAK_INDEX])
    legacy = o.perf_env.peak_hbm_bw_giga_bytes_per_second
    return float(legacy) if legacy > 0 else None


def peak_ici_link_gbps_from_xprof(xs: xplane_pb2.XSpace) -> Optional[float]:
    """Search XStats across device + Task Environment + host planes for an
    ICI peak link bandwidth.

    Per the user's roofline-source convention, ICI peak BW lives in xprof
    XStats — most commonly on the device plane or the Task Environment plane,
    NOT in op_stats. Match any stat name containing both 'peak' and 'ici'
    (case-insensitive). Returns None if no matching stat is present.

    Note (verified on the dp8_fsdp128 fixture): the fixture publishes
    peak_hbm/cmem/sram/vmem/teraflops stats but NO peak_ici_* stat. In that
    case this function returns None, and callers (e.g. roofline computation)
    must accept the absence and either skip the ICI roofline or fall back
    to a CLI flag with a warning.
    """
    def _scan(plane: xplane_pb2.XPlane) -> Optional[float]:
        names = stat_name_by_id(plane)
        # Plane-level stats: resolve metadata-id -> name and match.
        for stat in plane.stats:
            nm = names.get(stat.metadata_id, "").lower()
            if "peak" in nm and "ici" in nm:
                v = _xstat_value(stat)
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
        return None

    for plane in xs.planes:
        if (
            plane.name.startswith("/device:")
            or plane.name.startswith("/host:")
            or "Task Environment" in plane.name
        ):
            v = _scan(plane)
            if v is not None:
                return v
    return None
