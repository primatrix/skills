"""
Extract HLO-op-level XEvents from the device plane's "XLA Ops" line.

Each event is a single HLO operation execution. For HLO events the rich
payload (`hlo_category`, `flops`, etc.) lives on the *EventMetadata*,
NOT on the per-event stats list — XLA shares one XEventMetadata across
every run of the same HLO op and attaches the op-level stats there. The
HLO op text itself is XEventMetadata.name (no separate `hlo_op` stat).
The per-XEvent stats list on this line carries only execution-time
counters: device_offset_ps, device_duration_ps, Time Scale Multiplier.

Schema shown:
    XPlane(name startswith '/device:') -> XLine(name='XLA Ops') -> XEvent
    XEvent.metadata_id -> XEventMetadata (.name = HLO op text;
        .stats carry hlo_category / flops / ... resolved via
        XPlane.stat_metadata).

Fields illustrated:
    XEvent.{metadata_id, offset_ps, duration_ps, stats}
    XEventMetadata.{name, stats}
    XStat.{metadata_id, value oneof}
    Stat names looked up on XEventMetadata.stats (verified present on
    /device:TPU:0 in dp8_fsdp128):
        hlo_category, tf_op, program_id, flops, model_flops,
        bytes_accessed, raw_bytes_accessed, shape_with_layout.

Source proto:
    _proto/xplane_pb2.XLine.events,
    _proto/xplane_pb2.XEventMetadata.stats
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
import xplane_pb2  # noqa: E402


INTERESTING_STATS = (
    "hlo_category", "tf_op", "program_id",
    "flops", "model_flops", "bytes_accessed", "raw_bytes_accessed",
    "shape_with_layout",
)


def _stat_value(stat):
    vf = stat.WhichOneof("value")
    return vf, (getattr(stat, vf) if vf else None)


def main(profile_dir: str, limit: int = 20) -> None:
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        print("[absent] no *.xplane.pb in", profile_dir)
        return
    xs = xplane_pb2.XSpace()
    with open(pbs[0], "rb") as f:
        xs.ParseFromString(f.read())

    device_plane = next((p for p in xs.planes if p.name.startswith("/device:")), None)
    if device_plane is None:
        print("[absent] no /device:* plane")
        return

    ops_line = next((l for l in device_plane.lines if l.name == "XLA Ops"), None)
    if ops_line is None:
        print(f"[absent] plane {device_plane.name!r} has no 'XLA Ops' line "
              f"(lines: {[l.name for l in device_plane.lines]})")
        return

    # Build name -> id reverse map from stat_metadata for quick lookup
    stat_name_by_id = {smid: sm.name for smid, sm in device_plane.stat_metadata.items()}

    print(f"Plane {device_plane.name!r}  Line 'XLA Ops'  "
          f"events={len(ops_line.events)}  (showing first {limit})")
    for ev in ops_line.events[:limit]:
        em = device_plane.event_metadata.get(ev.metadata_id)
        hlo_op = em.name if em is not None else "?"
        # Op-level stats live on XEventMetadata.stats, not XEvent.stats.
        meta_stats = {stat_name_by_id.get(s.metadata_id, "?"): _stat_value(s)
                      for s in (em.stats if em is not None else [])}
        shown = {k: meta_stats[k] for k in INTERESTING_STATS if k in meta_stats}
        print(f"  event hlo_op={hlo_op[:60]!r} duration_ps={ev.duration_ps}")
        for k, (vf, vv) in shown.items():
            print(f"    {k}: ({vf}) {vv!r}")
    if len(ops_line.events) > limit:
        print(f"  ... ({len(ops_line.events) - limit} more)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
