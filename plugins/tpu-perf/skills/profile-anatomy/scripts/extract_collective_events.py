"""
Extract async XEvents from the device plane's "Async XLA Ops" line and
pair start <-> done by their 'flow' stat.

XLA emits collectives (all-reduce, all-gather, reduce-scatter, all-to-all)
and copies as paired async events. The two events of a pair share a uint64
'flow' stat; the *-done event's duration_ps measures the EXPOSED stall
cost (compute waiting for the comm engine).

Schema shown:
    XPlane(name startswith '/device:') -> XLine(name='Async XLA Ops')
    -> XEvent (paired by 'flow' XStat).

Fields illustrated:
    XEvent.{metadata_id, offset_ps, duration_ps, stats}
    XStat names: flow (pairing key), device_offset_ps, device_duration_ps,
                 hlo_op, id.

Source proto:
    _proto/xplane_pb2.XLine.events,
    _proto/xplane_pb2.XEvent.stats
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
import xplane_pb2  # noqa: E402


def _get_stat(ev, stat_name_by_id, name):
    for s in ev.stats:
        if stat_name_by_id.get(s.metadata_id) == name:
            vf = s.WhichOneof("value")
            return getattr(s, vf) if vf else None
    return None


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

    async_line = next((l for l in device_plane.lines if l.name == "Async XLA Ops"), None)
    if async_line is None:
        print(f"[absent] plane {device_plane.name!r} has no 'Async XLA Ops' line "
              f"(lines: {[l.name for l in device_plane.lines]})")
        return

    stat_name_by_id = {smid: sm.name for smid, sm in device_plane.stat_metadata.items()}

    print(f"Plane {device_plane.name!r}  Line 'Async XLA Ops'  "
          f"events={len(async_line.events)}")

    # First pass: bucket events by their 'flow' stat
    by_flow = {}
    for ev in async_line.events:
        flow = _get_stat(ev, stat_name_by_id, "flow")
        if flow is None:
            continue
        by_flow.setdefault(flow, []).append(ev)

    print(f"  distinct flow IDs: {len(by_flow)}  (showing first {limit})")
    shown = 0
    for flow, evs in by_flow.items():
        if shown >= limit:
            break
        # Sort within a flow by offset_ps so 'start' comes before 'done'
        evs_sorted = sorted(evs, key=lambda e: e.offset_ps)
        print(f"  flow={flow}  pair_size={len(evs_sorted)}")
        for ev in evs_sorted:
            ev_name = device_plane.event_metadata[ev.metadata_id].name if ev.metadata_id in device_plane.event_metadata else "?"
            hlo_op = _get_stat(ev, stat_name_by_id, "hlo_op")
            dev_dur = _get_stat(ev, stat_name_by_id, "device_duration_ps")
            print(f"    event metadata.name={ev_name[:50]!r}  "
                  f"hlo_op={hlo_op!r}  offset_ps={ev.offset_ps}  "
                  f"duration_ps={ev.duration_ps}  device_duration_ps={dev_dur}")
        shown += 1
    if len(by_flow) > limit:
        print(f"  ... ({len(by_flow) - limit} more flows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
