"""
Extract per-step XEvents from the device plane's "Steps" line.

Each XEvent on the "Steps" XLine represents one training step. The event's
metadata.name typically encodes the step number; offset_ps + duration_ps
give the timing in picoseconds relative to XLine.timestamp_ns (nanoseconds
since UNIX epoch).

Schema shown:
    XPlane(name startswith '/device:') -> XLine(name='Steps') -> XEvent.

Fields illustrated:
    XLine.timestamp_ns, XLine.duration_ps,
    XEvent.metadata_id (resolved via XPlane.event_metadata),
    XEvent.offset_ps, XEvent.duration_ps.

Source proto:
    _proto/xplane_pb2.XLine.events
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
import xplane_pb2  # noqa: E402


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

    steps_line = next((l for l in device_plane.lines if l.name == "Steps"), None)
    if steps_line is None:
        print(f"[absent] plane {device_plane.name!r} has no 'Steps' line  "
              f"(lines available: {[l.name for l in device_plane.lines]})")
        return

    print(f"Plane {device_plane.name!r}  Line 'Steps'  "
          f"timestamp_ns={steps_line.timestamp_ns} "
          f"duration_ps={steps_line.duration_ps} "
          f"events={len(steps_line.events)}")
    for ev in steps_line.events[:limit]:
        name = device_plane.event_metadata[ev.metadata_id].name if ev.metadata_id in device_plane.event_metadata else "?"
        data_field = ev.WhichOneof("data")
        data_val = getattr(ev, data_field) if data_field else None
        dur_us = ev.duration_ps / 1_000_000  # ps -> us for human reading
        print(f"  step name={name!r}  {data_field}={data_val} "
              f"duration_ps={ev.duration_ps}  (~{dur_us:.1f} us)")
    if len(steps_line.events) > limit:
        print(f"  ... ({len(steps_line.events) - limit} more)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
