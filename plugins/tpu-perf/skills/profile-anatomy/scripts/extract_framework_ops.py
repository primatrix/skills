"""
Extract framework-op events from the /host:CPU plane.

These are JAX/XLA-level Python calls (e.g., 'jit(train_step)',
'XlaPipelineCall', 'HostExecutionTimer'); their stat names are NOT a
fixed schema and vary by profiling source, so this script does not
hard-code stat names — it shows whatever is on each event.

Schema shown:
    XPlane(name='/host:CPU') -> XLine -> XEvent -> XStat.

Fields illustrated:
    XEvent.{metadata_id, offset_ps, duration_ps, stats}
    XStat.{metadata_id, value oneof}.
    Stat names are discovered, not assumed.

Source proto:
    _proto/xplane_pb2.XLine.events,
    _proto/xplane_pb2.XEvent.stats
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

    host_plane = next((p for p in xs.planes if p.name == "/host:CPU"), None)
    if host_plane is None:
        print("[absent] no /host:CPU plane")
        return

    stat_name_by_id = {smid: sm.name for smid, sm in host_plane.stat_metadata.items()}

    print(f"Plane {host_plane.name!r}  lines={len(host_plane.lines)}  "
          f"event_metadata={len(host_plane.event_metadata)}")
    shown = 0
    for line in host_plane.lines:
        if shown >= limit:
            break
        print(f"  XLine name={line.name!r}  events={len(line.events)}")
        for ev in line.events:
            if shown >= limit:
                break
            name = host_plane.event_metadata[ev.metadata_id].name if ev.metadata_id in host_plane.event_metadata else "?"
            data_field = ev.WhichOneof("data")
            data_val = getattr(ev, data_field) if data_field else None
            print(f"    event metadata.name={name[:60]!r}  "
                  f"{data_field}={data_val} duration_ps={ev.duration_ps}")
            for stat in sorted(ev.stats, key=lambda s: stat_name_by_id.get(s.metadata_id, "")):
                sname = stat_name_by_id.get(stat.metadata_id, "?")
                vf = stat.WhichOneof("value")
                vv = getattr(stat, vf) if vf else None
                if isinstance(vv, str) and len(vv) > 80:
                    vv = vv[:77] + "..."
                print(f"      {sname}: ({vf}) {vv!r}")
            shown += 1


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
