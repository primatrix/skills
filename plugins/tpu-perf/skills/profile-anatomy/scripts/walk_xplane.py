"""
Walk the entire XSpace tree of a profile directory and print it indented.

Schema shown:
    XSpace -> XPlane -> XLine -> XEvent -> XStat (all five levels).

Fields illustrated:
    XSpace.{planes, errors, warnings, hostnames},
    XPlane.{id, name, lines, event_metadata, stat_metadata, stats},
    XLine.{id, name, timestamp_ns, duration_ps, events},
    XEvent.{metadata_id, offset_ps, num_occurrences, duration_ps, stats},
    XStat.{metadata_id, value oneof}.

Source proto:
    _proto/xplane_pb2.XSpace
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
import xplane_pb2  # noqa: E402


def main(profile_dir: str, limit: int = 5) -> None:
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        print("[absent] no *.xplane.pb in", profile_dir)
        return
    xs = xplane_pb2.XSpace()
    with open(pbs[0], "rb") as f:
        xs.ParseFromString(f.read())

    print(f"XSpace  source={pbs[0].name}")
    print(f"  hostnames={list(xs.hostnames)} errors={len(xs.errors)} warnings={len(xs.warnings)}")
    for p in xs.planes:
        print(f"  XPlane id={p.id} name={p.name!r}  "
              f"lines={len(p.lines)} event_metadata={len(p.event_metadata)} "
              f"stat_metadata={len(p.stat_metadata)} stats={len(p.stats)}")
        for line in p.lines[:limit]:
            print(f"    XLine id={line.id} name={line.name!r}  "
                  f"timestamp_ns={line.timestamp_ns} duration_ps={line.duration_ps} "
                  f"events={len(line.events)}")
            for ev in line.events[:limit]:
                ev_name = p.event_metadata[ev.metadata_id].name if ev.metadata_id in p.event_metadata else "?"
                data_field = ev.WhichOneof("data")
                data_val = getattr(ev, data_field) if data_field else None
                print(f"      XEvent metadata_id={ev.metadata_id} name={ev_name!r}  "
                      f"{data_field}={data_val} duration_ps={ev.duration_ps} "
                      f"stats={len(ev.stats)}")
                for stat in ev.stats[:limit]:
                    stat_name = p.stat_metadata[stat.metadata_id].name if stat.metadata_id in p.stat_metadata else "?"
                    vfield = stat.WhichOneof("value")
                    vval = getattr(stat, vfield) if vfield else None
                    print(f"        XStat name={stat_name!r}  {vfield}={vval!r}")
            if len(line.events) > limit:
                print(f"      ... ({len(line.events) - limit} more events)")
        if len(p.lines) > limit:
            print(f"    ... ({len(p.lines) - limit} more lines)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
