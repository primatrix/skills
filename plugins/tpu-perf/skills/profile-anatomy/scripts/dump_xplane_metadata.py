"""
Dump the event_metadata and stat_metadata reverse-lookup tables of every
XPlane in the profile. These are the tables every XEvent.metadata_id and
XStat.metadata_id resolves through; understanding them is the key to
reading any other field of any event or stat.

Schema shown:
    XPlane.event_metadata (map<int64, XEventMetadata>)
    XPlane.stat_metadata  (map<int64, XStatMetadata>)

Fields illustrated:
    XEventMetadata.{id, name, display_name, child_id}
    XStatMetadata.{id, name, description}

Source proto:
    _proto/xplane_pb2.XPlane.event_metadata
    _proto/xplane_pb2.XPlane.stat_metadata
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

    for p in xs.planes:
        print(f"=== Plane {p.name!r}  "
              f"event_metadata={len(p.event_metadata)} stat_metadata={len(p.stat_metadata)} ===")
        print(f"  -- event_metadata (showing up to {limit}) --")
        for emid, em in list(p.event_metadata.items())[:limit]:
            children = list(em.child_id) if em.child_id else []
            print(f"    [{emid}] name={em.name!r} display={em.display_name!r} "
                  f"child_id={children}")
        if len(p.event_metadata) > limit:
            print(f"    ... ({len(p.event_metadata) - limit} more)")
        print(f"  -- stat_metadata (showing up to {limit}) --")
        for smid, sm in list(p.stat_metadata.items())[:limit]:
            print(f"    [{smid}] name={sm.name!r} description={sm.description!r}")
        if len(p.stat_metadata) > limit:
            print(f"    ... ({len(p.stat_metadata) - limit} more)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
