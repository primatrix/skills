"""
Extract embedded HloProto module(s) from an xplane.pb capture.

Modern XProf/JAX captures embed every JIT-compiled HLO module's
HloProto inside the `*.xplane.pb` itself, on the `/host:metadata`
plane. They live on `XEventMetadata.stats` as a `bytes_value` whose
stat name is exactly `'Hlo Proto'`. The owning XEventMetadata's `name`
is the JAX module name (e.g. `jit_train_step(8722433274278871538)`),
which matches the standalone `*.hlo_proto.pb` file dropped next to the
xplane in the same directory. Both decode to the same `HloProto`
message (only field-ordering differs).

This script:
  * discovers every `Hlo Proto` blob on `/host:metadata`,
  * parses each as `xla.HloProto`,
  * prints a one-line summary per module
    (name, id, #computations, #instructions, entry computation),
  * additionally lists any sibling `*.hlo_proto.pb` files on disk so
    you can see at a glance which modules also have a standalone
    file.

Pass `--dump <module-name-substring>` to print the matching module's
HLO instructions (opcode + name + shape) to stdout.

Schema shown:
    XPlane(name='/host:metadata') -> XEventMetadata.stats[*]
    XStat(metadata_id resolves to 'Hlo Proto').bytes_value
        -> hlo_pb2.HloProto

Source proto:
    _proto/xplane_pb2.XEventMetadata.stats,
    _proto/hlo_pb2.HloProto
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
import xplane_pb2  # noqa: E402
import hlo_pb2  # noqa: E402


def _hlo_blobs(xspace):
    """Yield (module_name, hlo_proto_bytes) for every Hlo Proto blob."""
    for plane in xspace.planes:
        if plane.name != "/host:metadata":
            continue
        sid = next(
            (sm.id for sm in plane.stat_metadata.values() if sm.name == "Hlo Proto"),
            None,
        )
        if sid is None:
            continue
        for em in plane.event_metadata.values():
            for stat in em.stats:
                if stat.metadata_id == sid and stat.HasField("bytes_value"):
                    yield em.name, stat.bytes_value


def _summarize(name: str, payload: bytes) -> dict:
    hp = hlo_pb2.HloProto()
    hp.ParseFromString(payload)
    mod = hp.hlo_module
    return {
        "name": name,
        "module_name": mod.name,
        "module_id": mod.id,
        "entry_computation_id": mod.entry_computation_id,
        "computations": len(mod.computations),
        "instructions": sum(len(c.instructions) for c in mod.computations),
        "bytes": len(payload),
        "hlo_proto": hp,
    }


def main(profile_dir: str, dump_substr: str | None = None) -> None:
    profile_dir = pathlib.Path(profile_dir)
    pbs = sorted(profile_dir.glob("*.xplane.pb"))
    if not pbs:
        print("[absent] no *.xplane.pb in", profile_dir)
        return
    xs = xplane_pb2.XSpace()
    with open(pbs[0], "rb") as f:
        xs.ParseFromString(f.read())

    blobs = list(_hlo_blobs(xs))
    sums = []
    if not blobs:
        print(f"[absent] no 'Hlo Proto' XStat under /host:metadata in {pbs[0].name}")
    else:
        print(f"Hlo Proto blobs embedded in {pbs[0].name}: {len(blobs)}")
        for name, payload in blobs:
            try:
                sums.append(_summarize(name, payload))
            except Exception as e:  # corrupt / future-format blob
                print(f"  ! parse failed for {name!r}: {e}")
        sums.sort(key=lambda s: -s["bytes"])
        for s in sums:
            print(
                f"  {s['name'][:60]:<60}  "
                f"id={s['module_id']:<6}  "
                f"comps={s['computations']:<6}  "
                f"insts={s['instructions']:<8}  "
                f"bytes={s['bytes']}"
            )

    on_disk = sorted(profile_dir.glob("*.hlo_proto.pb"))
    if on_disk:
        print(f"\nStandalone *.hlo_proto.pb files in {profile_dir.name}: {len(on_disk)}")
        for p in on_disk:
            print(f"  {p.name}  ({p.stat().st_size} bytes)")
    else:
        print("\n[absent] no standalone *.hlo_proto.pb files (xplane is the only source)")

    if dump_substr:
        match = next((s for s in sums if dump_substr in s["name"]), None)
        if match is None:
            print(f"\n[absent] no module name contains {dump_substr!r}")
            return
        mod = match["hlo_proto"].hlo_module
        entry = next(
            (c for c in mod.computations if c.id == mod.entry_computation_id),
            mod.computations[0] if mod.computations else None,
        )
        if entry is None:
            print("[absent] module has no computations")
            return
        print(f"\n--- entry computation '{entry.name}' "
              f"({len(entry.instructions)} instructions) ---")
        for inst in entry.instructions[:50]:
            shape = inst.shape
            shape_dims = "x".join(str(d) for d in shape.dimensions) if shape.dimensions else "scalar"
            print(f"  {inst.opcode:<20} {inst.name[:50]:<50} {shape_dims}")
        if len(entry.instructions) > 50:
            print(f"  ... ({len(entry.instructions) - 50} more instructions)")


if __name__ == "__main__":
    args = sys.argv[1:]
    profile = args[0] if args else "/Users/xl/tensorboard/tensorboard/plugins/profile/2026_05_26_11_29_35"
    dump = None
    if "--dump" in args:
        i = args.index("--dump")
        dump = args[i + 1] if i + 1 < len(args) else None
    main(profile, dump)
