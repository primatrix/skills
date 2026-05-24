"""
Parse the Chrome trace JSON (gzipped) and show its top-level structure.

Schema shown:
    Top-level JSON object with keys: displayTimeUnit, metadata, traceEvents.
    traceEvents are objects with at least 'ph' (phase) in:
      'M' = metadata (process_name, thread_name, process_sort_index, ...)
      'X' = complete event with start ts and dur
      'i' = instant event
      'B' / 'E' = paired begin/end (rare in TPU profiles).

Fields illustrated:
    Top-level: displayTimeUnit, metadata.{highres-ticks, ...}, traceEvents[].
    Per event: ph, pid, tid, name, cat, ts, dur, args.

Source proto:
    n/a -- this is plain JSON, not a protobuf.
"""
import gzip
import json
import sys
import pathlib


def main(profile_dir: str, limit: int = 5) -> None:
    gzs = sorted(pathlib.Path(profile_dir).glob("*.trace.json.gz"))
    if not gzs:
        print("[absent] no *.trace.json.gz in", profile_dir)
        return
    with gzip.open(gzs[0], "rt") as f:
        doc = json.load(f)

    print(f"trace file: {gzs[0].name}")
    print(f"top-level keys: {sorted(doc.keys())}")
    print(f"displayTimeUnit: {doc.get('displayTimeUnit')!r}")
    print(f"metadata: {doc.get('metadata')!r}")
    events = doc.get("traceEvents", [])
    print(f"traceEvents: {len(events)} total")

    # Build pid/tid name maps from ph='M' metadata events
    pid_name = {}
    tid_name = {}
    for ev in events:
        if ev.get("ph") != "M":
            continue
        if ev.get("name") == "process_name":
            pid_name[ev.get("pid")] = ev.get("args", {}).get("name")
        elif ev.get("name") == "thread_name":
            tid_name[(ev.get("pid"), ev.get("tid"))] = ev.get("args", {}).get("name")

    print(f"\n-- pid -> process_name ({len(pid_name)} entries) --")
    for pid, name in list(pid_name.items())[:limit]:
        print(f"  pid={pid}  name={name!r}")

    print(f"\n-- (pid, tid) -> thread_name (showing first {limit}) --")
    for (pid, tid), name in list(tid_name.items())[:limit]:
        print(f"  pid={pid} tid={tid}  name={name!r}")

    print(f"\n-- sample 'X' (complete) events (showing first {limit}) --")
    x_count = 0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        print(f"  name={ev.get('name')[:60]!r} cat={ev.get('cat')!r} "
              f"pid={ev.get('pid')} tid={ev.get('tid')} ts={ev.get('ts')} "
              f"dur={ev.get('dur')} args_keys={sorted(list((ev.get('args') or {}).keys()))[:5]}")
        x_count += 1
        if x_count >= limit:
            break

    print(f"\n-- sample 'i' (instant) events (showing first {limit}) --")
    i_count = 0
    for ev in events:
        if ev.get("ph") != "i":
            continue
        print(f"  name={ev.get('name')!r} pid={ev.get('pid')} tid={ev.get('tid')} ts={ev.get('ts')}")
        i_count += 1
        if i_count >= limit:
            break

    # 1M-event truncation warning
    if len(events) >= 999_000:
        print("\nWARNING: traceEvents close to or at the 1M cap — file may be truncated.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
