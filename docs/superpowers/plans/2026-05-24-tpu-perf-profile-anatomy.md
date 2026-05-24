# tpu-perf profile-anatomy Skill Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first skill of the new `tpu-perf` plugin — `profile-anatomy` — which documents the on-disk schema of TPU pretraining profiles (`xplane.pb`, `trace.json.gz`) and ships seven Python reference scripts that demonstrate, by direct execution, how to read each slice of the schema.

**Architecture:** Static skill (no runtime services). One plugin manifest, one `SKILL.md` (the schema dictionary), one vendored protobuf module (`xplane_pb2.py` + its `.proto` source under `scripts/_proto/`), and seven small standalone Python scripts each acting as executable schema documentation. Marketplace is updated to register the new plugin.

**Tech Stack:** Python 3 stdlib (`gzip`, `json`, `pathlib`, `sys`); `protobuf` runtime (already on the system, transitively via `xprof`); the upstream `tensorflow.profiler` xplane proto schema, vendored as a single generated `xplane_pb2.py`.

---

## Source spec

`docs/superpowers/specs/2026-05-24-tpu-perf-profile-anatomy-design.md` (round 2, reviewer-approved). Anything that conflicts between this plan and the spec — the spec wins; correct the plan before proceeding.

## Sample profile directories used by every test

- **Full profile** (used by every script's primary verification): `/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`
  - Contains `gke-tpu-4233cc6e-d8q7.xplane.pb`, `gke-tpu-4233cc6e-d8q7.trace.json.gz`, plus three `*.hlo_proto.pb` files (which we ignore).
- **Reduced profile** (used to verify the "graceful absence" contract for scripts #1, #3, #6): `/Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16/`
  - Contains only `gke-tpu-74a3f8a5-4kzv.xplane.pb` and `gke-tpu-74a3f8a5-4kzv.trace.json.gz`.

These directories are external to the repo. Treat them as read-only fixtures; never copy them into the repo.

## File structure (locked)

Every file added/modified by this plan:

```
plugins/tpu-perf/
├── .claude-plugin/
│   └── plugin.json                                              # NEW
└── skills/
    └── profile-anatomy/
        ├── SKILL.md                                             # NEW
        └── scripts/
            ├── _proto/
            │   ├── __init__.py                                  # NEW (empty)
            │   ├── xplane.proto                                 # NEW (vendored, verbatim)
            │   └── xplane_pb2.py                                # NEW (vendored, generated)
            ├── walk_xplane.py                                   # NEW
            ├── dump_xplane_metadata.py                          # NEW
            ├── extract_step_events.py                           # NEW
            ├── extract_hlo_events.py                            # NEW
            ├── extract_framework_ops.py                         # NEW
            ├── extract_collective_events.py                     # NEW
            └── read_trace_json.py                               # NEW

.claude-plugin/marketplace.json                                  # MODIFIED (+1 plugin entry)
```

Total: 13 new files, 1 modified.

## Per-script uniform contract

Every one of the 7 reference scripts (`walk_xplane.py`, `dump_xplane_metadata.py`, `extract_step_events.py`, `extract_hlo_events.py`, `extract_framework_ops.py`, `extract_collective_events.py`, `read_trace_json.py`) shares this contract:

1. **Module docstring** at the top with three labelled blocks:
   ```python
   """
   <one-line summary>

   Schema shown:
       <which slice of XSpace/XPlane/XLine/XEvent/XStat or trace.json this illustrates>

   Fields illustrated:
       <explicit list of proto field names or JSON keys, comma-separated>

   Source proto:
       _proto/xplane_pb2.<MessageType>.<field>
       (or "n/a — JSON only" for read_trace_json.py)
   """
   ```
2. **Boilerplate after the docstring** (xplane scripts only — `read_trace_json.py` skips the `xplane_pb2` import):
   ```python
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
   import xplane_pb2  # noqa: E402
   ```
3. **Entry point:** `def main(profile_dir: str, limit: int = 20) -> None`. The `limit` parameter caps how many events / stats are printed when there are many; it is a print cap, not a filter on what is loaded.
4. **CLI:**
   ```python
   if __name__ == "__main__":
       main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
   ```
5. **Graceful absence:** if the script needs a specific plane (e.g., `/device:TPU:0`) or line (e.g., `"Steps"`) and it is missing from the input, print exactly `[absent]` and `return` — do not raise. This contract is exercised by scripts #1, #3, #6 against the reduced fixture `dp4_fsdp16/`.
6. **Field naming in output:** print proto field names verbatim (`offset_ps`, not `start_us`). The reader of the output should be able to grep the source `.proto` file for any name they see in the output.
7. **No external dependencies beyond `protobuf` + stdlib.** Specifically: do **not** import `tensorflow`, `xprof`, or `tensorboard*`. The `protobuf` runtime is satisfied by whatever `xprof` already depends on — verified working on this machine.

---

## Chunk 1: Plugin scaffold + vendored proto

This chunk gets the plugin registered and proves we can parse an xplane file before any of the seven scripts exist.

### Task 1: Register the new plugin in marketplace.json

**Files:**
- Modify: `.claude-plugin/marketplace.json` (append one entry to the `plugins` array)

- [ ] **Step 1: Read the current marketplace.json**

  Run: open `.claude-plugin/marketplace.json` and confirm it ends with the `agent-recap` entry (closing brace of that object on the line before `]`).

- [ ] **Step 2: Append the new plugin entry**

  Insert this object as the last element of the `plugins` array. Add a trailing comma to the previous last entry (`agent-recap`) so the JSON remains valid.

  ```json
  {
    "name": "tpu-perf",
    "source": "./plugins/tpu-perf",
    "description": "Systematic TPU pretraining efficiency analysis — profile schema reference and (future) optimization-point detection skills",
    "version": "0.1.0",
    "license": "Apache-2.0",
    "keywords": ["tpu", "profiling", "xplane", "pretraining", "performance"],
    "category": "performance"
  }
  ```

- [ ] **Step 3: Validate JSON**

  Run: `python3 -m json.tool .claude-plugin/marketplace.json > /dev/null && echo OK`
  Expected: `OK` (zero stderr, zero exit).

- [ ] **Step 4: Commit**

  ```bash
  git add .claude-plugin/marketplace.json
  git commit -m "feat(tpu-perf): register new plugin in marketplace"
  ```

### Task 2: Create plugin manifest

**Files:**
- Create: `plugins/tpu-perf/.claude-plugin/plugin.json`

- [ ] **Step 1: Create the directory**

  ```bash
  mkdir -p plugins/tpu-perf/.claude-plugin
  ```

- [ ] **Step 2: Write the manifest**

  Write to `plugins/tpu-perf/.claude-plugin/plugin.json`:

  ```json
  {
    "name": "tpu-perf",
    "description": "Systematic analysis of TPU pretraining efficiency. Starts with profile-anatomy: schema dictionary and reference scripts for xplane.pb / trace.json.gz.",
    "version": "0.1.0",
    "license": "Apache-2.0"
  }
  ```

- [ ] **Step 3: Validate JSON**

  Run: `python3 -m json.tool plugins/tpu-perf/.claude-plugin/plugin.json > /dev/null && echo OK`
  Expected: `OK`.

- [ ] **Step 4: Commit**

  ```bash
  git add plugins/tpu-perf/.claude-plugin/plugin.json
  git commit -m "feat(tpu-perf): add plugin manifest"
  ```

### Task 3: Vendor xplane.proto source

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/__init__.py`
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto`

The `.proto` source already exists on this machine at `/Users/xl/Code/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto` (from the `openxla/xla` checkout). Copy it verbatim — keep the package, syntax, license header, comments, and field numbers exactly as they are. Do not edit it.

- [ ] **Step 1: Create the directory and empty package marker**

  ```bash
  mkdir -p plugins/tpu-perf/skills/profile-anatomy/scripts/_proto
  : > plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/__init__.py
  ```

- [ ] **Step 2: Copy the upstream .proto verbatim**

  ```bash
  cp /Users/xl/Code/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto \
     plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto
  ```

- [ ] **Step 3: Sanity-check the copy**

  Run: `diff -q /Users/xl/Code/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto`
  Expected: empty output (files identical).

  Run: `grep -c '^import' plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto`
  Expected: `0` (no transitive imports — single-file vendoring is valid).

- [ ] **Step 4: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/__init__.py \
          plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto
  git commit -m "feat(tpu-perf): vendor upstream xplane.proto for profile-anatomy"
  ```

### Task 4: Vendor a generated xplane_pb2.py

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py`

`protoc` is **not installed** on this machine, so do not try to run it. A pre-generated `xplane_pb2.py` ships inside an unrelated venv at
`/Users/xl/Code/ant-pretrain/.venv/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/xplane_pb2.py`. It was confirmed during planning that this file:
- Is 57 lines, generated from the same `xplane.proto` we vendored.
- Imports only `google.protobuf.*` (no package-relative imports), so it's a drop-in.
- Successfully parses `dp8_fsdp128/gke-tpu-4233cc6e-d8q7.xplane.pb` (8 planes, 101 lines).

Copy it verbatim — the auto-generated header `# Generated by the protocol buffer compiler. DO NOT EDIT!` should remain.

- [ ] **Step 1: Copy the generated file verbatim**

  ```bash
  cp /Users/xl/Code/ant-pretrain/.venv/lib/python3.12/site-packages/tensorflow/tsl/profiler/protobuf/xplane_pb2.py \
     plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py
  ```

- [ ] **Step 2: Confirm it imports standalone**

  Run:
  ```bash
  python3 -c "
  import sys
  sys.path.insert(0, 'plugins/tpu-perf/skills/profile-anatomy/scripts/_proto')
  import xplane_pb2
  xs = xplane_pb2.XSpace()
  print('XSpace fields:', sorted(f.name for f in xs.DESCRIPTOR.fields))
  "
  ```
  Expected: `XSpace fields: ['errors', 'hostnames', 'planes', 'warnings']` (set-equal to {planes, errors, warnings, hostnames}; ordering not significant).

- [ ] **Step 3: Confirm it parses the real fixture**

  Run:
  ```bash
  python3 -c "
  import sys
  sys.path.insert(0, 'plugins/tpu-perf/skills/profile-anatomy/scripts/_proto')
  import xplane_pb2
  xs = xplane_pb2.XSpace()
  with open('/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/gke-tpu-4233cc6e-d8q7.xplane.pb', 'rb') as f:
      xs.ParseFromString(f.read())
  print('planes:', [p.name for p in xs.planes])
  "
  ```
  Expected: a list including at least `/device:TPU:0`, `/host:CPU`, `/host:metadata`. (The exact full list observed in planning: `/host:metadata`, `/device:TPU:0`, `/device:TPU:1`, `/device:TPU:0 SparseCore 0`, `/device:TPU:0 SparseCore 1`, `/device:CUSTOM:Megascale Trace`, `/host:CPU`, `Task Environment`.)

- [ ] **Step 4: Add a one-line comment block at the very top recording the source**

  Prepend (above the existing auto-generated header) these 5 lines, then save:

  ```python
  # Vendored from upstream tensorflow.tsl.profiler.protobuf.xplane_pb2.
  # Source .proto: ./xplane.proto (also vendored in this directory).
  # To regenerate: `protoc --python_out=. xplane.proto` from this directory.
  # Do NOT edit by hand — regenerate from xplane.proto if the schema changes.
  #
  ```

  After the edit, re-run Step 2 to confirm it still imports cleanly.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py
  git commit -m "feat(tpu-perf): vendor generated xplane_pb2.py for profile-anatomy"
  ```

---

## Chunk 2: Reference scripts (all 7)

Each script is small (≤80 lines including docstring). Each is committed individually so a reviewer can isolate what changed per script. The TDD cycle here is unusual: the "test" is `python <script>.py <fixture>` itself — there is no pytest, because these scripts are executable schema documentation, not library code (this matches the spec's explicit decision to not write unit tests).

The order below is dependency-friendly: `walk_xplane.py` first (broadest, lets us discover plane/line names empirically before writing the more focused scripts).

### Task 5: walk_xplane.py — full XSpace tree dump

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py`

**Schema illustrated:** `XSpace → XPlane → XLine → XEvent → XStat`, the full five-level proto tree, indented.

**Fields illustrated** (per spec, "Authoritative schema content" section): `XSpace.{planes, errors, warnings, hostnames}`, `XPlane.{id, name, lines, event_metadata, stat_metadata, stats}`, `XLine.{id, name, timestamp_ns, duration_ps, events}`, `XEvent.{metadata_id, offset_ps, num_occurrences, duration_ps, stats}`, `XStat.{metadata_id, value oneof}`.

- [ ] **Step 1: Find the first xplane.pb in the directory**

  The fixture file name varies (`gke-tpu-4233cc6e-d8q7.xplane.pb` for dp8_fsdp128, `gke-tpu-74a3f8a5-4kzv.xplane.pb` for dp4_fsdp16). Use `pathlib.Path(profile_dir).glob("*.xplane.pb")`, take the first match, print `[absent]` and return if none.

- [ ] **Step 2: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py`:

  ```python
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
  ```

- [ ] **Step 3: Run on dp8_fsdp128 — must print non-empty tree**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128`
  Expected: at least one `XPlane` line whose name is `/device:TPU:0`, with nested `XLine`s including `Steps`, `XLA Modules`, `XLA Ops`, `Async XLA Ops`. Exit 0.

- [ ] **Step 4: Run on dp4_fsdp16 — must also work (graceful absence contract)**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16`
  Expected: prints a tree (this fixture also has an xplane.pb), exit 0.

- [ ] **Step 5: Run on a directory with no xplane.pb — must print [absent] and exit 0**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py /tmp`
  Expected: `[absent] no *.xplane.pb in /tmp`, exit 0 (no traceback).

- [ ] **Step 6: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py
  git commit -m "feat(tpu-perf): add walk_xplane.py reference script"
  ```

### Task 6: dump_xplane_metadata.py — event_metadata & stat_metadata tables

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py`

**Schema illustrated:** the two reverse-lookup tables on each plane: `event_metadata{id → (name, display_name, child_id)}` and `stat_metadata{id → (name, description)}`.

- [ ] **Step 1: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py`:

  ```python
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
  ```

- [ ] **Step 2: Run on dp8_fsdp128 and verify expected stat names appear**

  Run:
  ```bash
  python3 plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 100 2>&1 | \
    grep -E "name='(flops|hlo_category|hlo_op|tf_op|flow|device_duration_ps)'" | head -10
  ```
  Expected: at least 6 grep matches (one per stat name listed). The `100` argument is the limit; the script's `main` accepts a positional `limit` if invoked from Python — for CLI just rely on the default and show the first matches.

  If the limit-via-CLI is wanted, simply re-run with `python3 -c "import sys; sys.path.insert(0,'plugins/tpu-perf/skills/profile-anatomy/scripts'); from dump_xplane_metadata import main; main(sys.argv[1], 100)" <fixture>`.

- [ ] **Step 3: Run on dp4_fsdp16 — non-empty output, exit 0**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16`
  Expected: prints metadata tables for the planes that exist in this profile.

- [ ] **Step 4: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py
  git commit -m "feat(tpu-perf): add dump_xplane_metadata.py reference script"
  ```

### Task 7: extract_step_events.py — Steps line on the device plane

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py`

**Schema illustrated:** the `"Steps"` line on the `/device:TPU:0` plane (or any plane name beginning with `/device:`). Each XEvent in this line is one training step. Useful for "how long is one step", "how many steps did we capture".

- [ ] **Step 1: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py`:

  ```python
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
  ```

- [ ] **Step 2: Run on dp8_fsdp128 — must list at least one step**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128`
  Expected: a header line `Plane '/device:TPU:0' ... events=N` with N ≥ 1, followed by per-step entries.

- [ ] **Step 3: Run on dp4_fsdp16 — gracefully handles whatever it has**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16`
  Expected: either prints step events, or prints `[absent] plane ... has no 'Steps' line ...`. Exit 0 either way; no traceback.

- [ ] **Step 4: Run on /tmp — must print [absent] and exit 0**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py /tmp`
  Expected: `[absent] no *.xplane.pb in /tmp`, exit 0.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py
  git commit -m "feat(tpu-perf): add extract_step_events.py reference script"
  ```

### Task 8: extract_hlo_events.py — XLA Ops line on the device plane

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_hlo_events.py`

**Schema illustrated:** `"XLA Ops"` line on `/device:TPU:0`. Each event is one HLO operation; the rich metadata is in its stats.

**Stat names this script knows are present** (verified during planning by direct parse of `dp8_fsdp128`): `hlo_category`, `hlo_op`, `tf_op`, `program_id`, `flops`, `model_flops`, `bytes_accessed`, `raw_bytes_accessed`, `shape_with_layout`. The script must handle the case where any of them is missing on a particular event (just skip that field in the print).

- [ ] **Step 1: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_hlo_events.py`:

  ```python
  """
  Extract HLO-op-level XEvents from the device plane's "XLA Ops" line.

  Each event is a single HLO operation execution. The interesting payload
  is in XEvent.stats, which is keyed via XPlane.stat_metadata. This script
  illustrates the most informative stat names commonly found on HLO events.

  Schema shown:
      XPlane(name startswith '/device:') -> XLine(name='XLA Ops') -> XEvent
      XEvent.stats -> XStat (resolved via XPlane.stat_metadata).

  Fields illustrated:
      XEvent.{metadata_id, offset_ps, duration_ps, stats}
      XStat.{metadata_id, value oneof}
      Stat names actually present in dp8_fsdp128 we look for:
          hlo_category, hlo_op, tf_op, program_id, flops, model_flops,
          bytes_accessed, raw_bytes_accessed, shape_with_layout.

  Source proto:
      _proto/xplane_pb2.XLine.events,
      _proto/xplane_pb2.XEvent.stats
  """
  import sys
  import pathlib

  sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
  import xplane_pb2  # noqa: E402


  INTERESTING_STATS = (
      "hlo_category", "hlo_op", "tf_op", "program_id",
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
          ev_name = device_plane.event_metadata[ev.metadata_id].name if ev.metadata_id in device_plane.event_metadata else "?"
          stats = {stat_name_by_id.get(s.metadata_id, "?"): _stat_value(s) for s in ev.stats}
          shown = {k: stats[k] for k in INTERESTING_STATS if k in stats}
          print(f"  event metadata.name={ev_name[:60]!r} duration_ps={ev.duration_ps}")
          for k, (vf, vv) in shown.items():
              print(f"    {k}: ({vf}) {vv!r}")
      if len(ops_line.events) > limit:
          print(f"  ... ({len(ops_line.events) - limit} more)")


  if __name__ == "__main__":
      main(sys.argv[1] if len(sys.argv) > 1 else "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
  ```

- [ ] **Step 2: Run on dp8_fsdp128 — must print at least one event with `hlo_category` and `hlo_op` stats**

  Run:
  ```bash
  python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_hlo_events.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 | \
    head -50
  ```
  Expected: at least one event block has `hlo_category:` and `hlo_op:` lines underneath.

- [ ] **Step 3: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/extract_hlo_events.py
  git commit -m "feat(tpu-perf): add extract_hlo_events.py reference script"
  ```

### Task 9: extract_framework_ops.py — host plane (`/host:CPU`) framework events

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_framework_ops.py`

**Schema illustrated:** events on the `/host:CPU` plane. These are JAX/XLA Python-side framework calls. Stat names on the host plane vary by generator — do **not** assume a fixed list. Print every stat we see, sorted by name, for the first `limit` events.

- [ ] **Step 1: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_framework_ops.py`:

  ```python
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
  ```

- [ ] **Step 2: Run on dp8_fsdp128 — must print at least one host event**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_framework_ops.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128`
  Expected: header `Plane '/host:CPU' ...` followed by at least one `event metadata.name=...` block.

- [ ] **Step 3: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/extract_framework_ops.py
  git commit -m "feat(tpu-perf): add extract_framework_ops.py reference script"
  ```

### Task 10: extract_collective_events.py — Async XLA Ops, async-start ↔ async-done pairing

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py`

**Schema illustrated:** `"Async XLA Ops"` line on `/device:TPU:0`. Async events come in matched pairs (`*-start` / `*-done`) connected by the `flow` stat (uint64 flow id). The `*-done` event's `duration_ps` (or its `device_duration_ps` stat — same value, different access path) is the **exposed communication stall**.

**Stat names we know exist** (verified during planning): `flow`, `device_offset_ps`, `device_duration_ps`, `hlo_op`, `id`. We must not invent `is_root` or `occupancy_pct`.

- [ ] **Step 1: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py`:

  ```python
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
  ```

- [ ] **Step 2: Run on dp8_fsdp128 — must show at least one flow with pair_size ≥ 1**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 | head -30`
  Expected: at least one `flow=<int>  pair_size=...` line, exit 0.

- [ ] **Step 3: Run on dp4_fsdp16 — graceful (either lists flows or `[absent]`)**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16`
  Expected: exit 0, no traceback.

- [ ] **Step 4: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py
  git commit -m "feat(tpu-perf): add extract_collective_events.py reference script"
  ```

### Task 11: read_trace_json.py — Chrome trace JSON top-level

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py`

**Schema illustrated:** `trace.json.gz` top-level fields (`displayTimeUnit`, `metadata`, `traceEvents[]` with phases `M`, `X`, `i`); pid/tid resolution via `ph='M'` `process_name` / `thread_name` events; sampling of complete (`X`) and instant (`i`) events.

- [ ] **Step 1: Write the script**

  Write to `plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py`:

  ```python
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
  ```

- [ ] **Step 2: Run on dp8_fsdp128 — non-empty pid/tid map**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128`
  Expected: prints `top-level keys: ['displayTimeUnit', ...]`, a non-empty pid map, and at least one `X` event sample.

- [ ] **Step 3: Run on dp4_fsdp16 — also works**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py /Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16`
  Expected: similar output, exit 0.

- [ ] **Step 4: Run on /tmp — `[absent]` and exit 0**

  Run: `python3 plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py /tmp`
  Expected: `[absent] no *.trace.json.gz in /tmp`, exit 0.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py
  git commit -m "feat(tpu-perf): add read_trace_json.py reference script"
  ```

---

## Chunk 3: SKILL.md and final verification

### Task 12: Write SKILL.md

**Files:**
- Create: `plugins/tpu-perf/skills/profile-anatomy/SKILL.md`

This is the schema dictionary. It cites only proto fields and stat names that the upstream `xplane.proto` defines or that we have empirically observed in `dp8_fsdp128`. **It does not mention HLO proto files at all** (per spec, out of scope).

- [ ] **Step 1: Write the file**

  Write to `plugins/tpu-perf/skills/profile-anatomy/SKILL.md`:

  ````markdown
  ---
  name: profile-anatomy
  description: Use when reading TPU pretraining profiles (xplane.pb, trace.json.gz) — describes the on-disk layout, the XSpace/XPlane/XLine/XEvent/XStat hierarchy, and provides reference scripts that future tpu-perf skills can read as schema documentation.
  ---

  # Profile Anatomy

  Reference for what's inside a TPU pretraining profile directory and how
  to parse each artifact. This skill is **schema documentation**, not an
  analysis tool — it answers "what's in here and what does each field
  mean", not "is my training fast".

  Future `tpu-perf` skills (MFU, comm overlap, HBM pressure, …) build on
  the schema described here.

  ## 1. What's in a profile directory

  A typical capture (e.g. `/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`)
  contains:

  | File pattern | What it is | When to read it |
  |---|---|---|
  | `*.xplane.pb` | The authoritative protobuf trace. Contains all profiled hosts and devices, all events, all metadata. | Whenever you need anything reliable. This is the source of truth. |
  | `*.trace.json.gz` | Chrome-trace-format JSON gzipped. A flattened, browser-viewable export of the same data, **capped at ~1M events**. | Quick browser inspection, manual scripts that don't need every event. Do not use it for total-time accounting if the cap was hit. |

  ## 2. xplane.pb schema

  Five-level proto tree, defined in [`scripts/_proto/xplane.proto`](scripts/_proto/xplane.proto). Quote of the field shape:

  - **`XSpace`** (top-level container)
    - `repeated XPlane planes`
    - `repeated string errors`
    - `repeated string warnings`
    - `repeated string hostnames`

  - **`XPlane`** (one timeline source — a host, a device, a metadata
    plane)
    - `int64 id`
    - `string name` (e.g. `"/device:TPU:0"`, `"/host:CPU"`,
      `"/device:CUSTOM:Megascale Trace"`, `"Task Environment"`)
    - `repeated XLine lines`
    - `map<int64, XEventMetadata> event_metadata` — every
      `XEvent.metadata_id` resolves through this map.
    - `map<int64, XStatMetadata> stat_metadata` — every
      `XStat.metadata_id` resolves through this map.
    - `repeated XStat stats` — plane-level stats (e.g. device
      capabilities).

  - **`XLine`** (a single timeline within a plane — e.g. `"Steps"`,
    `"XLA Ops"`, `"Async XLA Ops"`)
    - `int64 id`, `int64 display_id`, `string name`,
      `string display_name`
    - `int64 timestamp_ns` — start of this line, **nanoseconds** since
      epoch. `XEvent.offset_ps` is **picoseconds** relative to this.
    - `int64 duration_ps`
    - `repeated XEvent events`
    - `reserved 5, 6, 7, 8`

  - **`XEvent`** (one event on a timeline)
    - `int64 metadata_id` → `event_metadata[metadata_id].name` for the
      human label
    - `oneof data { int64 offset_ps | int64 num_occurrences }` —
      `offset_ps` for normal events, `num_occurrences` for aggregated
      counts. Use `WhichOneof("data")` to discriminate.
    - `int64 duration_ps`
    - `repeated XStat stats`

  - **`XStat`** (a named value attached to an event or plane)
    - `int64 metadata_id` → `stat_metadata[metadata_id].name`
    - `oneof value` with **six** variants:
      `double_value`, `uint64_value`, `int64_value`, `str_value`,
      `bytes_value`, `ref_value`. Use `WhichOneof("value")` to discriminate.
      `ref_value` is a back-reference whose payload is stored in
      `XStatMetadata.name`.

  - **`XEventMetadata`** (shared metadata per event-type-id within a
    plane)
    - `int64 id`, `string name`, `string display_name`,
      `bytes metadata`, `repeated XStat stats`,
      `repeated int64 child_id`.

  - **`XStatMetadata`** (shared metadata per stat-type-id within a
    plane)
    - `int64 id`, `string name`, `string description`.
    - **No `value_type` field** — value type is determined per-XStat at
      the use site, via `WhichOneof("value")`.

  ### Real planes observed in `dp8_fsdp128`

  `/host:metadata`, `/device:TPU:0`, `/device:TPU:1`,
  `/device:TPU:0 SparseCore 0`, `/device:TPU:0 SparseCore 1`,
  `/device:CUSTOM:Megascale Trace`, `/host:CPU`, `Task Environment`.

  ### Real lines on `/device:TPU:0`

  `_counters_`, `Scalar Unit`, `Steps`, `XLA Modules`, `XLA Ops`,
  `Async XLA Ops`, `TC Overlay`, `XLA TraceMe`, `counters_0`.

  ### Stat-metadata names observed on `/device:TPU:0`

  Real stat names that show up in this fixture (81 total; non-exhaustive
  highlights):

  - **Compute / FLOPs**: `flops`, `model_flops`, `bytes_accessed`,
    `raw_bytes_accessed`,
    `peak_teraflops_per_second`,
    `peak_hbm_bw_gigabytes_per_second`,
    `peak_sram_rd_bw_gigabytes_per_second`,
    `peak_sram_wr_bw_gigabytes_per_second`,
    `peak_vmem_rd_bw_gigabytes_per_second`,
    `peak_vmem_wr_bw_gigabytes_per_second`,
    `peak_cmem_rd_bw_gigabytes_per_second`,
    `peak_cmem_wr_bw_gigabytes_per_second`.
  - **Op identity**: `hlo_category`, `hlo_op`, `tf_op`, `program_id`,
    `symbol_id`, `deduplicated_name`, `shape_with_layout`, `source`,
    `source_stack`.
  - **Async / collective**: `flow` (uint64 flow id used to pair
    `*-start` ↔ `*-done` events), `device_offset_ps`,
    `device_duration_ps`, `all_reduce_id`, `all_reduce_unique_id`,
    `dcn_collective_info`.
  - **Identity / topology**: `device_id`, `core_type`, `core_details`,
    `global_chip_id`, `process_id`, `replica_id`, `run_id`, `queue_id`.
  - **Counters & power**: `counter_value`, `% util`, `power`,
    `temperature`, `throttle %`, various `HBM FW *`, `VDD Core FW *`,
    `PCIe FW *`.

  Names you might **see in older docs but that are not present** in this
  capture: `is_root`, `occupancy_pct`. Don't write code that depends on
  them without first verifying via `dump_xplane_metadata.py`.

  ## 3. trace.json.gz schema

  After `gzip.open(...).read()` → `json.loads(...)`:

  ```json
  {
    "displayTimeUnit": "ns",
    "metadata": { "highres-ticks": true },
    "traceEvents": [ ...up to ~1,000,000 events... ]
  }
  ```

  Each event has a `ph` (phase) field:

  | `ph` | Meaning | Notable fields |
  |---|---|---|
  | `M` | Metadata. Names processes & threads. | `name` ∈ {`process_name`, `process_sort_index`, `thread_name`, `thread_sort_index`}, `args.name` |
  | `X` | Complete event (one start + one duration). | `name`, `cat`, `pid`, `tid`, `ts` (µs), `dur` (µs), `args` |
  | `i` | Instant event. | `name`, `pid`, `tid`, `ts` |
  | `B` / `E` | Paired begin/end (rare in TPU profiles). | matched by name within a tid |

  `pid` ↔ `XPlane.name` and `tid` ↔ `XLine.name` are established by the
  `M` events; you must scan all `ph='M'` events first to build the
  `pid → process_name` and `(pid, tid) → thread_name` maps before
  reading any `X`/`i` event.

  **Truncation caveat:** if `len(traceEvents)` is at the 1M cap, the
  trace is incomplete (events at the tail of capture were dropped). Do
  not compute totals from a truncated trace.

  ## 4. Reference scripts

  All seven scripts under [`scripts/`](scripts/) accept a profile
  directory as argv[1] and run standalone with stdlib + `protobuf`.
  They print `[absent]` and exit 0 (no traceback) when the slice they
  cover is missing.

  | Script | What it shows |
  |---|---|
  | [`walk_xplane.py`](scripts/walk_xplane.py) | Full `XSpace → planes → lines → events → stats` tree, indented. First-look overview. |
  | [`dump_xplane_metadata.py`](scripts/dump_xplane_metadata.py) | The `event_metadata{}` and `stat_metadata{}` reverse-lookup tables of every plane. |
  | [`extract_step_events.py`](scripts/extract_step_events.py) | Per-step events on the device plane's `"Steps"` line. |
  | [`extract_hlo_events.py`](scripts/extract_hlo_events.py) | HLO-level events on `"XLA Ops"` with `hlo_category`/`hlo_op`/`tf_op`/`flops`/`model_flops`/`bytes_accessed`/etc. |
  | [`extract_framework_ops.py`](scripts/extract_framework_ops.py) | `/host:CPU` framework events, with stat names discovered (not assumed). |
  | [`extract_collective_events.py`](scripts/extract_collective_events.py) | `"Async XLA Ops"` paired by the `flow` stat — measures exposed comm stall via `device_duration_ps` of `*-done` events. |
  | [`read_trace_json.py`](scripts/read_trace_json.py) | `trace.json.gz` top-level plus pid/tid name maps and sample `X`/`i` events. |

  ### Sample invocation

  ```bash
  python3 plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
  ```

  ## 5. Common gotchas

  - **Protobuf parsing is required.** `*.xplane.pb` is binary protobuf;
    you must use `xplane_pb2`. The vendored module is at
    `scripts/_proto/xplane_pb2.py` (regeneratable from the adjacent
    `xplane.proto`).
  - **`XStat.value` is a 6-variant oneof.** Always
    `WhichOneof("value")` first; never assume `int64_value`.
  - **Time units mix ns and ps.** `XLine.timestamp_ns` is nanoseconds
    since epoch; `XEvent.offset_ps` and `XEvent.duration_ps` are
    picoseconds **relative to that nanosecond timestamp**. Convert
    carefully.
  - **`trace.json.gz` may be truncated** at ~1M events. Do not compute
    totals from it; use `*.xplane.pb` for accurate counts.
  - **`"TC Overlay"` is a derived line**, not raw hardware events —
    don't double-count its events against `"XLA Ops"`.
  - **Async pairing uses `flow`, not `is_root`.** Don't write code that
    looks for an `is_root` stat — it doesn't exist in current captures.
  ````

- [ ] **Step 2: Validate frontmatter parses as YAML**

  Run:
  ```bash
  python3 -c "
  import re, sys
  src = open('plugins/tpu-perf/skills/profile-anatomy/SKILL.md').read()
  m = re.match(r'^---\n(.*?)\n---\n', src, re.S)
  assert m, 'no frontmatter fence'
  import yaml
  fm = yaml.safe_load(m.group(1))
  assert 'name' in fm and 'description' in fm, fm
  print('OK', fm['name'])
  "
  ```
  Expected: `OK profile-anatomy`. If `pyyaml` is not installed, instead
  visually verify the two `---` fences and the two fields.

- [ ] **Step 3: Commit**

  ```bash
  git add plugins/tpu-perf/skills/profile-anatomy/SKILL.md
  git commit -m "feat(tpu-perf): add profile-anatomy SKILL.md schema dictionary"
  ```

### Task 13: End-to-end verification

**Files:** none (verification only).

This task runs every script against both fixtures and confirms exit codes and key signal lines. It is the spec's "Verification" section turned into runnable form.

- [ ] **Step 1: Validate both JSON manifests**

  Run:
  ```bash
  python3 -m json.tool plugins/tpu-perf/.claude-plugin/plugin.json > /dev/null && \
  python3 -m json.tool .claude-plugin/marketplace.json > /dev/null && \
  echo "JSON OK"
  ```
  Expected: `JSON OK`.

- [ ] **Step 2: Self-test the vendored proto**

  Run:
  ```bash
  python3 -c "
  import sys
  sys.path.insert(0, 'plugins/tpu-perf/skills/profile-anatomy/scripts/_proto')
  import xplane_pb2
  xplane_pb2.XSpace()
  print('proto OK')
  "
  ```
  Expected: `proto OK`.

- [ ] **Step 3: Run every script on dp8_fsdp128 — every one must exit 0 with non-empty output**

  Run:
  ```bash
  set -e
  for s in plugins/tpu-perf/skills/profile-anatomy/scripts/*.py; do
    echo "=== $s on dp8_fsdp128 ==="
    out=$(python3 "$s" /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128)
    if [ -z "$out" ]; then
      echo "FAIL: $s produced empty output"
      exit 1
    fi
    echo "$out" | head -5
    echo "..."
  done
  echo "all 7 scripts OK on dp8_fsdp128"
  ```
  Expected: `all 7 scripts OK on dp8_fsdp128` at the end.

- [ ] **Step 4: Run scripts #1, #3, #6, #7 on dp4_fsdp16 — must exit 0 (graceful absence)**

  Run:
  ```bash
  set -e
  for s in walk_xplane.py extract_step_events.py extract_collective_events.py read_trace_json.py; do
    echo "=== $s on dp4_fsdp16 ==="
    python3 "plugins/tpu-perf/skills/profile-anatomy/scripts/$s" \
      /Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16 | head -3
  done
  echo "graceful-absence-on-reduced-fixture OK"
  ```
  Expected: `graceful-absence-on-reduced-fixture OK`.

- [ ] **Step 5: Run all 7 scripts on /tmp (no profile files) — must exit 0 with `[absent]` line**

  Run:
  ```bash
  set -e
  for s in plugins/tpu-perf/skills/profile-anatomy/scripts/*.py; do
    out=$(python3 "$s" /tmp)
    case "$out" in
      *"[absent]"*) ;;
      *) echo "FAIL: $s did not print [absent] on /tmp"; echo "$out"; exit 1 ;;
    esac
  done
  echo "graceful-absence-on-empty-dir OK"
  ```
  Expected: `graceful-absence-on-empty-dir OK`.

- [ ] **Step 6: Final tree check**

  Run:
  ```bash
  find plugins/tpu-perf -type f | sort
  ```
  Expected (exactly these 12 files):
  ```
  plugins/tpu-perf/.claude-plugin/plugin.json
  plugins/tpu-perf/skills/profile-anatomy/SKILL.md
  plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/__init__.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto
  plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/extract_framework_ops.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/extract_hlo_events.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py
  plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py
  ```
  (12 files total: `plugin.json`, `SKILL.md`, the 3 vendored proto files, and 7 scripts.)

  And `git diff --stat main..HEAD -- .claude-plugin/marketplace.json` should show `1 file changed, ~9 insertions(+)`.

- [ ] **Step 7: No commit (this is verification only).**

  If any of the prior steps fails, fix the underlying issue and re-run from the failing step. Do not advance to "open the PR" until every check is green.

---

## Out-of-scope reminders (do not silently expand)

- Do **not** add HLO proto parsing. The `*.hlo_proto.pb` files in
  `dp8_fsdp128/` are intentionally ignored.
- Do **not** add MFU, roofline, comm-overlap analysis, HBM diagnostics,
  or any other interpretation. This skill is schema-only.
- Do **not** add unit tests. The seven scripts are themselves the test
  suite; the verification task above runs them.
- Do **not** modify `plugins/xprof-profiling-analysis/` or any other
  existing plugin.
- Do **not** vendor any other proto file. `xplane.proto` has no transitive
  imports — verified during planning.
- Do **not** use `xprof.protobuf.xplane_pb2` — that module does **not**
  exist on this system; round-1 review caught this. Use the vendored
  `_proto/xplane_pb2`.

## Open-the-PR (only after Task 13 is fully green)

The user will probably want to open the PR via `/beaver-pr` or by hand.
This plan does not prescribe the PR step — leave the branch ready, with
all 11 commits clean (one per file/script + 1 for marketplace + 1 for
manifest), and stop.
