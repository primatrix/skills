# tpu-perf: profile-anatomy (basic skill) — Design

Date: 2026-05-24
Status: Draft (pending spec review)

## Background

`tpu-perf` will be a new collection of Agent Skills that systematically
analyzes the efficiency of TPU pretraining runs and identifies optimization
points. This document specifies the **first** skill in the collection,
`profile-anatomy`, which is intentionally a *basic / reference* skill: it
documents what a TPU profile directory contains and how to read it.

Future tpu-perf skills (MFU calculation, communication-overlap analysis,
HBM pressure, etc.) will assume the schema knowledge in this skill, so
`profile-anatomy` must be authoritative and complete for the file types it
covers.

A separate `xprof-profiling-analysis` plugin already exists; it is built
around an XProf MCP server and provides high-level analysis workflows.
`tpu-perf` is **independent** of that plugin and focuses on offline,
direct-protobuf-and-JSON parsing of the on-disk profile artifacts.

The previously-existing `tpu-perf-model` plugin (theoretical TPU v7x
performance modeling) was removed in commit `4f4faa2`; the `tpu-perf`
namespace is therefore unused and free for this new collection.

## Scope

### In scope (this skill)

- A new plugin `tpu-perf` with one skill `profile-anatomy`.
- A `SKILL.md` that describes:
  - The on-disk layout of a TPU profile directory.
  - The `XSpace → XPlane → XLine → XEvent → XStat` hierarchy of
    `xplane.pb`.
  - The top-level structure of `trace.json.gz` (Chrome trace JSON).
  - Common gotchas when reading these files.
- Seven Python reference scripts under `scripts/` that each demonstrate
  one slice of the schema by parsing a real profile directory.

### Out of scope (this skill)

- Any analysis or interpretation (MFU, roofline, communication exposure,
  HBM diagnostics) — those belong to future tpu-perf skills.
- `*.hlo_proto.pb` parsing. The HLO proto files are not covered; the
  SKILL.md does not even mention them.
- `memory_viewer_preprocess`, `op_stats`, `pod_viewer`, `kernel_stats`
  and other xprof-derived artifacts.
- Unit tests. The scripts are themselves demos; running them on a real
  profile is the test.
- Any change to the existing `xprof-profiling-analysis` plugin.

## File layout

```
plugins/tpu-perf/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── profile-anatomy/
        ├── SKILL.md
        └── scripts/
            ├── walk_xplane.py
            ├── dump_xplane_metadata.py
            ├── extract_step_events.py
            ├── extract_hlo_events.py
            ├── extract_framework_ops.py
            ├── extract_collective_events.py
            └── read_trace_json.py
```

`marketplace.json` is updated to register the new plugin.

## plugin.json

```json
{
  "name": "tpu-perf",
  "description": "Systematic analysis of TPU pretraining efficiency. Starts with profile-anatomy: schema dictionary and reference scripts for xplane.pb / trace.json.gz.",
  "version": "0.1.0",
  "license": "Apache-2.0"
}
```

`version` is `0.1.0` because this is the first skill in a planned
collection; major version 1.0 is reserved for when the collection is
considered feature-complete.

## marketplace.json entry

Append to the end of the `plugins` array in
`.claude-plugin/marketplace.json`:

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

`homepage` and `repository` fields are intentionally omitted to match the
style of every existing entry in the marketplace.

## SKILL.md structure

Frontmatter:

```yaml
---
name: profile-anatomy
description: Use when reading TPU pretraining profiles (xplane.pb, trace.json.gz) — describes the on-disk layout, the XSpace/XPlane/XLine/XEvent/XStat hierarchy, and provides reference scripts that future tpu-perf skills can read as schema documentation.
---
```

Body sections (each scaled to its complexity):

1. **What's in a profile directory** — table of files actually found
   (xplane.pb, trace.json.gz). One short paragraph each: what it is,
   when to use it, when to avoid it. (HLO proto files are not listed.)
2. **xplane.pb schema** — explains the five-level hierarchy
   (`XSpace → XPlane → XLine → XEvent → XStat`). For each level: what
   it is, key fields, which reference script demonstrates reading it.
3. **trace.json.gz schema** — `displayTimeUnit`, `metadata`,
   `traceEvents[]` with `ph=M/X/i`; pid/tid naming via metadata events;
   the 1M-event truncation caveat.
4. **Reference scripts (index)** — one line per script: invocation,
   what schema it shows, sample-output snippet.
5. **Common gotchas** — protobuf parsing requirement, `XStat` oneof,
   picosecond vs nanosecond units across planes, 1M-event truncation,
   "TC Overlay" being a derived line, etc.

## Reference scripts

All seven scripts follow a uniform contract:

- **Entry point**: `def main(profile_dir: str, limit: int = 20) -> None`
- **CLI usage**: `python <script>.py <profile_dir>` (under
  `if __name__ == "__main__":`).
- **Top-of-file module docstring** must contain three labelled blocks:
  - `Schema shown:` — the XPlane/line/stat slice illustrated.
  - `Fields illustrated:` — explicit list of fields by name.
  - `Source proto:` — fully-qualified protobuf type path
    (e.g. `xprof.protobuf.xplane_pb2.XPlane.lines`).
- **Field naming**: print proto field names verbatim (`offset_ps`, not
  `start_us`) so the reader can map output back to the `.proto` file.
- **Graceful absence**: when the expected plane/line is not present in
  the input directory, print `[absent]` and return — never raise. This
  is verified by running scripts #1, #3, #6 against `dp4_fsdp16/`,
  which has fewer artifacts than `dp8_fsdp128/`.
- **Dependencies**: only `xprof.protobuf.xplane_pb2`, `gzip`, `json`,
  `sys`, `pathlib` from stdlib. No tensorflow, no extra installs.

| # | Script                          | Schema shown                                                                                                                                                                                                                |
|---|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `walk_xplane.py`                | The full five-level tree `XSpace → planes → lines → events → stats`, indented; gives a first-look overview.                                                                                                                 |
| 2 | `dump_xplane_metadata.py`       | The two reverse-lookup tables on each plane: `event_metadata{id → name/description}` and `stat_metadata{id → name/value_type}`. These are how every `XEvent.metadata_id` and `XStat.metadata_id` resolve to a human name.   |
| 3 | `extract_step_events.py`        | Device plane's `"Steps"` line: per-step XEvents with `offset_ps`, `duration_ps`, and the `step_num` stat. Source for "how long is one step" / "how many steps did we capture".                                              |
| 4 | `extract_hlo_events.py`         | `"XLA Ops"` line on the device plane: HLO-level XEvents with stats `hlo_category`, `tf_op`, `program_id`, `hlo_module_id`, `flops`, `bytes_accessed`, `occupancy_pct`.                                                      |
| 5 | `extract_framework_ops.py`      | `/host:CPU` plane: framework-op XEvents (JAX/XLA Python-side calls) with stats like `long_name`, `source`, `is_eager`, `producer`/`consumer` IDs.                                                                           |
| 6 | `extract_collective_events.py`  | `"Async XLA Ops"` line: paired `*-start` / `*-done` async events with `is_root` and `hlo_category ∈ {all-reduce, all-gather, reduce-scatter, all-to-all}`. Highlights that **`async-done` `duration_ps` = exposed comm stall** — the core communication-cost metric. |
| 7 | `read_trace_json.py`            | `trace.json.gz` top-level: `displayTimeUnit`, `metadata`, `traceEvents[]`. Resolves `pid → process_name` and `tid → thread_name` from `ph=M` metadata events; samples a few `ph=X` (complete) and `ph=i` (instant) events to show their fields.                       |

## Verification (must run before declaring complete)

1. `python -m json.tool plugins/tpu-perf/.claude-plugin/plugin.json`
   exits 0.
2. `python -m json.tool .claude-plugin/marketplace.json` exits 0.
3. `SKILL.md` frontmatter parses as valid YAML (two fences, two
   fields).
4. Every script runs end-to-end on
   `/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`
   without raising and produces non-empty output.
5. Scripts #1, #3, #6 also run end-to-end on
   `/Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16/` (no
   `hlo_proto.pb`, slightly different content) without raising —
   verifying the "graceful absence" contract.

## Submission

Single commit on a feature branch (already in a worktree). Files
touched:

- `plugins/tpu-perf/.claude-plugin/plugin.json` (new)
- `plugins/tpu-perf/skills/profile-anatomy/SKILL.md` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/walk_xplane.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/dump_xplane_metadata.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_step_events.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_hlo_events.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_framework_ops.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/extract_collective_events.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/read_trace_json.py` (new)
- `.claude-plugin/marketplace.json` (one entry appended)

## Design rationale notes

- **Why a separate plugin and not a sub-skill of `xprof-profiling-analysis`?**
  `xprof-profiling-analysis` is built around an MCP server and presupposes
  a running XProf instance. `tpu-perf` is offline-first: scripts read
  files directly via `xplane_pb2`. Mixing the two would force readers to
  understand both worlds at once. Keeping them separate also lets
  `tpu-perf` evolve without disturbing the existing MCP workflow.

- **Why "scripts as documentation" instead of a callable library?**
  The user explicitly framed these scripts as *schema records*, not
  reusable utilities. Future tpu-perf skills will read these scripts to
  learn the field layout, then write their own analysis code on top of
  `xplane_pb2` directly. A single-script-per-topic layout (vs one big
  `reference.py`) means a future skill can `Read` exactly the slice it
  needs without skimming a 600-line file.

- **Why not cover HLO proto files?**
  The user removed them from scope. They're a substantially different
  proto schema (`HloModuleProto`, buffer assignment) and warrant a
  dedicated future skill rather than being squeezed into the basic one.
