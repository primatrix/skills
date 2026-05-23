# tpu-perf: profile-anatomy (basic skill) — Design

Date: 2026-05-24
Status: Draft (round 2 — addresses spec-reviewer round 1 BLOCK on D3/D6)

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

## Proto module — resolved

Round-1 review correctly flagged that `xprof.protobuf.xplane_pb2` is
**not** importable on this system (the installed `xprof` 2.22.2 ships
its profile parser as a C++ shim `_pywrap_profiler_plugin`, not as a
Python proto module). The authoritative `.proto` source on disk is at
`/Users/xl/Code/xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto`
(package `tensorflow.profiler`); a generated module is shipped inside
the `tensorflow` pip package as
`tensorflow.tsl.profiler.protobuf.xplane_pb2`.

To avoid a hard dependency on the heavy `tensorflow` package, the skill
**vendors a single generated `xplane_pb2.py`** alongside its scripts:

```
plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/
├── __init__.py              # empty
├── xplane.proto             # exact copy of the upstream .proto, license header preserved
└── xplane_pb2.py            # protoc-generated; regeneratable from xplane.proto
```

All scripts import `from _proto import xplane_pb2` (relative to the
script's own directory, via a 1-line `sys.path` insertion at the top of
each script — see "Per-script contract" below).

The vendored proto is **single-file and self-contained**: `xplane.proto`
has no `import` statements (verified against the source), so no other
.proto files are needed.

The README-style top of `_proto/xplane_pb2.py` records the source SHA
and the regeneration command:

```text
# Generated from xplane.proto (sha256: <hash>) using protoc <version>.
# Regenerate:
#   protoc --python_out=. xplane.proto
```

This means the verification step "every script runs end-to-end" works
with **only stdlib + the `protobuf` Python package** (which `xprof` and
many other ML packages already pull in transitively). No tensorflow
install required.

## File layout

```
plugins/tpu-perf/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── profile-anatomy/
        ├── SKILL.md
        └── scripts/
            ├── _proto/
            │   ├── __init__.py
            │   ├── xplane.proto
            │   └── xplane_pb2.py
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
   (`XSpace → XPlane → XLine → XEvent → XStat`). For each level: the
   exact proto-defined fields, which reference script demonstrates
   reading it. Quotes the upstream `xplane.proto` field-by-field rather
   than paraphrasing.
3. **trace.json.gz schema** — `displayTimeUnit`, `metadata`,
   `traceEvents[]` with `ph=M/X/i`; pid/tid naming via metadata events;
   the 1M-event truncation caveat.
4. **Reference scripts (index)** — one line per script: invocation,
   what schema it shows, sample-output snippet.
5. **Common gotchas** — protobuf parsing requirement, `XStat.value`
   oneof shape (six variants), picosecond vs nanosecond convention,
   1M-event truncation, "TC Overlay" being a derived line, etc.

### Authoritative schema content (must match exactly)

Section 2 of the SKILL.md must reproduce the field lists below
verbatim, since these are the canonical proto definitions and any
deviation would mislead future skills:

- **`XSpace`**: `repeated XPlane planes`, `repeated string errors`,
  `repeated string warnings`, `repeated string hostnames`.
- **`XPlane`**: `int64 id`, `string name`, `repeated XLine lines`,
  `map<int64, XEventMetadata> event_metadata`, `map<int64, XStatMetadata>
  stat_metadata`, `repeated XStat stats`.
- **`XLine`**: `int64 id`, `int64 display_id`, `string name`, `string
  display_name`, `int64 timestamp_ns` (start of line, ns since epoch),
  `int64 duration_ps`, `repeated XEvent events`. (Field 5–8 reserved.)
- **`XEvent`**: `int64 metadata_id`, oneof `data { int64 offset_ps |
  int64 num_occurrences }`, `int64 duration_ps`, `repeated XStat stats`.
  Note **both `offset_ps` and `duration_ps` are picoseconds**, while
  the line's `timestamp_ns` is nanoseconds — Section 5 (gotchas) calls
  this out.
- **`XStat`**: `int64 metadata_id`, oneof `value { double double_value |
  uint64 uint64_value | int64 int64_value | string str_value | bytes
  bytes_value | uint64 ref_value }`. Six variants — must use
  `WhichOneof('value')` to discriminate. The `ref_value` variant is a
  back-reference to a string stored in `XStatMetadata.name`.
- **`XEventMetadata`**: `int64 id`, `string name`, `string display_name`,
  `bytes metadata` (serialized opaque payload), `repeated XStat stats`,
  `repeated int64 child_id`.
- **`XStatMetadata`**: `int64 id`, `string name`, `string description`.
  (No `value_type` field — value type is determined at the use site,
  not in metadata.)

### Real plane and line names observed in dp8_fsdp128

The SKILL.md cites the actual planes/lines present in
`/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`,
verified by direct parse:

- **Planes**: `/host:metadata`, `/device:TPU:0`, `/device:TPU:1`,
  `/device:TPU:0 SparseCore 0`, `/device:TPU:0 SparseCore 1`,
  `/device:CUSTOM:Megascale Trace`, `/host:CPU`, `Task Environment`.
- **`/device:TPU:0` lines**: `_counters_`, `Scalar Unit`, `Steps`,
  `XLA Modules`, `XLA Ops`, `Async XLA Ops`, `TC Overlay`,
  `XLA TraceMe`, `counters_0`.

### Real stat names observed (cited in SKILL.md and per-script docstrings)

The following stat-metadata names appear on the `/device:TPU:0` plane
of dp8_fsdp128 and are the only stat names the spec / scripts may
claim exist:

`% util`, `_a`, `_c`, `_ct`, `_p`, `_pt`, `all_reduce_id`,
`all_reduce_unique_id`, `bytes_accessed`, `core_details`, `core_type`,
`counter_value`, `dcn_collective_info`, `deduplicated_name`,
`device_duration_ps`, `device_id`, `device_offset_ps`,
`device_type_string`, `dropped_traces`, `flops`, `flow`,
`global_chip_id`, `has_megacore`, `has_merged_vmem`, `hlo_category`,
`hlo_op`, `id`, `memory_access_breakdown`, `model_flops`,
`offload_core_id`, `offload_duration_ps`, `offload_execution_index`,
`offload_type`, `peak_*_bw_gigabytes_per_second`,
`peak_teraflops_per_second`, `performance_counter_*`, `power`,
`process_id`, `program_id`, `queue_id`, `raw_bytes_accessed`,
`replica_id`, `run_id`, `shape_with_layout`, `source`, `source_stack`,
`symbol_id`, `tc_offload_start_id`, `temperature`, `tf_op`,
`throttle %`.

(81 stat metadata entries total. The full list is reproduced in the
SKILL.md "Stat metadata reference" subsection.)

**Stat names previously listed in round 1 that do NOT exist and must
not be cited**: `is_root`, `occupancy_pct`. These were fabrications.
Async-start/async-done pairing is done via the `flow` stat (uint64
flow id), as observed in real `Async XLA Ops` events.

## Reference scripts

All seven scripts follow a uniform contract:

- **Top-of-file boilerplate** (4 lines, identical in every script):
  ```python
  import sys, pathlib
  sys.path.insert(0, str(pathlib.Path(__file__).parent / "_proto"))
  import xplane_pb2  # noqa: E402
  ```
- **Entry point**: `def main(profile_dir: str, limit: int = 20) -> None`
- **CLI usage**: `python <script>.py <profile_dir>` (under
  `if __name__ == "__main__":`).
- **Top-of-file module docstring** must contain three labelled blocks:
  - `Schema shown:` — the XPlane/line/stat slice illustrated.
  - `Fields illustrated:` — explicit list of fields by name (using
    only the proto-verified names from the section above).
  - `Source proto:` — fully-qualified path within the vendored proto,
    e.g. `_proto/xplane_pb2.XPlane.lines`.
- **Field naming**: print proto field names verbatim (`offset_ps`, not
  `start_us`) so the reader can map output back to `xplane.proto`.
- **Graceful absence**: when the expected plane/line is not present in
  the input directory, print `[absent]` and return — never raise. This
  is verified by running scripts #1, #3, #6 against `dp4_fsdp16/`.
- **Dependencies**: only `protobuf` (Python package; transitively
  installed by xprof) plus stdlib (`gzip`, `json`, `sys`, `pathlib`).
  No tensorflow, no xprof Python API. The vendored `_proto/xplane_pb2`
  satisfies the proto import.

| # | Script                          | Schema shown                                                                                                                                                                                                                                                |
|---|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `walk_xplane.py`                | The full five-level tree `XSpace → planes → lines → events → stats`, indented; gives a first-look overview.                                                                                                                                                 |
| 2 | `dump_xplane_metadata.py`       | The two reverse-lookup tables on each plane: `event_metadata{id → (name, display_name, child_id)}` and `stat_metadata{id → (name, description)}`. These are how every `XEvent.metadata_id` and `XStat.metadata_id` resolve to a human name.                 |
| 3 | `extract_step_events.py`        | Device plane's `"Steps"` line: per-step XEvents with `offset_ps`, `duration_ps`. Resolves event-name via `event_metadata[metadata_id].name` (typically containing the step number). Source for "how long is one step" / "how many steps did we capture".    |
| 4 | `extract_hlo_events.py`         | `"XLA Ops"` line on the device plane: HLO-level XEvents with stats `hlo_category`, `hlo_op`, `tf_op`, `program_id`, `flops`, `model_flops`, `bytes_accessed`, `raw_bytes_accessed`, `shape_with_layout`. (All names verified present in dp8_fsdp128.)        |
| 5 | `extract_framework_ops.py`      | `/host:CPU` plane: framework-op XEvents (JAX/XLA Python-side calls). The script enumerates whatever stat names the host plane exposes (e.g., `tf_op`, `source`, `source_stack`); it does not assume a fixed list, since host-plane stats vary by generator. |
| 6 | `extract_collective_events.py`  | `"Async XLA Ops"` line: paired async events (`*-start` ↔ `*-done`) matched by the `flow` stat (uint64 flow id). For each `-done` event prints `device_duration_ps` (the exposed comm stall) and `hlo_op`. Highlights that **`-done` events with non-zero `device_duration_ps` measure exposed communication cost.** |
| 7 | `read_trace_json.py`            | `trace.json.gz` top-level: `displayTimeUnit`, `metadata`, `traceEvents[]`. Resolves `pid → process_name` and `tid → thread_name` from `ph=M` metadata events; samples a few `ph=X` (complete) and `ph=i` (instant) events to show their fields.                                                              |

## Verification (must run before declaring complete)

1. `python -m json.tool plugins/tpu-perf/.claude-plugin/plugin.json`
   exits 0.
2. `python -m json.tool .claude-plugin/marketplace.json` exits 0.
3. `SKILL.md` frontmatter parses as valid YAML (two fences, two
   fields).
4. `python3 -c "import sys; sys.path.insert(0,
   'plugins/tpu-perf/skills/profile-anatomy/scripts/_proto');
   import xplane_pb2; xplane_pb2.XSpace()"` exits 0 (vendored proto
   self-test).
5. Every script runs end-to-end on
   `/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`
   without raising and produces non-empty output.
6. Scripts #1, #3, #6 also run end-to-end on
   `/Users/xl/tensorboard/tensorboard/plugins/profile/dp4_fsdp16/` (no
   `hlo_proto.pb`, slightly different content) without raising —
   verifying the "graceful absence" contract.

## Submission

Single commit on a feature branch (already in a worktree). Files
touched:

- `plugins/tpu-perf/.claude-plugin/plugin.json` (new)
- `plugins/tpu-perf/skills/profile-anatomy/SKILL.md` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/__init__.py` (new)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto` (new, vendored verbatim)
- `plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py` (new, generated)
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
  files directly via the vendored `xplane_pb2`. Mixing the two would
  force readers to understand both worlds at once. Keeping them separate
  also lets `tpu-perf` evolve without disturbing the existing MCP
  workflow.

- **Why "scripts as documentation" instead of a callable library?**
  The user explicitly framed these scripts as *schema records*, not
  reusable utilities. Future tpu-perf skills will read these scripts to
  learn the field layout, then write their own analysis code on top of
  `xplane_pb2` directly. A single-script-per-topic layout (vs one big
  `reference.py`) means a future skill can `Read` exactly the slice it
  needs without skimming a 600-line file.

- **Why vendor `xplane_pb2.py` instead of importing from a system package?**
  Round-1 review verified that none of the installed packages on the
  target machine ship `xplane_pb2` as a Python module (`xprof` ships only
  the C++ shim; tensorflow is not installed). Vendoring a single 6 KB
  generated file plus the upstream `.proto` (no transitive imports)
  removes the runtime dependency entirely while still letting future
  maintainers regenerate from a known source.

- **Why not cover HLO proto files?**
  The user removed them from scope. They're a substantially different
  proto schema (`HloModuleProto`, buffer assignment) and warrant a
  dedicated future skill rather than being squeezed into the basic one.

## Round-1 reviewer feedback — disposition

| Round-1 fix request | Disposition in this revision |
|---|---|
| 1. Resolve xplane proto import path | **Resolved** — vendored `xplane_pb2.py` under `scripts/_proto/`; verification step #4 added. |
| 2. Update every script's `Source proto:` and "Dependencies" bullet | **Resolved** — see "Per-script contract": import path is now `_proto/xplane_pb2`, dependencies are stdlib + `protobuf`. |
| 3. Cite a concrete source for `is_root` (script #6) | **Resolved by removal** — empirical inspection of dp8_fsdp128 confirms `is_root` is **not** a stat name in real data. Replaced with the actually-present `flow` stat for async start/done pairing, plus `device_duration_ps` as the comm-stall metric. The fabricated `occupancy_pct` (script #4) was likewise removed and replaced with verified stat names. |
| Nit: stale `tpu-perf-model` entry in marketplace.json | Acknowledged but **out of scope** for this skill; will be addressed in a separate cleanup commit if the user requests. |
