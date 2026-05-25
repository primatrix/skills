# tpu-perf: comm-analysis — Design

Date: 2026-05-24
Status: Draft

## Background

`tpu-perf` is a Claude Code plugin that systematically analyzes TPU
pretraining efficiency by parsing profile-directory artifacts directly.
The first skill in the plugin, `profile-anatomy`, is a schema dictionary:
it teaches future skills how to read `*.xplane.pb` and `*.trace.json.gz`,
but it performs no analysis.

This document specifies the **second** skill, `comm-analysis`, which is
the first analytical skill in the plugin. It answers the question "is my
training run communication-bound, and where?" by inspecting every comm
primitive in a profile, attributing each to a logical mesh axis, and
quantifying compute/comm overlap.

The skill consumes `profile-anatomy` as a schema reference and reuses
its vendored `xplane_pb2.py`. It introduces two new vendored protos
(`hlo.proto`, a subset of `op_stats.proto`) for HLO-side and
peak-bandwidth metadata.

## Scope

### In scope

- A new skill `plugins/tpu-perf/skills/comm-analysis/` with three
  reference scripts.
- Detection and classification of every communication primitive in the
  capture: async collectives, sync/inline collectives, point-to-point
  ops, and async copies. SparseCore (SC0/SC1) and TensorCore (TC) events
  both covered.
- Per-primitive attributes: kind, mode (async/sync), core (TC/SC0/SC1),
  axis (physical X/Y/Z + logical name from optional mesh-spec), group
  size, bidirectional dual-issue flag (heuristic), wall time, stall time
  (exposed), hidden time, byte count, source file:line.
- Three aggregation views: by kind, by source location, by individual
  op_name.
- Per-axis bandwidth utilization: NCCL-style bus BW formula vs peak ICI
  link BW. Peak BW resolved from xprof XStats first, then op_stats.pb
  PerfEnv, then mesh-spec, then `--peak-ici-link-gbps` flag.
- Per-step and per-collective overlap analysis: compute_busy,
  comm_inflight, overlapped, exposed_comm, overlap_ratio.
- All scripts standalone-runnable (`python3 X.py <profile_dir>`),
  graceful `[absent]` for missing inputs, optional `--json` output,
  optional `--mesh-spec` YAML.

### Out of scope (deferred)

- DCN / megascale collectives (`/device:CUSTOM:Megascale Trace` plane).
- Per-physical-link (per-port) BW breakdown.
- Rigorous bidirectional detection via HLO `frontend_attributes`.
  v1 uses a `(opcode, shape, replica_groups)` clustering heuristic; a
  `--strict-bidir` flag may be added later.
- `trace.json.gz` parsing — `xplane.pb` is the source of truth.
- MFU, HBM-pressure, roofline analyses (separate future skills).

## Architecture

```
plugins/tpu-perf/
├── .claude-plugin/plugin.json         # add comm-analysis to skills list
└── skills/
    ├── profile-anatomy/               # existing
    └── comm-analysis/                 # NEW
        ├── SKILL.md
        └── scripts/
            ├── _proto/
            │   ├── hlo.proto                # vendored XLA HLO subset
            │   ├── hlo_pb2.py
            │   ├── op_stats.proto           # vendored OpStats/PerfEnv subset
            │   └── op_stats_pb2.py
            ├── _comm_common.py        # shared parsing helpers
            ├── list_comm_primitives.py
            ├── axis_bandwidth.py
            └── overlap_report.py
```

`xplane_pb2.py` is **reused** from `profile-anatomy/scripts/_proto/`
via `sys.path.insert`, not re-vendored. This is deliberate; both the
proto file and the generated module belong to the schema-dictionary
skill and that's where they stay.

`_comm_common.py` is private to this skill and holds:

- `load_xspace(profile_dir) -> xplane_pb2.XSpace`
- `iter_device_planes(xs) -> Iterator[XPlane]` (matches `/device:TPU:N`,
  `/device:TPU:N SparseCore 0/1`)
- `stat_name_by_id(plane) -> dict[int, str]`
- `event_metadata_stats(plane, event) -> dict[str, value]`
  (resolves the `XEventMetadata.stats` payload — the op-level stats
  bucket like `flops`, `bytes_accessed`, `source`)
- `event_stats(plane, event) -> dict[str, value]`
  (resolves the per-execution `XEvent.stats` bucket)
- `pair_async_events(line) -> list[(start, done)]` (paired by `flow`;
  unpaired entries returned as `(None, ev)` so callers can decide)
- `load_hlo_module(profile_dir) -> hlo_pb2.HloModuleProto | None`
  (selects the most relevant `*.hlo_proto.pb` by program_id match)
- `join_hlo(events, hlo_module) -> dict[op_name, HloInstructionProto]`
  (canonicalizes `*.call-done` / `*.call-start` suffixes)

## Inputs and detection

Each script accepts a profile directory as `argv[1]` (or via
`--profile-dir DIR`).

### Required input — xprof

- Glob `<dir>/*.xplane.pb`, take the first match.
- If absent: print `[absent] no *.xplane.pb in <dir>` and exit 0.

### Optional input — HLO module

Used by `axis_bandwidth.py`.

- Glob `<dir>/*.hlo_proto.pb`. Selection heuristic:
  1. Prefer the module whose `program_id` (top-level field) matches the
     program_id of the most events on `XLA Modules`.
  2. Tie-break: largest file size.
  3. Final tie-break: lexicographic name.
- If absent: `[absent] no *.hlo_proto.pb — axis attribution
  unavailable; falling back to physical-group sizes only` and continue
  with degraded output.

### Optional input — op_stats.pb

Used as a fallback peak-BW source.

- Glob `<dir>/*op_stats.pb`. Parse `OpStats.perf_env`,
  read `peak_bw_giga_bytes_per_second_list[ICI_INDEX]`. The exact ICI
  index depends on the proto schema version; we will document the
  index in `op_stats.proto` comments after verifying against the live
  fixture during implementation.

### Optional input — mesh spec YAML (`--mesh-spec path.yaml`)

```yaml
topology: [4, 4, 8]              # physical chip dims (X, Y, Z)
axes:
  fsdp:  {dims: [Y, Z], size: 32}
  dp:    {dims: [X],    size: 4}
  tp:    {dims: []}              # not used in this run
peak_link_gbps: 90               # optional override
links_per_axis: 2                # hardware constant: 2 links per torus axis (one per ring direction). Independent of whether any given collective uses both directions — that is the per-collective `bidir` heuristic.
```

All fields optional. Missing mesh-spec ⇒ physical axis names only
(`X`, `Y`, `Z`, or `stride-N group`); no logical names.

### Peak-BW resolution order

For `axis_bandwidth.py`:

1. xprof XStat `peak_ici_*` / `peak_link_*` on the device plane (scan
   names; do not assume a specific name).
2. `*op_stats.pb` `PerfEnv.peak_bw_giga_bytes_per_second_list[ICI_INDEX]`.
3. `mesh_spec.peak_link_gbps`.
4. `--peak-ici-link-gbps N` CLI flag.
5. None of the above ⇒ omit the utilization column, print `[warn] peak
   ICI BW unknown — utilization omitted`.

xprof wins over CLI flag because xprof represents what the hardware
actually reported during this run.

### Common flags

- `--limit N` (default 20) — for human-readable output truncation.
- `--json out.json` — also emit structured JSON.
- `--mesh-spec path.yaml` — optional mesh definition.
- `--by {kind,source,op}` — aggregation view for
  `list_comm_primitives.py`.
- `--include-copies` — show `Copy` events in
  `list_comm_primitives.py` (suppressed by default).
- `--peak-ici-link-gbps N` — peak BW fallback for `axis_bandwidth.py`.

## Components

### `list_comm_primitives.py` — capability #1

The **spine** of the skill: emits the rich per-primitive table that
the other two scripts consume.

**Sources scanned on every device-like plane** (`/device:TPU:N`,
`/device:TPU:N SparseCore 0/1`):

| Line | Event class | Pairing |
|---|---|---|
| `Async XLA Ops` | async collectives, async copies | `flow` stat (`*-start` ↔ `*-done`) |
| `XLA Ops` | sync/inline collectives, send/recv | none — single event |

**Classification.** Map `hlo_op` (per-event stat) and
`XEventMetadata.stats.hlo_category` to a kind:

```
all-reduce            → AllReduce
all-gather            → AllGather
reduce-scatter        → ReduceScatter
all-to-all            → AllToAll
collective-permute    → CollectivePermute
send | recv           → P2P
copy-(start|done)     → Copy   # excluded from default output; --include-copies opts in
```

If `hlo_category` is missing, fall back to a regex on the op name; if
still ambiguous, classify as `Unknown` and count.

**Per-event row** (full schema, also the `--json` payload):

| Field | Source |
|---|---|
| `op_name` | `hlo_op` stat, `.call-done` / `.call-start` suffix stripped |
| `kind` | classification rule above |
| `mode` | `async` (on `Async XLA Ops`) or `sync` (on `XLA Ops`) |
| `core` | derived from plane: `TC`, `SC0`, `SC1` |
| `axis` | from HLO `replica_groups` join + mesh-spec; `—` if HLO absent |
| `group_size` | length of the matching replica group |
| `bidir` | heuristic: yes if HLO clusters `(opcode, shape, replica_groups, sharding)` produces ≥2 instructions with distinct `channel_id`s; otherwise no |
| `bytes` | `bytes_accessed` from `XEventMetadata.stats` |
| `wall_ps` | `done.offset_ps − start.offset_ps` for paired async; `duration_ps` for sync or unpaired |
| `stall_ps` | `done.device_duration_ps` for async; full `device_duration_ps` for sync (always exposed) |
| `hidden_ps` | `wall_ps − stall_ps` |
| `source` | `XEventMetadata.stats.source` (and/or `source_stack`); the exact value format (e.g. `"file:line"` vs structured) is undocumented in `profile-anatomy` and must be confirmed against the live fixture during implementation. Fallback to HLO `OpMetadata.{source_file, source_line}` joined via `op_name` |
| `flow` | the pairing key, for debugging |
| `program_id` | `XEventMetadata.stats.program_id` |
| `channel_id` | from joined HLO instruction |

**Aggregation views:**

- `--by kind` (default): roll up by `(kind, axis, core)` with count,
  Σwall, Σstall, p50/p99 of stall.
- `--by source`: roll up by `source` (`file:line`). Σwall, Σstall,
  count, dominant kind/axis. Answers "which line of my model is
  causing comm?".
- `--by op`: per individual `op_name`, top N by Σstall.

**Edge cases:**

- Flow with `pair_size=1` (observed in the live fixture): treat as
  fully exposed (`wall_ps = stall_ps`, `hidden_ps = 0`); tag
  `unpaired=true` in `--json`. Do not drop.
- Events on `Async XLA Ops` with no `flow` stat: bucket as `unpaired`,
  count separately, keep their stall data.
- Sync collective with both `hlo_category` and op-name regex
  failing: classify `Unknown`, keep the row.
- `Async XLA Ops` line absent on a plane: skip it for that plane,
  continue with sync events.

### `axis_bandwidth.py` — capability #2

Reads the per-primitive spine from `list_comm_primitives.py`'s
extractor, joins with HLO and peak BW, adds two columns:
`bus_BW(GB/s)` and `util%`. Reports two tables:

1. **Per-axis aggregate.** For each `(axis, logical_name)`: Σbytes,
   Σtime, bus BW, peak axis BW (`peak_link_gbps × links_per_axis`),
   utilization.
2. **Top-N per-collective table.** Sorted by Σstall.

**Bus-bandwidth formulas** (NCCL/XLA convention):

| Kind | Bus BW |
|---|---|
| AllReduce | `2 × (N−1)/N × message_bytes / time` |
| AllGather | `(N−1)/N × output_bytes / time` |
| ReduceScatter | `(N−1)/N × input_bytes / time` |
| AllToAll | `(N−1)/N × message_bytes / time` |
| CollectivePermute / P2P | `message_bytes / time` |

`N = group_size`. Time is `wall_ps` (in-flight, not stall) — the
collective's actual on-the-wire duration.

**Axis attribution.** For each replica group:

1. Translate replica IDs to physical chip coords via the mesh-spec
   `topology`, or if no mesh-spec: row-major
   `(replica_id // (Y*Z), replica_id % (Y*Z) // Z, replica_id % Z)`
   using a topology range derived from `device_id` stats.
2. Identify which dim(s) vary within the group:
   - One varying dim ⇒ named axis (`X`/`Y`/`Z`).
   - Two varying dims ⇒ plane (`YZ`, `XZ`, `XY`) of size `dim1 × dim2`.
   - All three varying ⇒ `full mesh` (rare).
3. If mesh-spec gave logical names, match the varying-dims set to a
   logical axis name (e.g. `fsdp`).

**Bidirectional dual-issue (heuristic).** Cluster
`HloInstructionProto`s by `(opcode, shape, replica_groups, sharding)`
within one `HloModuleProto`. Cluster size ≥ 2 with distinct
`channel_id`s ⇒ tag the corresponding xprof event(s) `bidir=yes`.
Per-event `channel_id` is reported so users can verify visually.

**Degradation.** No HLO module ⇒ skip steps 1–3; group size inferred
from how many distinct devices share a `flow` ID within a
`program_id`; logical names dropped; output collapses to a flatter
"by group-size bucket" table.

### `overlap_report.py` — capability #3

xprof-only. No HLO needed.

**Pipeline:**

1. **Step boundaries.** Each XEvent on the `Steps` line of the device
   plane is one step; window = `[step.offset_ps,
   step.offset_ps + step.duration_ps]`.
2. **Interval sets per step:**
   - `compute_intervals`: `XLA Ops` events whose `hlo_category`
     ∉ {all-reduce, all-gather, reduce-scatter, all-to-all,
     collective-permute, send, recv}.
   - `comm_intervals`: paired async collectives from `Async XLA Ops`
     (interval = `[start.offset_ps, done.offset_ps + done.duration_ps]`)
     plus sync collectives from `XLA Ops`.
   - Both sets clipped to the step window.
3. **Sweep-line union math:**
   - `compute_busy_ps = ∪(compute_intervals).total_length`
   - `comm_inflight_ps = ∪(comm_intervals).total_length`
   - `overlapped_ps = ∪(compute_intervals ∩ comm_intervals).total_length`
   - `exposed_comm_ps = comm_inflight_ps − overlapped_ps`
   - `overlap_ratio = overlapped_ps / comm_inflight_ps` (NaN-safe)
4. **Sanity check:** compare `exposed_comm_ps` (sweep) to
   `Σ done.device_duration_ps within the step` (metadata). Mismatch
   above 5 % triggers a `[warn] step N` line; sweep value is
   authoritative.
5. **Per-collective view.** Joined to the spine, with a per-event
   `hidden_ratio = hidden / wall`.

**Output:**

- Step-level table: per step + total row with the five derived
  metrics.
- Top-N exposed-comm contributors across all steps.
- Separate sub-table for SparseCore comm (SC0/SC1 don't compete
  with TC compute, so we don't muddle the math).

**Edge cases:**

- `Steps` line absent ⇒ synthesize one global window covering the
  full event range. Print `[fallback] no Steps line; using global
  window`.
- `pair_size=1` async events ⇒ treat as fully exposed (`wall =
  stall`, `hidden = 0`), tag `unpaired=true`.
- SparseCore lines processed independently; never mixed into the
  TC compute-vs-comm sweep.

## Vendored protos

| File | Source | Subset shipped |
|---|---|---|
| `_proto/hlo.proto` | `tensorflow/compiler/xla/service/hlo.proto` | `HloModuleProto`, `HloComputationProto`, `HloInstructionProto`, `ReplicaGroup`, `ShapeProto`, `OpMetadata`, `FrontendAttributes` |
| `_proto/op_stats.proto` | `tensorflow/core/profiler/protobuf/op_stats.proto` | `OpStats`, `PerfEnv` (specifically `peak_bw_giga_bytes_per_second_list`) |
| `_proto/hlo_pb2.py`, `op_stats_pb2.py` | generated via `protoc --python_out=` | shipped, regeneratable from the `.proto` files |

`xplane_pb2.py` is reused from `profile-anatomy/scripts/_proto/` via
`sys.path.insert` — explicit dependency on the schema-dictionary
skill.

## Error handling

Graceful degradation cascade:

| Missing input | Behavior |
|---|---|
| `*.xplane.pb` absent | print `[absent] no *.xplane.pb in <dir>`, exit 0 |
| `Async XLA Ops` line absent on a plane | skip async section for that plane; sync events still reported |
| `Steps` line absent (overlap_report) | synthesize one global window; `[fallback] no Steps line` |
| `*.hlo_proto.pb` absent (axis_bandwidth) | drop axis-attribution columns; physical-group sizes only; `[absent] no *.hlo_proto.pb — axis attribution unavailable` |
| `*op_stats.pb` absent + no peak XStat + no flag | drop utilization column; `[warn] peak ICI BW unknown — utilization omitted` |
| Mesh-spec missing | drop logical axis names; physical X/Y/Z only |

No traceback exits except for genuine programming errors. Same
convention as `profile-anatomy`.

## Testing / verification

The repo has no test framework. Verification is one-shot human-driven
against the live fixture at
`/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128`.

1. **Smoke runs:**
   ```
   python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py /tmp/.../dp8_fsdp128
   python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py /tmp/.../dp8_fsdp128 --by source
   python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py /tmp/.../dp8_fsdp128
   python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py /tmp/.../dp8_fsdp128
   ```
   All four exit 0; non-empty tables (or `[absent]` lines for
   genuinely-missing slices).
2. **Cross-check sanity:** `Σ stall_ps` from `list_comm_primitives.py`
   should equal `Σ exposed across all steps` from `overlap_report.py`
   within 5 % (the same epsilon the overlap warning uses).
3. **Empty-input behavior:** run each script against a directory with
   no `*.xplane.pb` — expect `[absent]` and exit 0.
4. **JSON validity:** `--json out.json` output passes
   `python3 -m json.tool`.
5. **Manifest validity:** `python3 -c "import json;
   json.load(open('plugins/tpu-perf/.claude-plugin/plugin.json'))"`
   and the same for `marketplace.json`.
6. **Frontmatter validity:** `python3 -c "import yaml;
   yaml.safe_load(open('plugins/tpu-perf/skills/comm-analysis/SKILL.md').read().split('---')[1])"`.

## Open questions

The implementation phase will resolve:

1. **The exact ICI index** in
   `PerfEnv.peak_bw_giga_bytes_per_second_list`. Verify against the
   live fixture's `*op_stats.pb`; document the index in
   `op_stats.proto`.
2. **Whether `peak_ici_*` XStats appear on any plane in any TPU
   generation.** v6e fixture has none. Implementation will keep the
   scan defensive — it is a no-op when absent.
3. **Whether `Async XLA Ops` `flow` IDs ever yield real pairs**
   (`pair_size=2`) or always `pair_size=1` as observed. The design
   handles both; the implementation must not regress the unpaired
   case.
4. **Exact value format of the `source` / `source_stack` XStats** —
   the `profile-anatomy` schema dictionary lists their names but not
   their value shape. Implementation must verify against the live
   fixture and document the format in `_comm_common.py`.
