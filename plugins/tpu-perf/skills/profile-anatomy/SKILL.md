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
    human label (and op-level stats — see note below)
  - `oneof data { int64 offset_ps | int64 num_occurrences }` —
    `offset_ps` for normal events, `num_occurrences` for aggregated
    counts. Use `WhichOneof("data")` to discriminate.
  - `int64 duration_ps`
  - `repeated XStat stats` — **per-execution counters only** (e.g.
    `device_offset_ps`, `device_duration_ps`). Op-level stats like
    `hlo_category` / `flops` / `shape_with_layout` are NOT here — they
    are on `XEventMetadata.stats` (see below). XLA shares one
    `XEventMetadata` across every execution of the same HLO op.

- **`XStat`** (a named value attached to an event, plane, or
  event-metadata)
  - `int64 metadata_id` → `stat_metadata[metadata_id].name`
  - `oneof value` with **six** variants:
    `double_value`, `uint64_value`, `int64_value`, `str_value`,
    `bytes_value`, `ref_value`. Use `WhichOneof("value")` to discriminate.
    `ref_value` is a back-reference whose payload is stored in
    `XStatMetadata.name`.

- **`XEventMetadata`** (shared metadata per event-type-id within a
  plane)
  - `int64 id`, `string name` (the HLO op text for `XLA Ops`),
    `string display_name`,
    `bytes metadata`, `repeated XStat stats`,
    `repeated int64 child_id`.
  - **`stats` here is the op-level payload** for HLO events:
    `hlo_category`, `tf_op`, `program_id`, `flops`, `model_flops`,
    `bytes_accessed`, `raw_bytes_accessed`, `shape_with_layout`, etc.
    Resolve names via the same `XPlane.stat_metadata` map used by
    `XEvent.stats`.

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
| [`extract_hlo_events.py`](scripts/extract_hlo_events.py) | HLO-level events on `"XLA Ops"`. **Op-level stats** (`hlo_category`, `tf_op`, `program_id`, `flops`, `model_flops`, `bytes_accessed`, `raw_bytes_accessed`, `shape_with_layout`) are read from `XEventMetadata.stats`, not `XEvent.stats`. The HLO op text itself is `XEventMetadata.name`. |
| [`extract_framework_ops.py`](scripts/extract_framework_ops.py) | `/host:CPU` framework events, with stat names discovered (not assumed). |
| [`extract_collective_events.py`](scripts/extract_collective_events.py) | `"Async XLA Ops"` paired by the `flow` stat (per-`XEvent.stats`) — measures exposed comm stall via `device_duration_ps` of `*-done` events. |
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
- **`XEvent.stats` vs `XEventMetadata.stats`.** This is the most
  common mistake. `XEvent.stats` carries only per-execution counters
  (e.g. `device_offset_ps`, `device_duration_ps`). Op-level facts
  about an HLO op — `hlo_category`, `flops`, `shape_with_layout` etc.
  — live on `XEventMetadata.stats`, because XLA shares one
  `XEventMetadata` across every execution of the same op. Resolve
  names from the same `XPlane.stat_metadata` map either way. The
  `Async XLA Ops` line is the exception: its events carry per-event
  stats (`flow`, `hlo_op`, `device_duration_ps`) directly on
  `XEvent.stats` — verify with `dump_xplane_metadata.py` if unsure.
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
