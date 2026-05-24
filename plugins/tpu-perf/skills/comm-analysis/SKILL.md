---
name: comm-analysis
description: Use when analyzing communication on a TPU pretraining profile — extracts every comm primitive (async + sync, TC + SparseCore), attributes axes via HLO replica_groups, computes per-axis NCCL bus BW vs peak ICI link BW, and reports per-step compute/comm overlap. Builds on profile-anatomy.
---

# Communication Analysis

Three reference scripts for analyzing the communication portion of a TPU
pretraining profile. Each script accepts a profile directory as `argv[1]`
(or `--profile-dir DIR`) and runs standalone with stdlib + `protobuf` +
optional `pyyaml` (only for `--mesh-spec`).

This skill builds on [`profile-anatomy`](../profile-anatomy/SKILL.md);
read that first for the xplane.pb / xplane.proto schema.

## 1. What's covered

| Capability | Script |
|---|---|
| List every comm primitive (async + sync, TC + SC) with rich attributes | [`scripts/list_comm_primitives.py`](scripts/list_comm_primitives.py) |
| Per-axis bandwidth utilization (NCCL bus BW vs peak ICI link BW) | [`scripts/axis_bandwidth.py`](scripts/axis_bandwidth.py) |
| Per-step compute/comm overlap (sweep-line union) | [`scripts/overlap_report.py`](scripts/overlap_report.py) |

ICI only. DCN/megascale collectives are deferred to a future skill.

## 2. Per-primitive row schema

`list_comm_primitives.py` builds rows with these fields (also the `--json`
payload):

| Field | Source |
|---|---|
| `op_name` | `hlo_op` stat (canonicalized — `.call-start` / `.call-done` / `.start` / `.done` stripped) |
| `kind` | `AllReduce` / `AllGather` / `ReduceScatter` / `AllToAll` / `CollectivePermute` / `P2P` / `Copy` / `Unknown` |
| `mode` | `async` (Async XLA Ops) or `sync` (XLA Ops) |
| `core` | `TC`, `SC0`, or `SC1` |
| `axis` | logical or physical mesh axis (set by `axis_bandwidth.py`); `—` when unattributed |
| `group_size` | size of the first replica group (walks `collective_device_list` / `iota_collective_device_list` / legacy `replica_groups`) |
| `bidir` | `yes` / `no` heuristic from `(opcode, shape, replica_groups, sharding)` cluster having ≥2 distinct channel_ids |
| `bytes` | `bytes_accessed` from `XEventMetadata.stats` |
| `wall_ps` | `done.offset_ps + done.duration_ps − start.offset_ps` for paired async; `duration_ps` for sync |
| `stall_ps` | `done.device_duration_ps` for async; full `duration_ps` for sync (sync = always exposed) |
| `hidden_ps` | `wall_ps − stall_ps` |
| `bus_bw_gbps` | NCCL bus BW for this row's `kind` and `group_size`, computed by `axis_bandwidth.py` |
| `effective_bus_bw_gbps` | `2 × bus_bw_gbps` when `bidir == yes` (both ICI directions carry traffic), else `bus_bw_gbps` |
| `source` | `XEventMetadata.stats.source` / `source_stack`; falls back to HLO `OpMetadata.source_file:line` |
| `flow` | the `flow` XStat used to pair async events |
| `program_id` | `XEventMetadata.stats.program_id` |
| `channel_id` | from joined HLO instruction (int64 scalar; `0` ⇒ unset) |

## 3. Aggregation views

`list_comm_primitives.py --by {kind,source,op}`:

- `kind` (default): roll up by `(kind, axis, core)` with count, Σwall, Σstall, p50/p99 stall.
- `source`: roll up by source `file:line` — answers "which line of the model is causing comm?".
- `op`: per individual `op_name`, top N by Σstall.

## 4. Bus-bandwidth formulas (NCCL/XLA convention)

| Kind | Bus BW |
|---|---|
| AllReduce | `2 × (N−1)/N × message_bytes / time` |
| AllGather | `(N−1)/N × output_bytes / time` |
| ReduceScatter | `(N−1)/N × input_bytes / time` |
| AllToAll | `(N−1)/N × message_bytes / time` |
| CollectivePermute / P2P | `message_bytes / time` |

`N = group_size`; `time = wall_ps` (in-flight, not stall). Peak axis BW =
`peak_link_gbps × links_per_axis` (default 2). When `bidir == yes`, the
displayed `util%` doubles the single-direction bus BW because both ICI
directions carry traffic simultaneously.

## 5. Peak-BW resolution order

1. xprof XStat (`peak_ici_*` / `peak_link_*` scanned across device, host, and Task Environment planes via `cc.peak_ici_link_gbps_from_xprof`).
2. `--mesh-spec` YAML `peak_link_gbps:`.
3. `--peak-ici-link-gbps N` flag.
4. None ⇒ utilization column omitted, `[warn]` printed.

`*op_stats.pb` does NOT carry ICI peak BW: its
`PerfEnv.peak_bws_giga_bytes_per_second` list is keyed by upstream
`MemBwType` (HBM_RW / SRAM_* / CMEM_* / VMEM_*) and has no ICI entry.

## 6. Optional mesh-spec YAML

```yaml
topology: [4, 4, 8]              # physical chip dims (X, Y, Z)
axes:
  fsdp:  {dims: [Y, Z], size: 32}
  dp:    {dims: [X],    size: 4}
peak_link_gbps: 90
links_per_axis: 2
```

All fields optional. Without a mesh-spec, axes are reported as physical
`X`/`Y`/`Z` (or `stride-N group` if topology is unknown).

## 7. Sample invocations

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --by source

python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --mesh-spec mesh.yaml

python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
```

## 8. Common gotchas

- **Async pairing uses `flow` (uint64), not `is_root`.** See profile-anatomy.
- **`pair_size=1` is observed in current captures.** Treated as fully
  exposed (`wall = stall`, `hidden = 0`); the row's `bidir` is "no" and
  it still appears in the Top-N exposed contributors table.
- **HLO module is optional but recommended.** Without `*.hlo_proto.pb`,
  axis attribution and the bidir heuristic degrade gracefully — `axis`
  stays `—` and a `[warn] N collective rows have no HLO counterpart`
  line is emitted.
- **Cloned-wrapper join failure is a known limitation.** xprof events
  often reference op names like `all-reduce.3008.cloned.1`, which exist
  in HLO as opcode=`call` wrappers around the actual collective rather
  than as the collective itself. The wrapper has no replica info, so
  axis stays `—` and group_size stays `0` for those rows.
  `axis_bandwidth.py` counts these and emits a single
  `[warn] N collective rows could not be axis-attributed (cloned-wrapper join failure)` line.
- **Modern HLO uses `collective_device_list` / `iota_collective_device_list`,
  not legacy `replica_groups` (field 49).** The vendored helpers walk
  all three locations; the legacy field is empty in current captures.
- **SparseCore comm is reported in a separate sub-table** in
  `overlap_report.py` because SC and TC compute don't compete; mixing
  them would muddle the math.
- **The sweep-derived `exposed_comm` is authoritative** when it disagrees
  with `Σ done.device_duration_ps` by >5%; the metadata sum doesn't
  account for parallel streams.
- **`Σ stall_ps` (from `list_comm_primitives.py`) ≠ `Σ exposed_comm_ps` (from
  `overlap_report.py`) on captures where async events are flow-singletons.**
  When every Async XLA Op is unpaired (no flow start/done pair), each row's
  `stall_ps` falls back to `device_duration_ps`, which measures the comm
  engine's busy time — NOT the exposed (un-overlapped) slice. Multiple ICI
  links and SC lanes can be busy in parallel and overlap with TC compute,
  so summed `stall_ps` legitimately exceeds wall-clock. Always trust
  `overlap_report.py`'s sweep-derived `exposed_comm` for true exposed time.
- **`xplane_pb2.py` is reused from profile-anatomy** via
  `sys.path.insert`. Don't re-vendor it.
