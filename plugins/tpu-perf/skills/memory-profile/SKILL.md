---
name: memory-profile
description: Use when analyzing TPU pretraining HBM occupancy from a profile directory — locates the static HBM peak (the same number TensorBoard's Memory Viewer shows), enumerates every buffer alive at the peak schedule moment with size / HLO instruction / opcode / op_name, and rolls the alive set up by opcode and op_name. Reads compile-time `*.hlo_proto.pb` (BufferAssignmentProto) as the primary source; runtime `*.xplane.pb` allocator events are a secondary, often-truncated signal.
argument-hint: "<profile_dir> [--top K] [--no-runtime] [runtime opts: --step N | --step-policy {peak,last,first} | --all-trace]"
---

# Memory Profile

Answer "what is the HBM peak, what HLO instructions own it, and what
fraction of it is unavoidable static residency vs the per-step
activation spike" for a TPU pretraining profile, in a form Claude can
read structurally and turn into optimization recommendations. One
Python entry script, single JSON object on stdout, `status: ok | absent`.

This skill is built on top of `profile-anatomy`, which documents the
XSpace/XPlane/XLine/XEvent/XStat hierarchy. Read that first if you
need to know what an XEvent is, where allocator events live, or how
`XLine.timestamp_ns` and `XEvent.offset_ps` combine into a wall clock.

## Primary data source: `*.hlo_proto.pb`

The profile directory contains one `*.hlo_proto.pb` per compiled jit;
the largest is the train-step module. Its `BufferAssignmentProto` is
**exactly** the data TensorBoard's Memory Viewer renders. It enumerates
every buffer XLA reserves at compile time — weights, optimizer state,
activations, communication scratch — with size, lifetime, and HLO
instruction attribution. This is the authoritative source for HBM
peak.

The runtime allocator events on `/host:CPU` (`MemoryAllocation` /
`MemoryDeallocation`) are kept as a secondary signal because they are
routinely truncated by the trace window — they miss every buffer
allocated before capture started (typically all weights / optimizer
state). Use them only as a cross-check; if they disagree with the HLO
block by >5%, `consistency_warnings` flags it.

## When to use

"We want to reduce HBM peak — what is the peak, what HLO instructions
own it, and which slice is static residency we cannot remat away."

## Concepts you need first

- **`hlo.static_peak_bytes`** = `Σ buffer_allocations[*].size`. This is
  the compile-time HBM total Memory Viewer displays. Allocations are
  classified into `entry_params` / `constants` / `thread_local` /
  `temp_pool`; the temp pool is the single largest non-thread-local
  non-param non-const allocation and holds all activation / scratch
  traffic.
- **`hlo.schedule_sweep`** walks the entry-computation schedule in
  order and finds the position where `Σ live-buffer-sizes` is maximal.
  The reported `peak_alive_bytes_entry_level` is the entry-level peak
  (typically lower than `static_peak_bytes` because logical buffers
  defined inside while-bodies / fusion / scan-bodies cannot be placed
  on the entry schedule and are counted as part of their wrapping
  while/call output — see `n_subcomputation_lbs_skipped`). The
  authoritative HBM peak remains `static_peak_bytes`.
- **`hlo.always_alive`** is the static-residency floor: bytes inside
  the temp pool owned by exactly one logical buffer in the address
  space. Two logical buffers can share an `(offset, size)` range only
  if XLA proved their lifetimes disjoint, so unique-occupant regions
  are alive at every schedule position by construction. No remat
  policy can eliminate them.
- **`runtime.alive_at_peak`** (secondary): set of buffers with
  `alloc_ts_ns ≤ peak.ts_ns < dealloc_ts_ns` from `/host:CPU`
  allocator events within the chosen step window. Use only when the
  HLO block is unavailable or as a cross-check.
- **`runtime.lifetime_class`** (secondary, runtime block only):
  - `persistent` ⇐ `crossed_step_boundaries ≥ persistent_threshold_steps`
    (default 2) **and** never deallocated within the trace.
  - `transient` ⇐ alloc and dealloc both within the same step interval.
  - `unknown` ⇐ otherwise. Trace truncation biases this ↑.

## CLI and examples

```bash
# Default: HLO peak + Top-30 alive buffers + secondary runtime block
python3 .../memory_profile.py <profile_dir>

# HLO only (skip runtime allocator block)
python3 .../memory_profile.py <profile_dir> --no-runtime

# Larger Top-K
python3 .../memory_profile.py <profile_dir> --top 100

# Runtime-block options (only affect the secondary block):
python3 .../memory_profile.py <profile_dir> --all-trace
python3 .../memory_profile.py <profile_dir> --step 3
python3 .../memory_profile.py <profile_dir> --step-policy last
```

## JSON schema cheat-sheet (schema v2)

```json
{
  "status": "ok",
  "skill": "memory-profile",
  "version": 2,
  "inputs": { "profile_dir": "...", "xplane_pb": "...", "hlo_proto_pb": "..." },
  "primary_source": "hlo_buffer_assignment",
  "hlo": {
    "hlo_proto_path": "...", "module_name": "jit_train_step",
    "static_peak_bytes": ...,                      /* the Memory Viewer total */
    "decomposition": {
      "entry_params_bytes": ...,                   /* weights + optimizer state passed in */
      "constants_bytes": ...,
      "thread_local_bytes": ...,
      "temp_pool_bytes": ...,                      /* the activation / scratch arena */
      "temp_pool_alloc_index": ...
    },
    "n_logical_buffers": ..., "n_buffer_allocations": ...,
    "schedule_sweep": {
      "schedule_present": true,
      "entry_schedule_length": ...,
      "peak_schedule_pos": ...,
      "peak_instruction": { "id":..., "name":..., "opcode":..., "op_name":... },
      "peak_alive_bytes_entry_level": ...,
      "n_subcomputation_lbs_skipped": ...,
      "scope_note": "..."
    },
    "alive_at_peak": {
      "n_buffers": ..., "total_bytes": ...,
      "buffers": [ /* Top-K HloAliveBuffer: logical_buffer_id, size_bytes,
                      allocation_index, offset_in_allocation,
                      instruction_id, instruction_name, opcode, op_name,
                      shape_index */ ],
      "tail": { "n_buffers": ..., "total_bytes": ... },
      "rollups": { "by_opcode": [...], "by_op_name": [...] }
    },
    "always_alive": {
      "total_bytes": ...,                          /* static-residency floor */
      "pct_of_temp_pool": ...,
      "buffers": [ /* Top-K owners; size_bytes is unique-occupancy bytes */ ],
      "rollups": { "by_opcode": [...], "by_op_name": [...] },
      "definition": "..."
    },
    "top_allocations": [ /* informational: largest buffer_allocations with flags */ ]
  },
  "hlo_absent_reason": null,
  "runtime": {                                      /* secondary; may be null */
    "step":     { "id": ..., "policy": "...", "range_ns": [lo, hi], "source": "..." },
    "pool":     { "id": 0, "bytes_reserved": ... },
    "peak":     { "ts_ns": ..., "bytes_total": ..., "bytes_by_pool": {"0": ...},
                  "fragmentation_at_peak": ..., "is_global_peak": ... },
    "alive_at_peak": { "n_buffers": ..., "total_bytes": ..., "buffers": [...], "tail": {...} },
    "rollups":  { "by_lifetime_class": [...], "by_shape": [...], "by_tf_op": [...],
                  "by_parent_jit": [...], "by_dtype": [...] },
    "timeline": { "samples": [...], "events_of_interest": [...], "axis_units": {...} },
    "n_planes": ..., "host_plane_present": true
  },
  "runtime_diagnostics": {
    "alloc_accounting_drift_pct": ..., "unmatched_dealloc_count": ...,
    "pretrace_dealloc_count": ..., "unmatched_alloc_count": ...,
    "trace_end_live_bytes": ..., "n_pools_seen": ...,
    "pools_summary": [...], "step_line_present": ...,
    "shape_missing_count": ..., "tf_op_missing_count": ..., "warnings": [...]
  },
  "runtime_absent_reason": null,
  "consistency_warnings": [ /* fires when runtime_peak vs static_peak diverge >5% */ ]
}
```

## Invariants (consistency gates)

HLO block (always exact when `hlo` is present):

| # | Invariant |
|---|---|
| H1 | `static_peak_bytes == entry_params_bytes + constants_bytes + thread_local_bytes + temp_pool_bytes + Σ(other allocations)` (other allocations are sub-pool / aliased; usually 0) |
| H2 | `peak_alive_bytes_entry_level ≤ temp_pool_bytes` |
| H3 | `Σ alive_at_peak.buffers[*].size_bytes + tail.total_bytes == alive_at_peak.total_bytes` |
| H4 | `always_alive.total_bytes ≤ temp_pool_bytes` |
| H5 | `Σ rollups.by_opcode[*].total_bytes == alive_at_peak.total_bytes` (each rollup partitions the alive set) |

Runtime block (when present):

| # | Invariant | Tolerance |
|---|---|---|
| R1 | `Σ buffers[*].size_bytes + tail.total_bytes == alive_at_peak.total_bytes` | exact |
| R2 | `alive_at_peak.total_bytes == peak.bytes_total` | exact |
| R2b | `\|peak.bytes_total − allocator's bytes_allocated at peak_ts\| / peak.bytes_total ≤ 0.01` | soft |
| R3 | `peak.bytes_total ≤ pool.bytes_reserved` | exact |
| R4 | `step.range_ns[0] ≤ peak.ts_ns ≤ step.range_ns[1]` (skipped under `--all-trace`) | exact |

Cross-block:

| # | Invariant |
|---|---|
| C1 | If both blocks present and `\|static_peak − runtime_peak\| / max(...) > 0.05`, a `consistency_warnings` entry is emitted (typical signature: trace started after model init, so weights are missing from the runtime peak — trust HLO). |

## Reading guide

- **"What is the actual HBM peak?"** → `hlo.static_peak_bytes`. This
  is the Memory Viewer number. Compare against the HBM pool size
  (32 GB on v5e, 95 GB on v5p, etc.) to gauge headroom.
- **"How is the peak split — what's avoidable?"** → `hlo.decomposition`.
  `entry_params_bytes` is weights + optimizer state passed in (not
  remattable; reduce via FSDP / lower precision / fewer optimizer
  states). `temp_pool_bytes` holds activations and scratch (reduce via
  remat, smaller microbatch, or sharding intermediates).
- **"Within the temp pool, what is unavoidable?"** →
  `hlo.always_alive.total_bytes / temp_pool_bytes`. High ratio means
  most of the pool is static residency that no remat policy can
  eliminate; remat will mostly trade compute for the *non*-always-alive
  remainder.
- **"At the peak schedule moment, which HLO instructions are alive?"**
  → `hlo.alive_at_peak.rollups.by_opcode` and `by_op_name`. The
  `op_name` is the JAXPR call site (e.g. `jit(train_step)/.../decoder/.../while`).
- **"Where in the program does the peak occur?"** →
  `hlo.schedule_sweep.peak_instruction.{name, opcode, op_name}` and
  `peak_schedule_pos / entry_schedule_length` (gives the fractional
  position). A peak under a `while` op_name = inside a scan / decoder
  layer loop.
- **"Why is `peak_alive_bytes_entry_level < static_peak_bytes`?"** →
  `n_subcomputation_lbs_skipped`. The entry-level sweep can only see
  the wrapping `while`/`call` output buffer, not the per-iteration
  internals. The authoritative peak remains `static_peak_bytes`.
- **"Did the runtime trace see the same peak?"** → if `runtime` is
  present, compare `runtime.peak.bytes_total` to
  `hlo.static_peak_bytes`. A large gap (typically captured under
  `consistency_warnings`) means the trace window missed model init.

## Common gotchas

- **`alive_at_peak.tail` cannot be ignored.** `buffers` is Top-K only;
  `n_buffers` and `total_bytes` are the truth.
- **`peak_alive_bytes_entry_level` is an under-estimate.** Logical
  buffers defined inside while/fusion/scan bodies are skipped from
  the entry-level sweep — see `n_subcomputation_lbs_skipped`. Use
  `static_peak_bytes` as the authoritative peak.
- **`always_alive` `size_bytes` ≠ logical buffer size.** It is the
  number of bytes of the temp pool's address space that the buffer
  uniquely occupies. A large logical buffer can have a small
  always-alive footprint if most of its range is shared with other
  short-lived buffers.
- **Runtime block absent or much smaller than HLO?** Trace truncation.
  The runtime allocator on `/host:CPU` only logs allocs that happen
  *during* the capture; long-lived buffers (weights, optimizer state)
  allocated before capture started are invisible. The
  `consistency_warnings` array calls this out when it happens. Trust
  the HLO block.
- **`runtime.step.source == "execute_event"`** means the `Steps` line
  was missing and the runtime block fell back to the outermost
  `Execute (jit_*)` event. `step.id` is then a sequential index, NOT
  the user's training step number. (HLO block is unaffected.)

## Limitations

- **No per-device split.** XLA buffer assignment is per-module, not
  per-shard. For the multi-host total, multiply by
  `num_hosts × num_devices_per_host` only if the module is
  data-parallel-replicated; FSDP / TP modules already encode the
  per-device residency.
- **No source-line attribution beyond `op_name`.** The JAXPR
  `op_name` carries the call-site path but no `file:line`. For exact
  source pointers, cross-reference `op_name` with the model code.
- **`hlo_proto.pb` must be present in the profile directory.** If the
  capture is xplane-only (rare), only the (truncated) runtime block is
  available; the skill warns and degrades gracefully.

## Files

- `scripts/memory_profile.py` — main entry script.
- `scripts/_hlo_loader.py` — `*.hlo_proto.pb` loader: parse
  BufferAssignmentProto, classify allocations, sweep entry schedule,
  sweep address-space for always-alive bytes.
- `scripts/_loader.py` — xplane load, plane/line lookup, step window
  picker, runtime allocator sweep, runtime rollups (secondary block).
- `scripts/_proto/` — vendored protobuf bindings. `hlo.proto` /
  `hlo_pb2.py` are reused from `comm-analysis/scripts/_proto/`.
- `scripts/tests/` — unit + e2e tests (stdlib `unittest`).
