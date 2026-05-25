---
name: memory-profile
description: Use when analyzing TPU pretraining HBM occupancy from xplane.pb — locates the peak HBM moment, lists every buffer alive at that moment with size/shape/tf_op/parent_jit/lifetime_class, and rolls the alive set up by lifetime / shape / tf_op / parent jit / dtype. Reads schema documented by profile-anatomy.
argument-hint: "<profile_dir> [--step N | --step-policy {peak,last,first} | --all-trace] [--top K]"
---

# Memory Profile

Answer "when does HBM peak, and what is alive at that moment" for a TPU
pretraining profile, in a form Claude can read structurally and turn
into optimization recommendations. One Python entry script, single JSON
object on stdout, `status: ok | absent`.

This skill is built on top of `profile-anatomy`, which documents the
XSpace/XPlane/XLine/XEvent/XStat hierarchy. Read that first if you
need to know what an XEvent is, where allocator events live, or how
`XLine.timestamp_ns` and `XEvent.offset_ps` combine into a wall clock.

## When to use

Single purpose: "we want to reduce HBM peak — what is sitting in HBM at
the worst moment, and which call sites own it." If you need static
peak (XLA layout) or HLO-instruction-level attribution, that requires
`xla_dump/` artifacts and is **out of scope** for this skill — see
[Limitations](#limitations).

## Concepts you need first

- **`alive_at_peak`** is the set of buffers with `alloc_ts_ns ≤ peak.ts_ns < dealloc_ts_ns` (or no dealloc seen). The peak is the moment `Σ requested_bytes` of live buffers maximises within the chosen step window. The set is taken from runtime allocator events on `/host:CPU` (`MemoryAllocation` / `MemoryDeallocation`).
- **`lifetime_class`** is a heuristic over each alive buffer:
  - `persistent` ⇐ `crossed_step_boundaries ≥ persistent_threshold_steps` (default 2) **and** never deallocated within the trace.
  - `transient` ⇐ alloc and dealloc both within the same step interval.
  - `unknown` ⇐ otherwise (allocated near a step boundary; or trace truncation hides the dealloc — common, since the fixture has 3,698 allocs vs only 106 deallocs).
  Trace truncation biases `unknown` ↑. Use `crossed_step_boundaries` to separate truly-persistent (weights, optimizer state) from trace-truncated-unknown.
- **Timeline vs peak scope.** `timeline.samples` and `timeline.events_of_interest` span the **full trace** so cross-step trend is visible (plateau = persistent baseline; spike = per-step transient). `peak`, `alive_at_peak`, and `rollups` are scoped to the **chosen step** (default: the step containing the global peak).
- **Pool model.** HBM is `id=0`. Other pools are omitted by default; pass `--include-host-pools` to surface them. The fixture has only `id=0`.
- **Dealloc events do not carry pool `id`.** The skill matches deallocs to allocs by `addr` alone. Single-pool captures (the common case) are unambiguous; if a future capture surfaces multiple pools simultaneously, a `warnings` entry flags any same-addr-in-two-pools ambiguity.

## CLI and examples

```bash
# Default: peak step, Top-30 alive buffers, full-trace timeline
python3 .../memory_profile.py /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

# Whole trace, no step scoping
python3 .../memory_profile.py <profile_dir> --all-trace

# A specific Steps-line index
python3 .../memory_profile.py <profile_dir> --step 3

# Last step on the Steps line
python3 .../memory_profile.py <profile_dir> --step-policy last

# Larger Top-K and finer timeline
python3 .../memory_profile.py <profile_dir> --top 100 --time-samples 500
```

## JSON schema cheat-sheet

```json
{
  "status": "ok",
  "skill": "memory-profile",
  "version": 1,
  "inputs":   { "profile_dir": "...", "xplane_pb": "...", "n_planes": 8, "host_plane_present": true },
  "step":     { "id": 3, "policy": "peak", "range_ns": [lo, hi], "source": "steps_line" },
  "pool":     { "id": 0, "bytes_reserved": ... },
  "peak":     { "ts_ns": ..., "bytes_total": ..., "bytes_by_pool": {"0": ...},
                "fragmentation_at_peak": 0.07, "is_global_peak": true },
  "alive_at_peak": {
    "n_buffers": ..., "total_bytes": ...,
    "buffers": [ /* Top-K records with addr, size_bytes, shape, tf_op,
                    data_type, alloc_ts_ns, age_ns_at_peak,
                    crossed_step_boundaries, parent_chain,
                    lifetime_class, deallocated */ ],
    "tail": { "n_buffers": ..., "total_bytes": ... }
  },
  "rollups": {
    "by_lifetime_class": [...],   /* full set, no Top-K */
    "by_shape":          [...],   /* Top-K by total_bytes desc + tail */
    "by_tf_op":          [...],   /* Top-K + tail; <no tf_op> collapses */
    "by_parent_jit":     [...],   /* Top-K + tail */
    "by_dtype":          [...]    /* full set, no Top-K */
  },
  "timeline": {
    "samples": [ {"ts_ns": ..., "bytes_allocated": ..., "live_count": ..., "fragmentation": ...} ],
    "events_of_interest": [ {"kind": "global_peak"|"step_start"|"step_end"|"step_local_peak", ...} ],
    "axis_units": { "ts_ns": "nanoseconds since epoch", "bytes": "bytes (base-2)" }
  },
  "diagnostics": {
    "alloc_accounting_drift_pct": 0.42,
    "unmatched_dealloc_count": 0, "pretrace_dealloc_count": 103, "unmatched_alloc_count": 3592,
    "trace_end_live_bytes": ..., "n_pools_seen": 1,
    "pools_summary": [ {"pool_id": 0, "n_alloc": 3698, "n_dealloc": 106,
                        "max_peak_bytes_in_use": ...} ],
    "step_line_present": true,
    "shape_missing_count": 0, "tf_op_missing_count": 12,
    "warnings": []
  }
}
```

## Invariants (consistency gates)

| # | Invariant | Tolerance |
|---|---|---|
| I1 | `Σ buffers[*].size_bytes + tail.total_bytes == alive_at_peak.total_bytes` | exact |
| I2 | `alive_at_peak.total_bytes == peak.bytes_total` | exact |
| I2b | `\|peak.bytes_total − allocator's bytes_allocated at peak_ts\| / peak.bytes_total ≤ 0.01` | soft; >1% raises a warning. Recorded as `diagnostics.alloc_accounting_drift_pct`. |
| I3 | `Σ rollups.by_shape[*].total_bytes == alive_at_peak.total_bytes` | exact |
| I4 | each of `by_tf_op`, `by_parent_jit`, `by_lifetime_class`, `by_dtype` partitions `alive_at_peak.total_bytes` | exact |
| I5 | `peak.bytes_total ≤ pool.bytes_reserved` | exact |
| I6 | `max(timeline.samples.bytes_allocated) ≥ peak.bytes_total` | exact (timeline is full trace, peak is step-scoped) |
| I7 | every alive buffer has `alloc_ts_ns ≤ peak.ts_ns`; if `deallocated`, dealloc `ts_ns ≥ peak.ts_ns` | exact |
| I8 | `diagnostics.unmatched_dealloc_count == 0` | exact; nonzero is a real bug |
| I9 | `step.range_ns[0] ≤ peak.ts_ns ≤ step.range_ns[1]` (skipped under `--all-trace`) | exact |

## Reading guide

- **"Why is peak this high?"** → `rollups.by_lifetime_class`. `persistent` is the baseline (weights / optimizer state); `transient` is the per-step activation spike. The ratio tells you whether to attack persistent footprint (sharding, lower-precision weights) or activation footprint (rematerialization, smaller microbatch).
- **"Where to cut for biggest win?"** → `rollups.by_tf_op` Top few × `pct_of_peak`. The `parent_jit` rollup answers the same question scoped to one jit boundary.
- **"Is fragmentation severe?"** → `peak.fragmentation_at_peak` (allocator's own score 0..1) and `pool.bytes_reserved − peak.bytes_total` (raw headroom).
- **"Is this step an outlier?"** → `peak.is_global_peak` (chosen step holds the global peak) and `timeline.events_of_interest` step-local peaks across steps.

## Common gotchas

- **`alive_at_peak.tail` cannot be ignored.** `buffers` is Top-K only; `n_buffers` and `total_bytes` are the truth.
- **`step.source == "execute_event"`** means the `Steps` line was missing and the skill fell back to the outermost `Execute (jit_*)` event. In that case `step.id` is a sequential index, NOT the user's training step number.
- **Large `unmatched_alloc_count`** is the expected truncated-trace state, not a bug — these are allocations whose matching dealloc never came because the buffer was still live at trace end.
- **Large `pretrace_dealloc_count`** is also a trace-truncation artifact, NOT a producer bug — these are deallocations whose matching alloc happened before the trace started, so we never recorded the alloc.
- **Single HBM pool reported by default.** Multi-pool captures need `--include-host-pools` to surface host-side or other pools.
- **Dealloc events have no pool `id`.** Match-by-addr is unambiguous when only one pool is in flight (the common case). Same-addr-in-two-pools simultaneously is not produced by XLA's allocator today; the skill flags it via `warnings` if it ever appears.
- **No source-line attribution.** `MemoryAllocation` carries `tf_op` (JAXPR identity) but not `source_stack`. File:line attribution would need HLO buffer assignment, which lives in `xla_dump` and is out of scope.

## Limitations

- **No XLA static peak / no `addr → static slice` cross-check** — both need `xla_dump/`'s `*-memory-usage-report.txt`, which the user's normal capture flow does not produce.
- **No HLO-instruction-level attribution.** `hlo_proto.pb` in current captures lacks `buffer_assignment`. The runtime path uses `tf_op` instead.
- **No per-device split.** Allocator events live on `/host:CPU` and are not split per TPU core.
- **Trace truncation biases `lifetime_class`.** With 3,698 allocs and 106 deallocs in the reference fixture, almost every alive-at-peak buffer is also alive at end-of-trace. The `crossed_step_boundaries` field separates truly-persistent from trace-truncated-unknown.

## Files

- `scripts/memory_profile.py` — main entry script.
- `scripts/_loader.py` — xplane load, plane/line lookup, step window picker, two-pass sweep, rollups.
- `scripts/_proto/` — vendored xplane protobuf bindings (reused from `profile-anatomy/_proto/` via `sys.path.insert`).
- `scripts/tests/` — unit + e2e tests (stdlib `unittest`).
