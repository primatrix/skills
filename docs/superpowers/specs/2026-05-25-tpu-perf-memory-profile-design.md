# tpu-perf `memory-profile` skill — Design

Date: 2026-05-25
Plugin: `tpu-perf`
Status: Approved (pending user review of this written spec)

## 0. Goal

Add a fourth skill under `plugins/tpu-perf/` that answers — for a TPU
pretraining profile — **when does HBM peak, and what is alive at that
moment**, in a form Claude can read structurally and turn into
optimization recommendations.

The skill works on captures that contain only `xplane.pb` (no
`xla_dump/`, no `op_stats.pb` memory-profile sub-message). XLA-static
peak / HLO-slice attribution / runtime↔static cross-check are
explicitly **out of scope** for this skill — they require dump
artifacts that the user's normal capture flow does not produce.

## 1. Scope and non-goals

### In scope

- Locate the peak HBM occupancy moment within a chosen step (or across
  the whole trace).
- List every buffer alive at that moment, each annotated with
  `(size, shape, tf_op, parent_jit, lifetime_class)`.
- Roll up the alive set by lifetime-class / shape / tf_op / parent
  jit / dtype.
- Sample a `bytes_allocated` timeline across the full trace so Claude
  can see the shape (plateau = persistent baseline; spike =
  per-step transient).
- Report runtime fragmentation indicators
  (`bytes_reserved − bytes_allocated`, allocator's own `fragmentation`
  field at peak).
- Cross-mode invariants (sums match, monotonicity, step containment).

### Out of scope (and why)

| Excluded capability | Reason |
|---|---|
| XLA static peak from `*-memory-usage-report.txt` | User's capture flow does not dump `xla_dump/`. Adding it would create an opt-in path the fixture does not exercise. |
| `addr → static slice` cross-check | Same — needs static report. |
| HLO-instruction-level attribution via `BufferAssignmentProto` | `hlo_proto.pb` in the reference fixture has `memory_space` but no `buffer_assignment` block. Captures at this site do not include it. |
| Per-device split (TPU:0 vs TPU:1) | Allocator events live on `/host:CPU` and are not split per device. This is a capture-format limitation, not a skill choice. |

## 2. Data sources (verified against fixture)

Fixture: `/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/gke-tpu-4233cc6e-d8q7.xplane.pb`.

Memory data lives entirely on `/host:CPU` plane:

- **`MemoryAllocation` events** on `pjrt_tpu_execute/*` and `main/*`
  XLines (3,698 events in fixture). Per-event stats:
  - `addr` (int64) — device address. Together with `id` (pool) it
    keys a buffer.
  - `id` (int64) — allocator pool. Fixture shows only `id=0` (HBM).
  - `requested_bytes` (int64) — what user code asked for.
  - `allocation_bytes` (int64) — what the allocator carved out (≥
    `requested_bytes` due to alignment padding).
  - `bytes_allocated` (int64) — running pool occupancy after this
    event.
  - `bytes_reserved` (int64) — pool size.
  - `bytes_available` (int64) — pool free.
  - `peak_bytes_in_use` (int64) — allocator's running peak so far.
  - `fragmentation` (double) — pool fragmentation score 0..1.
  - `shape` (str) — e.g. `bf16[2,8192,2560]`. Empty for
    allocator-internal buffers.
  - `tf_op` (str) — JAX/JAXPR op identity. Empty for some internal
    buffers.
  - `data_type` (int64) — XLA dtype enum (resolved to string in
    output).
  - `index_on_host` (int64) — internal sequencing.

- **`MemoryDeallocation` events** on `futex-default-SDomainT/*` and
  occasionally on `main/*` (106 events in fixture, vs 3,698 allocs —
  the trace window is short, most buffers stay live to the end). Per
  event:
  - `addr`, `bytes_allocated`, `bytes_available`, `bytes_reserved`,
    `peak_bytes_in_use`, `fragmentation`, `index_on_host`.
  - **No `id` (pool) stat on dealloc events.** `addr` alone is the
    join key. The fixture has a single pool, so this is unambiguous;
    if a future capture has multiple pools, see §6 limitations.

- **Step boundaries** come from `/device:TPU:0` plane's `Steps`
  XLine, same epoch as `/host:CPU` (both `XLine.timestamp_ns` are
  nanoseconds since epoch — verified by profile-anatomy).

## 3. Skill layout

Path: `plugins/tpu-perf/skills/memory-profile/`

```
memory-profile/
  SKILL.md
  scripts/
    memory_profile.py      # single CLI entry, no --mode
    _loader.py             # xplane load, plane/line lookup, step window picker
    _proto/                # sys.path.insert reuse of profile-anatomy/_proto
    tests/
      test_snapshot.py
      test_step_selection.py
      test_alive_set.py
      test_invariants.py
```

Register `memory-profile` in `plugins/tpu-perf/.claude-plugin/plugin.json`'s
`skills` list. `marketplace.json` is unchanged (plugin already registered).

## 4. CLI

```
python3 memory_profile.py <profile_dir>
  [--step N | --step-policy {peak,last,first}]   # default: peak
  [--all-trace]                                  # disable step scoping
  [--top K]                                      # default: 30
  [--persistent-threshold-steps N]               # default: 2
  [--include-host-pools]                         # default: HBM pool only (id=0)
  [--time-samples N]                             # default: 200
```

Single top-level JSON object on stdout. `status: "ok"` or
`status: "absent"` (with `reason` and exit 0). No subcommands.

## 5. Algorithm

### 5.1 Load (`_loader.py`)

1. Parse `xplane.pb` → `XSpace`.
2. Locate `/host:CPU` plane. If absent, return absent.
3. Build reverse maps `stat_name → metadata_id`,
   `event_name → metadata_id`.
4. Collect every `MemoryAllocation` and `MemoryDeallocation` event
   into a flat record list. `ts_ns = line.timestamp_ns + ev.offset_ps // 1000`.
5. Within each XLine, sort events by `(offset_ps, -duration_ps)` and
   sweep with a nesting stack to derive each alloc/dealloc's
   `parent_chain` — the metadata names of every event whose
   `[start, start+duration]` window contains it. This makes the
   parent jit naturally appear in the chain.

### 5.2 Step window selection

Order:

1. `--all-trace` → `range_ns = (-inf, +inf)`, `step.source = "all_trace"`.
2. `--step N` → take the N-th event on `/device:TPU:0`'s `Steps`
   line. Out-of-range → absent.
3. `--step-policy peak` (default) →
   - First do a full-trace alloc/dealloc sweep to find
     `t* = argmax(bytes_allocated)`.
   - Match `t*` to a `Steps` line event by interval containment.
   - If no `Steps` line, fall back to the outermost
     `[N] CommonPjRtLoadedExecutable::Execute (jit_*)` event that
     contains `t*` — record `step.source = "execute_event"`.
4. `--step-policy {last, first}` → simple Steps-line index.

`step.id` = the matched event's metadata.name when available
(e.g. `step_3`); else its sequential index, recorded with a flag.

### 5.3 Main sweep

**Pass 1 — full trace.** Merge all alloc/dealloc events sorted by
`ts_ns`. Maintain:

- `live: dict[(pool_id, addr)] → AllocEvent`
- `bytes_now: dict[pool_id] → int` (sum of `requested_bytes` of live
  buffers)
- `peak_now: dict[pool_id] → (ts_ns, bytes)`
- `samples: list` of `(ts_ns, bytes_allocated, live_count, fragmentation)`
  at N equally-spaced timestamps.

At each event, reconcile our `bytes_now` with the allocator's own
`bytes_allocated` stat. The relative error is recorded as
`diagnostics.alloc_accounting_drift_pct`. >1% raises a warning but
does not fail (alignment padding & internal allocator metadata are
expected drift sources).

**Pass 2 — step-scoped peak.** Inside `step.range_ns`, find
`peak_ts_ns = argmax(bytes_now)`. Snapshot `live` at that instant.

### 5.4 Alive buffer record

```json
{
  "addr": 97729695232,
  "pool_id": 0,
  "size_bytes": 65536,
  "alloc_bytes": 65536,
  "shape": "bf16[2,8192,2560]",
  "tf_op": "jit(train_step)/...",
  "data_type": "bf16",
  "alloc_ts_ns": 102083809,
  "age_ns_at_peak": 12345678,
  "crossed_step_boundaries": 2,
  "parent_chain": ["[3] Execute (jit_train_step)", "AllocateOutputBuffersWithInputReuse", "AllocateRawBuffer"],
  "lifetime_class": "persistent",
  "deallocated": false
}
```

`lifetime_class` heuristic:

- `persistent` ⇐ `crossed_step_boundaries ≥ persistent_threshold_steps`
  (default 2) **and** never deallocated within the trace.
- `transient` ⇐ alloc and dealloc both within the same step interval.
- `unknown` ⇐ otherwise (e.g. allocated near a step boundary, or
  trace truncation hides the dealloc).

### 5.5 Rollups

Each row carries `n_buffers`, `total_bytes`, `pct_of_peak`, and
where applicable `lifetime_mix = {persistent, transient, unknown}`
in bytes. Buffers are not double-counted; per-rollup sums equal
`alive_at_peak.total_bytes` (invariants I3, I4).

- `by_lifetime_class` — full set, no Top-K.
- `by_shape` — Top-K by `total_bytes` desc, plus `tail`.
- `by_tf_op` — Top-K by `total_bytes` desc, plus `tail`. Empty
  `tf_op` collapses to a single `<no tf_op>` row.
- `by_parent_jit` — Top-K. `parent_jit` is the first event in
  `parent_chain` matching `jit_*`; falls back to the chain root.
- `by_dtype` — full set, no Top-K (dtype cardinality is small).

## 6. Top-level JSON schema

```json
{
  "status": "ok",
  "skill": "memory-profile",
  "version": 1,
  "inputs": {
    "profile_dir": "...",
    "xplane_pb": "...",
    "n_planes": 8,
    "host_plane_present": true
  },
  "step": {
    "id": 3,
    "policy": "peak",
    "range_ns": [102000000000, 102680000000],
    "source": "steps_line"
  },
  "pool": {
    "id": 0,
    "bytes_reserved": 51848807424
  },
  "peak": {
    "ts_ns": 102556789012,
    "bytes_total": 12108808192,
    "bytes_by_pool": {"0": 12108808192},
    "fragmentation_at_peak": 0.0731,
    "is_global_peak": true
  },
  "alive_at_peak": {
    "n_buffers": 312,
    "total_bytes": 12108808192,
    "buffers": [/* Top-K records, see §5.4 */],
    "tail": {"n_buffers": 282, "total_bytes": 4023456789}
  },
  "rollups": {
    "by_lifetime_class": [...],
    "by_shape":          [...],
    "by_tf_op":          [...],
    "by_parent_jit":     [...],
    "by_dtype":          [...]
  },
  "timeline": {
    "samples": [
      {"ts_ns": 102083809, "bytes_allocated": 9059102208,
       "live_count": 312, "fragmentation": 0.0731}
    ],
    "events_of_interest": [
      {"kind": "global_peak",      "ts_ns": ..., "bytes": ...},
      {"kind": "step_start",       "ts_ns": ..., "step_id": 3},
      {"kind": "step_end",         "ts_ns": ..., "step_id": 3},
      {"kind": "step_local_peak",  "ts_ns": ..., "step_id": 3, "bytes": ...}
    ],
    "axis_units": {
      "ts_ns": "nanoseconds since epoch",
      "bytes": "bytes (base-2)"
    }
  },
  "diagnostics": {
    "alloc_accounting_drift_pct": 0.42,
    "unmatched_dealloc_count": 0,
    "unmatched_alloc_count": 3592,
    "trace_end_live_bytes": 9058376704,
    "n_pools_seen": 1,
    "pools_summary": [{"pool_id": 0, "n_alloc": 3698, "n_dealloc": 106,
                       "max_peak_bytes_in_use": 12109042176}],
    "step_line_present": true,
    "step_selection_fallback": null,
    "shape_missing_count": 0,
    "tf_op_missing_count": 12,
    "warnings": []
  }
}
```

`timeline.samples` are taken across the full trace window regardless
of step scoping, because cross-step trend is the whole point of this
sub-section. `peak`, `alive_at_peak`, and `rollups` are scoped to
the chosen step.

### 6.1 Absent envelope

```json
{
  "status": "absent",
  "skill": "memory-profile",
  "version": 1,
  "reason": "no MemoryAllocation events found on /host:CPU plane",
  "inputs": {...}
}
```

Trigger conditions (all exit 0, no traceback):

1. `xplane.pb` missing or unparseable.
2. `/host:CPU` plane absent.
3. Zero `MemoryAllocation` events on `/host:CPU` (some old captures
   ship without allocator instrumentation).
4. `--step N` out of range and `--all-trace` not given.

## 7. Invariants

| # | Invariant | Tolerance |
|---|---|---|
| I1 | `Σ buffers[*].size_bytes + tail.total_bytes == alive_at_peak.total_bytes` | exact |
| I2 | `alive_at_peak.total_bytes == peak.bytes_total` | ≤ 1% drift; warning above |
| I3 | `Σ rollups.by_shape[*].total_bytes == alive_at_peak.total_bytes` | exact |
| I4 | `Σ rollups.by_tf_op` and `by_parent_jit` and `by_lifetime_class` and `by_dtype` each = `alive_at_peak.total_bytes` | exact |
| I5 | `peak.bytes_total ≤ pool.bytes_reserved` | exact |
| I6 | `max(timeline.samples.bytes_allocated) ≥ peak.bytes_total` | exact (timeline is full trace, peak is step-scoped) |
| I7 | every alive buffer has `alloc_ts_ns ≤ peak.ts_ns`; if `deallocated`, dealloc `ts_ns ≥ peak.ts_ns` | exact |
| I8 | `diagnostics.unmatched_dealloc_count == 0` | exact; nonzero is a real bug |
| I9 | `step.range_ns[0] ≤ peak.ts_ns ≤ step.range_ns[1]` (skipped under `--all-trace`) | exact |

I2/I3/I4 are the consistency gates of the whole skill. Violations
are recorded in `diagnostics.warnings` with a human-readable
message including the drift percentage.

## 8. Tests

`tests/` with stdlib `unittest`. All tests run end-to-end against
the real `dp8_fsdp128` fixture; no mocking.

| File | Cases |
|---|---|
| `test_step_selection.py` | (a) `policy=peak` selects the step containing the global peak; (b) `policy=last` selects the last `Steps` event; (c) `--step N` out-of-range → absent; (d) `Steps` line absent → falls back to `execute_event` (using a temp xplane fixture variant) |
| `test_alive_set.py` | (a) buffer count == `n_buffers`; (b) every record has all required fields; (c) `lifetime_class` ∈ `{persistent, transient, unknown}` |
| `test_snapshot.py` | end-to-end: `status=ok`, `peak.bytes_total > 0`, `rollups.by_shape` non-empty, Top-K sorted by `total_bytes desc` |
| `test_invariants.py` | I1–I9, with readable failure messages including drift percentages |

## 9. SKILL.md outline

Sections to write (modeled on `compute-breakdown/SKILL.md`):

1. When to use — single purpose: peak moment + alive buffers + attribution.
2. Concepts you need first:
   - Definition of `alive_at_peak`.
   - `lifetime_class` heuristic and its limits (trace truncation
     biases unknown↑).
   - Why `timeline` spans full trace but `peak` is step-anchored.
   - Pool model: HBM is `id=0`; host pools omitted by default.
3. CLI and examples (full-trace; explicit step; default peak step).
4. JSON schema cheat-sheet (paste §6 skeleton).
5. Invariants table (paste §7).
6. Reading guide:
   - "Why is peak this high?" → `rollups.by_lifetime_class`;
     persistent is the baseline (weights / optimizer state);
     transient is the activation spike.
   - "Where to cut for biggest win?" → `rollups.by_tf_op` Top
     few × `pct_of_peak`.
   - "Is fragmentation severe?" → `peak.fragmentation_at_peak`
     and `pool.bytes_reserved − peak.bytes_total`.
   - "Is this step an outlier?" → `peak.is_global_peak` and
     `timeline.events_of_interest.step_local_peak` across steps.
7. Common gotchas:
   - `alive_at_peak.tail` cannot be ignored; `buffers` is Top-K only.
   - When `step.source == "execute_event"`, `step.id` is a sequential
     index (not the user's training step number).
   - Large `unmatched_alloc_count` is the expected truncated-trace
     state, not a bug.
   - The skill reports a single HBM pool; multi-pool captures need
     `--include-host-pools`.
   - `MemoryDeallocation` events do not carry pool `id`. Single-pool
     captures (the common case) are unambiguous; if a future capture
     surfaces multiple pools, allocs are pool-tagged but deallocs
     match by `addr` alone — a `warnings` entry will flag any
     ambiguity (same addr live in two pools simultaneously, which
     XLA's allocator does not currently produce).
8. Files (entry script, helpers, tests).

## 10. Limitations

- **Trace truncation biases attribution.** With 3,698 allocs and
  106 deallocs in the fixture, almost every alive buffer at peak is
  also alive at end-of-trace, so `lifetime_class` distinguishes
  truly-persistent (kept across many step boundaries) from
  trace-truncated-unknown using `crossed_step_boundaries`.
- **No source-line attribution.** `MemoryAllocation` events carry
  `tf_op` (JAXPR identity) but not `source_stack`. Binding allocs
  to user-code file:line would require HLO buffer assignment, which
  is in `xla_dump` and out of scope.
- **No per-device split.** All TPU device allocations route through
  one host allocator and appear on `/host:CPU` without device
  tagging at the allocator-event level. If multiple devices ever
  produce distinct pools, `--include-host-pools` exposes them all
  with `pool_id` separation.
- **Pool semantics inferred from `id=0` only.** The fixture has a
  single pool. Schema names like "HBM" / "host pinned" are not
  written into output — we report the raw `id` and let the reader
  interpret. If future captures show multiple pools, we add a
  `pool_kind` field with a best-effort label.

## 11. Open questions

None at design time. All ambiguous choices (step policy default,
single-mode vs multi-mode, xla_dump opt-in vs drop) were resolved
in brainstorming.
