# tpu-perf: compute-breakdown skill — design

**Status:** draft
**Date:** 2026-05-24
**Author:** brainstorming session
**Predecessor skill:** `tpu-perf:profile-anatomy` (schema dictionary; provides the XSpace/XPlane/XLine/XEvent/XStat reading reference)
**Target plugin:** `plugins/tpu-perf/`

## 1. Purpose

Add a second skill to the `tpu-perf` plugin that turns a captured TPU pretraining profile (`*.xplane.pb`) into actionable compute-efficiency analyses. Four user-stated capabilities:

1. **Top time-eaters by source line.** Aggregate non-communication HLO duration by source code location; report execution count, total / min / max / avg per source.
2. **Layer-scoped breakdown.** Given user-provided code, scope the aggregation to a sub-tree (e.g. one attention layer).
3. **Non-compute audit.** Surface duration spent on padding, type conversion, copy, layout transforms, broadcasts — the "data wrangling" overhead.
4. **v7x roofline shortfall.** Using TPU v7x BF16/FP8 peak compute and HBM bandwidth, compute MFU / HBM utilization / compute-vs-memory bound classification, and report shortfall vs roofline.

The skill consumes the schema documented by `profile-anatomy`; it does **not** re-document protobuf layout. It is the first analysis skill in the `tpu-perf` plugin.

## 2. Approach (decided during brainstorming)

**One skill, one main script, four `--mode` subcommands.** The four capabilities share a single load → step-pick → event-iterate → normalize pipeline (stages 1-3 below); only the final projection (stage 4) differs. Decision rationale: high overlap of underlying parsing logic, shared data contract across modes lets Claude cross-compare, lowest cognitive load (one skill / one script / four modes).

Rejected: 4 separate scripts (4× the load cost of a 298 MB protobuf), 4 separate skills (excessive duplication of "step picking" and "while exclusion" docs).

## 3. Pipeline architecture

```
xplane.pb (~300 MB)
   │
   ▼
Stage 1  load & locate
   • parse XSpace (protobuf)
   • pick device plane (default /device:TPU:0)
   • build stat_metadata reverse map
   • build event_metadata cache (per md_id):
       category, source, source_stack, tf_op,
       flops, model_flops, bytes_accessed,
       raw_bytes_accessed, shape_with_layout, dtype
   │
   ▼
Stage 2  step window
   • read 'Steps' line
   • pick middle step (default) or --step N
   • compute [t_start_ps, t_end_ps] window
   │
   ▼
Stage 3  iterate 'XLA Ops' line, per-event records
   for each XEvent in window:
     resolve metadata → category, source_stack, ...
     skip 'while' (container; double-counts)
     classify into kind ∈ {compute, data_move, comm}
     resolve aggregation key:
       source_stack → tf_op → '<no source>:'+category
   emit one normalized record per event
   │
   ▼
Stage 4  mode-specific projection
   summary       → top-K compute groups + totals
   by_source     → full per-key aggregation table (no truncation)
   non_compute   → data_move (+ async-done by default), by-source roll-up
   roofline      → per-key actual vs theoretical, using v7x peaks + dtype
   │
   ▼
JSON to stdout (one top-level object per invocation)
```

`comm` (async-* / all-reduce) is excluded from all modes by default. `--include-comm` re-enables it. Mode 3 includes `async-done` by default as a "comm-stall" non-compute category; pass `--no-comm-stalls` to exclude.

## 4. Per-event record schema (Stage 3 output)

All four modes derive from the same per-event normalized record:

```python
{
  "duration_ps":        int,       # XEvent.duration_ps
  "device_offset_ps":   int,       # from XEvent.stats (relative to XLine.timestamp_ns)
  "step_id":            int,       # the step (Stage 2) the event was assigned to
  "hlo_category":       str,       # 'loop fusion' | 'custom-call' | ...
  "kind":               str,       # 'compute' | 'data_move' | 'comm'
  "hlo_op":             str,       # XEventMetadata.name (full HLO IR text, not truncated)
  "tf_op":              str|None,  # JAX/XLA jaxpr op-path
  "source":             str|None,  # innermost frame of source_stack
  "source_stack":       str|None,  # full multi-line stack
  "source_stack_hash":  str|None,  # sha1(source_stack)[:16] when source_stack present
  "agg_key":            str,       # unified aggregation key after fallback
  "agg_key_kind":       str,       # 'stack' | 'tf_op' | 'no_source'
  "flops":              int|None,
  "model_flops":        int|None,
  "bytes_accessed":     int|None,
  "raw_bytes_accessed": int|None,
  "shape_with_layout":  str|None,
  "dtype":              str|None,  # parsed from shape_with_layout, see §4.2
  "dtype_uncertain":    bool,      # True for fusion categories where input dtype may differ
  "program_id":         int|None,
  "deduplicated_name":  str|None,
}
```

### 4.1 `agg_key` three-tier fallback

| Priority | Condition | `agg_key` value | `agg_key_kind` |
|---|---|---|---|
| 1 | `source_stack` non-empty | `"stack:" + source_stack_hash` | `"stack"` |
| 2 | `source_stack` empty, `tf_op` non-empty | `"tfop:" + tf_op` | `"tf_op"` |
| 3 | both absent | `"nosrc:" + hlo_category` | `"no_source"` |

This three-tier fallback achieves ~100% event coverage. Real-data validation on `dp8_fsdp128`: 79.7% of events carry `source_stack`, 97.3% carry `tf_op`, 100% carry `hlo_category`.

### 4.2 `dtype` parsing

`shape_with_layout` is e.g. `"bf16[8192,4096]{1,0}"` / `"f8e4m3fn[1024,4096]{1,0}"` / `"f32[]"`. Regex `^([a-z][a-z0-9]*)\[` extracts prefix; mapping:

- `bf16` → `"bf16"`
- `f8e4m3fn` / `f8e5m2` → `"fp8"`
- `f32` → `"fp32"`
- `f16` → `"fp16"`
- everything else (`s32`, `s8`, `pred`, `tuple`, parse failure) → `"other"` (roofline skips these)

### 4.3 `dtype_uncertain` flag

Set `dtype_uncertain=true` when `hlo_category` ∈ `{"convolution fusion", "custom-call", "custom fusion", "output fusion"}` and `dtype` ∈ `{"bf16", "fp32"}`. Rationale: `shape_with_layout` describes the output; mixed-precision matmul/conv may take fp8 inputs and write bf16 outputs, so picking `peak_tflops_bf16` would over-estimate the theoretical ceiling. Other categories (elementwise, data_move) are typically input-output dtype-consistent and stay `false`. The roofline mode still computes, but propagates the flag for Claude's interpretation; the script does not silently switch to a "best-case" peak.

### 4.4 `kind` classification

| `kind` | Member `hlo_category` |
|---|---|
| `compute` | `loop fusion`, `convolution fusion`, `custom fusion`, `output fusion`, `non-fusion elementwise`, `reduce`, `reduce-window`, `sort`, `rng-bit-generator`, `custom-call` |
| `data_move` | `copy-start`, `copy-done`, `data formatting`, `pad`, `broadcast`, `slice`, `dynamic-slice`, `dynamic-update-slice`, `iota`, `convert` |
| `comm` | `async-start`, `async-done`, `all-reduce`, `all-gather`, `reduce-scatter`, `collective-permute` |
| (skipped) | `while` (container; sub-events already counted under their own categories) |

`while` is identified at Stage 3 and never produces a record. Stage 1 separately accumulates `while_total_ps` for reporting in `totals`.

### 4.5 Step window

- Read device plane's `Steps` line, sort events by `device_offset_ps`.
- Default: pick `events[len(events)//2]`.
- `--step N` (0-indexed): pick `events[N]`. Out-of-range → stderr error + exit 1.
- `--step-id ID`: match by `XEventMetadata.name` (e.g. step number string).
- The chosen step's `[start_ps, end_ps]` defines the window; events with `device_offset_ps` falling inside are admitted.
- If `Steps` line is missing or empty, fall back to the full plane time range; record `notes: ["no Steps line; falling back to full-plane window"]` in output.

## 5. Mode 1 — `summary` (capability 1)

**Purpose:** top time-eating compute groups, source-line aggregated.

**CLI:**
```
python3 compute_breakdown.py <profile_dir> --mode summary
  [--device /device:TPU:0] [--step N] [--top 50] [--include-comm]
```

`--top` defaults to 50.

**JSON output (top-level object):**

```json
{
  "status": "ok",
  "mode": "summary",
  "profile_dir": "...",
  "device": "/device:TPU:0",
  "step_id": 7,
  "step_window_ps": [123456789, 234567890],
  "step_duration_ps": 111111101,
  "notes": [],

  "totals": {
    "n_events_total":      45123,
    "n_events_compute":    32100,
    "n_events_data_move":   9876,
    "n_events_comm":        3147,
    "compute_duration_ps":  87654321,
    "data_move_duration_ps": 5432100,
    "comm_duration_ps":     12345678,
    "while_container_duration_ps": 56789012,
    "non_while_duration_ps":      54322089,
    "while_pct_of_step":          51.1
  },

  "agg_key_coverage": {"stack": 25600, "tf_op": 6480, "no_source": 20},

  "top_compute_groups": [
    {
      "rank": 1,
      "agg_key":      "stack:a3f8...",
      "agg_key_kind": "stack",
      "source":       "/root/maxtext/.../attention.py:312",
      "tf_op":        "jit(loss_fn)/.../FlashAttention",
      "source_stack": "/root/.../attention.py:312:18\n...",
      "n_executions": 1024,
      "total_dur_ps": 12345678,
      "min_dur_ps":      890,
      "max_dur_ps":    23456,
      "avg_dur_ps":    12056,
      "pct_of_compute":  14.1,
      "pct_of_step":      8.3,
      "hlo_categories": {"loop fusion": 1024},
      "flops_sum":             12345678901234,
      "bytes_accessed_sum":     9876543210,
      "example_hlo_op": "fusion.123 = bf16[8192,4096] fusion(...) kind=kLoop ..."
    }
  ],

  "tail_compute": {"n_groups_omitted": 782, "dur_ps": 1234567},

  "by_kind_rollup": {
    "compute":   {"n": 32100, "dur_ps": 87654321, "pct_of_step": 78.9},
    "data_move": {"n":  9876, "dur_ps":  5432100, "pct_of_step":  4.9},
    "comm":      {"n":  3147, "dur_ps": 12345678, "pct_of_step": 11.1}
  }
}
```

**Notes:**
- `top_compute_groups` only covers `kind=compute`.
- `pct_of_compute` denominator: `compute_duration_ps`. `pct_of_step` denominator: `step_duration_ps` (includes while).
- `flops_sum` / `bytes_accessed_sum` per-group: skip individual events with null fields when summing; emit `null` only if all events in the group lack the field.
- `example_hlo_op`: first hlo_op text seen for the agg_key (not full enumeration).

## 6. Mode 2 — `by_source` (capability 2)

**Purpose:** complete per-`agg_key` table for Claude-side filtering. Claude reads the user's code (e.g. attention.py), infers a scope (file path / function name / jaxpr path fragment), then post-filters this table.

**CLI:**
```
python3 compute_breakdown.py <profile_dir> --mode by_source
  [--device /device:TPU:0] [--step N] [--include-comm] [--include-data-move]
```

`--include-data-move` defaults to false (this mode is `kind=compute` only by default).

**JSON output:**

```json
{
  "status": "ok",
  "mode": "by_source",
  "profile_dir": "...",
  "device": "/device:TPU:0",
  "step_id": 7,
  "step_window_ps": [...],
  "step_duration_ps": 111111101,
  "notes": [],

  "totals": {
    "compute_duration_ps":   87654321,
    "data_move_duration_ps":  5432100,
    "comm_duration_ps":      12345678,
    "while_container_duration_ps": 56789012,
    "n_groups_total":             832
  },

  "groups": [
    {
      "agg_key":      "stack:a3f8...",
      "agg_key_kind": "stack",
      "source":       "/root/.../attention.py:312",
      "source_stack": "/root/.../attention.py:312:18\n...",
      "tf_op":        "jit(loss_fn)/.../FlashAttention",
      "kind":         "compute",
      "hlo_categories": {"loop fusion": 1024, "convolution fusion": 32},
      "n_executions": 1056,
      "total_dur_ps": 12345678,
      "min_dur_ps":   890,
      "max_dur_ps":   23456,
      "avg_dur_ps":   11689,
      "flops_sum":          12345678901234,
      "model_flops_sum":    11111111111111,
      "bytes_accessed_sum":  9876543210,
      "shapes": ["bf16[8192,4096]{1,0}", "bf16[1024,8192]{1,0}"],
      "shapes_truncated": false,
      "dtypes": {"bf16": 1056},
      "dtype_uncertain": false,
      "example_hlo_op": "fusion.123 = bf16[8192,4096] fusion(...) ..."
    }
  ]
}
```

**Notes:**
- **Not sorted, not truncated.** Claude sorts and selects.
- `shapes`: deduped list of `shape_with_layout` values; cap at 8; if exceeded set `shapes_truncated: true`.
- `dtypes`: histogram over the group's events.
- Layer-scoping recipe (in SKILL.md): (1) read user code → infer file path + function name; (2) run by_source; (3) filter `groups` where `source_stack` contains file path OR `tf_op` contains function name; (4) sum `total_dur_ps`; (5) report % of compute / step.

## 7. Mode 3 — `non_compute` (capability 3)

**Purpose:** all `kind=data_move` (plus `async-done` by default) events, broken down by category and by source.

**CLI:**
```
python3 compute_breakdown.py <profile_dir> --mode non_compute
  [--device /device:TPU:0] [--step N] [--include-comm] [--no-comm-stalls]
```

By default `async-done` events are included as `hlo_category="async-done (comm stall)"`. `--no-comm-stalls` excludes them. When included:
- `totals.non_compute_pct_of_step` numerator includes async-done duration.
- `notes` includes `"async-done included as comm-stall non-compute time; pass --no-comm-stalls to exclude"`.

**JSON output:**

```json
{
  "status": "ok",
  "mode": "non_compute",
  "profile_dir": "...",
  "device": "/device:TPU:0",
  "step_id": 7,
  "step_window_ps": [...],
  "step_duration_ps": 111111101,
  "notes": ["async-done included as comm-stall non-compute time; pass --no-comm-stalls to exclude"],

  "totals": {
    "compute_duration_ps":      87654321,
    "data_move_duration_ps":     5432100,
    "comm_duration_ps":         12345678,
    "non_compute_pct_of_step":      4.9,
    "non_compute_pct_of_compute":   6.2
  },

  "by_category": [
    {
      "hlo_category":  "data formatting",
      "n_executions":  11028,
      "total_dur_ps":  306613000000,
      "min_dur_ps":    150,
      "max_dur_ps":    98765,
      "avg_dur_ps":    27800,
      "n_groups":      48,
      "agg_key_coverage": {"stack": 6440, "tf_op": 4500, "no_source": 88}
    }
  ],

  "by_source_within_category": [
    {
      "hlo_category":  "data formatting",
      "agg_key":       "stack:a3f8...",
      "agg_key_kind":  "stack",
      "source":        "/root/.../attention.py:312",
      "source_stack":  "...",
      "tf_op":         "jit(loss_fn)/.../transpose",
      "n_executions":  1024,
      "total_dur_ps":  234560000,
      "min_dur_ps":    120,
      "max_dur_ps":    9999,
      "avg_dur_ps":    229,
      "shapes_in":  ["bf16[8192,4096]{1,0}"],
      "shapes_out": ["bf16[4096,8192]{0,1}"],
      "dtype_change":  false,
      "layout_change": true,
      "example_hlo_op": "transpose.123 = bf16[4096,8192]{0,1} transpose(...)"
    }
  ]
}
```

**Notes:**
- Two layers: `by_category` (one row per hlo_category, no thresholding), `by_source_within_category` (full (category, agg_key) breakdown, not truncated, not sorted).
- `dtype_change` / `layout_change` heuristic: regex-parse the `hlo_op` IR text to extract `op_name = T_out[shape_out]{layout_out} <op>(T_in[shape_in]{layout_in}, ...)`. Compare dtype prefixes and `{...}` layout. If parse fails → both fields `null` (Claude is told null = unknown, not "no change").
- `shapes_in` / `shapes_out`: cap at 4 each; `null` if not parseable.
- SKILL.md note: `copy-start` / `copy-done` carry no source — XLA-internal DMA, not user-code-driven; real copy waste shows up in `data formatting` and `broadcast`.

## 8. Mode 4 — `roofline` (capability 4)

**Purpose:** dtype-aware roofline analysis on v7x peaks; per-group MFU, HBM utilization, compute-vs-memory bound, shortfall vs theoretical.

### 8.1 v7x peak table (per-device = per-TensorCore)

Source: https://docs.cloud.google.com/tpu/docs/tpu7x. v7x chip has 2 TensorCores; `/device:TPU:N` is one TensorCore, so per-chip values are divided by 2.

| Spec | per chip | per device (÷2) |
|---|---:|---:|
| Peak BF16 (TFLOPS) | 2307 | 1153.5 |
| Peak FP8  (TFLOPS) | 4614 | 2307.0 |
| HBM bandwidth (GiB/s) | 7380 | 3690 |
| HBM capacity (GiB) | 192 | 96 |

`fp32` and `fp16` peaks are not officially listed → table value `None`; roofline mode skips groups with these dtypes unless `--peak-tflops-fp32` / `--peak-tflops-fp16` is provided.

**Unit discipline:** `bytes_accessed` is in bytes (byte = 8 bits). HBM bandwidth is GiB/s (base-1024). The conversion `bytes / (gib_per_sec * 1024**3) seconds` MUST use base-1024 throughout; do not mix GiB and GB. The `peaks_used` block in output explicitly tags `"unit": "GiB/s (base-1024) per device"`.

### 8.2 Formulas

For each group with `flops_sum` not null and `bytes_accessed_sum` not null and `dtype` ∈ {"bf16", "fp8", "fp16", "fp32"} (where peak is known):

```
arithmetic_intensity = flops_sum / bytes_accessed_sum                       [FLOPs/byte]
ridge_point          = peak_tflops * 1e12 / (peak_hbm_gibps * (1024**3))    [FLOPs/byte]

t_compute_theory_ps  = flops_sum / (peak_tflops * 1e12) * 1e12              [ps]
t_hbm_theory_ps      = bytes_accessed_sum / (peak_hbm_gibps * (1024**3)) * 1e12   [ps]
t_roofline_theory_ps = max(t_compute_theory_ps, t_hbm_theory_ps)

bound = "compute" if arithmetic_intensity >= ridge_point else "memory"

mfu           = t_compute_theory_ps  / total_dur_ps         # 0..1, higher = better
hbm_util      = t_hbm_theory_ps      / total_dur_ps         # 0..1, higher = better
roofline_util = t_roofline_theory_ps / total_dur_ps         # the achievement ratio
shortfall_ps  = total_dur_ps - t_roofline_theory_ps         # absolute waste vs roofline
shortfall_pct = (1 - roofline_util) * 100
```

The shortfall formula is the user's "actual vs theoretical" decomposition: `shortfall_ps` is the slack between observed time and the roofline-theoretical lower bound, given the per-group dtype.

### 8.3 CLI

```
python3 compute_breakdown.py <profile_dir> --mode roofline
  [--device /device:TPU:0] [--step N]
  [--chip v7x]
  [--peak-tflops-bf16 ...] [--peak-tflops-fp8 ...]
  [--peak-tflops-fp32 ...] [--peak-tflops-fp16 ...]
  [--peak-hbm-gibps ...]
```

### 8.4 JSON output

```json
{
  "status": "ok",
  "mode": "roofline",
  "profile_dir": "...",
  "device": "/device:TPU:0",
  "step_id": 7,
  "step_window_ps": [...],
  "step_duration_ps": 111111101,
  "notes": [],

  "chip": "v7x",
  "peaks_used": {
    "peak_tflops_bf16": 1153.5,
    "peak_tflops_fp8":  2307.0,
    "peak_tflops_fp32": null,
    "peak_tflops_fp16": null,
    "peak_hbm_gibps":   3690.0,
    "unit":             "GiB/s (base-1024) per device",
    "ridge_points":     {"bf16": 320.4, "fp8": 640.7},
    "source":           "builtin v7x table"
  },

  "step_summary": {
    "step_compute_duration_ps":   87654321,
    "weighted_avg_mfu":              0.412,
    "weighted_avg_hbm_util":         0.683,
    "weighted_avg_roofline_util":    0.701,
    "step_shortfall_ps":          26200000,

    "top_shortfall_groups": [
      {
        "agg_key":      "stack:a3f8...",
        "source":       "/root/.../attention.py:312",
        "tf_op":        "jit(loss_fn)/.../FlashAttention",
        "total_dur_ps": 12345678,
        "shortfall_ps":  4567890,
        "bound":        "memory"
      }
    ]
  },

  "groups": [
    {
      "agg_key":      "stack:a3f8...",
      "agg_key_kind": "stack",
      "source":       "/root/.../attention.py:312",
      "tf_op":        "jit(loss_fn)/.../FlashAttention",
      "hlo_categories": {"convolution fusion": 32},
      "n_executions": 32,
      "total_dur_ps": 12345678,
      "flops_sum":          12345678901234,
      "bytes_accessed_sum":  9876543210,
      "dtype":              "bf16",
      "dtype_uncertain":    true,

      "arithmetic_intensity": 1249.7,
      "ridge_point":           320.4,
      "bound":                 "compute",
      "t_compute_theory_ps":  10703440,
      "t_hbm_theory_ps":       2486040,
      "t_roofline_theory_ps": 10703440,
      "mfu":                       0.867,
      "hbm_util":                  0.201,
      "roofline_util":             0.867,
      "shortfall_ps":           1642238,
      "shortfall_pct":            13.3
    }
  ],

  "skipped_groups": {
    "n_no_flops":           42,
    "n_no_bytes":            7,
    "n_dtype_other":         9,
    "n_peak_unknown_for_dtype": 0,
    "total_dur_ps_skipped": 234567
  }
}
```

**Notes:**
- `weighted_avg_*` use `total_dur_ps` as weights (not arithmetic mean).
- Roofline mode runs on `kind=compute` only by default. `data_move` is excluded (flops typically 0). To inspect data_move HBM utilization, use mode 3.
- `dtype_uncertain=true` groups are still computed; the flag is propagated but the script does not silently switch to a higher peak.
- `peaks_used.source` ∈ `{"builtin v7x table", "cli override", "profile peak_*"}`. Default precedence: builtin → cli override (if any flag passed). Profile-derived peaks are not used by default; reserved for a future `--peak-source profile` flag.
- `top_shortfall_groups` carries 5 fields per entry (source, tf_op, total_dur_ps, shortfall_ps, bound), top 10 by shortfall_ps. Lets Claude answer "where is the biggest waste" in one glance.

## 9. SKILL.md structure

File: `plugins/tpu-perf/skills/compute-breakdown/SKILL.md`.

```
---
name: compute-breakdown
description: Use when analyzing TPU pretraining compute efficiency from
  xplane.pb — produces source-line-aggregated HLO duration tables, layer-
  scoped breakdowns, non-compute (padding/cast/copy) audits, and v7x
  roofline shortfall vs theoretical peak. Reads schema documented by
  profile-anatomy.
---

# Compute Breakdown

[1-paragraph overview: 4 modes built on profile-anatomy schema.]

## When to use which mode

| Question | Mode |
|---|---|
| "Top time-eaters in this profile" | `summary` |
| "How much time does X layer / module spend" | `by_source`, then filter |
| "How much time goes to padding/cast/copy/transpose" | `non_compute` |
| "Are we compute- or memory-bound; what's MFU on v7x" | `roofline` |

## Concepts you need first

- agg_key: source_stack hash → tf_op → '<no source>:'+category fallback
- `while` HLO is a container, skipped from per-event tables; reported as
  `while_container_duration_ps` separately
- v7x peaks are per-device (TensorCore), chip values ÷ 2
- `copy-start` / `copy-done` carry no source — XLA-internal DMA; real
  copy waste shows up in `data formatting` / `broadcast`
- `dtype_uncertain=true` for fusions whose inputs may differ in precision
  from outputs; roofline still computes but the flag is conveyed
- HBM bandwidth uses GiB/s (base-1024); do not mix with GB

## Mode 1 — summary
[invocation, JSON shape, reading guide]

## Mode 2 — by_source (layer scoping)
[invocation, then 5-step layer scoping recipe]

## Mode 3 — non_compute
[invocation, JSON shape, dtype_change/layout_change interpretation]

## Mode 4 — roofline
[invocation, peaks_used, MFU/HBM_util/bound interpretation,
 shortfall formula recap]

## Common gotchas
[mirror profile-anatomy's gotchas + new ones for this skill]

## Files
- scripts/compute_breakdown.py — main entry
- scripts/_peaks.py — v7x peak table
- scripts/_proto/ — vendored xplane protobuf (copy of profile-anatomy's)
```

## 10. Error handling

Output contract: **always emit a top-level-valid JSON object on stdout**, even on absent inputs. Top-level fields `"status": "ok" | "absent"`, `"reason": str|null`, `"notes": str[]`. Differs from profile-anatomy's `[absent]` text style because this skill's contract is "consumed as JSON by Claude"; profile-anatomy is "schema docs for humans".

| Situation | Behavior |
|---|---|
| `profile_dir` missing or no `*.xplane.pb` | `{"status":"absent","reason":"no_xplane_pb",...}`, exit 0 |
| selected device plane absent | `{"status":"absent","reason":"device_not_found","notes":["have: [...]"]}`, exit 0 |
| `XLA Ops` line absent on plane | `{"status":"absent","reason":"no_xla_ops_line",...}`, exit 0 |
| `Steps` line missing or empty | degrade to full-plane window, `notes:["no Steps line; falling back to full-plane window"]`, status `ok` |
| `--step N` out of range | stderr error + exit 1 (explicit user-arg error, not silent) |
| roofline mode + all dtypes "other" | `groups: []`, fully populated `skipped_groups`, status `ok` |
| roofline mode + dtype peak is null and no override | group goes to `skipped_groups.n_peak_unknown_for_dtype`, status `ok` |

## 11. Testing strategy

No build/lint/test framework in repo. Manual verification steps for the spec's implementer:

1. **Smoke** — each mode parses to valid JSON:
   ```bash
   for m in summary by_source non_compute roofline; do
     python3 .../compute_breakdown.py \
       /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
       --mode $m | python3 -m json.tool > /dev/null
   done
   ```
2. **Cross-mode consistency:**
   - `summary.totals.compute_duration_ps` ≈ `by_source.totals.compute_duration_ps`
   - `summary.totals.data_move_duration_ps` ≈ `non_compute.totals.data_move_duration_ps` (with `--no-comm-stalls`)
   - All four modes agree on `step_window_ps`.
3. **Error paths:**
   - non-existent dir → `status=absent`, `reason=no_xplane_pb`, exit 0
   - non-existent device → `status=absent`, `reason=device_not_found`, exit 0
   - `--step 999` (oob) → stderr message, exit 1
4. **Sanity bounds:**
   - all `mfu` ∈ [0, 1.05] (slack for measurement noise)
   - all `hbm_util` ∈ [0, 1.05]
   - `weighted_avg_roofline_util` ∈ [0, 1.05]
   - `shortfall_ps` ≥ −small_epsilon
5. **v7x peak fidelity:** confirm `peaks_used.peak_tflops_bf16 == 1153.5` and `peaks_used.peak_hbm_gibps == 3690.0` when invoked without overrides.

## 12. Relationship with profile-anatomy

- **Documentation**: SKILL.md does NOT duplicate the protobuf schema. It assumes the reader (Claude) has the profile-anatomy skill available, and references it for "what's an XEvent / XEventMetadata.stats / source_stack semantics".
- **Code**: `_proto/` is **copied** (not symlinked) into compute-breakdown's `scripts/` directory. Rationale: skill must be self-contained for distribution. The two `_proto/` copies must stay in sync; `xplane.proto` upstream changes require updating both. Recorded as a maintenance note in this spec.

## 13. Out of scope (explicitly deferred)

These were considered and removed per YAGNI:

- **Multi-device aggregation across all `/device:TPU:N`** — only `/device:TPU:0` by default. Adding `--device all` would 8x the JSON size for the dp8_fsdp128 fixture.
- **Multi-step weighted analysis** — middle step only by default. Cross-step variance is a separate question; future skill.
- **Threshold-based anomaly flagging in mode 3** — explicitly rejected; full-table output, Claude judges.
- **Layer-scope CLI flag** — Claude does scoping client-side via post-filter. Keeps the script's data contract uniform across modes.
- **Auto-switching peak by dtype_uncertain** — flag propagated, no silent peak swap.
- **Reading `peak_*` from XPlane.stats by default** — defaults to builtin table (deterministic); profile-derived only if a future opt-in flag is added.

## 14. Implementation notes for the plan phase

- Single Python file `compute_breakdown.py` (~400-500 lines projected). Stages 1-3 in shared helper functions; stage 4 dispatched on `--mode`.
- One dataclass `EventRecord` for the normalized record (§4).
- Module `_peaks.py` holds the v7x table and `--override` resolution.
- Vendored `_proto/` copied from profile-anatomy at scaffolding time.
- Invariant: every mode's output passes `json.loads()` and has `status` ∈ `{"ok","absent"}`.
- Performance budget: a single 298 MB `.xplane.pb` parses + analyzes within ~10 s on a laptop. The 4-mode shared parsing makes "run all 4 then cross-check" feasible without re-parsing.
