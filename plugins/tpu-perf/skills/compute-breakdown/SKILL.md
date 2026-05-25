---
name: compute-breakdown
description: Use when analyzing TPU pretraining compute efficiency from xplane.pb — produces source-line-aggregated HLO duration tables, layer-scoped breakdowns, non-compute (padding/cast/copy) audits, and v7x roofline shortfall vs theoretical peak. Reads schema documented by profile-anatomy.
argument-hint: "<profile_dir> --mode {summary|by_source|non_compute|roofline} [--step N] [--top K]"
---

# Compute Breakdown

**回答语言要求：调用此 skill 时，所有面向用户的回答必须使用中文。**

Analyze the compute portion of a TPU pretraining profile. One Python entry script with four `--mode` subcommands sharing a single load → step-pick → event-iterate → normalize pipeline. Always emits a single top-level JSON object on stdout (`status: ok | absent`), so output is consumed structurally — Claude reads the JSON, filters/sums client-side, and reports.

This skill is built on top of `profile-anatomy`, which documents the XSpace/XPlane/XLine/XEvent/XStat hierarchy. Read that first if you need to know what an XEvent is, where source_stack lives, or how `XEventMetadata.stats` differs from `XEvent.stats`.

## When to use which mode

| Question | Mode |
|---|---|
| "Top time-eaters in this profile" | `summary` |
| "How much time does X layer / module spend" | `by_source`, then filter |
| "How much time goes to padding/cast/copy/transpose" | `non_compute` |
| "Are we compute- or memory-bound; what's MFU on v7x" | `roofline` |

## Concepts you need first

- **`agg_key`**: groups events by source location with a 3-tier fallback. Tier 1: SHA-256 hash of `source_stack` (`stack:<8-hex>`). Tier 2: `tf_op` string (`tfop:<value>`). Tier 3: `<no source>:<hlo_category>`. The group's `agg_key_kind` field reports which tier was used.
- **`while` HLO is a container**, not a leaf op. Its events are excluded from per-event tables; their total duration is reported separately as `while_container_duration_ps`. Do **not** double-count it against `compute_duration_ps`.
- **TPU concurrency**: events on the device plane can overlap (Scalar Unit, vector core, async scheduler). Per-kind sums therefore can exceed wall-clock step duration. Treat the durations as throughput proxies, not exclusive time.
- **v7x peaks are per-device** (= per-TensorCore). The v7x chip has 2 TensorCores; `/device:TPU:N` is one of them. Per-chip values are divided by 2: BF16 peak 1153.5 TFLOPS/device, FP8 peak 2307.0 TFLOPS/device, HBM 3690 GiB/s/device.
- **HBM bandwidth uses GiB/s (base-1024)**. Do not mix with GB (base-10). The roofline formulas keep TFLOPS (10^12) and GiB (2^30) separate; the constants do not cancel.
- **`copy-start` / `copy-done` carry no source** — XLA-internal DMA, not user-code-driven. Real copy waste shows up in `data formatting` and `broadcast` categories.
- **`dtype_uncertain=true`** flags a fusion whose inputs may differ in precision from the output (e.g. fp8 inputs, bf16 accumulation). Roofline still computes the per-group MFU using the dominant dtype; flag is propagated so Claude can present a caveat (true peak may be ~2× higher).

## Mode 1 — `summary`

```bash
python3 .../compute_breakdown.py <profile_dir> --mode summary [--step N] [--top K] [--include-comm]
```

Top-K compute groups by source line. JSON has `totals` (per-kind durations, while accounting, agg_key coverage), `top_compute_groups` (top K, sorted by total_dur_ps desc), `tail_compute` (rollup of the rest), `by_kind_rollup` (4-row table over compute / data_move / comm / other).

Reading guide: walk `top_compute_groups` for the biggest time-eaters; check `tail_compute.dur_ps` against the top-K sum to see how concentrated the workload is; check `unknown_categories` and `n_events_unresolved` for spec-coverage gaps.

## Mode 2 — `by_source` (layer scoping)

```bash
python3 .../compute_breakdown.py <profile_dir> --mode by_source [--step N] [--include-data-move]
```

Full per-`agg_key` table — not sorted, not truncated. Each group carries its `source_stack`, `tf_op`, `kind`, `hlo_categories`, durations, sums (flops/model_flops/bytes_accessed), `shapes` (cap 8), `dtypes` histogram, `dtype_uncertain`, `example_hlo_op`.

**Layer-scoping recipe** (the canonical use):
1. Read the user's code (e.g. `attention.py`) — note the file path and function names.
2. Run `--mode by_source`.
3. In the JSON, filter `groups` where `source_stack` contains the file path OR `tf_op` contains the function name (be permissive — JAX adds wrapper prefixes).
4. Sum `total_dur_ps` over the filtered set.
5. Report: layer total / `step_duration_ps` (% of step), and layer total / `totals.compute_duration_ps` (% of compute).

## Mode 3 — `non_compute`

```bash
python3 .../compute_breakdown.py <profile_dir> --mode non_compute [--step N] [--no-comm-stalls]
```

Two-layer output:
- `by_category`: one row per `hlo_category` (`data formatting`, `copy`, `convert`, `pad`, `broadcast`, …) with execution count, durations, group count, agg_key coverage.
- `by_source_within_category`: full (category, agg_key) breakdown with `dtype_change` / `layout_change` (parsed from the HLO IR text), `shapes_in` / `shapes_out` (cap 4), `example_hlo_op`.

**`dtype_change` / `layout_change` semantics:**
- `true`: detected from the IR text (e.g. `f32[...] convert(bf16[...] ...)` — dtype changes from bf16 to f32).
- `false`: detected, no change.
- `null`: undetectable (HLO IR didn't include both layouts, or text wasn't parseable). **`null` is NOT "no change"** — it means we couldn't decide. Don't claim a layout change is absent when this field is `null`.

By default `async-done` events are included as `hlo_category="async-done (comm stall)"` (with a `notes` entry telling Claude how to flip it off). Pass `--no-comm-stalls` to exclude them.

## Mode 4 — `roofline`

```bash
python3 .../compute_breakdown.py <profile_dir> --mode roofline [--step N]
  [--chip v7x]
  [--peak-tflops-bf16 ...] [--peak-tflops-fp8 ...]
  [--peak-tflops-fp32 ...] [--peak-tflops-fp16 ...]
  [--peak-hbm-gibps ...]
```

v7x peaks are built in (per-device: BF16=1153.5, FP8=2307.0, HBM=3690 GiB/s). FP32/FP16 peaks are not officially listed; pass `--peak-tflops-fp32 ...` to include those groups (otherwise they go to `skipped_groups.n_peak_unknown_for_dtype`).

Per-group output: `arithmetic_intensity` (FLOPs/byte), `ridge_point` (where compute and memory roofs meet), `bound` ∈ `{compute, memory}`, `t_compute_theory_ps`, `t_hbm_theory_ps`, `t_roofline_theory_ps`, `mfu`, `hbm_util`, `roofline_util`, `shortfall_ps`, `shortfall_pct`.

Step summary: `weighted_avg_mfu`, `weighted_avg_hbm_util`, `weighted_avg_roofline_util` (weighted by `total_dur_ps`); `top_shortfall_groups` (top 10 by absolute `shortfall_ps`).

**Reading guide:**
- High `weighted_avg_mfu` → workload is using compute; gains come from reducing wall-clock (kernel fusion, less padding) not from algorithmic changes.
- High `weighted_avg_hbm_util` with low `weighted_avg_mfu` → memory-bound; gains come from raising arithmetic intensity (fusion to keep activations in SRAM, larger contraction dims, lower-precision activations).
- Both low → other bottleneck (scheduling, dependencies, control flow). Look at `summary.totals.while_pct_of_step` and the `non_compute` audit.
- When a group has `dtype_uncertain=true`, present both the bf16-peak MFU **and a note** that the true peak may be fp8 (~2× higher), making the MFU number an upper bound on under-utilization, not a definitive figure.

## Common gotchas

- **`XEvent.stats` vs `XEventMetadata.stats`**: see profile-anatomy. Op-level fields (`flops`, `bytes_accessed`, `hlo_category`, `shape_with_layout`) live on `XEventMetadata.stats`, not `XEvent.stats`.
- **`while` HLO is a container**: `while_container_duration_ps` is reported separately. Don't add it to `compute_duration_ps`.
- **Concurrency caveat**: per-kind durations can sum > step duration. The field is named `non_while_duration_ps_sum` (not `total`) for this reason.
- **`copy-start` / `copy-done` carry no source** — XLA-internal DMA. Real copy waste appears in `data formatting`.
- **GiB vs GB**: HBM is GiB/s (base-1024). The peak table block tags `unit: "GiB/s (base-1024) per device"` to make this explicit.
- **Cross-mode equality**: `summary.totals.compute_duration_ps == by_source.totals.compute_duration_ps` exactly. `summary.totals.data_move_duration_ps == non_compute.totals.data_move_duration_ps` only when mode 3 was invoked with `--no-comm-stalls`.

## Files

- `scripts/compute_breakdown.py` — main entry script.
- `scripts/_peaks.py` — v7x peak table and CLI override resolver.
- `scripts/_proto/` — vendored xplane protobuf bindings (copy of profile-anatomy's `_proto/`).
- `scripts/tests/` — unit + e2e tests (stdlib `unittest`).
