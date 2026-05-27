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

## Units — read this first

**Every duration field ends in `_ps` and is picoseconds.** Convert before printing:

| Want | Divisor |
|---|---|
| Microseconds | `/ 1e6` |
| Milliseconds | `/ 1e9` |
| **Seconds** | `/ 1e12` |

A 6-second step is `6_000_000_000_000` ps. Dividing a ps value by `1e9` gives milliseconds, not seconds — easy 1000× error. Other unit-bearing fields: `*_pct`/`pct_of_*` are already in percent (multiply by 100 already applied); `*_util` (`mfu`, `hbm_util`, `roofline_util`) are fractions in [0, 1] — multiply by 100 for percent display.

## Concepts you need first

- **`agg_key`**: groups events by source location with a 3-tier fallback. Tier 1: SHA-1 hash of `source_stack` (`stack:<16-hex>`). Tier 2: `tf_op` string (`tfop:<value>`). Tier 3: `<no source>:<hlo_category>`. The group's `agg_key_kind` field reports which tier was used.
- **`tf_op` is a CALL HIERARCHY, not a leaf identifier** — see "Layer-scoping recipe" below before substring-matching on it. Outer scopes (jit → vmap → shard_map → layer-block-name → kernel) all appear in the path of every nested op. A naive `'kda' in tf_op` against a layer named `moe_layers_kda_cycle_5` matches **everything inside that block**, including the MoE FFN's GMM kernel. This single mistake can shift attribution by 2–3×.
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

The trap: `tf_op` is the JAX call hierarchy from outermost jit to the leaf op (e.g. `jit(train_step)/jvp/.../moe_layers_kda_cycle_5/shard_map/jit(gmm)/select_n`). Outer scope names appear in every nested op. Substring-matching keywords like `kda`, `mla`, or `expert` against the full `tf_op` will overcount whenever those names also appear in **block / layer-cycle names** that wrap unrelated math.

**Real example from this skill's RED data:** a layer named `moe_layers_kda_cycle_N` is a transformer block that contains *both* KDA attention *and* MoE FFN. A naive `'kda' in tf_op.lower()` filter attributed 79% of compute to "KDA"; the actual KDA-kernel math was 27.5%. The 51-point gap was MoE GMM (FFN expert math) inside KDA-style layer blocks.

**Use these signals, in order of reliability:**
1. **`source_stack` (file:line)** — points to the *source code that emitted the op*, not the call hierarchy. Filter on the file path of the kernel/module you care about (`kernels/kda/pallas.py`, `layers/attention.py`). This is the strongest signal.
2. **`source_inner`** — the innermost frame of `source_stack`, already extracted. Use when you want the exact emitting site.
3. **Leaf segment of `tf_op`** — split `tf_op` on `/` and look at the **last 2-3 segments only** (the actual op name + its immediate JAX wrapper, e.g. `jit(gmm)/pallas_call`). Use this to classify *what kind of op* a group is.
4. **`hlo_categories`** — for op-kind classification (`custom-call`, `loop fusion`, `dot`, `convolution`, etc).

**Avoid** matching layer-block names (`moe_layers_*_cycle_N`, `decoder.body`, etc) against the full `tf_op` for layer attribution — those are scopes, not leaf identifiers. If you must match a block-name, anchor it: split on `/` first and check that the matched name is *not* followed by deeper scopes that re-classify the op (e.g. `.../moe_layers_kda_cycle_5/shard_map/jit(gmm)/...` is GMM math, not KDA math, despite "kda" in the path).

**Procedure:**
1. Read the user's code (e.g. `attention.py`, `moe.py`) — note the file path and the leaf function/kernel names.
2. Run `--mode by_source`.
3. Filter `groups` where `source_stack` contains the file path. Cross-check by inspecting the leaf segment of `tf_op` (`tf_op.split('/')[-1]`) — it should look like the op kind you expect.
4. Sum `total_dur_ps` over the filtered set.
5. Report: layer total / `step_duration_ps` (% of step), and layer total / `totals.compute_duration_ps` (% of compute).
6. **Sanity check:** the buckets must sum to ≤ `compute_duration_ps`. If your buckets sum to >100% of compute, you are double-counting via overlapping substring matches.

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

Step summary: `weighted_avg_mfu`, `weighted_avg_hbm_util`, `weighted_avg_roofline_util` (weighted by `total_dur_ps`, fractions in [0, 1]); `top_shortfall_groups` (top 10 by absolute `shortfall_ps`); coverage fields `rooflined_dur_ps`, `step_compute_dur_ps_total`, `rooflined_pct_of_compute`, `skipped_pct_of_compute`.

**`top_shortfall_groups` has a slim schema** — only `agg_key`, `source_inner`, `tf_op`, `total_dur_ps`, `shortfall_ps`, `bound`. To access the full per-group fields (`mfu`, `arithmetic_intensity`, `dtype_uncertain`, etc.) for a top-shortfall group, look up its `agg_key` in the full `groups` array.

**Roofline coverage** — `weighted_avg_mfu` is computed only over rooflined-eligible groups, not the full step compute. When `rooflined_pct_of_compute` is well below 100%, the averages reflect only that subset; the rest is binned into `skipped_groups` (`n_no_flops`, `n_no_bytes`, `n_dtype_other`, `n_peak_unknown_for_dtype`). Always report `rooflined_pct_of_compute` alongside the MFU number — a 22% MFU over 28% coverage tells a different story than the same MFU over 95% coverage.

**Reading guide:**
- High `weighted_avg_mfu` → workload is using compute; gains come from reducing wall-clock (kernel fusion, less padding) not from algorithmic changes.
- High `weighted_avg_hbm_util` with low `weighted_avg_mfu` → memory-bound; gains come from raising arithmetic intensity (fusion to keep activations in SRAM, larger contraction dims, lower-precision activations).
- Both low → other bottleneck (scheduling, dependencies, control flow). Look at `summary.totals.while_pct_of_step` and the `non_compute` audit.
- When a group has `dtype_uncertain=true`, present both the bf16-peak MFU **and a note** that the true peak may be fp8 (~2× higher), making the MFU number an upper bound on under-utilization, not a definitive figure.

## JSON schema cheat-sheet

Field names are stable; consult before writing inspectors so you don't guess. Common cross-mode fields: `status`, `mode`, `profile_dir`, `device`, `step_id`, `step_window_ps` (`[start_ps, end_ps]`), `step_duration_ps`, `notes` (list — includes the auto-step-pick reason), `totals`.

**`totals` block (all modes):** `n_events_{total,compute,data_move,comm,other,unresolved}`, `{compute,data_move,comm,other}_duration_ps`, `while_container_duration_ps`, `non_while_duration_ps_sum`, `while_pct_of_step`, `unknown_categories`. Mode 3 also adds `non_compute_pct_of_{step,compute}`.

**Group records — note `n_executions`, NOT `n_events`.** Per-mode group schemas:

| Mode | Array | Per-row fields |
|---|---|---|
| `summary` | `top_compute_groups`, `tail_compute` | `rank`, `agg_key`, `agg_key_kind`, `source_inner`, `tf_op`, `source_stack`, `n_executions`, `total_dur_ps`, `min/max/avg_dur_ps`, `pct_of_compute`, `pct_of_step`, `hlo_categories`, `flops_sum`, `bytes_accessed_sum`, `example_hlo_op`, `example_hlo_op_dur_ps`, `hlo_op_breakdown` (top-N), `hlo_op_breakdown_overflow` |
| `by_source` | `groups` | above + `dtypes` (histogram), `dtype_uncertain`, `shapes`, `kind` |
| `non_compute` | `by_category` | `hlo_category`, `n_executions`, `total_dur_ps`, `min/max/avg_dur_ps`, `n_groups`, `agg_key_coverage` |
| `non_compute` | `by_source_within_category` | `hlo_category`, `agg_key`, `agg_key_kind`, `source_inner`, `source_stack`, `tf_op`, `n_executions`, `total_dur_ps`, `min/max/avg_dur_ps`, `shapes_in`, `shapes_out`, `dtype_change`, `layout_change`, `example_hlo_op` |
| `roofline` | `groups` | `agg_key`, `agg_key_kind`, `source_inner`, `tf_op`, `hlo_categories`, `n_executions`, `total_dur_ps`, `flops_sum`, `bytes_accessed_sum`, `dtype`, `dtype_uncertain`, `arithmetic_intensity`, `ridge_point`, `bound`, `t_compute_theory_ps`, `t_hbm_theory_ps`, `t_roofline_theory_ps`, `mfu`, `hbm_util`, `roofline_util`, `shortfall_ps`, `shortfall_pct` |
| `roofline` | `step_summary.top_shortfall_groups` | **slim**: `agg_key`, `source_inner`, `tf_op`, `total_dur_ps`, `shortfall_ps`, `bound` only |

## HLO-level verification (when group numbers aren't enough)

`agg_key` groups can mix many distinct HLO ops (Pallas kernel + buffer placeholders + format-conversion fusions). Group-level numbers (`total_dur_ps`, `hlo_categories`, `example_hlo_op`) **describe the bag, not the dominant cost**. When attributing cost inside a hot group, **read `hlo_op_breakdown`** (top-N HLO signatures with their measured `total_dur_ps` and `pct_of_group`) rather than guessing from `example_hlo_op` or `hlo_categories`.

**When to drop further to raw HLO text** (use `profile-anatomy/scripts/walk_xplane.py` to read `XEventMetadata.name` directly):
- The dominant signature is `custom-call:tpu_custom_call` — Pallas kernel internals are not visible to XLA, so flops/bytes_accessed are 0 and roofline is uninformative. Read the HLO text to identify the kernel by name (`%vmap_jit__kda_intra_chunk_bwd_subchunk_pallas__.NN`).
- The dominant signature is a `*_fusion` you don't recognize — read its body to see what shapes/dtypes it actually moves.
- You need to confirm a buffer materialization vs. cast vs. layout change — group-level `dtype_uncertain` and `shapes` are summaries, not proofs.

```python
# Skeleton for inspecting a specific group's raw HLO ops:
#   1. Find the agg_key in the by_source/summary JSON.
#   2. Filter walk_xplane events by source_stack (or tf_op).
#   3. Print top-K events by duration_ps with their HLO text.
events = [(em.name, ev.duration_ps)
          for ev in ops_line.events
          for em in [device_plane.event_metadata.get(ev.metadata_id)]
          if em and matches_group(em)]
events.sort(key=lambda x: -x[1])
for name, dur in events[:10]:
    print(f"{dur/1e9:6.2f} ms  {name[:120]}")
```

## Common gotchas

- **Auto step-picking** — when `--step` and `--step-id` are both omitted, the script picks the step with the most XLA Ops events (busiest), falling back to middle when the ops line is empty. The picked step is reported in `step_id` and the reason appears in `notes` (e.g. `"auto-picked busiest step (idx=7, n_xla_ops_events=...)"`). If the auto-pick disagrees with what you expected, override with `--step N`. Earlier versions of this skill picked the middle step unconditionally, which landed in idle warmup windows on profiles with long compile/warmup tails.
- **`while_pct_of_step` can exceed 100%.** Events are admitted to the step window when their *start* falls in `[step_start, step_end)`, but their full duration is summed. A `while` event that begins inside the step but extends past `step_end` contributes its entire duration. This is expected — the field is a coarse "how dominated by control flow is this step" indicator, not an exclusive percentage. Don't try to subtract it from 100%.
- **`XEvent.stats` vs `XEventMetadata.stats`**: see profile-anatomy. Op-level fields (`flops`, `bytes_accessed`, `hlo_category`, `shape_with_layout`) live on `XEventMetadata.stats`, not `XEvent.stats`.
- **`while` HLO is a container**: `while_container_duration_ps` is reported separately. Don't add it to `compute_duration_ps`.
- **Concurrency caveat**: per-kind durations can sum > step duration. The field is named `non_while_duration_ps_sum` (not `total`) for this reason.
- **`copy-start` / `copy-done` carry no source** — XLA-internal DMA. Real copy waste appears in `data formatting`.
- **GiB vs GB**: HBM is GiB/s (base-1024). The peak table block tags `unit: "GiB/s (base-1024) per device"` to make this explicit.
- **Cross-mode equality**: `summary.totals.compute_duration_ps == by_source.totals.compute_duration_ps` exactly. `summary.totals.data_move_duration_ps == non_compute.totals.data_move_duration_ps` only when mode 3 was invoked with `--no-comm-stalls`.
- **`example_hlo_op` is a sample, not a summary.** It now tracks the *single longest-duration* HLO event in the group (alongside `example_hlo_op_dur_ps`), but a group may pool many distinct HLO ops with very different cost profiles. **For groups mixing zero-cost placeholders (`custom_call_target="AllocateBuffer"`), Pallas kernels, and surrounding fusions, never attribute the group's behavior to `example_hlo_op` alone — consult `hlo_op_breakdown` for the full time distribution.** Past failure mode: the first event in a hot group was a 75 ps `AllocateBuffer`, leading to "the bottleneck is buffer allocation" when the actual cost was a 316 ms Pallas backward kernel.
- **`hlo_categories` is event counts, not durations.** It is `{category: n_executions}`. **Do not extrapolate time from category counts.** A group with `{"custom-call": 240, "loop fusion": 60}` may spend 99% of its time in 60 fusion events if 120 of the custom-calls were 0-duration `AllocateBuffer` placeholders. To attribute time across categories within a group, sum `total_dur_ps` from `hlo_op_breakdown` rows whose signature carries the matching category.
- **Pallas `tpu_custom_call` is a black box to XLA.** Pallas/Mosaic kernels emit `%... = custom-call(...) custom_call_target="tpu_custom_call"`; XLA does not see inside them. Consequences: `flops`, `model_flops`, `bytes_accessed` are always `0` (or absent) for these events; `arithmetic_intensity`, `mfu`, and `hbm_util` are uninterpretable; roofline mode skips them under `skipped_groups.n_no_flops` / `n_no_bytes`. To analyze a Pallas kernel you must (a) read the kernel source code, (b) compute its theoretical FLOPs/bytes by hand, or (c) profile inside the kernel with Mosaic-side tooling — the xplane profile only gives you the wall-clock duration.
- **`hlo_op_breakdown` field**: present in `summary.top_compute_groups[].hlo_op_breakdown` and `by_source.groups[].hlo_op_breakdown`. Top-8 (cap 64 distinct signatures internally) HLO signatures inside the group, each row `{signature, hlo_category, total_dur_ps, n_executions, pct_of_group, example_hlo_op, example_hlo_op_dur_ps}`. Signature normalization: custom-calls become `custom-call:<target>` (so `AllocateBuffer` ≠ `tpu_custom_call`); fusions become `<fusion-name-prefix> [<category>]` (SSA index `.NNN` stripped); other ops become `<opcode> [<category>]`. If the group has more than 64 distinct signatures, the overflow is reported in `hlo_op_breakdown_overflow` with `n_signatures` and `total_dur_ps`.

## Files

- `scripts/compute_breakdown.py` — main entry script.
- `scripts/_peaks.py` — v7x peak table and CLI override resolver.
- `scripts/_proto/` — vendored xplane protobuf bindings (copy of profile-anatomy's `_proto/`).
- `scripts/tests/` — unit + e2e tests (stdlib `unittest`).
