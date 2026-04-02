---
name: xprof-profiling-analysis
description: Use when analyzing TPU/GPU profiling performance — XProf MCP tools for operator breakdowns, memory profiles, A/B comparisons; offline methodology for trace parsing, MFU calculation, HBM memory analysis, XLA flag auditing, communication bottleneck identification
---

# XProf Profiling Analysis

TPU/GPU training performance analysis. Primary workflow uses XProf MCP tools (`xprof_*`) for live queries; appendix sections provide deep domain knowledge for offline analysis and result interpretation.

## Step 0: Connection Check (MUST do first)

Before any analysis, verify that XProf MCP tools are available:

1. **Try calling `xprof_list_runs()`.**
2. **If it succeeds** — skip to Step 1.
3. **If the tool is not found** (no `xprof_*` tools available) — the plugin is not enabled. Guide the user:

   > XProf MCP tools are not available. To enable them:
   >
   > 1. Add the plugin to your Claude Code settings:
   >    ```bash
   >    claude settings set enabledPlugins.xprof-profiling-analysis@primatrix-skills true
   >    ```
   > 2. Restart Claude Code (the plugin starts automatically).
   >
   > The plugin will auto-connect to the XProf K8s service via `kubectl port-forward`.
   > **Prerequisite**: `kubectl` must be configured with access to the GKE cluster:
   > ```bash
   > gcloud container clusters get-credentials tpu7x-cluster --region us-central1 --project tpu-service-473302
   > ```

   **Stop here** — wait for the user to enable the plugin and restart before continuing.

4. **If the tool exists but returns an error** (connection refused, timeout, etc.) — XProf is unreachable. Diagnose:

   > XProf MCP server is running but cannot reach the XProf backend. Common causes:
   >
   > **a) kubectl not configured or wrong cluster:**
   > ```bash
   > kubectl config current-context
   > # Expected: gke_tpu-service-473302_us-central1_tpu7x-cluster
   > # If wrong:
   > gcloud container clusters get-credentials tpu7x-cluster --region us-central1 --project tpu-service-473302
   > ```
   >
   > **b) XProf K8s service not running:**
   > ```bash
   > kubectl get pods -l app=xprof-service
   > # Should show a Running pod with 3/3 READY
   > ```
   >
   > **c) Port conflict (another process on port 8080):**
   > ```bash
   > lsof -i :8080
   > # If occupied by something else, set a custom port:
   > # In .claude/settings.json under the plugin env, change XPROF_LOCAL_PORT
   > ```
   >
   > **d) Manual port-forward as workaround:**
   > ```bash
   > kubectl port-forward svc/xprof-service 8080:8080
   > ```
   > Then retry the analysis.

   **Stop here** — help the user resolve the connection issue before continuing.

## Analysis Workflow

### Step 1: Identify Runs

```
xprof_list_runs()
```

Pick the runs to analyze. If comparing optimizations, identify:
- **Baseline run**: the `main` branch or previous best
- **Experiment run**: the optimization branch

### Step 2: Overview

```
xprof_overview(run="<run_name>")
```

Check key metrics:
- **MXU utilization** — target >50% for compute-heavy workloads
- **Device idle time** — high idle = pipeline stall or host bottleneck
- **Top ops by time** — identify the dominant operations

### Step 3: Framework Op Breakdown

```
xprof_framework_ops(run="<run_name>", top=30)
```

This gives JAX-level operation statistics. Filter by category to focus:

```
xprof_framework_ops(run="<run_name>", category="gla")       # GLA kernels
xprof_framework_ops(run="<run_name>", category="moe")       # MoE/GMM kernels
xprof_framework_ops(run="<run_name>", category="matmul")    # Dense matmuls
xprof_framework_ops(run="<run_name>", category="collective") # Communication
```

**Categories**: `gla`, `moe`, `attention`, `norm`, `embedding`, `collective`, `matmul`, `other`

### Step 4: A/B Comparison

```
xprof_compare(run_a="<baseline>", run_b="<experiment>", top=20)
```

Focus on:
- Ops with **largest absolute delta** (biggest time savings/regressions)
- Ops where **delta_pct > 5%** (significant relative change)
- **Category-level trends** (did GLA get faster? Did communication increase?)

### Step 5: Memory Analysis

```
xprof_memory(run="<run_name>")
```

Check:
- **Peak HBM usage** vs capacity — how much headroom?
- **Fragmentation** — >10% may indicate allocation pattern issues
- **Stack reserved** vs **heap allocated** — stack is XLA buffers, heap is dynamic

### Step 6: Detailed Inspection

For deeper analysis, get browser URLs for human inspection:

```
xprof_trace_url(run="<run_name>")
```

Share these URLs with the user for:
- **Trace viewer** — timeline of all operations (zoom into specific steps)
- **Memory viewer** — allocation/deallocation timeline
- **Pod viewer** — cross-host communication patterns

### Step 7: Report

Generate a comparison table:

```markdown
| Metric | Baseline | Experiment | Delta |
|--------|----------|------------|-------|
| Step time | X.XXs | X.XXs | -X.X% |
| MXU utilization | XX% | XX% | +X.X% |
| Top hotspot | op_name (XX%) | op_name (XX%) | -X.X% |
| Peak HBM | XX.X GiB | XX.X GiB | -X.X GiB |
```

Include:
1. Summary of what changed and why
2. Per-category breakdown (GLA, MoE, collective, etc.)
3. Top 5 ops by time change
4. XProf UI links for human verification

## Quick Recipes

### "How fast is the latest profiling run?"
```
runs = xprof_list_runs()
xprof_overview(run=runs[-1])
```

### "What's the GLA kernel time?"
```
xprof_framework_ops(run="<run>", category="gla")
```

### "Did my optimization help?"
```
xprof_compare(run_a="<baseline>", run_b="<experiment>")
```

### "Is there a memory issue?"
```
xprof_memory(run="<run>")
xprof_trace_url(run="<run>")  # memory viewer URL for human inspection
```

### "Show me the trace timeline"
```
xprof_trace_url(run="<run>")  # returns browser URLs for all XProf views
```

## Roofline Analysis

| Bound Type | Indicator | Optimization Direction |
|------------|-----------|----------------------|
| **Compute bound** | High FLOP rate, arithmetic intensity > 313 FLOPs/byte (TPUv7x) | Reduce FLOPs (algorithmic), increase MXU utilization |
| **Memory bound** | High bandwidth utilization, low arithmetic intensity | Reduce data movement, fuse ops, quantize |
| **Latency bound** | Low both FLOP rate and bandwidth | Pipeline better, reduce sync barriers, batch ops |

Ridge Point = Peak_FLOPS / HBM_BW: TPU v7x = 2307T / 7380 GB/s = 313 FLOPs/byte

## Common Bottleneck Patterns

| Pattern | Symptom | Action |
|---------|---------|--------|
| **Stalled communication** | High device idle, large collective ops | Overlap compute + comms, check XLA flags (see Appendix D) |
| **Kernel regression** | Same op category significantly slower after change | Check pallas-kernel version, chunk_size, block_size |
| **Memory pressure** | Peak HBM near capacity, high fragmentation | Enable remat, reduce batch size, chunked CE (see Appendix E) |
| **Load imbalance** | Cross-device step time variance >5% | Check MoE expert routing balance, PP stage sizing |
| **Host bottleneck** | High host idle time | Check data pipeline, increase grain workers |

---

# Appendix: Deep Analysis Reference

Domain knowledge for interpreting MCP results and offline analysis when MCP data is insufficient.

## Appendix A: HLO Operator Classification

XLA-compiled op names lose JAX semantics. Classification by `hlo_category` + name patterns:

| Category | hlo_category / Name Pattern | Meaning |
|----------|---------------------------|---------|
| `matmul` | `convolution fusion` | Matrix multiply (XLA uses convolution for matmul) |
| `attention` | `splash_mha`, `flash_attention` | Attention kernel |
| `communication` | `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all` (incl. `async-start`/`async-done`) | Collective comms |
| `custom_kernel` | `custom-call` (excl. GMM/TGMM/splash_mha) | Pallas kernel, offload |
| `custom_fusion` | `custom fusion` | XLA custom fusions |
| `elementwise_fusion` | `loop fusion`, `non-fusion elementwise` | Elementwise fusions |
| `data_formatting` | `data formatting` | Tensor layout conversion (FSDP sharding copies) |
| `routing` | `sort` | MoE TopK routing |

**Key**: `async-done` events for communication (e.g., `all-reduce.*.call-done`) have duration = **stall time** (compute engine waiting). This is the core metric for exposed communication cost.

### Key Operator Patterns

| Pattern | Meaning | Phase |
|---------|---------|-------|
| `convolution_bitcast_fusion.*` | MatMul (fwd or bwd dlhs) | fwd/bwd |
| `tgmm.*` | MoE weight gradient (Transposed GMM) | **backward only** |
| `splash_mha_fwd_*` | SplashAttention forward | forward |
| `splash_mha_dkv_*` / `splash_mha_dq_*` | SplashAttention gradients | backward |
| `ragged-all-to-all` | MoE expert routing (EP) | fwd/bwd |

### Forward/Backward Identification

Via `tf_op` field (most reliable):
- Contains `transpose(jvp(...))` → **backward**
- Contains `jvp(...)` without `transpose` → **forward**
- Also extracts layer number (e.g., `moe_layers_0`)

Via op name (quick): `tgmm.*` → always backward; `splash_mha_d*` → backward; `splash_mha_fwd_*` → forward.

## Appendix B: MFU Calculation

```
MFU = F_useful / (Peak_cluster x T_step)

F_useful = GBS x seq_len x FLOPs_per_token(fwd+bwd)
Peak_cluster = num_devices x peak_flops_per_device
```

**MoE caveat**: Per-token FLOPs much lower than equivalent dense model (only top-k experts activated). MoE MFU is naturally low (~10%) — this is architectural, not poor hardware utilization.

**Pallas kernel caveat**: SplashAttention `model_flops` reports 0. FLOPs from trace accumulation underestimates actual compute. GMM/TGMM `model_flops` correct in newer versions.

## Appendix C: Communication Analysis

### Primitive-to-Strategy Mapping

| Primitive | Parallelism | Purpose |
|-----------|------------|---------|
| AllGather | FSDP | Collect sharded weights |
| ReduceScatter | FSDP | Aggregate and distribute gradients |
| AllReduce | DP (non-FSDP) | Gradient sync |
| AllToAll | EP | Expert routing data exchange |

### Overlap Measurement

Single TPU device executes all ops serially. Communication overlap = between communication engine (ICI/DCN) and compute engine (TensorCore):

- `async-done` with **duration = 0**: Communication completed, no stall → effectively overlapped
- `async-done` with **duration > 0**: Compute engine waiting = **exposed communication cost**

**Wrong**: Don't analyze interval overlaps on single-device timeline (events are serial). Measure async-done duration instead.

### Data Formatting

`data formatting` ops = tensor layout conversions. When >5% of step time: check for unnecessary FSDP sharding layout copies.

## Appendix D: XLA Flags for Communication Overlap

Check `LIBTPU_INIT_ARGS` when communication is not overlapped.

**Continuation Fusion (CF):**

| Flag | Purpose |
|------|---------|
| `xla_tpu_enable_async_collective_fusion=true` | Enable async collective fusion |
| `xla_tpu_enable_async_collective_fusion_fuse_all_gather=true` | AllGather CF |
| `xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true` | AllReduce CF |
| `xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` | ReduceScatter CF |
| `xla_tpu_overlap_compute_collective_tc=true` | TensorCore overlap compute/comm |

**SparseCore Offloading (v7x recommended):**

| Flag | Purpose |
|------|---------|
| `xla_tpu_enable_sparse_core_collective_offload_all_gather=true` | AllGather via SparseCore |
| `xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true` | ReduceScatter via SparseCore |
| `xla_tpu_enable_sparse_core_collective_offload_all_reduce=true` | AllReduce via SparseCore |

**CF and SparseCore are mutually exclusive** for the same primitive.

| Symptom | Likely Missing Flags |
|---------|---------------------|
| AllReduce stall high | `fuse_all_reduce=true` or SparseCore offload |
| AllGather stall high | `fuse_all_gather=true` or SparseCore offload |
| All comms not overlapped | `async_collective_fusion=true` base flag |
| MoE comms not optimized | `DATA_PARALLEL_OVERLAP` flags |

## Appendix E: HBM Memory Deep Analysis

### Data Sources (priority order)

| Source | Best For | Notes |
|--------|----------|-------|
| `xprof_memory()` MCP | Quick peak/capacity/fragmentation | Use first |
| `memory_viewer.json` | Peak composition, optimization targets | Static upper bound |
| xplane.pb `/host:CPU` | Runtime actual HBM | `actual = reserved - available` |
| `hlo_proto.pb` | Buffer-level detail | Large files, slow |

**Critical**: `bytes_allocated` only tracks dynamic I/O buffers, NOT internal workspace (~70 GiB). Actual HBM = `bytes_reserved - bytes_available`.

### Buffer Classification (memory_viewer.json `maxHeap`)

| Category | Rule (`tfOpName` / `groupName`) | Typical Shape |
|----------|--------------------------------|---------------|
| Parameters | `groupName=="Parameter"` + `state.params` | Weight shapes |
| Optimizer | `groupName=="Parameter"` + `opt_state` | Same as weights, f32 |
| Logits/CE | `logits_dense` / `cross_entropy` / `softmax` | `[B,T,vocab]` |
| MoE experts | `moe` / `expert` / `gmm` / `router` | `[num_experts,dim,dim]` |
| Attention/MLA | `attention` / `mla` / `wq_` / `wkv_` | `[B,T,H,D]` |

### Optimization Targets

| Optimization | Expected Savings | Impact |
|-------------|-----------------|--------|
| Chunked cross-entropy | ~18 GiB (3x vocab logits) | No accuracy impact |
| save_out_proj remat | Large activation savings | ~33% recompute increase |
| Increase FSDP | Linear per-chip param reduction | More communication |
| bf16 optimizer (mu_dtype) | ~2 GiB (mu f32 -> bf16) | May affect stability |

**Vocab-sized buffer detection**: Search `maxHeap` for any dimension >= 100K → chunked CE targets.

### Memory Anomaly Diagnostics

From `memory_viewer.json` heapSizes timeline:

1. **Spike detection**: Find largest delta jumps — usually multiple vocab-sized buffers briefly overlapping
2. **Buffer overlap**: Group peak-time buffers by shape to quantify per-group contribution
3. **Lifetime anomaly**: Temporary buffers with >80% lifetime + >100 MiB = critical (but MoE expert AllGather is a known false positive)
4. **Phase curve**: Sample heapSizes by schedule % to see memory across compute phases

## Appendix F: MoE Backward Pass Structure

### GMM / TGMM Relationship

```
Forward:  Y = gmm(X, W)
Backward:
  dlhs = gmm(dY, W^T)   # activation gradient → convolution_bitcast_fusion
  drhs = tgmm(X^T, dY)  # weight gradient → tgmm.*
```

- **dlhs**: On critical path, serial across layers
- **drhs**: Independent per layer

Layer count: `MoE_layers = GMM_count / 4 = TGMM_count / 2`

XLA schedules backward in two loops: (1) all dlhs (serial), then (2) all tgmm (parallelizable).

## Appendix G: Offline CLI Tool

When MCP is insufficient, use `tpu-profiling/scripts/xprof.py`:

```bash
xprof memory peak --run-dir <run>           # Peak HBM composition
xprof memory diagnose --run-dir <run>       # Anomaly detection
xprof compute breakdown --run-dir <run>     # Op category breakdown
xprof compute mfu --run-dir <run> --gbs 5120 --num-chips 128
xprof comm overlap --run-dir <run>          # Async-done stall analysis
xprof compare --runs 115,183               # A/B comparison
xprof audit flags --flags-file launch.sh   # XLA flags check
```

Data source priority: **xplane.pb** (complete) > trace.json.gz (1M event hard limit — check `Complete (X) events`).

## Common Pitfalls

| Pitfall | Correct |
|---------|---------|
| XLA `convolution` = CNN | XLA uses convolution for all matmul |
| `hlo_category` distinguishes fwd/bwd | Cannot — use op name patterns or `tf_op` |
| Low MFU = poor utilization | MoE MFU naturally low; understand per-token FLOPs |
| `bytes_allocated` = actual HBM | Only dynamic I/O buffers; actual = `reserved - available` |
| `memory_viewer` peak = runtime peak | Static upper bound; runtime 5-15% lower |
| trace.json.gz is complete | 1M event hard limit — use xplane.pb if truncated |
| Comms not overlapped = need algorithm change | Check XLA flags first (CF/SparseCore) |
| Single-device interval overlap = comm overlap | Single device is serial; measure async-done duration |
| Logits size is fixed | Chunked CE: `[B,T,vocab]` → `[B,chunk,vocab]`, saving ~18 GiB |
