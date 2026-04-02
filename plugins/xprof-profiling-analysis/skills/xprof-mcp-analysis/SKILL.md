---
name: xprof-analysis
description: Use when analyzing TPU/GPU profiling performance via XProf MCP tools — operator breakdowns, memory profiles, A/B comparisons, optimization bottleneck identification
---

# XProf Profiling Analysis

Analyze TPU/GPU training performance using XProf MCP tools. This skill defines the methodology for profiling analysis — the actual data is fetched via MCP tools (`xprof_*`).

## Prerequisites

The `xprof` MCP server must be configured and connected. Verify by calling `xprof_list_runs()` — if it returns runs, the connection is working.

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

## Roofline Analysis Guide

When analyzing specific operators:

| Bound Type | Indicator | Optimization Direction |
|------------|-----------|----------------------|
| **Compute bound** | High FLOP rate, arithmetic intensity > 192 FLOPS/byte (TPUv7x) | Reduce FLOPs (algorithmic), increase MXU utilization |
| **Memory bound** | High bandwidth utilization, low arithmetic intensity | Reduce data movement, fuse ops, quantize |
| **Latency bound** | Low both FLOP rate and bandwidth | Pipeline better, reduce sync barriers, batch ops |

## Common Bottleneck Patterns

| Pattern | Symptom | Action |
|---------|---------|--------|
| **Stalled communication** | High device idle, large collective ops | Overlap compute + comms, reduce FSDP degree |
| **Kernel regression** | Same op category significantly slower after change | Check pallas-kernel version, chunk_size, block_size |
| **Memory pressure** | Peak HBM near capacity, high fragmentation | Enable remat, reduce batch size, chunked operations |
| **Load imbalance** | Cross-device step time variance >5% | Check MoE expert routing balance, PP stage sizing |
| **Host bottleneck** | High host idle time | Check data pipeline, increase grain workers |

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
# Then get memory viewer URL for human inspection:
xprof_trace_url(run="<run>")
```

### "Show me the trace timeline"
```
xprof_trace_url(run="<run>")
# Returns browser URLs for all XProf views
```
