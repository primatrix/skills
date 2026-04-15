---
name: tpu-perf-model
description: Use when analyzing theoretical TPU v7x performance for a mathematical formula or comparing kernel performance against theoretical bounds. Trigger when the user asks about TPU performance modeling, roofline analysis, data flow optimization, or tiling strategy.
---

# TPU Performance Model

Theoretical performance modeling tool for TPU v7x centered on the Register ↔ VMEM ↔ HBM data flow hierarchy.

## When to Use

- Before writing a Pallas kernel: predict theoretical performance, identify bottleneck, guide tiling
- After profiling a kernel: compare theoretical vs measured to find optimization opportunities

## TPU v7x Hardware Quick Reference

| Component | Spec |
|-----------|------|
| HBM | 192 GB, 3690 GB/s |
| VMEM | 64 MiB on-chip scratchpad |
| SPR | 4096 scalar registers (32bit) |
| VPR | 32 vector registers (8×128×32bit = 4KB each) |
| MXU | 2307 TFLOPS BF16 (dual MXU) |
| VPU | Vector processing unit (elementwise/reduce) |
| Ridge Point | ~625 FLOPs/byte |
| Alignment | Block dims must be divisible by 128 |

## Phase 1: Formula Decomposition

Given the user's math formula, decompose it into a list of `ComputeStep` objects.

### FLOPs Reference Table

| Op Type | FLOPs Formula | Compute Unit | Notes |
|---------|---------------|--------------|-------|
| matmul [M,K]×[K,N] | 2×M×N×K | MXU | Includes multiply + accumulate |
| elementwise (unary) | N (elements) | VPU | exp, log, scale, sqrt |
| elementwise (binary) | N (elements) | VPU | add, mul, sub, div |
| reduce (sum/max/min) | N (elements) | VPU | Along one dimension |
| softmax [M,N] | 5×M×N | VPU | max + sub + exp + sum + div |
| layer_norm [M,N] | 7×M×N | VPU | mean + var + sub + div + scale + shift |

### Fusion Rules

Determine `fusable_with_prev` for each step. Fusion keeps intermediate tensors in VMEM/VPR instead of writing back to HBM.

| Pattern | Fusable? | VPR Pressure |
|---------|----------|-------------|
| matmul → elementwise | YES | Low |
| matmul → reduce | MAYBE | Medium (accumulator VPRs) |
| elementwise → elementwise | YES | Low |
| elementwise → reduce | YES | Low |
| matmul → matmul | NO | Very high |
| reduce → elementwise | YES | Low |

### ComputeStep JSON Format

Write a JSON file with array of steps:

```json
[
  {
    "name": "descriptive_name",
    "op_type": "matmul|elementwise|reduce|softmax",
    "inputs": [{"name": "A", "shape": [M, K], "dtype": "bf16"}],
    "outputs": [{"name": "C", "shape": [M, N], "dtype": "bf16"}],
    "flops_formula": "2*M*N*K",
    "flops_vars": {"M": 4096, "N": 4096, "K": 128},
    "compute_unit": "MXU|VPU",
    "fusable_with_prev": false
  }
]
```

Save to a temporary file, e.g., `steps.json`.

## Phase 2: Run Simulation

```bash
# Basic analysis
python scripts/cli.py --steps steps.json

# With detailed tiling analysis
python scripts/cli.py --steps steps.json --tiling

# JSON output
python scripts/cli.py --steps steps.json --format json

# Compare with measured profile data
python scripts/cli.py --steps steps.json --eval eval_result.json
```

The `scripts/` directory is at: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/`

## Phase 3: Interpret Results

### Per-Step Analysis

For each step, the report shows:
- **T(HBM)**: Time to transfer data between HBM and VMEM
- **T(compute)**: Time for MXU/VPU computation
- **T(step)**: Effective time with double-buffer pipeline overlap
- **Bottleneck**: `HBM_BW` (memory-bound) or `COMPUTE` (compute-bound)
- **Arithmetic Intensity**: FLOPs/byte — compare against ridge point (~625)

### Key Decisions from Results

| Observation | Action |
|-------------|--------|
| Bottleneck = HBM_BW | Increase tile size, enable fusion, reduce data movement |
| Bottleneck = COMPUTE | Current tiling is good, focus on MXU utilization |
| Pipeline balance ratio ≫ 1 | DMA dominates — increase compute per tile |
| Pipeline balance ratio ≪ 1 | Compute dominates — tiles are large enough |
| Fusion savings > 0 | Verify fusion is implemented in actual kernel |
| Low efficiency vs peak | Multiple optimization opportunities exist |

## Phase 4: Gap Analysis (Optional)

When comparing against `eval_result.json` from pallas-evolve profiling:

| Gap | Diagnosis |
|-----|-----------|
| HBM bytes: measured > theoretical | Missing fusion or redundant loads |
| MXU util: measured < theoretical | Tile too small, alignment issues |
| Vector spills > 0 | Register pressure — reduce fusion or tile size |
| Total time: measured ≫ theoretical | Significant optimization headroom |

## Example: Flash Attention

Formula: Y = softmax(QK^T / sqrt(d)) @ V, shapes Q/K/V = [4096, 128]

See `scripts/examples/flash_attention.json` for the decomposed steps.
