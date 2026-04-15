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

## Layer A: Formula -> ComputeStep

Given the user's math formula, decompose it into a list of `ComputeStep` objects. This layer defines the mathematical pipeline, the FLOPs model, and which steps can be fused before any fragment-level scheduling begins.

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

## Layer B: ComputeStep -> TensorFragment -> MicroOp -> Schedule

After the `ComputeStep` list is defined, refine the analysis into fragment-level dataflow. Your job is to explain the schedule as a resource-constrained mathematical model, not as a vague optimization intuition.

At this layer, you must explicitly reason about:

- Which tensor fragments or tiles exist at each stage
- Which fragments live in `HBM`, `VMEM`, and `REG`
- Which `DMA`, `MXU`, and `VPU` micro-ops consume each fragment
- Which dependencies block later micro-ops from issuing
- Which fragments are retained, evicted, or reloaded under VMEM pressure
- Why the reported critical path determines total latency

When describing the optimal schedule, use the VMEM and register constraints directly:

- `sum(vmem_live_bytes(t)) <= VMEM_CAPACITY`
- `sum(reg_groups_live(t)) <= REG_GROUP_CAPACITY`
- `start(B) >= end(A)` for each dependency edge `A -> B`
- `makespan = max(end(op_i))`

The point of this layer is to answer:

- At time `t`, which data is in registers?
- Which buffer slots are occupied?
- Which compute unit is active or stalled?
- Why is this schedule optimal or near-optimal under the current VMEM limit?

## Run Simulation

```bash
# Basic analysis
python scripts/cli.py --steps steps.json

# JSON output
python scripts/cli.py --steps steps.json --format json

# Micro-op analysis
python scripts/cli.py --steps steps.json --analysis-level micro

# Micro-op JSON output
python scripts/cli.py --steps steps.json --format json \
  --analysis-level \
  micro

# Micro-op analysis with timeline details
python scripts/cli.py --steps steps.json --show-timeline \
  --analysis-level \
  micro

# With detailed tiling analysis
python scripts/cli.py --steps steps.json --tiling

# Compare with measured profile data
python scripts/cli.py --steps steps.json --eval eval_result.json
```

The `scripts/` directory is at: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/`

## Interpret Results

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

### Micro-Op Analysis

When using micro-op mode, interpret the report as a fragment-level execution plan:

- **Timeline**: the ordered micro-op schedule with start and end times
- **Residency and Occupancy**: which `VMEM` slots, register groups, and units are active over time
- **Critical Path**: the dependency chain that determines total makespan
- **Stall Breakdown**: whether time is lost to `WAIT_DATA`, `WAIT_UNIT`, `WAIT_VMEM`, or `WAIT_REG`
- **Optimization Hints**: which resource or dependency bottleneck should be attacked first

Use this mode when the user asks for finer-grained dataflow, explicit dependency reasoning, or a proof-like explanation of why one schedule is faster than another.

## Required Output Sections

When you answer with the micro-op model, use these sections in order:

1. Fragment Inventory
2. Micro-Op Expansion
3. Residency Timeline
4. Dependency Graph
5. Critical Path
6. Optimality Argument Under VMEM Constraint

Do not collapse this into a generic summary. The point is to make the dataflow explicit enough that the user can see which fragments, units, and constraints control performance.

## Gap Analysis (Optional)

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
