# TPU Performance Model — Design Document

**Date:** 2026-04-15
**Target:** TPU v7x (single chip)
**Deliverable:** Claude Code skill + Python scripts (`plugins/tpu-perf-model/`)

## Goal

A theoretical performance modeling tool centered on the **Register ↔ VMEM ↔ HBM** data flow hierarchy. Given a mathematical formula and tensor shapes, it produces:

1. **Pre-implementation analysis:** Theoretical optimal data flow, pipeline schedule, and performance upper bound
2. **Post-implementation gap analysis:** Compare theoretical model vs measured `eval_result.json` to locate optimization opportunities

## Architecture: Three-Level Storage Simulator

### Approach

Model TPU execution as a data flow problem across three storage levels with capacity and bandwidth constraints. AI decomposes the user's formula into `ComputeStep` objects; Python scripts simulate optimal scheduling.

**Why this approach over alternatives:**
- **vs Operator DAG + solver:** Too abstract — misses instruction-level pipeline overlap
- **vs Template framework:** Not general enough for arbitrary formula decomposition
- **Three-level simulator** directly maps the physical storage hierarchy, naturally supports fusion analysis (fusion = skip HBM round-trip), and models pipeline overlap at instruction granularity

## Hardware Model: TPU v7x

| Component | Spec | Notes |
|-----------|------|-------|
| HBM | 192 GB, 3690 GB/s | All inputs/outputs reside here |
| VMEM | 64 MiB | On-chip scratchpad, tiling target |
| SPR (Scalar Register) | 4096 × 32bit | Scalar control/indexing |
| VPR (Vector Register) | 32 × (8×128×32bit) = 32 × 4KB | Vector data, MXU/VPU direct input |
| MXU (Matrix Unit) | 2307 TFLOPS BF16, dual MXU | Matrix multiply |
| VPU (Vector Processing Unit) | — | Elementwise/reduce operations |

**Data flow rules:**
- All computation happens in Registers (VPR)
- MXU/VPU inputs are loaded from VMEM → VPR
- VMEM data is loaded from HBM via DMA
- Write-back: VPR → VMEM → HBM (DMA)
- Fusion = intermediate results stay in VMEM/VPR, skip HBM round-trip

## Compute Step Abstraction

AI decomposes formulas into a list of `ComputeStep` objects:

```python
@dataclass
class TensorRef:
    name: str
    shape: tuple[int, ...]
    dtype: str  # "bf16", "f32"

    @property
    def size_bytes(self) -> int: ...

@dataclass
class ComputeStep:
    name: str                    # e.g. "qk_matmul"
    op_type: str                 # "matmul" | "reduce" | "elementwise" | "softmax"
    inputs: list[TensorRef]
    outputs: list[TensorRef]
    flops_formula: str           # e.g. "2*M*N*K"
    compute_unit: str            # "MXU" | "VPU"
    fusable_with_prev: bool      # can fuse with previous step
```

**Division of labor:**
- **AI (guided by Skill):** Formula → ComputeStep list, including FLOPs formulas and fusion decisions
- **Python scripts:** Numerical computation — pipeline simulation, tiling optimization, gap analysis

## Instruction-Level Pipeline Modeling

### Functional Units & Micro-Operations

| Unit | Micro-Op | Duration |
|------|----------|----------|
| DMA Engine | `dma_load(HBM→VMEM)` | bytes / 3690 GB/s |
| DMA Engine | `dma_store(VMEM→HBM)` | bytes / 3690 GB/s |
| MXU | `mxu_compute(VPR→VPR)` | flops / 2307 TFLOPS |
| VPU | `vpu_compute(VPR→VPR)` | flops / VPU_peak |
| VMEM Port | `vmem_load(VMEM→VPR)` | — (fast, rarely bottleneck) |
| VMEM Port | `vmem_store(VPR→VMEM)` | — |

### Pipeline Overlap (Double Buffering)

Each ComputeStep is tiled. Across tiles, DMA and compute overlap:

```
DMA Engine:  |load_tile0|load_tile1|load_tile2|....|store_tile0|store_tile1|
MXU/VPU:               |compute_0 |compute_1 |compute_2|....|
```

**Steady-state throughput** is limited by the slower of DMA and compute per tile:
```
T_steady = max(T_dma_per_tile, T_compute_per_tile)
T_total = T_startup + num_tiles * T_steady + T_drain
```

### Buffering Strategy

Only **single buffer** and **double buffer** are modeled:
- **Single buffer:** No overlap — `T = T_dma + T_compute` per tile
- **Double buffer:** Full overlap — `T = max(T_dma, T_compute)` per tile, costs 2× VMEM

Constraint: `tile_data × num_buffers ≤ 64 MiB`

### Tiling Optimization

Search for optimal block shape under constraints:
- `block_dims % 128 == 0` (alignment)
- `tile_data × num_buffers ≤ 64 MiB` (VMEM capacity)
- `active_VPRs ≤ 32` (VPR limit)
- **Objective:** Balance DMA time ≈ compute time (pipeline equilibrium)

## Fusion Analysis

### Fusion Benefit

Fusing adjacent ops eliminates HBM round-trip for intermediate tensors:
```
Unfused: op_A writes intermediate to HBM, op_B reads it back
Fused: intermediate stays in VMEM or VPR
Savings = 2 × intermediate_size_bytes / HBM_BW
```

### VPR Pressure Check

```python
def can_fuse(op_a, op_b, vpr_limit=32):
    intermediate_vprs = ceil(op_a.output_elements / VPR_LANE_COUNT)
    fused_vprs = op_a.input_vprs + intermediate_vprs + op_b.output_vprs
    return fused_vprs <= vpr_limit
```

**Typical cases:**
- **matmul + elementwise:** Low pressure, usually fusable
- **matmul + reduce:** Medium pressure (accumulator VPRs)
- **matmul + matmul:** Very high pressure, usually not fusable

## Gap Analysis (vs Measured Data)

Input: `eval_result.json` from pallas-evolve.

| Metric | Theoretical | Measured | Diagnosis |
|--------|------------|---------|-----------|
| HBM transfer | Model-computed | `hbm_bandwidth_bytes` | Excess = fusion opportunity |
| MXU utilization | Tile-alignment based | `mxu_utilization_pct` | Low = tile too small or padding |
| VPR spills | 0 (ideal) | `vector_spills` | >0 = reduce fusion or adjust tile |
| Total time | Pipeline schedule | `total_time_us` | Gap = optimization headroom |

Output:
```python
@dataclass
class GapAnalysis:
    step_name: str
    metric: str
    theoretical: float
    measured: float
    gap_pct: float
    diagnosis: str

@dataclass
class ComparisonReport:
    theoretical: PipelineSchedule
    measured: MeasuredMetrics
    gaps: list[GapAnalysis]
    top_opportunities: list[str]
    achievable_speedup: float
```

## Project Structure

```
plugins/tpu-perf-model/
  .claude-plugin/
    plugin.json
  skills/
    tpu-perf-model/
      SKILL.md                     # AI guidance: formula decomposition, script invocation, result interpretation
      scripts/
        hw_params.py               # TPU v7x hardware constants
        compute_step.py            # ComputeStep / TensorRef dataclasses
        pipeline_simulator.py      # Core: instruction-level pipeline simulator
        tiling_optimizer.py        # Optimal tiling under VMEM/VPR constraints
        gap_analyzer.py            # Theoretical vs measured comparison
        report.py                  # Output formatting (JSON + text)
        cli.py                     # CLI entry point
```

## Skill Responsibilities (SKILL.md)

1. Guide AI to decompose user's math formula into `ComputeStep` list
2. Provide FLOPs/data-volume formula reference table per operator type
3. Describe when/how to call scripts and interpret output
4. Describe the `eval_result.json` comparison workflow

## Usage Example

```
User: "Analyze flash attention on TPU v7x, Q/K/V shape = [1, 32, 4096, 128]"

AI (skill-guided):
1. Decompose: Y = softmax(QK^T / sqrt(d)) @ V
2. Generate ComputeSteps:
   - step1: matmul Q@K^T [4096,128]×[128,4096] → MXU
   - step2: elementwise scale ÷sqrt(128) → VPU (fuse with step1)
   - step3: softmax (max→sub→exp→sum→div) → VPU (fuse with step2)
   - step4: matmul S@V [4096,4096]×[4096,128] → MXU
3. Call: python cli.py --steps steps.json --analyze
4. Interpret output: tiling recommendation, bottleneck diagnosis, pipeline schedule
```
