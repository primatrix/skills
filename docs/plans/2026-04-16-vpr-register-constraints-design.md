# VPR Register Constraint Modeling for TPU Perf Model

## Problem

The TPU perf model tracks register groups as abstract named slots without modeling physical VPR limits. It cannot detect register pressure, predict spills, or constrain tiling choices based on the 32 VPR limit (each 8x128x32bit = 4096 bytes).

## Key Constraint

TPU v7x has 32 VPRs. Each VPR holds 4096 bytes regardless of dtype:
- bf16: 2 x 8 x 128 = 2048 elements per VPR
- f32: 1 x 8 x 128 = 1024 elements per VPR

VPR count for a fragment: `ceil(size_bytes / 4096)`

MXU accumulator is internal to the MXU — it does NOT occupy VPRs. Only Q and K inputs occupy VPRs during MXU compute. Result occupies VPRs after writeback.

### VPR Pressure Examples

| Matmul tile | Q VPRs | K VPRs | Peak (Q+K) | Result VPRs |
|-------------|--------|--------|------------|-------------|
| [128,128] bf16 | 8 | 8 | 16 | 8 (bf16) or 16 (f32) |
| [256,128] bf16 | 16 | 8 | 24 | 16 (bf16) |
| [256,256] bf16 | 16 | 16 | 32 (at limit) | 16 (bf16) |

## Design

### 1. Data Model (micro_op_ir.py)

`TensorFragment` gains `vpr_count: int` field — computed as `ceil(size_bytes / hw.vpr_size_bytes)`.

`MicroOp` gains `required_vpr_count: int` field — total physical VPRs held during this op's execution.

### 2. micro_op_builder.py

New helper `_calc_vpr_count(size_bytes, hw) -> int`.

**Matmul graph** changes:
- MXU accumulator no longer modeled as REG fragment during compute
- New `mxu_writeback` micro-op: writes MXU result to VPR after compute
- VPR count per micro-op:

| Micro-op | VPR held | Count |
|----------|----------|-------|
| vmem_to_reg_q | Q_reg | vpr(Q) |
| vmem_to_reg_k | Q_reg + K_reg | vpr(Q) + vpr(K) |
| mxu_compute | Q_reg + K_reg | vpr(Q) + vpr(K) |
| mxu_writeback | result_reg | vpr(result) |
| reg_to_vmem | result_reg | vpr(result) |

**VPU graph** similarly computes VPR counts for input + output fragments.

### 3. Scheduler (micro_op_scheduler.py)

Physical VPR pool management:
- Tracks `vpr_pool_used: int` — current VPR occupation
- Fragment VPRs allocated on REG entry, released after last consumer
- When `vpr_pool_used + needed > 32`: trigger spill

**Spill strategy** (furthest-next-use):
1. Find live REG fragment with latest next use
2. Insert `spill_to_vmem` micro-op (~10ns per VPR)
3. Free the spilled fragment's VPRs
4. If spilled fragment needed later, insert `fill_from_vmem` micro-op

**ScheduleResult** new fields: `spill_count: int`, `spill_cost_ns: float`.

### 4. Tiling Optimizer (tiling_optimizer.py)

VPR feasibility check in candidate evaluation:

```python
def _matmul_tile_vpr_count(bm, bn, bk, dtype_b, hw):
    q_vpr = ceil(bm * bk * dtype_b / hw.vpr_size_bytes)
    k_vpr = ceil(bk * bn * dtype_b / hw.vpr_size_bytes)
    acc_vpr = ceil(bm * bn * 4 / hw.vpr_size_bytes)  # f32 writeback
    return max(q_vpr + k_vpr, acc_vpr)
```

Skip candidates where `vpr_count > hw.vpr_count`. Two-layer defense: optimizer avoids spill-prone tilings, scheduler handles edge cases.

### 5. Report & Visualization — Register-Centric View

**Text report** — new `=== VPR Register Map ===` section:
```
Time   0ns: VPR[ 0.. 7] <- Q[128,128] bf16 (load)
Time   0ns: VPR[ 8..15] <- K[128,128] bf16 (load)
Time  50ns: VPR[ 0..15] feeding MXU (Q+K)
Time 100ns: VPR[ 0.. 7] <- result[128,128] bf16 (writeback)
Time 120ns: VPR[ 0.. 7] -> VMEM spill (result), freed
```

**Mermaid Gantt** — VPR groups as rows:
- Each section is a VPR range (e.g., `VPR 0-7`)
- Bars show stored content and time span
- Spills shown as red `crit` bars
- Peak usage and spill count in comments

**Mermaid Flowchart** — fragment nodes labeled with VPR numbers:
```
REG VPR[0..7]: Q[128,128] bf16
```

**JSON output** — new `vpr_pressure` object with peak count, utilization %, spill count, spill cost.

## Files Changed

| File | Change |
|------|--------|
| micro_op_ir.py | Add vpr_count to TensorFragment, required_vpr_count to MicroOp |
| micro_op_builder.py | Compute VPR counts, add mxu_writeback op, fix acc as MXU-internal |
| micro_op_scheduler.py | Physical VPR pool tracking, spill insertion, new result fields |
| tiling_optimizer.py | VPR feasibility pruning in candidate search |
| micro_op_report.py | Register-centric text/JSON/Mermaid output |
| SKILL.md | Update docs for VPR analysis |
| test_*.py | Update tests for new fields and behavior |
