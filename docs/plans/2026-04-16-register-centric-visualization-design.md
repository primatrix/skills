# Register-Centric Visualization Design

**Date**: 2026-04-16
**Status**: Approved
**Scope**: `plugins/tpu-perf-model` Mermaid diagram output

## Problem

Current visualization groups by execution unit (DMA/MXU/VPU). Registers appear only as labels — there is no view of register lifecycle, occupancy over time, data dependencies through registers, or capacity pressure.

## Solution

Replace existing Mermaid output with two register-centric diagrams:

1. **Resource Occupancy Gantt** — rows are VMEM slots and REG groups (not execution units)
2. **Register Data Flow Flowchart** — nodes are data fragments at each memory level (HBM/VMEM/REG), edges show data movement and stalls

## Design

### Diagram 1: Resource Occupancy Gantt

Replaces `micro_schedule_to_mermaid()`.

**Structure**:
- `section VMEM Slots` — one bar per VMEM slot occupancy interval
- `section REG Groups` — one bar per REG group occupancy interval
- `section Capacity` — comments with peak occupancy stats and warnings

**Bar labels**: `slot_name [step_name data tile_idx]`
**Stall bars**: red `crit` bars between occupancy intervals, labeled with wait reason (WAIT_VMEM, WAIT_REG, WAIT_DATA).
**Capacity annotation**: `Peak VMEM: N/M slots (X%)`, `Peak REG: N/32 groups (X%)`. WARNING when exceeding hardware limits.

### Diagram 2: Register Data Flow Flowchart

Replaces `micro_schedule_to_mermaid_flowchart()`.

**Structure**:
- One `subgraph` per tile
- Fused steps share the same tile subgraph
- Nodes represent data fragments at specific memory levels: `HBM: tensor[shape] dtype`, `VMEM slot_name: tensor[shape]`, `REG group_name: tensor[shape]`
- Solid edges (`-->`) = data transfer with `|"op_kind latency_ns"|` label
- Dashed edges (`-.->`) = stall/wait with `|"WAIT_REASON duration"|` label

### Scheduler Enhancement

Add to `micro_op_scheduler.py`:
- Track active VMEM slots and REG groups at each time point
- Return `peak_vmem_slots: int` and `peak_reg_groups: int` in schedule result
- Emit WARNING when peak exceeds hardware capacity

## Files Changed

| File | Change |
|------|--------|
| `micro_op_report.py` | Rewrite `micro_schedule_to_mermaid()` (resource Gantt) and `micro_schedule_to_mermaid_flowchart()` (data flow) |
| `micro_op_scheduler.py` | Add peak resource tracking, return peak stats |
| `SKILL.md` | Update Pipeline Diagram documentation |
| `test_micro_op_report.py` | Update tests for new diagram format |
