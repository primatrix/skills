# TPU Performance Model Micro-Op Dataflow Design

**Date:** 2026-04-15
**Target:** `plugins/tpu-perf-model/skills/tpu-perf-model`
**Deliverable:** Updated skill guidance plus a runnable micro-op and buffer-slot dataflow model

## Goal

Extend the existing TPU v7x performance model from step-level estimates into a finer-grained dataflow analysis that can answer:

- At any time, which tensor fragments are in registers, VMEM, or HBM
- Which compute unit is currently consuming each fragment
- Which dependencies prevent the next operation from issuing
- How VMEM capacity constrains the feasible schedule
- Why a given schedule is time-optimal or near-optimal under those constraints

The result should remain a mathematical, explainable model. It should not claim cycle-accurate fidelity.

## Problem Statement

The current model operates at `ComputeStep` granularity. It can estimate step-level DMA time, compute time, tiling, and fusion savings, but it cannot represent:

- Register residency of intermediate fragments
- Explicit VMEM ping-pong or staging slot usage
- Per-unit occupancy for DMA, MXU, and VPU over time
- Critical-path dependencies across sub-operations within a step
- Tradeoffs between retaining a fragment in VMEM or evicting and reloading it later

This leaves a gap between "the step is memory-bound" and the more useful question: "what exact dataflow and resource schedule minimizes time while staying within VMEM limits?"

## Proposed Approach

Keep `ComputeStep` as the user-facing abstraction, then add a second internal layer that expands steps into tensor fragments and schedulable micro-operations.

The overall flow becomes:

1. User formula
2. Skill-guided `ComputeStep` decomposition
3. Tiling selection under VMEM constraints
4. Step expansion into fragment-level `MicroOp` graph
5. Resource-constrained scheduling
6. Human-readable and machine-readable dataflow report

This preserves the current workflow while making the internal model capable of explaining residency, dependency, and optimality.

## Architecture

### Layer 1: Step-Level Input

User input stays as a list of `ComputeStep` objects. The user does not need to describe registers, slots, or micro-ops directly.

This layer continues to describe:

- Operator type
- Inputs and outputs
- FLOPs formulas
- Intended compute unit
- Fusion eligibility

### Layer 2: Fragment and Micro-Op Expansion

Each `ComputeStep` is expanded into:

- `TensorFragment` objects representing tiles or partial results
- `MicroOp` nodes representing data motion or compute actions
- Dependency edges between micro-ops
- Resource requirements for execution

This layer is responsible for producing the real dataflow graph used by the scheduler.

## Core Data Model

### `TensorFragment`

Represents a tile-sized or partial tensor value, for example:

- `Q[m0,k0]`
- `K[n0,k0]`
- `acc[m0,n0]`
- `softmax_stats[m0]`

Each fragment tracks:

- `fragment_id`
- `tensor_name`
- `step_name`
- `shape`
- `dtype`
- `size_bytes`
- `home_level`: `HBM`, `VMEM`, or `REG`
- `producer_op`
- `consumer_ops`

### `MicroOp`

Represents a schedulable unit of work. Initial operation kinds:

- `dma_load_hbm_to_vmem`
- `vmem_to_reg`
- `mxu_compute`
- `vpu_compute`
- `reg_to_vmem`
- `dma_store_vmem_to_hbm`
- `retain_fragment`
- `evict_fragment`
- `free_vmem_slot`
- `free_reg_group`

Each micro-op tracks:

- `op_id`
- `step_name`
- `op_kind`
- `depends_on`
- `input_fragments`
- `output_fragments`
- `required_units`
- `required_vmem_slots`
- `required_reg_groups`
- `latency_model`
- `start_ns`
- `end_ns`

### `ResourceState`

Represents the machine state at a moment in simulated time:

- `vmem_slots`
- `reg_groups`
- `dma_state`
- `mxu_state`
- `vpu_state`
- `live_fragments`

This is the basis for answering "what is resident where at time `t`?"

### `ScheduleResult`

Represents the outcome of scheduling:

- Ordered timeline of micro-ops
- Per-resource occupancy intervals
- Fragment residency intervals
- Stall breakdown
- Critical path
- Total makespan

## Scheduling Semantics

### Readiness Conditions

A micro-op can issue only when all of the following are true:

1. All data dependencies are complete
2. Required units are idle
3. Required VMEM slots and register groups are available
4. Launching the op does not violate VMEM or register capacity

This replaces the current step-level timing heuristic with an explicit resource-constrained execution model.

### Fragment State Transitions

Fragments move through explicit storage states:

- `not_ready`
- `in_hbm`
- `in_vmem`
- `in_reg`
- `consumed`
- `released`

The scheduler decides whether a produced fragment is:

- retained in place for reuse
- moved to another storage level
- evicted to free capacity
- recomputed or reloaded later if that shortens the critical path

### Resource Constraints

The model is defined by explicit inequalities instead of hidden heuristics:

- `sum(vmem_live_bytes(t)) <= VMEM_CAPACITY`
- `sum(reg_groups_live(t)) <= REG_GROUP_CAPACITY`
- `unit_busy(unit, t) in {0,1}`
- `start(B) >= end(A)` for each dependency edge `A -> B`
- `makespan = max(end(op_i))`

The objective is to minimize `makespan`.

### Stall Taxonomy

The scheduler records why a ready-looking operation still cannot issue:

- `WAIT_DATA`
- `WAIT_UNIT`
- `WAIT_VMEM`
- `WAIT_REG`

This gives a sharper diagnosis than simply labeling a whole step as compute-bound or memory-bound.

### Optimality Model

The model should explain schedule quality, not just output a number. A report must be able to state:

- which micro-op chain forms the critical path
- which fragments remain live on that path
- which resource bottlenecks force serialization
- why a candidate retain or evict decision shortens or lengthens the path

This is the mathematical basis for "how to do it fastest under VMEM limits."

## CLI Design

Retain the existing CLI entry point and add an analysis-depth switch.

Examples:

```bash
python scripts/cli.py --steps steps.json
python scripts/cli.py --steps steps.json --analysis-level micro
python scripts/cli.py --steps steps.json --analysis-level micro --format json
python scripts/cli.py --steps steps.json --analysis-level micro --show-timeline
python scripts/cli.py --steps steps.json --analysis-level micro --show-residency
python scripts/cli.py --steps steps.json --analysis-level micro --show-critical-path
```

Behavior:

- Default mode remains step-level for backward compatibility
- `--analysis-level micro` enables fragment and micro-op scheduling
- Extra display flags control output verbosity without changing the underlying simulation

## Report Design

### Text Output

The micro-op mode report should include:

1. `Macro Summary`
2. `Micro-Op Schedule Summary`
3. `Timeline`
4. `Residency and Occupancy`
5. `Critical Path and Optimization Hints`

This allows users to read the same run at three levels:

- Overall throughput summary
- Step-to-step reasoning
- Exact timeline and residency details

### JSON Output

The JSON schema should expose:

- `summary`
- `step_results`
- `micro_ops`
- `timeline`
- `resource_occupancy`
- `fragment_residency`
- `critical_path`
- `stall_breakdown`
- `optimization_hints`

This makes the report consumable by both humans and downstream agents.

## Module Changes

Keep current step-level modules intact where possible and add new micro-op modules.

Existing modules to preserve:

- `compute_step.py`
- `pipeline_simulator.py`
- `tiling_optimizer.py`
- `report.py`
- `cli.py`

New modules:

- `micro_op_ir.py`
- `micro_op_builder.py`
- `micro_op_scheduler.py`
- `micro_op_report.py`

Responsibilities:

- `micro_op_ir.py`: dataclasses and enums for fragments, micro-ops, resources, and schedule results
- `micro_op_builder.py`: expand a `ComputeStep` plus tiling choice into fragment graph and micro-op graph
- `micro_op_scheduler.py`: run dependency-aware, resource-constrained scheduling
- `micro_op_report.py`: render text and JSON views for timeline, residency, and critical path

## Skill Changes

`SKILL.md` should move from a step-only interpretation to a two-layer analysis workflow:

1. `Formula -> ComputeStep`
2. `ComputeStep -> TensorFragment -> MicroOp -> Schedule`

The updated skill should explicitly require the agent to describe:

- fragment inventory
- micro-op expansion
- residency timeline
- dependency graph
- critical path
- optimality argument under VMEM constraints

The output format should make the agent answer in these sections so the reasoning is reproducible instead of impressionistic.

## Testing Strategy

Add tests in three groups.

### 1. IR and Expansion Tests

Verify that representative steps expand into the expected fragments, micro-ops, and dependency edges.

Examples:

- matmul
- elementwise
- matmul followed by fused postprocessing

### 2. Scheduling Constraint Tests

Verify that:

- VMEM overcommit prevents illegal concurrent issuance
- register overcommit blocks dependent or competing ops
- dependency edges are honored
- resource release enables later operations

### 3. Report Integrity Tests

Verify that micro-op mode produces:

- timeline records
- fragment residency intervals
- resource occupancy intervals
- critical path output
- stall breakdown output

An end-to-end test should use the existing flash attention example or a reduced variant.

## Implementation Order

1. Add tests that define the micro-op IR and scheduling behavior
2. Add `micro_op_ir.py` and `micro_op_builder.py`
3. Add `micro_op_scheduler.py`
4. Integrate with `cli.py`
5. Add `micro_op_report.py`
6. Update `SKILL.md` so documentation matches the new runtime model
7. Run verification on unit and integration tests

## Non-Goals

This design intentionally does not attempt:

- full cycle-accurate TPU simulation
- speculative hardware behavior not supported by the current model
- exhaustive global optimization over all possible schedules with an external solver

The target is an explainable, mathematical, resource-constrained model that is finer than the current step-level simulator while remaining practical to maintain.
