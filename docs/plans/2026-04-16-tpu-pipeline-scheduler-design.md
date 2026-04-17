# TPU Pipeline Scheduler — Design Document

**Date**: 2026-04-16
**Plugin**: tpu-perf-model (new skill: `tpu-pipeline-scheduler`)
**Status**: Approved

## Problem Statement

The existing tpu-perf-model skill analyzes performance at the ComputeStep → MicroOp level with automatic VPR allocation. There is no tool for analyzing **explicit, instruction-level register-based pipeline scheduling** — where the user specifies exact VPR assignments and the tool analyzes data dependencies, hardware unit utilization, and register pressure to guide optimal pipeline design.

## Goals

1. Analyze data dependencies (RAW/WAR/WAW hazards) on explicitly-numbered VPRs and named VMEM slots
2. Schedule instructions across DMA/MXU/VPU units respecting dependency and resource constraints
3. Produce VPR occupancy analysis (liveness intervals, peak pressure, dead intervals)
4. Suggest optimal instruction ordering to minimize total latency and register pressure

## Non-Goals

- Automatic VPR allocation (user provides explicit VPR numbers)
- Integration with the existing MicroOp pipeline (clean separation)
- Pallas source code parsing (input is hand-authored JSON IR)

## Architecture

New skill within tpu-perf-model plugin, sharing only `hw_params.py`.

```
Pipeline IR JSON
    │
    ▼
pipeline_ir.py (PipelineOp, PipelineSpec — parse + validate)
    │
    ▼
dependency_analyzer.py (RAW/WAR/WAW detection → DependencyGraph)
    │
    ▼
pipeline_scheduler.py (greedy list scheduler → ScheduleResult)
    │
    ▼
vpr_analyzer.py (liveness analysis → VPROccupancy)
    │
    ▼
pipeline_report.py (text / JSON / Mermaid output)
    │
    ▼
pipeline_ir_cli.py (CLI entry point)
```

## IR Format

### PipelineOp

```python
@dataclass(frozen=True)
class PipelineOp:
    op_id: str              # Unique identifier, e.g. "load_q_0"
    op_kind: str            # DMA_LOAD | DMA_STORE | MXU | VPU | VMEM_TO_REG | REG_TO_VMEM
    input_vprs: list[int]   # VPR numbers read (0-31)
    output_vprs: list[int]  # VPR numbers written (0-31)
    input_vmem: list[str]   # VMEM slot names read
    output_vmem: list[str]  # VMEM slot names written
    latency_ns: float       # Instruction latency
    unit: str               # Execution unit: DMA | MXU | VPU
    label: str = ""         # Human-readable description
```

### JSON Input

```json
{
  "name": "flash_attention_tile",
  "hw": "v7x",
  "ops": [
    {
      "op_id": "load_q",
      "op_kind": "DMA_LOAD",
      "input_vprs": [],
      "output_vprs": [],
      "input_vmem": [],
      "output_vmem": ["q_buf"],
      "latency_ns": 200,
      "unit": "DMA",
      "label": "Load Q tile [128,128] from HBM"
    },
    {
      "op_id": "q_to_reg",
      "op_kind": "VMEM_TO_REG",
      "input_vprs": [],
      "output_vprs": [0, 1, 2, 3],
      "input_vmem": ["q_buf"],
      "output_vmem": [],
      "latency_ns": 10,
      "unit": "VPU",
      "label": "Q tile -> VPR[0:3]"
    }
  ]
}
```

## Dependency Analysis

### Hazard Types

| Hazard | Condition | Meaning |
|--------|-----------|---------|
| RAW | op_j reads VPR[n], op_i writes VPR[n], i < j | True dependency |
| WAR | op_j writes VPR[n], op_i reads VPR[n], i < j | Anti-dependency |
| WAW | op_j writes VPR[n], op_i writes VPR[n], i < j | Output dependency |

Same hazard analysis applies to VMEM slots (string name matching).

### DependencyGraph

```python
@dataclass
class Dependency:
    from_op: str
    to_op: str
    hazard_type: str    # RAW | WAR | WAW
    resource_type: str  # VPR | VMEM
    resource_id: str    # "VPR[3]" or "q_buf"

@dataclass
class DependencyGraph:
    ops: list[PipelineOp]
    edges: list[Dependency]
```

Transitive reduction applied to keep DAG minimal.

## Scheduler

Greedy list scheduler:
- Each unit (DMA, MXU, VPU) can run one instruction at a time
- `start_time = max(all dependency end times, unit available time)`
- Tracks wait reason per op: NONE | WAIT_DATA | WAIT_UNIT
- Computes critical path through the dependency DAG

```python
@dataclass
class ScheduleEntry:
    op_id: str
    start_ns: float
    end_ns: float
    unit: str
    wait_reason: str
    stall_ns: float
```

### Optimal Reordering

Within topological order constraints, try alternative ready-op selection strategies:
- Critical-path-first
- VPR-release-first (prefer ops that free VPRs)

Report best ordering vs original ordering with delta on total latency and peak VPR pressure.

## VPR Liveness Analysis

For each VPR[0..31]:
- **Define point**: end_time of the op that writes it
- **Last use point**: start_time of the last op that reads it
- **Dead interval**: from last use to end of program (or redefinition)

```python
@dataclass
class VPRLiveness:
    vpr_id: int
    defined_by: str
    last_used_by: str
    live_start_ns: float
    live_end_ns: float

@dataclass
class VPROccupancy:
    liveness: list[VPRLiveness]
    peak_concurrent: int
    peak_time_ns: float
    utilization_ratio: float
    pressure_warnings: list[str]
```

## Output Formats

### 1. Dependency Graph
- Text: table of edges `from → to [RAW on VPR[n]]`
- Mermaid: flowchart, RAW=solid, WAR=dashed, WAW=dotted

### 2. Pipeline Gantt
- Text: ASCII Gantt with stall markers
- Mermaid: gantt diagram, DMA/MXU/VPU sections, stalls as `crit`

### 3. VPR Heatmap
- Text: ASCII grid, rows=VPR[0..31], cols=time steps, `█`=live `·`=idle
- JSON: liveness intervals + peak/utilization stats

### 4. Optimal Reorder Suggestion
- Text: side-by-side comparison of original vs suggested schedule
- JSON: both schedules with delta metrics

## CLI Interface

```
python scripts/pipeline_ir_cli.py \
  --pipeline kernel.json \
  --format text|json \
  --show deps|gantt|vpr|suggest|all \
  --mermaid
```

## File Structure

```
plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/
  SKILL.md
  scripts/
    pipeline_ir.py
    dependency_analyzer.py
    pipeline_scheduler.py
    vpr_analyzer.py
    pipeline_report.py
    pipeline_ir_cli.py
    examples/
      flash_attention_tile.json
    test_pipeline_ir.py
    test_dependency_analyzer.py
    test_pipeline_scheduler.py
    test_vpr_analyzer.py
    test_pipeline_report.py
    test_integration.py
```

## Relationship to Existing Skill

- **Shared**: `hw_params.py` (TPU v7x hardware constants)
- **Independent**: all other modules — no coupling with MicroOp/MicroOpGraph/MicroOpScheduler
- **Plugin manifest**: update `plugin.json` to list both skills

## Testing

- Unit tests per module (IR parsing, dependency analysis, scheduling, VPR analysis, reporting)
- Integration test: end-to-end CLI with flash_attention_tile.json example
- All tests runnable with `python -m pytest`
