---
name: tpu-pipeline-scheduler
description: >
  Use when analyzing register-level pipeline scheduling for TPU v7x kernels.
  Trigger when the user asks about instruction-level pipeline analysis,
  VPR register pressure, data hazard detection (RAW/WAR/WAW),
  or optimal instruction ordering for TPU pipelines.
---

# TPU Pipeline Scheduler

Analyze register-level pipeline scheduling for TPU v7x kernels. Given an explicit sequence of instructions with VPR assignments, this skill detects data hazards, schedules across hardware units, analyzes VPR pressure, and suggests optimal ordering.

## When to Use

- Designing optimal instruction interleaving for a Pallas kernel tile
- Analyzing VPR register pressure to determine if a tiling strategy is feasible
- Identifying data dependency bottlenecks (RAW/WAR/WAW hazards)
- Comparing alternative instruction orderings for pipeline efficiency

## Input Format: Pipeline IR

The input is a JSON file describing a sequence of hardware instructions with explicit VPR assignments:

```json
{
  "name": "kernel_tile_name",
  "hw": "v7x",
  "ops": [
    {
      "op_id": "unique_name",
      "op_kind": "DMA_LOAD | DMA_STORE | MXU | VPU | VMEM_TO_REG | REG_TO_VMEM",
      "input_vprs": [0, 1, 2, 3],
      "output_vprs": [4, 5, 6, 7],
      "input_vmem": ["slot_name"],
      "output_vmem": ["slot_name"],
      "latency_ns": 500,
      "unit": "DMA | MXU | VPU",
      "label": "Human-readable description"
    }
  ]
}
```

### Fields

| Field | Description |
|-------|-------------|
| `op_id` | Unique instruction identifier |
| `op_kind` | Instruction type (DMA_LOAD, DMA_STORE, MXU, VPU, VMEM_TO_REG, REG_TO_VMEM) |
| `input_vprs` | VPR numbers read (0-31) |
| `output_vprs` | VPR numbers written (0-31) |
| `input_vmem` | VMEM slot names read |
| `output_vmem` | VMEM slot names written |
| `latency_ns` | Instruction latency in nanoseconds |
| `unit` | Execution unit (DMA, MXU, VPU) |
| `label` | Optional human-readable description |

### TPU v7x Hardware Reference

- 32 VPRs (Vector Pipeline Registers), 4 KiB each
- 3 execution units: DMA, MXU, VPU — each runs one instruction at a time
- Dual MXU at 2307 TFLOPS BF16
- 64 MiB VMEM, 192 GB HBM at 3690 GB/s

## CLI Usage

```bash
# All analyses (text)
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show all

# Dependency graph only (JSON)
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format json --show deps

# Gantt + Mermaid diagrams
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show deps,gantt --mermaid

# VPR pressure only
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show vpr

# Reorder suggestion
python scripts/pipeline_ir_cli.py --pipeline kernel.json --format text --show suggest

# VPR timeline plot (PNG)
python scripts/pipeline_ir_cli.py --pipeline kernel.json --plot

# Custom output path
python scripts/pipeline_ir_cli.py --pipeline kernel.json --plot --plot-output my_chart.png
```

### CLI Options

| Flag | Values | Description |
|------|--------|-------------|
| `--pipeline` | path | Pipeline IR JSON file (required) |
| `--format` | text, json | Output format (default: text) |
| `--show` | deps, gantt, vpr, suggest, all | Sections to show (comma-separated, default: all) |
| `--mermaid` | flag | Include Mermaid diagrams (text format only) |
| `--plot` | flag | Generate VPR timeline heatmap as PNG image |
| `--plot-output` | path | Output path for plot (default: `<name>_vpr_timeline.png`) |

## Output Sections

### 1. Data Dependency Graph

Detects three types of data hazards:

| Hazard | Condition | Impact |
|--------|-----------|--------|
| **RAW** (Read-After-Write) | Op B reads VPR[n] that Op A writes | True dependency — B must wait for A |
| **WAR** (Write-After-Read) | Op B writes VPR[n] that Op A reads | Anti-dependency — B can't overwrite before A reads |
| **WAW** (Write-After-Write) | Op B writes VPR[n] that Op A writes | Output dependency — ordering must be preserved |

Same analysis applies to VMEM slots. Transitive reduction is applied to keep the DAG minimal.

Mermaid output uses: solid arrows for RAW, dashed for WAR, dotted for WAW.

### 2. Pipeline Gantt

Shows each hardware unit's timeline with instruction placement and stall markers. Each instruction reports:
- Start/end time in ns
- Wait reason: NONE, WAIT_DATA (blocked on dependency), WAIT_UNIT (unit busy)
- Stall duration

### 3. VPR Occupancy Heatmap

ASCII grid showing which VPRs are live at each time step. Reports:
- Peak concurrent VPR count and when it occurs
- Utilization ratio (average live VPRs / 32)
- Pressure warnings when >75% VPRs are simultaneously live

### 4. Reorder Suggestion

Compares original instruction ordering against analysis:
- Critical path identification and latency
- Parallelism efficiency (critical path / total latency)
- Stall breakdown

### 5. VPR Timeline Plot (PNG)

Matplotlib-rendered 2D heatmap with:
- **X-axis**: Time (ns), continuous scale
- **Y-axis**: VPR registers, one row per used VPR
- **Cell color**: 3-state × 3-unit color matrix
  - Write (deep): op is actively writing this VPR
  - Read (mid): op is actively reading this VPR
  - Live (light): VPR holds data but no op is accessing it
  - Colors: DMA=blue, MXU=red, VPU=green
- **Top band**: Gantt strips showing DMA/MXU/VPU unit utilization
- **Dependency arrows**: Arc arrows between VPR rows (RAW=solid, WAR=dashed, WAW=dotted)
- **Title bar**: Kernel name, total latency, peak VPR count, stall time

Requires `matplotlib` (`pip install matplotlib`).

## Workflow

1. **Decompose** your kernel tile into Pipeline IR instructions
2. **Assign VPRs** explicitly — this is where the design happens
3. **Run analysis** to identify hazards, stalls, and pressure points
4. **Iterate** on VPR assignments and instruction ordering
5. **Validate** that peak VPR pressure stays within hardware limits (32 VPRs)

## Output Language

Narrative text in Chinese, technical terms (VPR, RAW, WAR, WAW, DMA, MXU, VPU, VMEM, HBM) in English.

## Example

See `scripts/examples/flash_attention_tile.json` for a complete Flash Attention tile decomposition with 11 instructions across DMA/MXU/VPU units using VPR[0:23].
