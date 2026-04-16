# TPU Perf Model v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Chinese output directive to SKILL.md and Mermaid Gantt pipeline diagram generation to the CLI tool.

**Architecture:** Two independent changes — (1) a SKILL.md-only text addition for Chinese output, (2) a new `micro_schedule_to_mermaid()` function in `micro_op_report.py` wired into `cli.py` via `--mermaid` flag. The Mermaid function groups micro-ops by execution unit (DMA/MXU/VPU) and shows the first N tiles to visualize pipeline overlap.

**Tech Stack:** Python 3, unittest, Mermaid gantt syntax

---

### Task 1: Add `micro_schedule_to_mermaid()` — failing test

**Files:**
- Test: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

**Step 1: Write the failing test**

Add a test that imports and calls `micro_schedule_to_mermaid`. We need a richer fixture than the existing `_sample_schedule_result()` because we need ops with tile indices in their IDs and multiple resource types. Build a 2-tile matmul micro-op graph using the existing builder.

```python
def _sample_mermaid_schedule():
    """Build a 2-tile matmul schedule for Mermaid testing."""
    from compute_step import ComputeStep, TensorRef
    from hw_params import TPU_V7X
    from micro_op_builder import build_micro_op_graph_for_step
    from micro_op_scheduler import schedule_micro_op_graph
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul",
        op_type="matmul",
        inputs=[
            TensorRef(name="Q", shape=(256, 128), dtype="bf16"),
            TensorRef(name="K", shape=(128, 256), dtype="bf16"),
        ],
        outputs=[
            TensorRef(name="S", shape=(256, 256), dtype="bf16"),
        ],
        flops_formula="2*M*N*K",
        flops_vars={"M": 256, "N": 256, "K": 128},
        compute_unit="MXU",
        fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128},
        num_tiles=2,
        double_buffer=True,
    )
    graph = build_micro_op_graph_for_step(step, tile, step_idx=0)
    schedule = schedule_micro_op_graph(graph, TPU_V7X)
    return schedule, graph


class TestMermaidOutput(unittest.TestCase):
    def test_mermaid_contains_gantt_structure(self):
        from micro_op_report import micro_schedule_to_mermaid

        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("```mermaid", output)
        self.assertIn("gantt", output)
        self.assertIn("dateFormat x", output)
        self.assertIn("section DMA", output)
        self.assertIn("section MXU", output)
        self.assertIn("```\n", output.split("```mermaid")[1])

    def test_mermaid_filters_tiles_by_max(self):
        from micro_op_report import micro_schedule_to_mermaid

        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("tile0", output)
        self.assertNotIn("tile1", output)

    def test_mermaid_shows_ellipsis_when_truncated(self):
        from micro_op_report import micro_schedule_to_mermaid

        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("%%", output)
        self.assertIn("steady-state", output)
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_report.py::TestMermaidOutput -v`
Expected: FAIL with `ImportError: cannot import name 'micro_schedule_to_mermaid'`

**Step 3: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py
git commit -m "test: add failing tests for micro_schedule_to_mermaid"
```

---

### Task 2: Implement `micro_schedule_to_mermaid()`

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py`

**Step 1: Add the import for MicroOpGraph**

At the top of `micro_op_report.py`, add `MicroOpGraph` to imports:

```python
from micro_op_ir import MicroOpGraph
```

**Step 2: Implement the function**

Add `micro_schedule_to_mermaid()` at the end of `micro_op_report.py`:

```python
def _tile_index_from_op_id(op_id: str) -> int | None:
    """Extract tile index from op_id like 's0_qk_matmul_load_q_tile3'."""
    import re
    match = re.search(r"tile(\d+)", op_id)
    return int(match.group(1)) if match else None


def _unit_from_op(graph: MicroOpGraph, op_id: str) -> str | None:
    """Return the primary execution unit (DMA, MXU, VPU) for a micro-op."""
    op = graph.micro_ops.get(op_id)
    if not op or not op.required_units:
        return None
    return op.required_units[0]


def _short_label(op_id: str) -> str:
    """Shorten op_id for Mermaid bar labels. Strip step prefix like 's0_'."""
    import re
    return re.sub(r"^s\d+_", "", op_id)


def micro_schedule_to_mermaid(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
    max_tiles: int = 3,
) -> str:
    """Render the micro-op schedule as a Mermaid Gantt pipeline diagram."""
    # Determine total tile count
    all_tile_indices = set()
    for op_id in schedule.op_timings:
        idx = _tile_index_from_op_id(op_id)
        if idx is not None:
            all_tile_indices.add(idx)
    total_tiles = max(all_tile_indices, default=0) + 1 if all_tile_indices else 0

    # Collect step names for title
    step_names = []
    for op in graph.micro_ops.values():
        if op.step_name not in step_names:
            step_names.append(op.step_name)
    title = ", ".join(step_names)

    # Group ops by unit, filtered by tile range
    unit_ops: dict[str, list[tuple[str, float, float]]] = {}
    for op_id, timing in schedule.op_timings.items():
        tile_idx = _tile_index_from_op_id(op_id)
        if tile_idx is not None and tile_idx >= max_tiles:
            continue
        unit = _unit_from_op(graph, op_id)
        if unit is None:
            continue
        unit_ops.setdefault(unit, []).append(
            (op_id, timing.start_ns, timing.end_ns)
        )

    # Sort each unit's ops by start time
    for unit in unit_ops:
        unit_ops[unit].sort(key=lambda x: (x[1], x[0]))

    # Build Mermaid output
    lines = [
        "```mermaid",
        "gantt",
        f"    title Tile Pipeline: {title}",
        "    dateFormat x",
        "    axisFormat %s ns",
    ]

    section_order = ["DMA", "MXU", "VPU"]
    for unit in section_order:
        ops = unit_ops.get(unit)
        if not ops:
            continue
        lines.append(f"    section {unit}")
        for op_id, start_ns, end_ns in ops:
            label = _short_label(op_id)
            lines.append(
                f"        {label} :{int(start_ns)}, {int(end_ns)}"
            )

    if total_tiles > max_tiles:
        lines.append(
            f"    %% ... tiles {max_tiles}-{total_tiles - 1} follow steady-state pattern ..."
        )

    lines.append("```")
    return "\n".join(lines)
```

**Step 3: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_report.py -v`
Expected: ALL PASS (both existing and new tests)

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py
git commit -m "feat: add micro_schedule_to_mermaid() for pipeline diagram output"
```

---

### Task 3: Wire `--mermaid` and `--max-tiles` into CLI

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py`

**Step 1: Add the new arguments and logic**

In `cli.py`, add to the import line:
```python
from micro_op_report import micro_schedule_to_json, micro_schedule_to_text, micro_schedule_to_mermaid
```

Add two new argparse arguments after the `--tiling` line:
```python
parser.add_argument("--mermaid", action="store_true", help="Output Mermaid Gantt pipeline diagram (micro mode only)")
parser.add_argument("--max-tiles", type=int, default=3, help="Max tiles to show in Mermaid diagram (default: 3)")
```

After the micro-op text/json print block (line 45), add:
```python
        if args.mermaid:
            print()
            print(micro_schedule_to_mermaid(schedule, graph, max_tiles=args.max_tiles))
```

Before the micro block, add validation for step mode + mermaid:
```python
    if args.mermaid and args.analysis_level != "micro":
        parser.error("--mermaid requires --analysis-level micro")
```

**Step 2: Smoke test the CLI**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python cli.py --steps examples/flash_attention.json --analysis-level micro --mermaid`
Expected: Normal micro-op text output followed by a ````mermaid` ... ```` block with DMA/MXU/VPU sections.

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python cli.py --steps examples/flash_attention.json --mermaid 2>&1; echo "exit: $?"`
Expected: Error message about `--mermaid requires --analysis-level micro`, non-zero exit.

**Step 3: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py
git commit -m "feat: add --mermaid and --max-tiles CLI flags"
```

---

### Task 4: Update SKILL.md — Chinese output directive + Mermaid instructions

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md`

**Step 1: Add Output Language section after "Required Output Sections" (after line 180)**

```markdown
## Output Language

When writing analysis conclusions, use **Chinese** for all narrative text:
- Section headers, bottleneck diagnoses, optimization recommendations, and summary conclusions: 用中文
- Technical terms (HBM, VMEM, MXU, VPU, DMA, FLOPS, roofline) keep English spelling
- Numeric data, formulas, units (ns, us, ms, GB/s, TFLOPS), and code blocks remain unchanged
```

**Step 2: Add Mermaid Pipeline Diagram section after Output Language**

```markdown
## Pipeline Diagram

When using micro-op analysis, ALWAYS include the Mermaid pipeline diagram by adding `--mermaid` to the CLI command:

```bash
python scripts/cli.py --steps steps.json --analysis-level micro --mermaid
```

Include the generated Mermaid Gantt block in your output. The diagram shows the first 3 tiles by default (startup + steady-state overlap). Use `--max-tiles N` to adjust.

The diagram groups micro-ops by execution unit (DMA / MXU / VPU) and visually shows:
- Which operations overlap across different units (pipeline parallelism)
- Where stalls create gaps between bars
- How double-buffering enables tile overlap
```

**Step 3: Update Run Simulation section — add Mermaid example command**

After the existing `# Micro-op analysis with timeline details` example (around line 122-124), add:

```bash
# Micro-op analysis with pipeline diagram
python scripts/cli.py --steps steps.json --analysis-level micro --mermaid

# Pipeline diagram showing first 5 tiles
python scripts/cli.py --steps steps.json --analysis-level micro --mermaid --max-tiles 5
```

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md
git commit -m "docs: add Chinese output directive and Mermaid pipeline diagram instructions to SKILL.md"
```

---

### Task 5: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest -v`
Expected: ALL PASS

**Step 2: End-to-end smoke test**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python cli.py --steps examples/flash_attention.json --analysis-level micro --mermaid`
Expected: Full micro-op report followed by Mermaid Gantt block with DMA/MXU/VPU sections and tile0/tile1/tile2 ops.

**Step 3: Verify SKILL.md YAML frontmatter is valid**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model && python -c "import yaml; yaml.safe_load(open('SKILL.md').read().split('---')[1])"`
Expected: No errors
