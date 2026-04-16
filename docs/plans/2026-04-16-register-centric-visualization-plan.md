# Register-Centric Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace execution-unit-grouped Mermaid diagrams with register/VMEM-centric resource occupancy Gantt and data flow flowchart.

**Architecture:** Rewrite `micro_schedule_to_mermaid()` to group by VMEM slots and REG groups instead of DMA/MXU/VPU. Rewrite `micro_schedule_to_mermaid_flowchart()` to use fragment nodes at each memory level (HBM/VMEM/REG) with data movement edges. Add peak resource tracking to `ScheduleResult`.

**Tech Stack:** Python 3, Mermaid Gantt/Flowchart syntax, unittest

---

### Task 1: Add peak resource tracking to scheduler

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py:27-34`
- Test: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

**Step 1: Write failing test**

Add to `test_micro_op_scheduler.py`:

```python
class TestPeakResources(unittest.TestCase):
    def test_schedule_result_has_peak_fields(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[
                TensorRef(name="Q", shape=(256, 128), dtype="bf16"),
                TensorRef(name="K", shape=(128, 256), dtype="bf16"),
            ],
            outputs=[TensorRef(name="S", shape=(256, 256), dtype="bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 256, "N": 256, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=2, double_buffer=True,
            tile_input_bytes=65536, tile_output_bytes=32768,
            vmem_usage_bytes=196608,
        )
        graph = build_micro_op_graph_for_step(step, tile, step_idx=0)
        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertIsInstance(result.peak_vmem_slots, int)
        self.assertIsInstance(result.peak_reg_groups, int)
        self.assertGreater(result.peak_vmem_slots, 0)
        self.assertGreater(result.peak_reg_groups, 0)

    def test_peak_reg_groups_within_hardware_limit(self):
        from compute_step import ComputeStep, TensorRef
        from micro_op_builder import build_micro_op_graph_for_step
        from pipeline_simulator import TileConfig

        step = ComputeStep(
            name="qk_matmul", op_type="matmul",
            inputs=[
                TensorRef(name="Q", shape=(256, 128), dtype="bf16"),
                TensorRef(name="K", shape=(128, 256), dtype="bf16"),
            ],
            outputs=[TensorRef(name="S", shape=(256, 256), dtype="bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 256, "N": 256, "K": 128},
            compute_unit="MXU", fusable_with_prev=False,
        )
        tile = TileConfig(
            block_dims={"M": 128, "N": 128, "K": 128},
            num_tiles=2, double_buffer=True,
            tile_input_bytes=65536, tile_output_bytes=32768,
            vmem_usage_bytes=196608,
        )
        graph = build_micro_op_graph_for_step(step, tile, step_idx=0)
        result = schedule_micro_op_graph(graph, TPU_V7X)
        self.assertLessEqual(result.peak_reg_groups, TPU_V7X.reg_group_count)
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_scheduler.py::TestPeakResources -v`
Expected: FAIL — `AttributeError: ... has no attribute 'peak_vmem_slots'`

**Step 3: Add peak tracking to ScheduleResult and scheduler**

In `micro_op_scheduler.py`:

1. Add fields to `ScheduleResult` (line 33):
```python
    peak_vmem_slots: int
    peak_reg_groups: int
```

2. In `schedule_micro_op_graph()`, after the scheduling loop (after line 217), compute peaks:
```python
    # Compute peak concurrent resource usage
    all_events: list[tuple[float, int, str]] = []  # (time, +1/-1, type)
    for resource_id, intervals in resource_occupancy.items():
        if resource_id.startswith("VMEM:"):
            for iv in intervals:
                all_events.append((iv.start_ns, +1, "vmem"))
                all_events.append((iv.end_ns, -1, "vmem"))
        elif resource_id.startswith("REG:"):
            for iv in intervals:
                all_events.append((iv.start_ns, +1, "reg"))
                all_events.append((iv.end_ns, -1, "reg"))
    all_events.sort(key=lambda e: (e[0], e[1]))
    peak_vmem = 0
    peak_reg = 0
    cur_vmem = 0
    cur_reg = 0
    for _, delta, rtype in all_events:
        if rtype == "vmem":
            cur_vmem += delta
            peak_vmem = max(peak_vmem, cur_vmem)
        else:
            cur_reg += delta
            peak_reg = max(peak_reg, cur_reg)
```

3. Pass peaks to `ScheduleResult`:
```python
    return ScheduleResult(
        ...,
        peak_vmem_slots=peak_vmem,
        peak_reg_groups=peak_reg,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_scheduler.py::TestPeakResources -v`
Expected: PASS

**Step 5: Run all scheduler tests to avoid regressions**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_scheduler.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py \
       plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py
git commit -m "feat(tpu-perf-model): add peak VMEM/REG resource tracking to scheduler"
```

---

### Task 2: Rewrite Gantt to resource-centric view

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py:270-348`
- Test: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

**Step 1: Write failing tests for the new Gantt format**

Replace `TestMermaidOutput` and `TestEnhancedGantt` in `test_micro_op_report.py` with:

```python
class TestResourceGantt(unittest.TestCase):
    def test_gantt_has_vmem_section(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("section VMEM Slots", output)

    def test_gantt_has_reg_section(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("section REG Groups", output)

    def test_gantt_has_no_unit_sections(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertNotIn("section DMA", output)
        self.assertNotIn("section MXU", output)
        self.assertNotIn("section VPU", output)

    def test_gantt_has_capacity_comments(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("Peak VMEM", output)
        self.assertIn("Peak REG", output)

    def test_gantt_contains_slot_names(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertTrue(
            "q_slot" in output and "k_slot" in output,
            f"Expected VMEM slot names in output:\n{output}",
        )

    def test_gantt_contains_reg_group_names(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertTrue(
            "q_reg" in output and "acc_reg" in output,
            f"Expected REG group names in output:\n{output}",
        )

    def test_gantt_includes_stall_bars(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph)
        self.assertIn("crit", output)

    def test_gantt_filters_tiles_by_max(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("tile0", output)
        self.assertNotIn("tile1", output)

    def test_gantt_shows_ellipsis_when_truncated(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
        self.assertIn("%%", output)
        self.assertIn("steady-state", output)

    def test_gantt_rejects_non_positive_max_tiles(self):
        from micro_op_report import micro_schedule_to_mermaid
        schedule, graph = _sample_mermaid_schedule()
        with self.assertRaises(ValueError):
            micro_schedule_to_mermaid(schedule, graph, max_tiles=0)
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_report.py::TestResourceGantt -v`
Expected: FAIL — `section VMEM Slots` not found

**Step 3: Rewrite `micro_schedule_to_mermaid()`**

Replace the function body (lines 270-348 of `micro_op_report.py`) with logic that:

1. Groups `resource_occupancy` intervals by prefix: `VMEM:*` and `REG:*`
2. For each resource key (e.g., `VMEM:q_slot0`), iterate its `OccupancyInterval` list sorted by `start_ns`
3. Output under `section VMEM Slots` / `section REG Groups`
4. Bar label: `resource_name [short_op_label]` — use `_short_label(interval.op_id)` for the op part
5. Insert `crit` stall bars between consecutive intervals on the same resource when there is a gap, using `_detect_op_stalls` to label the wait reason
6. Filter intervals by `max_tiles` using `_tile_index_from_op_id(interval.op_id)`
7. At the end, add capacity comments from `schedule.peak_vmem_slots` and `schedule.peak_reg_groups`

Key implementation detail — the resource occupancy dict already has entries like `VMEM:q_slot0`, `REG:acc_reg0` etc. Group and sort these.

```python
def micro_schedule_to_mermaid(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
    max_tiles: int = 3,
) -> str:
    """Render the micro-op schedule as a resource-centric Mermaid Gantt."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")

    all_tile_indices = set()
    for op_id in schedule.op_timings:
        idx = _tile_index_from_op_id(op_id)
        if idx is not None:
            all_tile_indices.add(idx)
    total_tiles = max(all_tile_indices, default=0) + 1 if all_tile_indices else 0

    step_names = []
    for op in graph.micro_ops.values():
        if op.step_name not in step_names:
            step_names.append(op.step_name)
    title = ", ".join(step_names)

    stalls = _detect_op_stalls(schedule, graph)

    # Collect intervals grouped by resource type
    vmem_resources: dict[str, list[OccupancyInterval]] = {}
    reg_resources: dict[str, list[OccupancyInterval]] = {}
    for res_id, intervals in schedule.resource_occupancy.items():
        filtered = []
        for iv in intervals:
            tile_idx = _tile_index_from_op_id(iv.op_id)
            if tile_idx is not None and tile_idx >= max_tiles:
                continue
            filtered.append(iv)
        if not filtered:
            continue
        if res_id.startswith("VMEM:"):
            vmem_resources[res_id] = sorted(filtered, key=lambda iv: iv.start_ns)
        elif res_id.startswith("REG:"):
            reg_resources[res_id] = sorted(filtered, key=lambda iv: iv.start_ns)

    lines = [
        "```mermaid",
        "gantt",
        f"    title Resource Occupancy: {title} (ns)",
        "    dateFormat x",
        "    axisFormat %Q",
    ]

    # VMEM section
    if vmem_resources:
        lines.append("    section VMEM Slots")
        for res_id in sorted(vmem_resources):
            slot_name = res_id.split(":", 1)[1]
            intervals = vmem_resources[res_id]
            prev_end = None
            for iv in intervals:
                if prev_end is not None and iv.start_ns > prev_end:
                    wait_reasons = stalls.get(iv.op_id, [])
                    wait_label = ",".join(wait_reasons) if wait_reasons else "WAIT"
                    lines.append(f"        {wait_label} :crit, {int(prev_end)}, {int(iv.start_ns)}")
                label = f"{slot_name} [{_short_label(iv.op_id)}]"
                lines.append(f"        {label} :{int(iv.start_ns)}, {int(iv.end_ns)}")
                prev_end = iv.end_ns

    # REG section
    if reg_resources:
        lines.append("    section REG Groups")
        for res_id in sorted(reg_resources):
            reg_name = res_id.split(":", 1)[1]
            intervals = reg_resources[res_id]
            prev_end = None
            for iv in intervals:
                if prev_end is not None and iv.start_ns > prev_end:
                    wait_reasons = stalls.get(iv.op_id, [])
                    wait_label = ",".join(wait_reasons) if wait_reasons else "WAIT"
                    lines.append(f"        {wait_label} :crit, {int(prev_end)}, {int(iv.start_ns)}")
                label = f"{reg_name} [{_short_label(iv.op_id)}]"
                lines.append(f"        {label} :{int(iv.start_ns)}, {int(iv.end_ns)}")
                prev_end = iv.end_ns

    # Capacity section
    peak_vmem = getattr(schedule, 'peak_vmem_slots', 0)
    peak_reg = getattr(schedule, 'peak_reg_groups', 0)
    lines.append(f"    %% Peak VMEM: {peak_vmem} slots")
    lines.append(f"    %% Peak REG: {peak_reg}/32 groups ({peak_reg * 100 // 32}%)")

    if total_tiles > max_tiles:
        lines.append(f"    %% ... tiles {max_tiles}-{total_tiles - 1} follow steady-state pattern ...")

    lines.append("```")
    return "\n".join(lines) + "\n"
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_report.py::TestResourceGantt -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py \
       plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py
git commit -m "feat(tpu-perf-model): rewrite Gantt to resource-centric VMEM/REG view"
```

---

### Task 3: Rewrite Flowchart to register data flow view

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py:378-435`
- Test: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

**Step 1: Write failing tests for new flowchart format**

Replace `TestFlowchart` in `test_micro_op_report.py` with:

```python
class TestDataFlowChart(unittest.TestCase):
    def test_flowchart_has_memory_level_nodes(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
        self.assertIn("HBM:", output)
        self.assertIn("VMEM", output)
        self.assertIn("REG", output)

    def test_flowchart_has_data_transfer_edges(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
        # Should have solid edges with latency labels
        self.assertIn("-->", output)
        self.assertIn("ns", output)

    def test_flowchart_has_tile_subgraph(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
        self.assertIn("subgraph", output)
        self.assertIn("Tile 0", output)

    def test_flowchart_shows_stall_edges(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph)
        # Dashed edges for stalls (may or may not exist depending on schedule)
        # At minimum, verify the function runs without error
        self.assertIn("flowchart TD", output)

    def test_flowchart_per_tile_count(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=2)
        self.assertEqual(output.count("subgraph"), 2)

    def test_flowchart_rejects_non_positive_max_tiles(self):
        from micro_op_report import micro_schedule_to_mermaid_flowchart
        schedule, graph = _sample_mermaid_schedule()
        with self.assertRaises(ValueError):
            micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=0)
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_report.py::TestDataFlowChart -v`
Expected: FAIL — `HBM:` not found

**Step 3: Rewrite `micro_schedule_to_mermaid_flowchart()`**

The new flowchart creates nodes for each fragment (at HBM/VMEM/REG level) and edges for each micro-op that moves data between levels.

For each tile:
1. Collect all ops in that tile
2. For each op, get its input and output fragments from `graph.fragments`
3. Create nodes for each unique fragment: label = `"LEVEL: tensor[shape] dtype"` (for HBM) or `"VMEM slot_name: tensor[shape]"` (for VMEM) or `"REG group_name: tensor[shape]"` (for REG)
4. Create solid edges from input fragment nodes to output fragment nodes, labeled with `|"op_kind latency_ns"|`
5. For ops with stalls, create dashed edges labeled with `|"WAIT_REASON"|`

```python
def micro_schedule_to_mermaid_flowchart(
    schedule: ScheduleResult,
    graph: MicroOpGraph,
    max_tiles: int = 3,
) -> str:
    """Render per-tile data flow flowcharts with fragment nodes at each memory level."""
    if max_tiles < 1:
        raise ValueError("max_tiles must be >= 1")

    all_tile_indices = set()
    for op_id in schedule.op_timings:
        idx = _tile_index_from_op_id(op_id)
        if idx is not None:
            all_tile_indices.add(idx)
    total_tiles = max(all_tile_indices, default=0) + 1 if all_tile_indices else 0

    stalls = _detect_op_stalls(schedule, graph)
    blocks: list[str] = []

    for tile_idx in range(min(max_tiles, total_tiles)):
        tile_ops = [
            op_id for op_id in graph.micro_ops
            if _tile_index_from_op_id(op_id) == tile_idx
        ]
        if not tile_ops:
            continue

        lines = [
            "```mermaid",
            "flowchart TD",
            f'    subgraph tile{tile_idx}["Tile {tile_idx}"]',
        ]

        # Collect unique fragments referenced by this tile's ops
        seen_frags = set()
        for op_id in tile_ops:
            op = graph.micro_ops[op_id]
            for frag_id in op.input_fragments + op.output_fragments:
                seen_frags.add(frag_id)

        # Define fragment nodes
        for frag_id in sorted(seen_frags):
            frag = graph.fragments.get(frag_id)
            if not frag:
                continue
            node_id = _sanitize_node_id(frag_id)
            shape_str = "[" + ",".join(str(d) for d in frag.shape) + "]" if frag.shape else ""
            level = frag.home_level
            # Build label based on level
            if level == "HBM":
                label = f"HBM: {frag.tensor_name}{shape_str} {frag.dtype}"
            elif level == "VMEM":
                # Find VMEM slot from ops that use this fragment
                slot = _find_resource_for_fragment(graph, frag_id, tile_ops, "vmem")
                label = f"VMEM {slot}: {frag.tensor_name}{shape_str}"
            elif level in ("REG", "acc"):
                reg = _find_resource_for_fragment(graph, frag_id, tile_ops, "reg")
                label = f"REG {reg}: {frag.tensor_name}{shape_str}"
            else:
                label = f"{level}: {frag.tensor_name}{shape_str}"
            lines.append(f'        {node_id}["{label}"]')

        # Create edges: for each op, connect input frags -> output frags
        for op_id in tile_ops:
            op = graph.micro_ops[op_id]
            timing = schedule.op_timings.get(op_id)
            latency = f"{int(timing.end_ns - timing.start_ns)}ns" if timing else ""
            reasons = stalls.get(op_id, [])

            for in_frag in op.input_fragments:
                if in_frag not in seen_frags:
                    continue
                for out_frag in op.output_fragments:
                    if out_frag not in seen_frags:
                        continue
                    src = _sanitize_node_id(in_frag)
                    dst = _sanitize_node_id(out_frag)
                    edge_label = f"{op.op_kind} {latency}"
                    if reasons:
                        reason_str = ",".join(reasons)
                        lines.append(f'        {src} -."{reason_str} {edge_label}".-> {dst}')
                    else:
                        lines.append(f'        {src} -->|"{edge_label}"| {dst}')

        lines.append("    end")
        lines.append("```")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"
```

Also add the helper function `_find_resource_for_fragment()`:

```python
def _find_resource_for_fragment(
    graph: MicroOpGraph,
    frag_id: str,
    tile_ops: list[str],
    resource_type: str,
) -> str:
    """Find the VMEM slot or REG group associated with a fragment in tile ops."""
    for op_id in tile_ops:
        op = graph.micro_ops[op_id]
        if frag_id in op.input_fragments or frag_id in op.output_fragments:
            if resource_type == "vmem" and op.required_vmem_slots:
                return op.required_vmem_slots[0]
            if resource_type == "reg" and op.required_reg_groups:
                return op.required_reg_groups[0]
    return ""
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_micro_op_report.py::TestDataFlowChart -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py \
       plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py
git commit -m "feat(tpu-perf-model): rewrite flowchart to register data flow view"
```

---

### Task 4: Remove dead code and run full test suite

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

**Step 1: Remove old helper functions no longer used**

After the rewrites, these functions may be unused:
- `_unit_from_op()` — was for grouping by DMA/MXU/VPU
- `_enhanced_label()` — was for unit-based Gantt labels
- `_flowchart_node_label()` — was for op-centric flowchart nodes

Check each with grep. If unused, delete.

Also remove old test classes that were replaced:
- `TestMermaidOutput` → replaced by `TestResourceGantt`
- `TestEnhancedGantt` → replaced by `TestResourceGantt`
- `TestFlowchart` → replaced by `TestDataFlowChart`

**Step 2: Add `OccupancyInterval` import to micro_op_report.py**

The rewritten `micro_schedule_to_mermaid()` accesses `OccupancyInterval` objects from `schedule.resource_occupancy`. Ensure the import is present:

```python
from micro_op_scheduler import OccupancyInterval, ScheduleResult
```

**Step 3: Run all tests**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py \
       plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py
git commit -m "refactor(tpu-perf-model): remove dead code from old diagram format"
```

---

### Task 5: Update SKILL.md documentation

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md:195-247`

**Step 1: Replace Pipeline Diagram section**

Replace lines 195-247 of SKILL.md with documentation describing the new diagrams:

```markdown
## Pipeline Diagram

When using micro-op analysis, ALWAYS include the Mermaid pipeline diagrams by adding `--mermaid` to the CLI command:

\```bash
python scripts/cli.py --steps steps.json --analysis-level micro --mermaid
\```

Include the generated Mermaid blocks in your output. The `--mermaid` flag produces two complementary diagrams:

### Resource Occupancy Gantt

Shows VMEM slot and REG group occupancy over time. Each row is a storage resource, not an execution unit.

- **section VMEM Slots**: One bar per VMEM slot occupancy interval, labeled `slot_name [op_label]`
- **section REG Groups**: One bar per REG group occupancy interval, labeled `reg_name [op_label]`
- **Stall bars**: Red `crit` bars between intervals on the same resource, labeled with wait reason (`WAIT_DATA`, `WAIT_UNIT`, `WAIT_VMEM`, `WAIT_REG`)
- **Capacity comments**: Peak VMEM slots and REG groups with percentage of hardware limit

Shows first 3 tiles by default. Use `--max-tiles N` to adjust.

### Register Data Flow Flowchart

One flowchart per tile showing data movement through the memory hierarchy:

- **Nodes**: Data fragments at each memory level — `HBM: tensor[shape] dtype`, `VMEM slot: tensor[shape]`, `REG group: tensor[shape]`
- **Solid edges** (`-->`): Data transfers labeled with `op_kind latency`
- **Dashed edges** (`-. .-->`): Stall/wait relationships labeled with reason

Use the flowchart to trace how data flows from HBM through VMEM into registers, through compute, and back.
```

**Step 2: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md
git commit -m "docs(tpu-perf-model): update Pipeline Diagram docs for register-centric view"
```

---

### Task 6: End-to-end verification with flash_attention example

**Step 1: Run the CLI with the flash_attention example to verify output**

```bash
cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts
python cli.py --steps examples/flash_attention.json --analysis-level micro --mermaid
```

Expected: Output contains `section VMEM Slots`, `section REG Groups`, `Peak VMEM`, `Peak REG`, `subgraph`, `HBM:`, `VMEM`, `REG`.

**Step 2: Run full test suite one final time**

```bash
cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest -v
```

Expected: ALL PASS

**Step 3: Commit if any fixes were needed**

Only if fixes were required during verification.
