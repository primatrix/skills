# TPU Perf Model Micro-Op Dataflow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `plugins/tpu-perf-model` from step-level timing analysis to fragment-level micro-op, buffer-slot, and residency analysis while preserving the existing step-level workflow.

**Architecture:** Keep `ComputeStep` as the user-facing input and retain the existing step-level simulator. Add four new modules under `scripts/` for micro-op IR, step expansion, scheduling, and reporting; wire them into `cli.py` behind `--analysis-level micro`; update `SKILL.md` so the documented workflow matches the new runtime model.

**Tech Stack:** Python 3, `dataclasses`, `json`, `unittest`, Markdown (`SKILL.md`)

---

**Spec Reference:** `docs/superpowers/specs/2026-04-15-tpu-perf-model-micro-op-dataflow-design.md`

## File Structure

- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/hw_params.py`
  Responsibility: expose resource counts used by the micro-op scheduler, without breaking existing callers.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_ir.py`
  Responsibility: define `TensorFragment`, `MicroOp`, `MicroOpGraph`, occupancy records, and `ScheduleResult`.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_builder.py`
  Responsibility: expand `ComputeStep` plus tiling into fragment and micro-op graphs.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py`
  Responsibility: schedule micro-ops under dependency, unit, VMEM, and register constraints.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py`
  Responsibility: render micro-op text and JSON reports, including timeline, residency, stalls, and critical path.
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py`
  Responsibility: select between legacy step-level output and new micro-op output.
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md`
  Responsibility: teach the agent the two-layer workflow and the new analysis sections.
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py`
  Responsibility: lock down new resource fields.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_ir.py`
  Responsibility: validate the IR dataclasses and graph helpers.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`
  Responsibility: validate expansion from steps into fragments and micro-op sequences.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`
  Responsibility: validate legality of schedules, stall accounting, and critical-path extraction.
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`
  Responsibility: validate the text and JSON micro-op reports.
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py`
  Responsibility: cover the new CLI mode end-to-end while preserving the old mode.

## Chunk 1: Core Resource Model

### Task 1: Extend hardware parameters for micro-op scheduling

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/hw_params.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py`

- [ ] **Step 1: Add failing hardware-resource tests**

```python
def test_v7x_dma_engine_count(self):
    from hw_params import TPU_V7X
    self.assertEqual(TPU_V7X.dma_engine_count, 1)

def test_v7x_vpu_count(self):
    from hw_params import TPU_V7X
    self.assertEqual(TPU_V7X.vpu_count, 1)

def test_reg_group_count_aliases_vpr_count(self):
    from hw_params import TPU_V7X
    self.assertEqual(TPU_V7X.reg_group_count, TPU_V7X.vpr_count)
```

- [ ] **Step 2: Run the hardware test file to verify it fails**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py`

Expected: FAIL with `AttributeError` for `dma_engine_count`, `vpu_count`, or `reg_group_count`.

- [ ] **Step 3: Add the minimal hardware fields and helper property**

```python
@dataclass(frozen=True)
class TPUParams:
    ...
    dma_engine_count: int
    vpu_count: int

    @property
    def reg_group_count(self) -> int:
        return self.vpr_count


TPU_V7X = TPUParams(
    ...
    dma_engine_count=1,
    vpu_count=1,
)
```

- [ ] **Step 4: Re-run the hardware test file**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py`

Expected: PASS.

- [ ] **Step 5: Commit the hardware metadata change**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/hw_params.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py
git commit -m "feat: add TPU resource metadata for micro-op scheduling"
```

### Task 2: Introduce the micro-op IR

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_ir.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_ir.py`

- [ ] **Step 1: Write failing IR tests**

```python
def test_micro_op_graph_tracks_roots_and_leaves(self):
    from micro_op_ir import TensorFragment, MicroOp, MicroOpGraph

    q = TensorFragment(
        fragment_id="q_0_0",
        tensor_name="Q",
        step_name="qk_matmul",
        shape=(128, 128),
        dtype="bf16",
        size_bytes=128 * 128 * 2,
        home_level="HBM",
    )
    load_q = MicroOp(
        op_id="load_q",
        step_name="qk_matmul",
        op_kind="dma_load_hbm_to_vmem",
        depends_on=[],
        input_fragments=[],
        output_fragments=["q_0_0"],
        required_units=("DMA",),
        required_vmem_slots=("slot_q",),
        required_reg_groups=(),
        latency_ns=18.0,
    )
    graph = MicroOpGraph(fragments={"q_0_0": q}, micro_ops={"load_q": load_q})
    self.assertEqual(graph.root_ops(), ["load_q"])
    self.assertEqual(graph.leaf_ops(), ["load_q"])
```

- [ ] **Step 2: Run the new IR test file**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_ir.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'micro_op_ir'`.

- [ ] **Step 3: Implement the minimal IR layer**

```python
@dataclass
class TensorFragment:
    fragment_id: str
    tensor_name: str
    step_name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    home_level: str
    producer_op: str | None = None
    consumer_ops: tuple[str, ...] = ()


@dataclass
class MicroOp:
    op_id: str
    step_name: str
    op_kind: str
    depends_on: list[str]
    input_fragments: list[str]
    output_fragments: list[str]
    required_units: tuple[str, ...]
    required_vmem_slots: tuple[str, ...]
    required_reg_groups: tuple[str, ...]
    latency_ns: float


@dataclass
class MicroOpGraph:
    fragments: dict[str, TensorFragment]
    micro_ops: dict[str, MicroOp]

    def root_ops(self) -> list[str]:
        return sorted(op_id for op_id, op in self.micro_ops.items() if not op.depends_on)

    def leaf_ops(self) -> list[str]:
        parents = {dep for op in self.micro_ops.values() for dep in op.depends_on}
        return sorted(op_id for op_id in self.micro_ops if op_id not in parents)
```

- [ ] **Step 4: Re-run the IR tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_ir.py`

Expected: PASS.

- [ ] **Step 5: Commit the IR layer**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_ir.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_ir.py
git commit -m "feat: add micro-op IR primitives"
```

## Chunk 2: Step Expansion Into Fragments and Micro-Ops

### Task 3: Expand a matmul step into fragments and micro-ops

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_builder.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`

- [ ] **Step 1: Write a failing matmul-expansion test**

```python
def test_matmul_expands_into_dma_reg_mxu_store_pipeline(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul",
        op_type="matmul",
        inputs=[TensorRef("Q", (256, 128), "bf16"), TensorRef("K", (128, 256), "bf16")],
        outputs=[TensorRef("S", (256, 256), "bf16")],
        flops_formula="2*M*N*K",
        flops_vars={"M": 256, "N": 256, "K": 128},
        compute_unit="MXU",
        fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128},
        num_tiles=4,
        tile_input_bytes=128 * 128 * 2 * 2,
        tile_output_bytes=128 * 128 * 2,
        double_buffer=True,
        vmem_usage_bytes=128 * 128 * 2 * 5,
    )

    graph = build_micro_op_graph_for_step(step, tile)
    op_kinds = [graph.micro_ops[op_id].op_kind for op_id in sorted(graph.micro_ops)]
    self.assertIn("dma_load_hbm_to_vmem", op_kinds)
    self.assertIn("vmem_to_reg", op_kinds)
    self.assertIn("mxu_compute", op_kinds)
    self.assertIn("reg_to_vmem", op_kinds)
    self.assertIn("dma_store_vmem_to_hbm", op_kinds)
```

- [ ] **Step 2: Run the builder test file**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`

Expected: FAIL with `ModuleNotFoundError` or missing `build_micro_op_graph_for_step`.

- [ ] **Step 3: Implement the matmul builder**

```python
def build_micro_op_graph_for_step(step: ComputeStep, tile: TileConfig) -> MicroOpGraph:
    fragments = {}
    micro_ops = {}
    for tile_idx in range(tile.num_tiles):
        q_frag = _input_fragment(step, tile_idx, "Q", "HBM")
        k_frag = _input_fragment(step, tile_idx, "K", "HBM")
        acc_frag = _output_fragment(step, tile_idx, "acc", "REG")
        out_frag = _output_fragment(step, tile_idx, "out", "VMEM")
        ...
        _add_dma_load(micro_ops, q_frag, slot_id=f"slot_q_{tile_idx % 2}")
        _add_dma_load(micro_ops, k_frag, slot_id=f"slot_k_{tile_idx % 2}")
        _add_vmem_to_reg(micro_ops, q_frag, reg_group=f"reg_q_{tile_idx % 2}")
        _add_vmem_to_reg(micro_ops, k_frag, reg_group=f"reg_k_{tile_idx % 2}")
        _add_mxu_compute(micro_ops, q_frag, k_frag, acc_frag)
        _add_reg_to_vmem(micro_ops, acc_frag, out_frag, slot_id=f"slot_out_{tile_idx % 2}")
        _add_dma_store(micro_ops, out_frag)
    return MicroOpGraph(fragments=fragments, micro_ops=micro_ops)
```

- [ ] **Step 4: Re-run the matmul builder tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`

Expected: PASS.

- [ ] **Step 5: Commit the builder baseline**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_builder.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py
git commit -m "feat: expand matmul steps into micro-op graphs"
```

### Task 4: Add fused-step expansion without redundant HBM traffic

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_builder.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`

- [ ] **Step 1: Add a failing fused-expansion test**

```python
def test_fused_elementwise_reuses_previous_fragment(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_pipeline
    from pipeline_simulator import TileConfig

    matmul = ComputeStep(...)
    scale = ComputeStep(
        name="scale_scores",
        op_type="elementwise",
        inputs=[TensorRef("S", (256, 256), "bf16")],
        outputs=[TensorRef("S_scaled", (256, 256), "bf16")],
        flops_formula="M*N",
        flops_vars={"M": 256, "N": 256},
        compute_unit="VPU",
        fusable_with_prev=True,
    )

    graph = build_micro_op_graph_for_pipeline([matmul, scale], {"qk_matmul": tile, "scale_scores": tile})
    op_kinds = [op.op_kind for op in graph.micro_ops.values()]
    self.assertIn("vpu_compute", op_kinds)
    self.assertEqual(op_kinds.count("dma_store_vmem_to_hbm"), 1)
```

- [ ] **Step 2: Run the builder tests again**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`

Expected: FAIL because fused-step expansion still emits redundant store and reload operations.

- [ ] **Step 3: Extend the builder for fused pipelines**

```python
def build_micro_op_graph_for_pipeline(
    steps: list[ComputeStep],
    tile_configs: dict[str, TileConfig],
) -> MicroOpGraph:
    graph = build_micro_op_graph_for_step(steps[0], tile_configs[steps[0].name])
    for prev_step, step in zip(steps, steps[1:]):
        if step.fusable_with_prev:
            _reuse_previous_output_fragment(graph, prev_step.name, step.name)
            _append_vpu_or_mxu_ops_without_hbm_roundtrip(graph, step, tile_configs[step.name])
        else:
            _append_unfused_step(graph, step, tile_configs[step.name])
    return graph
```

- [ ] **Step 4: Re-run the builder tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py`

Expected: PASS.

- [ ] **Step 5: Commit fused graph expansion**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_builder.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_builder.py
git commit -m "feat: support fused micro-op expansion"
```

## Chunk 3: Scheduling, Stalls, and Critical Path

### Task 5: Schedule micro-ops under dependency and unit constraints

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

- [ ] **Step 1: Write a failing scheduling test**

```python
def test_scheduler_serializes_dma_and_respects_dependencies(self):
    from micro_op_ir import MicroOp, MicroOpGraph
    from micro_op_scheduler import schedule_micro_op_graph
    from hw_params import TPU_V7X

    graph = MicroOpGraph(
        fragments={},
        micro_ops={
            "load_q": MicroOp(..., op_kind="dma_load_hbm_to_vmem", depends_on=[], required_units=("DMA",), latency_ns=10.0),
            "load_k": MicroOp(..., op_kind="dma_load_hbm_to_vmem", depends_on=[], required_units=("DMA",), latency_ns=10.0),
            "mxu": MicroOp(..., op_kind="mxu_compute", depends_on=["load_q", "load_k"], required_units=("MXU",), latency_ns=20.0),
        },
    )
    result = schedule_micro_op_graph(graph, TPU_V7X)
    self.assertEqual(result.op_timings["load_q"].start_ns, 0.0)
    self.assertEqual(result.op_timings["load_k"].start_ns, 10.0)
    self.assertEqual(result.op_timings["mxu"].start_ns, 20.0)
```

- [ ] **Step 2: Run the scheduler tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

Expected: FAIL with `ModuleNotFoundError` or missing `schedule_micro_op_graph`.

- [ ] **Step 3: Implement the minimal scheduler**

```python
def schedule_micro_op_graph(graph: MicroOpGraph, hw: TPUParams) -> ScheduleResult:
    now = 0.0
    ready = _root_queue(graph)
    active = []
    state = ResourceState.empty(hw)
    while ready or active:
        _issue_ready_ops(ready, active, state, now, hw)
        now = _next_completion_time(active)
        finished = _pop_finished(active, now)
        _release_finished_resources(state, finished)
        _promote_newly_ready_ops(graph, ready, finished)
    return _build_schedule_result(...)
```

- [ ] **Step 4: Re-run the scheduler tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

Expected: PASS.

- [ ] **Step 5: Commit the scheduler baseline**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py
git commit -m "feat: add micro-op scheduler"
```

### Task 6: Add VMEM/register stalls and critical-path extraction

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

- [ ] **Step 1: Add failing stall and critical-path tests**

```python
def test_scheduler_records_wait_vmem_and_wait_reg(self):
    result = schedule_micro_op_graph(graph_with_slot_and_reg_conflict, TPU_V7X)
    self.assertGreater(result.stall_breakdown["WAIT_VMEM"], 0)
    self.assertGreater(result.stall_breakdown["WAIT_REG"], 0)

def test_scheduler_extracts_critical_path(self):
    result = schedule_micro_op_graph(graph_with_known_tail, TPU_V7X)
    self.assertEqual(result.critical_path[-1], "store_out")
    self.assertIn("mxu_main", result.critical_path)
```

- [ ] **Step 2: Run the scheduler tests again**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

Expected: FAIL because the scheduler does not yet track stall classes or critical path.

- [ ] **Step 3: Extend the scheduler result**

```python
@dataclass
class ScheduleResult:
    op_timings: dict[str, OpTiming]
    resource_occupancy: dict[str, list[OccupancyInterval]]
    fragment_residency: dict[str, list[OccupancyInterval]]
    stall_breakdown: dict[str, int]
    critical_path: list[str]
    total_time_ns: float


def _classify_wait(op: MicroOp, state: ResourceState) -> str:
    if not _deps_ready(op):
        return "WAIT_DATA"
    if not _units_available(op, state):
        return "WAIT_UNIT"
    if not _vmem_slots_available(op, state):
        return "WAIT_VMEM"
    return "WAIT_REG"
```

- [ ] **Step 4: Re-run the scheduler tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py`

Expected: PASS.

- [ ] **Step 5: Commit stall tracking and critical-path logic**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_scheduler.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_scheduler.py
git commit -m "feat: report stalls and critical path for micro-op schedules"
```

## Chunk 4: Reporting and CLI Integration

### Task 7: Render micro-op text and JSON reports

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

- [ ] **Step 1: Write failing report tests**

```python
def test_micro_report_json_contains_schedule_sections(self):
    from micro_op_report import micro_schedule_to_json
    payload = json.loads(micro_schedule_to_json(schedule_result))
    self.assertIn("micro_ops", payload)
    self.assertIn("timeline", payload)
    self.assertIn("fragment_residency", payload)
    self.assertIn("critical_path", payload)

def test_micro_report_text_contains_human_sections(self):
    from micro_op_report import micro_schedule_to_text
    text = micro_schedule_to_text(schedule_result)
    self.assertIn("Macro Summary", text)
    self.assertIn("Micro-Op Schedule Summary", text)
    self.assertIn("Residency and Occupancy", text)
    self.assertIn("Critical Path and Optimization Hints", text)
```

- [ ] **Step 2: Run the report tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'micro_op_report'`.

- [ ] **Step 3: Implement the micro-op report renderer**

```python
def micro_schedule_to_json(schedule: ScheduleResult, step_results: list[dict]) -> str:
    return json.dumps({
        "summary": _summary_dict(schedule, step_results),
        "step_results": step_results,
        "micro_ops": _micro_op_rows(schedule),
        "timeline": _timeline_rows(schedule),
        "resource_occupancy": _occupancy_dict(schedule),
        "fragment_residency": _residency_dict(schedule),
        "critical_path": schedule.critical_path,
        "stall_breakdown": schedule.stall_breakdown,
        "optimization_hints": _optimization_hints(schedule),
    }, indent=2)
```

- [ ] **Step 4: Re-run the report tests**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py`

Expected: PASS.

- [ ] **Step 5: Commit report rendering**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/micro_op_report.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_micro_op_report.py
git commit -m "feat: add micro-op reporting"
```

### Task 8: Wire micro-op mode into the CLI and integration tests

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py`

- [ ] **Step 1: Add failing integration coverage for micro mode**

```python
def test_cli_runs_in_micro_mode(self):
    result = subprocess.run(
        [
            "python", os.path.join(scripts_dir, "cli.py"),
            "--steps", example_path,
            "--analysis-level", "micro",
            "--format", "json",
        ],
        capture_output=True,
        text=True,
        cwd=scripts_dir,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    data = json.loads(result.stdout)
    self.assertIn("micro_ops", data)
    self.assertIn("critical_path", data)
```

- [ ] **Step 2: Run the integration test file**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py`

Expected: FAIL with `unrecognized arguments: --analysis-level micro`.

- [ ] **Step 3: Implement CLI branching**

```python
parser.add_argument("--analysis-level", choices=["step", "micro"], default="step")
parser.add_argument("--show-timeline", action="store_true")
parser.add_argument("--show-residency", action="store_true")
parser.add_argument("--show-critical-path", action="store_true")

if args.analysis_level == "micro":
    tile_configs = {step.name: find_optimal_tiling(step, TPU_V7X) for step in steps}
    graph = build_micro_op_graph_for_pipeline(steps, tile_configs)
    schedule = schedule_micro_op_graph(graph, TPU_V7X)
    if args.format == "json":
        print(micro_schedule_to_json(schedule, step_results))
    else:
        print(micro_schedule_to_text(schedule, step_results, args))
else:
    print(pipeline_report_to_text(report))
```

- [ ] **Step 4: Re-run the integration test file**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py`

Expected: PASS.

- [ ] **Step 5: Commit the CLI integration**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py
git commit -m "feat: expose micro-op analysis in the CLI"
```

## Chunk 5: Skill Documentation and Final Verification

### Task 9: Update the skill documentation to match the runtime model

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md`

- [ ] **Step 1: Replace the step-only workflow with the two-layer workflow**

Use these sections in `SKILL.md`:

```markdown
## Layer A: Formula -> ComputeStep

- Decompose the user formula into `ComputeStep` objects.
- Capture FLOPs, compute units, and fusion eligibility.

## Layer B: ComputeStep -> TensorFragment -> MicroOp -> Schedule

- Enumerate fragment inventory.
- Explain which fragments live in `HBM`, `VMEM`, and `REG`.
- Identify which `DMA`, `MXU`, and `VPU` micro-ops consume each fragment.
- Explain residency, reuse, eviction, and release timing.
- State the critical path and the VMEM-constrained optimality argument.
```

- [ ] **Step 2: Add the required answer template**

Use these output headings in `SKILL.md`:

```markdown
## Required Output Sections

1. Fragment Inventory
2. Micro-Op Expansion
3. Residency Timeline
4. Dependency Graph
5. Critical Path
6. Optimality Argument Under VMEM Constraint
```

- [ ] **Step 3: Add the new CLI usage examples**

Add examples for:

```bash
python scripts/cli.py --steps steps.json --analysis-level micro
python scripts/cli.py --steps steps.json --analysis-level micro --format json
python scripts/cli.py --steps steps.json --analysis-level micro --show-timeline
```

- [ ] **Step 4: Verify the documentation update**

Run: `rg -n "Layer A|Layer B|Fragment Inventory|Optimality Argument Under VMEM Constraint|--analysis-level micro" plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md`

Expected: each phrase appears exactly once in the new guidance sections.

- [ ] **Step 5: Commit the skill update**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md
git commit -m "docs: teach the TPU perf model skill the micro-op workflow"
```

### Task 10: Run the final regression suite

**Files:**
- Test: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_*.py`

- [ ] **Step 1: Run the full Python test suite**

Run: `python -m unittest discover -s plugins/tpu-perf-model/skills/tpu-perf-model/scripts -p 'test_*.py'`

Expected: PASS, including the new micro-op IR, builder, scheduler, report, and integration tests.

- [ ] **Step 2: Run the flash-attention example in micro mode**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py --steps plugins/tpu-perf-model/skills/tpu-perf-model/scripts/examples/flash_attention.json --analysis-level micro --format json`

Expected: JSON output containing `micro_ops`, `timeline`, `fragment_residency`, `critical_path`, and `stall_breakdown`.

- [ ] **Step 3: Run the flash-attention example in legacy step mode**

Run: `python plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py --steps plugins/tpu-perf-model/skills/tpu-perf-model/scripts/examples/flash_attention.json --format json`

Expected: JSON output containing only the existing step-level `steps` and `summary` fields.

- [ ] **Step 4: Review the final diff for scope control**

Run: `git log --stat --oneline --max-count=10`

Expected: the recent commit history only mentions the TPU perf model skill, its scripts and tests, plus the paired spec and plan docs.

- [ ] **Step 5: Create the final implementation commit**

```bash
git add docs/superpowers/plans/2026-04-15-tpu-perf-model-micro-op-dataflow.md \
        docs/superpowers/specs/2026-04-15-tpu-perf-model-micro-op-dataflow-design.md \
        plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts
git commit -m "feat: add micro-op dataflow analysis to TPU perf model"
```
