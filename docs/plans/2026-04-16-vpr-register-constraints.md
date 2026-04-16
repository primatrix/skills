# VPR Register Constraint Modeling — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add physical VPR (Vector Processing Register) pool tracking, spill simulation, tiling constraint pruning, and register-centric visualization to the TPU perf model.

**Architecture:** Extend the micro-op IR with physical VPR counts per fragment/op, add VPR pool tracking + spill insertion in the scheduler, prune infeasible tilings by VPR limit, and rewrite Mermaid/text reports around VPR occupancy. MXU accumulator is modeled as MXU-internal (not occupying VPRs).

**Tech Stack:** Python 3, dataclasses, unittest, Mermaid diagrams.

---

All file paths are relative to `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/`.

Run all tests from: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_*.py -q`

### Task 1: Add vpr_count to TensorFragment

**Files:**
- Modify: `micro_op_ir.py:9-18`
- Test: `test_micro_op_ir.py`

**Step 1: Write the failing test**

Add to `test_micro_op_ir.py`:

```python
def test_tensor_fragment_has_vpr_count(self):
    from micro_op_ir import TensorFragment
    frag = TensorFragment(
        fragment_id="q_tile0_reg",
        tensor_name="Q",
        step_name="matmul",
        shape=(128, 128),
        dtype="bf16",
        size_bytes=128 * 128 * 2,
        home_level="REG",
        vpr_count=8,
    )
    self.assertEqual(frag.vpr_count, 8)

def test_tensor_fragment_vpr_count_defaults_to_zero(self):
    from micro_op_ir import TensorFragment
    frag = TensorFragment(
        fragment_id="q_hbm",
        tensor_name="Q",
        step_name="matmul",
        shape=(128, 128),
        dtype="bf16",
        size_bytes=128 * 128 * 2,
        home_level="HBM",
    )
    self.assertEqual(frag.vpr_count, 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_ir.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'vpr_count'`

**Step 3: Write minimal implementation**

In `micro_op_ir.py`, add to `TensorFragment`:

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
    vpr_count: int = 0
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_micro_op_ir.py -q`
Expected: PASS (all tests)

**Step 5: Commit**

```
git add micro_op_ir.py test_micro_op_ir.py
git commit -m "feat(tpu-perf-model): add vpr_count field to TensorFragment"
```

---

### Task 2: Add required_vpr_count to MicroOp

**Files:**
- Modify: `micro_op_ir.py:22-32`
- Test: `test_micro_op_ir.py`

**Step 1: Write the failing test**

Add to `test_micro_op_ir.py`:

```python
def test_micro_op_has_required_vpr_count(self):
    from micro_op_ir import MicroOp
    op = MicroOp(
        op_id="mxu_tile0",
        step_name="matmul",
        op_kind="mxu_compute",
        depends_on=[],
        input_fragments=["q_reg", "k_reg"],
        output_fragments=["acc_reg"],
        required_units=("MXU",),
        required_vmem_slots=(),
        required_reg_groups=("q_reg0", "k_reg0"),
        latency_ns=20.0,
        required_vpr_count=16,
    )
    self.assertEqual(op.required_vpr_count, 16)

def test_micro_op_vpr_count_defaults_to_zero(self):
    from micro_op_ir import MicroOp
    op = MicroOp(
        op_id="load_q",
        step_name="matmul",
        op_kind="dma_load",
        depends_on=[],
        input_fragments=[],
        output_fragments=["q_vmem"],
        required_units=("DMA",),
        required_vmem_slots=("q_slot",),
        required_reg_groups=(),
        latency_ns=10.0,
    )
    self.assertEqual(op.required_vpr_count, 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_ir.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'required_vpr_count'`

**Step 3: Write minimal implementation**

In `micro_op_ir.py`, add to `MicroOp`:

```python
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
    required_vpr_count: int = 0
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS (all tests — default 0 is backward-compatible)

**Step 5: Commit**

```
git add micro_op_ir.py test_micro_op_ir.py
git commit -m "feat(tpu-perf-model): add required_vpr_count field to MicroOp"
```

---

### Task 3: Add _calc_vpr_count helper and update _add_fragment

**Files:**
- Modify: `micro_op_builder.py:1-56`
- Test: `test_micro_op_builder.py`

**Step 1: Write the failing test**

Add to `test_micro_op_builder.py`:

```python
def test_calc_vpr_count_bf16_128x128(self):
    from micro_op_builder import _calc_vpr_count
    from hw_params import TPU_V7X
    # 128*128*2 = 32768 bytes / 4096 = 8 VPRs
    self.assertEqual(_calc_vpr_count(128 * 128 * 2, TPU_V7X), 8)

def test_calc_vpr_count_f32_128x128(self):
    from micro_op_builder import _calc_vpr_count
    from hw_params import TPU_V7X
    # 128*128*4 = 65536 bytes / 4096 = 16 VPRs
    self.assertEqual(_calc_vpr_count(128 * 128 * 4, TPU_V7X), 16)

def test_calc_vpr_count_rounds_up(self):
    from micro_op_builder import _calc_vpr_count
    from hw_params import TPU_V7X
    # 4097 bytes -> ceil(4097/4096) = 2 VPRs
    self.assertEqual(_calc_vpr_count(4097, TPU_V7X), 2)

def test_calc_vpr_count_zero_bytes(self):
    from micro_op_builder import _calc_vpr_count
    from hw_params import TPU_V7X
    self.assertEqual(_calc_vpr_count(0, TPU_V7X), 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_builder.py::TestMicroOpBuilder::test_calc_vpr_count_bf16_128x128 -q`
Expected: FAIL with `ImportError: cannot import name '_calc_vpr_count'`

**Step 3: Write minimal implementation**

In `micro_op_builder.py`, add import and helper:

```python
import math
from hw_params import TPUParams, dtype_bytes

def _calc_vpr_count(size_bytes: int, hw: TPUParams) -> int:
    if size_bytes == 0:
        return 0
    return math.ceil(size_bytes / hw.vpr_size_bytes)
```

Also update `_add_fragment` to accept and pass through `vpr_count`:

```python
def _add_fragment(
    fragments: dict[str, TensorFragment],
    fragment_id: str,
    tensor_name: str,
    step_name: str,
    shape: tuple[int, ...],
    dtype: str,
    size_bytes: int,
    home_level: str,
    producer_op: str | None = None,
    vpr_count: int = 0,
) -> None:
    fragments[fragment_id] = TensorFragment(
        fragment_id=fragment_id,
        tensor_name=tensor_name,
        step_name=step_name,
        shape=shape,
        dtype=dtype,
        size_bytes=size_bytes,
        home_level=home_level,
        producer_op=producer_op,
        vpr_count=vpr_count,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS (all tests)

**Step 5: Commit**

```
git add micro_op_builder.py test_micro_op_builder.py
git commit -m "feat(tpu-perf-model): add _calc_vpr_count helper and vpr_count to _add_fragment"
```

---

### Task 4: Rewrite _build_matmul_graph with MXU-internal accumulator and VPR counts

**Files:**
- Modify: `micro_op_builder.py:96-239` (the `_build_matmul_graph` function)
- Modify: `micro_op_builder.py:58-82` (the `_add_micro_op` helper)
- Test: `test_micro_op_builder.py`

**Step 1: Write the failing tests**

Add to `test_micro_op_builder.py`:

```python
def test_matmul_graph_has_mxu_writeback_op(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul", op_type="matmul",
        inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
        outputs=[TensorRef("S", (128, 128), "bf16")],
        flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
        compute_unit="MXU", fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
        tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
        double_buffer=False, vmem_usage_bytes=128*128*2*3,
    )
    graph = build_micro_op_graph_for_step(step, tile)
    op_kinds = [op.op_kind for op in graph.micro_ops.values()]
    self.assertIn("mxu_writeback", op_kinds)

def test_matmul_mxu_compute_does_not_hold_acc_reg(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul", op_type="matmul",
        inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
        outputs=[TensorRef("S", (128, 128), "bf16")],
        flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
        compute_unit="MXU", fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
        tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
        double_buffer=False, vmem_usage_bytes=128*128*2*3,
    )
    graph = build_micro_op_graph_for_step(step, tile)
    mxu_ops = [op for op in graph.micro_ops.values() if op.op_kind == "mxu_compute"]
    for op in mxu_ops:
        # MXU compute should only hold Q and K reg groups, not acc
        self.assertNotIn("acc_reg0", op.required_reg_groups)
        self.assertNotIn("acc_reg1", op.required_reg_groups)

def test_matmul_fragments_have_vpr_counts(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul", op_type="matmul",
        inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
        outputs=[TensorRef("S", (128, 128), "bf16")],
        flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
        compute_unit="MXU", fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
        tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
        double_buffer=False, vmem_usage_bytes=128*128*2*3,
    )
    graph = build_micro_op_graph_for_step(step, tile)
    reg_frags = [f for f in graph.fragments.values() if f.home_level in ("REG", "acc")]
    for frag in reg_frags:
        self.assertGreater(frag.vpr_count, 0, f"{frag.fragment_id} has vpr_count=0")

def test_matmul_mxu_compute_vpr_count_is_q_plus_k(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="qk_matmul", op_type="matmul",
        inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
        outputs=[TensorRef("S", (128, 128), "bf16")],
        flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
        compute_unit="MXU", fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
        tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
        double_buffer=False, vmem_usage_bytes=128*128*2*3,
    )
    graph = build_micro_op_graph_for_step(step, tile)
    mxu_ops = [op for op in graph.micro_ops.values() if op.op_kind == "mxu_compute"]
    # Q=8 VPRs + K=8 VPRs = 16
    for op in mxu_ops:
        self.assertEqual(op.required_vpr_count, 16)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_builder.py -q`
Expected: FAIL — no `mxu_writeback` op kind, MXU still has `acc_reg` group, `vpr_count=0`

**Step 3: Write the implementation**

Rewrite `_build_matmul_graph` in `micro_op_builder.py`. Key changes:

1. Add `import math` and import `TPUParams` at top
2. Update `_add_micro_op` to accept `required_vpr_count: int = 0`
3. Rewrite `_build_matmul_graph`:
   - Remove `acc_reg` fragment from REG level (accumulator is MXU-internal)
   - Add `result_reg` fragment for MXU writeback output
   - Set `vpr_count` on all REG fragments via `_calc_vpr_count`
   - MXU compute op: `required_reg_groups=(q_reg_group, k_reg_group)` — no acc
   - MXU compute op: `required_vpr_count = q_vpr + k_vpr`
   - New `mxu_writeback` op: depends on `mxu`, outputs `result_reg`, `required_reg_groups=(acc_reg_group,)`, `required_vpr_count=result_vpr`
   - `reg_to_vmem` (spill) now depends on `writeback` instead of `mxu`

Full replacement for `_build_matmul_graph`:

```python
def _build_matmul_graph(step: ComputeStep, tile: TileConfig, step_idx: int | None) -> MicroOpGraph:
    fragments: dict[str, TensorFragment] = {}
    micro_ops: dict[str, MicroOp] = {}
    step_token = _step_token(step, step_idx)

    bm = tile.block_dims.get("M", 1)
    bn = tile.block_dims.get("N", 1)
    bk = tile.block_dims.get("K", 1)
    dtype = step.inputs[0].dtype
    dtype_b = dtype_bytes(dtype)
    bytes_per_input = bm * bk * dtype_b
    bytes_per_weight = bk * bn * dtype_b
    bytes_per_output = bm * bn * dtype_b

    from hw_params import TPU_V7X
    hw = TPU_V7X
    q_vpr = _calc_vpr_count(bytes_per_input, hw)
    k_vpr = _calc_vpr_count(bytes_per_weight, hw)
    result_vpr = _calc_vpr_count(bytes_per_output, hw)

    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)

        q_hbm = _fragment_id(step_token, step.inputs[0].name, suffix, "hbm")
        q_vmem = _fragment_id(step_token, step.inputs[0].name, suffix, "vmem")
        q_reg = _fragment_id(step_token, step.inputs[0].name, suffix, "reg")

        k_hbm = _fragment_id(step_token, step.inputs[1].name, suffix, "hbm")
        k_vmem = _fragment_id(step_token, step.inputs[1].name, suffix, "vmem")
        k_reg = _fragment_id(step_token, step.inputs[1].name, suffix, "reg")

        result_reg = _fragment_id(step_token, step.outputs[0].name, suffix, "reg")
        out_vmem = _fragment_id(step_token, step.outputs[0].name, suffix, "vmem")
        out_hbm = _fragment_id(step_token, step.outputs[0].name, suffix, "hbm")

        _add_fragment(fragments, q_hbm, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "HBM")
        _add_fragment(fragments, q_vmem, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "VMEM")
        _add_fragment(fragments, q_reg, step.inputs[0].name, step.name, (bm, bk), dtype, bytes_per_input, "REG", vpr_count=q_vpr)

        _add_fragment(fragments, k_hbm, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "HBM")
        _add_fragment(fragments, k_vmem, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "VMEM")
        _add_fragment(fragments, k_reg, step.inputs[1].name, step.name, (bk, bn), dtype, bytes_per_weight, "REG", vpr_count=k_vpr)

        _add_fragment(fragments, result_reg, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "REG", vpr_count=result_vpr)
        _add_fragment(fragments, out_vmem, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "VMEM")
        _add_fragment(fragments, out_hbm, step.outputs[0].name, step.name, (bm, bn), dtype, bytes_per_output, "HBM")

        q_slot = _buffer_slot("q", tile_idx, tile.double_buffer)
        k_slot = _buffer_slot("k", tile_idx, tile.double_buffer)
        out_slot = _buffer_slot("out", tile_idx, tile.double_buffer)
        q_reg_group = _reg_group("q", tile_idx, tile.double_buffer)
        k_reg_group = _reg_group("k", tile_idx, tile.double_buffer)
        acc_reg_group = _reg_group("acc", tile_idx, tile.double_buffer)

        load_q = f"{step_token}_load_q_{suffix}"
        load_k = f"{step_token}_load_k_{suffix}"
        move_q = f"{step_token}_vmem_to_reg_q_{suffix}"
        move_k = f"{step_token}_vmem_to_reg_k_{suffix}"
        mxu = f"{step_token}_mxu_{suffix}"
        writeback = f"{step_token}_mxu_writeback_{suffix}"
        spill = f"{step_token}_reg_to_vmem_{suffix}"
        store = f"{step_token}_store_{suffix}"

        _add_micro_op(micro_ops, load_q, step.name, "dma_load_hbm_to_vmem",
                       [], [q_hbm], [q_vmem], ("DMA",), (q_slot,), (), required_vpr_count=0)
        _add_micro_op(micro_ops, load_k, step.name, "dma_load_hbm_to_vmem",
                       [], [k_hbm], [k_vmem], ("DMA",), (k_slot,), (), required_vpr_count=0)
        _add_micro_op(micro_ops, move_q, step.name, "vmem_to_reg",
                       [load_q], [q_vmem], [q_reg], (), (q_slot,), (q_reg_group,), required_vpr_count=q_vpr)
        _add_micro_op(micro_ops, move_k, step.name, "vmem_to_reg",
                       [load_k], [k_vmem], [k_reg], (), (k_slot,), (k_reg_group,), required_vpr_count=k_vpr)
        _add_micro_op(micro_ops, mxu, step.name, "mxu_compute",
                       [move_q, move_k], [q_reg, k_reg], [], ("MXU",), (),
                       (q_reg_group, k_reg_group), required_vpr_count=q_vpr + k_vpr)
        _add_micro_op(micro_ops, writeback, step.name, "mxu_writeback",
                       [mxu], [], [result_reg], (), (), (acc_reg_group,), required_vpr_count=result_vpr)
        _add_micro_op(micro_ops, spill, step.name, "reg_to_vmem",
                       [writeback], [result_reg], [out_vmem], (), (out_slot,), (acc_reg_group,),
                       required_vpr_count=result_vpr)
        _add_micro_op(micro_ops, store, step.name, "dma_store_vmem_to_hbm",
                       [spill], [out_vmem], [out_hbm], ("DMA",), (out_slot,), (), required_vpr_count=0)

    return MicroOpGraph(fragments=fragments, micro_ops=micro_ops)
```

Also update `_add_micro_op` to accept and pass `required_vpr_count`:

```python
def _add_micro_op(
    micro_ops: dict[str, MicroOp],
    op_id: str,
    step_name: str,
    op_kind: str,
    depends_on: list[str],
    input_fragments: list[str],
    output_fragments: list[str],
    required_units: tuple[str, ...],
    required_vmem_slots: tuple[str, ...],
    required_reg_groups: tuple[str, ...],
    latency_ns: float = 1.0,
    required_vpr_count: int = 0,
) -> None:
    micro_ops[op_id] = MicroOp(
        op_id=op_id,
        step_name=step_name,
        op_kind=op_kind,
        depends_on=depends_on,
        input_fragments=input_fragments,
        output_fragments=output_fragments,
        required_units=required_units,
        required_vmem_slots=required_vmem_slots,
        required_reg_groups=required_reg_groups,
        latency_ns=latency_ns,
        required_vpr_count=required_vpr_count,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_micro_op_builder.py -q`
Expected: PASS

**Step 5: Fix broken tests in other files**

The old `acc_reg` fragment ID changed to `result_reg` (no more `_acc` suffix). The old op pipeline was `mxu → reg_to_vmem → store`. Now it's `mxu → mxu_writeback → reg_to_vmem → store`. Tests that check op_kind counts or specific op_ids need updating.

Run: `python -m pytest test_*.py -q`
Fix any failures by updating expected op kinds and counts in `test_micro_op_scheduler.py`, `test_micro_op_report.py`, `test_integration.py`, `test_pipeline_simulator.py`.

Key changes needed:
- Tests that count `"reg_to_vmem"` or `"mxu_compute"` ops: add `"mxu_writeback"` to expected set
- Tests that reference `acc_reg0`: this reg group name stays the same (used for writeback), but the fragment ID changes from `*_acc` to `*_reg`
- Tests that check `mxu` op's `required_reg_groups` should not include `acc_reg0`

**Step 6: Run full test suite**

Run: `python -m pytest test_*.py -q`
Expected: PASS (all tests)

**Step 7: Commit**

```
git add micro_op_builder.py test_micro_op_builder.py test_micro_op_scheduler.py test_micro_op_report.py test_integration.py
git commit -m "feat(tpu-perf-model): rewrite matmul graph with MXU-internal acc and VPR counts"
```

---

### Task 5: Update _build_vpu_graph with VPR counts

**Files:**
- Modify: `micro_op_builder.py:242-287` (the `_build_vpu_graph` function)
- Test: `test_micro_op_builder.py`

**Step 1: Write the failing test**

Add to `test_micro_op_builder.py`:

```python
def test_vpu_fragments_have_vpr_counts(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="scale", op_type="elementwise",
        inputs=[TensorRef("S", (128, 128), "bf16")],
        outputs=[TensorRef("S2", (128, 128), "bf16")],
        flops_formula="M*N", flops_vars={"M": 128, "N": 128},
        compute_unit="VPU", fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"dim0": 16384}, num_tiles=1,
        tile_input_bytes=16384*2, tile_output_bytes=16384*2,
        double_buffer=False, vmem_usage_bytes=16384*2*2,
    )
    graph = build_micro_op_graph_for_step(step, tile)
    reg_frags = [f for f in graph.fragments.values() if f.home_level == "REG"]
    for frag in reg_frags:
        self.assertGreater(frag.vpr_count, 0)

def test_vpu_compute_vpr_count_is_in_plus_out(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_step
    from pipeline_simulator import TileConfig

    step = ComputeStep(
        name="scale", op_type="elementwise",
        inputs=[TensorRef("S", (128, 128), "bf16")],
        outputs=[TensorRef("S2", (128, 128), "bf16")],
        flops_formula="M*N", flops_vars={"M": 128, "N": 128},
        compute_unit="VPU", fusable_with_prev=False,
    )
    tile = TileConfig(
        block_dims={"dim0": 16384}, num_tiles=1,
        tile_input_bytes=16384*2, tile_output_bytes=16384*2,
        double_buffer=False, vmem_usage_bytes=16384*2*2,
    )
    graph = build_micro_op_graph_for_step(step, tile)
    vpu_ops = [op for op in graph.micro_ops.values() if op.op_kind == "vpu_compute"]
    # 16384 elements * 2B = 32768 bytes / 4096 = 8 VPRs per tensor
    for op in vpu_ops:
        self.assertEqual(op.required_vpr_count, 16)  # in(8) + out(8)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_builder.py::TestMicroOpBuilder::test_vpu_fragments_have_vpr_counts -q`
Expected: FAIL — `vpr_count=0`

**Step 3: Write the implementation**

Update `_build_vpu_graph` to compute and set VPR counts on REG fragments and micro-ops. Same pattern as matmul but simpler:

```python
def _build_vpu_graph(step: ComputeStep, tile: TileConfig, step_idx: int | None) -> MicroOpGraph:
    fragments: dict[str, TensorFragment] = {}
    micro_ops: dict[str, MicroOp] = {}
    step_token = _step_token(step, step_idx)

    dtype = step.inputs[0].dtype
    dtype_b = dtype_bytes(dtype)
    total_numel = step.inputs[0].numel
    tile_numel = max(total_numel // max(tile.num_tiles, 1), 1)
    bytes_per_fragment = tile_numel * dtype_b

    from hw_params import TPU_V7X
    hw = TPU_V7X
    in_vpr = _calc_vpr_count(bytes_per_fragment, hw)
    out_vpr = _calc_vpr_count(tile_numel * dtype_bytes(step.outputs[0].dtype), hw)

    for tile_idx in range(tile.num_tiles):
        suffix = _tile_suffix(tile_idx)
        input_hbm = _fragment_id(step_token, step.inputs[0].name, suffix, "hbm")
        input_vmem = _fragment_id(step_token, step.inputs[0].name, suffix, "vmem")
        input_reg = _fragment_id(step_token, step.inputs[0].name, suffix, "reg")
        output_reg = _fragment_id(step_token, step.outputs[0].name, suffix, "reg")
        output_vmem = _fragment_id(step_token, step.outputs[0].name, suffix, "vmem")
        output_hbm = _fragment_id(step_token, step.outputs[0].name, suffix, "hbm")

        _add_fragment(fragments, input_hbm, step.inputs[0].name, step.name, (tile_numel,), dtype, bytes_per_fragment, "HBM")
        _add_fragment(fragments, input_vmem, step.inputs[0].name, step.name, (tile_numel,), dtype, bytes_per_fragment, "VMEM")
        _add_fragment(fragments, input_reg, step.inputs[0].name, step.name, (tile_numel,), dtype, bytes_per_fragment, "REG", vpr_count=in_vpr)
        _add_fragment(fragments, output_reg, step.outputs[0].name, step.name, (tile_numel,), step.outputs[0].dtype, tile_numel * dtype_bytes(step.outputs[0].dtype), "REG", vpr_count=out_vpr)
        _add_fragment(fragments, output_vmem, step.outputs[0].name, step.name, (tile_numel,), step.outputs[0].dtype, tile_numel * dtype_bytes(step.outputs[0].dtype), "VMEM")
        _add_fragment(fragments, output_hbm, step.outputs[0].name, step.name, (tile_numel,), step.outputs[0].dtype, tile_numel * dtype_bytes(step.outputs[0].dtype), "HBM")

        in_slot = _buffer_slot("in", tile_idx, tile.double_buffer)
        out_slot = _buffer_slot("out", tile_idx, tile.double_buffer)
        in_reg_group = _reg_group("vin", tile_idx, tile.double_buffer)
        out_reg_group = _reg_group("vout", tile_idx, tile.double_buffer)

        load = f"{step_token}_load_{suffix}"
        move_in = f"{step_token}_vmem_to_reg_{suffix}"
        vpu = f"{step_token}_vpu_{suffix}"
        spill = f"{step_token}_reg_to_vmem_{suffix}"
        store_op = f"{step_token}_store_{suffix}"

        _add_micro_op(micro_ops, load, step.name, "dma_load_hbm_to_vmem", [], [input_hbm], [input_vmem], ("DMA",), (in_slot,), (), required_vpr_count=0)
        _add_micro_op(micro_ops, move_in, step.name, "vmem_to_reg", [load], [input_vmem], [input_reg], (), (in_slot,), (in_reg_group,), required_vpr_count=in_vpr)
        _add_micro_op(micro_ops, vpu, step.name, "vpu_compute", [move_in], [input_reg], [output_reg], ("VPU",), (), (in_reg_group, out_reg_group), required_vpr_count=in_vpr + out_vpr)
        _add_micro_op(micro_ops, spill, step.name, "reg_to_vmem", [vpu], [output_reg], [output_vmem], (), (out_slot,), (out_reg_group,), required_vpr_count=out_vpr)
        _add_micro_op(micro_ops, store_op, step.name, "dma_store_vmem_to_hbm", [spill], [output_vmem], [output_hbm], ("DMA",), (out_slot,), (), required_vpr_count=0)

    return MicroOpGraph(fragments=fragments, micro_ops=micro_ops)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS (all tests)

**Step 5: Commit**

```
git add micro_op_builder.py test_micro_op_builder.py
git commit -m "feat(tpu-perf-model): add VPR counts to VPU graph builder"
```

---

### Task 6: Update _append_fused_vpu_step with VPR counts

**Files:**
- Modify: `micro_op_builder.py:314-439` (the `_append_fused_vpu_step` function)
- Test: `test_micro_op_builder.py`

**Step 1: Write the failing test**

Add to `test_micro_op_builder.py`:

```python
def test_fused_vpu_has_vpr_counts(self):
    from compute_step import ComputeStep, TensorRef
    from micro_op_builder import build_micro_op_graph_for_pipeline
    from pipeline_simulator import TileConfig

    matmul = ComputeStep(
        name="qk_matmul", op_type="matmul",
        inputs=[TensorRef("Q", (128, 128), "bf16"), TensorRef("K", (128, 128), "bf16")],
        outputs=[TensorRef("S", (128, 128), "bf16")],
        flops_formula="2*M*N*K", flops_vars={"M": 128, "N": 128, "K": 128},
        compute_unit="MXU", fusable_with_prev=False,
    )
    scale = ComputeStep(
        name="scale", op_type="elementwise",
        inputs=[TensorRef("S", (128, 128), "bf16")],
        outputs=[TensorRef("S2", (128, 128), "bf16")],
        flops_formula="M*N", flops_vars={"M": 128, "N": 128},
        compute_unit="VPU", fusable_with_prev=True,
    )
    tile = TileConfig(
        block_dims={"M": 128, "N": 128, "K": 128}, num_tiles=1,
        tile_input_bytes=128*128*2*2, tile_output_bytes=128*128*2,
        double_buffer=False, vmem_usage_bytes=128*128*2*3,
    )
    graph = build_micro_op_graph_for_pipeline([matmul, scale], [tile, tile])
    fused_vpu_ops = [op for op in graph.micro_ops.values()
                     if op.op_kind == "vpu_compute" and op.step_name == "scale"]
    for op in fused_vpu_ops:
        self.assertGreater(op.required_vpr_count, 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_builder.py -q`
Expected: FAIL — `required_vpr_count=0`

**Step 3: Write the implementation**

Update `_append_fused_vpu_step` to compute VPR counts and pass them to `_add_fragment` and `_add_micro_op`. Follow the same pattern as `_build_vpu_graph`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 5: Commit**

```
git add micro_op_builder.py test_micro_op_builder.py
git commit -m "feat(tpu-perf-model): add VPR counts to fused VPU step builder"
```

---

### Task 7: Add VPR pool tracking and spill fields to scheduler

**Files:**
- Modify: `micro_op_scheduler.py:27-36,151-259`
- Test: `test_micro_op_scheduler.py`

**Step 1: Write the failing test**

Add to `test_micro_op_scheduler.py`:

```python
def test_schedule_result_has_spill_fields(self):
    from hw_params import TPU_V7X
    from micro_op_ir import MicroOp, MicroOpGraph
    from micro_op_scheduler import schedule_micro_op_graph

    graph = MicroOpGraph(
        fragments={},
        micro_ops={
            "load_q": MicroOp(
                op_id="load_q", step_name="matmul",
                op_kind="dma_load_hbm_to_vmem", depends_on=[],
                input_fragments=[], output_fragments=["q_vmem"],
                required_units=("DMA",), required_vmem_slots=("q_slot",),
                required_reg_groups=(), latency_ns=10.0,
            ),
        },
    )
    result = schedule_micro_op_graph(graph, TPU_V7X)
    self.assertEqual(result.spill_count, 0)
    self.assertEqual(result.spill_cost_ns, 0.0)

def test_schedule_result_has_peak_vpr_count(self):
    from hw_params import TPU_V7X
    from micro_op_ir import MicroOp, MicroOpGraph
    from micro_op_scheduler import schedule_micro_op_graph

    graph = MicroOpGraph(
        fragments={},
        micro_ops={
            "move_q": MicroOp(
                op_id="move_q", step_name="matmul",
                op_kind="vmem_to_reg", depends_on=[],
                input_fragments=["q_vmem"], output_fragments=["q_reg"],
                required_units=(), required_vmem_slots=("q_slot",),
                required_reg_groups=("q_reg0",), latency_ns=1.0,
                required_vpr_count=8,
            ),
        },
    )
    result = schedule_micro_op_graph(graph, TPU_V7X)
    self.assertEqual(result.peak_vpr_count, 8)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_scheduler.py -q`
Expected: FAIL — `ScheduleResult` has no `spill_count`, `spill_cost_ns`, or `peak_vpr_count` attribute

**Step 3: Write the implementation**

Add new fields to `ScheduleResult`:

```python
@dataclass
class ScheduleResult:
    op_timings: dict[str, OpTiming]
    resource_occupancy: dict[str, list[OccupancyInterval]]
    fragment_residency: dict[str, list[OccupancyInterval]]
    stall_breakdown: dict[str, int]
    critical_path: list[str]
    total_time_ns: float
    peak_vmem_slots: int
    peak_reg_groups: int
    peak_vpr_count: int = 0
    spill_count: int = 0
    spill_cost_ns: float = 0.0
```

In `schedule_micro_op_graph`, after computing `peak_reg`, add VPR peak tracking:

```python
# Track peak VPR count from required_vpr_count on each op
peak_vpr = 0
for op_id, op in graph.micro_ops.items():
    if op.required_vpr_count > peak_vpr:
        peak_vpr = op.required_vpr_count
```

Pass `peak_vpr_count=peak_vpr` to the return value.

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 5: Commit**

```
git add micro_op_scheduler.py test_micro_op_scheduler.py
git commit -m "feat(tpu-perf-model): add spill and VPR peak fields to ScheduleResult"
```

---

### Task 8: Add VPR constraint pruning to tiling optimizer

**Files:**
- Modify: `tiling_optimizer.py:23-31,83-156,159-187`
- Test: `test_tiling_optimizer.py`

**Step 1: Write the failing test**

Add to `test_tiling_optimizer.py`:

```python
def test_matmul_tile_vpr_count(self):
    from tiling_optimizer import _matmul_tile_vpr_count
    from hw_params import TPU_V7X
    # [128,128] bf16: Q=8, K=8, peak(Q+K)=16, result=8
    self.assertEqual(_matmul_tile_vpr_count(128, 128, 128, 2, TPU_V7X), 16)

def test_matmul_tile_vpr_count_at_limit(self):
    from tiling_optimizer import _matmul_tile_vpr_count
    from hw_params import TPU_V7X
    # [256,256] bf16: Q[256,128]=16, K[128,256]=16, peak=32
    self.assertEqual(_matmul_tile_vpr_count(256, 256, 128, 2, TPU_V7X), 32)

def test_tiling_respects_vpr_limit(self):
    from tiling_optimizer import find_optimal_tiling
    from tiling_optimizer import _matmul_tile_vpr_count
    from compute_step import ComputeStep, TensorRef
    from hw_params import TPU_V7X
    step = ComputeStep(
        name="matmul", op_type="matmul",
        inputs=[TensorRef("A", (4096, 4096), "bf16"), TensorRef("B", (4096, 4096), "bf16")],
        outputs=[TensorRef("C", (4096, 4096), "bf16")],
        flops_formula="2*M*N*K", flops_vars={"M": 4096, "N": 4096, "K": 4096},
        compute_unit="MXU", fusable_with_prev=False,
    )
    result = find_optimal_tiling(step, TPU_V7X)
    bm = result.block_dims["M"]
    bn = result.block_dims["N"]
    bk = result.block_dims["K"]
    vpr = _matmul_tile_vpr_count(bm, bn, bk, 2, TPU_V7X)
    self.assertLessEqual(vpr, TPU_V7X.vpr_count)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_tiling_optimizer.py -q`
Expected: FAIL — `_matmul_tile_vpr_count` does not exist

**Step 3: Write the implementation**

Add to `tiling_optimizer.py`:

```python
def _matmul_tile_vpr_count(bm: int, bn: int, bk: int, dtype_b: int, hw: TPUParams) -> int:
    q_vpr = math.ceil(bm * bk * dtype_b / hw.vpr_size_bytes)
    k_vpr = math.ceil(bk * bn * dtype_b / hw.vpr_size_bytes)
    result_vpr = math.ceil(bm * bn * dtype_b / hw.vpr_size_bytes)
    return max(q_vpr + k_vpr, result_vpr)
```

Add VPR check in `_find_matmul_tiling` inner loop, right after the VMEM check:

```python
vpr_count = _matmul_tile_vpr_count(bm, bn, bk, dtype_b, hw)
if vpr_count > hw.vpr_count:
    continue
```

Add VPR check in `_find_elementwise_tiling`:

```python
in_vpr = math.ceil(tile_in / hw.vpr_size_bytes)
out_vpr = math.ceil(tile_out / hw.vpr_size_bytes)
if in_vpr + out_vpr > hw.vpr_count:
    continue  # skip, would require spills
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 5: Commit**

```
git add tiling_optimizer.py test_tiling_optimizer.py
git commit -m "feat(tpu-perf-model): add VPR constraint pruning to tiling optimizer"
```

---

### Task 9: Add VPR pressure to JSON output

**Files:**
- Modify: `micro_op_report.py:151-167`
- Test: `test_micro_op_report.py`

**Step 1: Write the failing test**

Add to `test_micro_op_report.py`:

```python
def test_json_contains_vpr_pressure(self):
    from micro_op_report import micro_schedule_to_json
    import json
    payload = json.loads(micro_schedule_to_json(_sample_schedule_result(), []))
    self.assertIn("vpr_pressure", payload)
    vpr = payload["vpr_pressure"]
    self.assertIn("peak_vpr_count", vpr)
    self.assertIn("vpr_capacity", vpr)
    self.assertIn("utilization_pct", vpr)
    self.assertIn("spill_count", vpr)
    self.assertIn("spill_cost_ns", vpr)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_report.py::TestMicroOpReport::test_json_contains_vpr_pressure -q`
Expected: FAIL — no `vpr_pressure` key

**Step 3: Write the implementation**

In `micro_schedule_to_json`, add `vpr_pressure` to `payload`:

```python
from hw_params import TPU_V7X

payload["vpr_pressure"] = {
    "peak_vpr_count": schedule.peak_vpr_count,
    "vpr_capacity": TPU_V7X.vpr_count,
    "utilization_pct": schedule.peak_vpr_count * 100.0 / TPU_V7X.vpr_count,
    "spill_count": schedule.spill_count,
    "spill_cost_ns": schedule.spill_cost_ns,
}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 5: Commit**

```
git add micro_op_report.py test_micro_op_report.py
git commit -m "feat(tpu-perf-model): add vpr_pressure to JSON output"
```

---

### Task 10: Add VPR Register Map to text report

**Files:**
- Modify: `micro_op_report.py:170-214`
- Test: `test_micro_op_report.py`

**Step 1: Write the failing test**

Add to `test_micro_op_report.py`:

```python
def test_text_report_contains_vpr_register_map(self):
    from micro_op_report import micro_schedule_to_text
    schedule, graph = _sample_mermaid_schedule()
    text = micro_schedule_to_text(schedule, [])
    self.assertIn("VPR Register Map", text)
    self.assertIn("VPR[", text)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_report.py -q`
Expected: FAIL — `VPR Register Map` not in text

**Step 3: Write the implementation**

Add a `_vpr_register_map` helper function and integrate it into `micro_schedule_to_text`. The function builds a timeline of VPR allocation events from fragment residency and op timings, outputting lines like:

```
=== VPR Register Map ===
Time   0.00 ns: VPR[ 0.. 7] <- Q[128,128] bf16 (vmem_to_reg)
Time   1.00 ns: VPR[ 8..15] <- K[128,128] bf16 (vmem_to_reg)
...
Peak: 16/32 VPRs (50%), Spills: 0 (0.00 ns)
```

Implementation approach:
1. Collect all REG-level fragments with their `vpr_count`, find the producer op's end time (= VPR allocation time) and last consumer's start time (= VPR free time)
2. Assign VPR ranges sequentially (offset counter)
3. Sort events by time, format output

This also needs the `graph` parameter in `micro_schedule_to_text` and `micro_schedule_to_json` — add `graph: MicroOpGraph | None = None` parameter, pass it through from `cli.py`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 5: Commit**

```
git add micro_op_report.py test_micro_op_report.py cli.py
git commit -m "feat(tpu-perf-model): add VPR Register Map section to text report"
```

---

### Task 11: Rewrite Mermaid Gantt to VPR-centric view

**Files:**
- Modify: `micro_op_report.py:228-319` (the `micro_schedule_to_mermaid` function)
- Test: `test_micro_op_report.py`

**Step 1: Write the failing test**

Add to `test_micro_op_report.py`:

```python
def test_gantt_has_vpr_sections(self):
    from micro_op_report import micro_schedule_to_mermaid
    schedule, graph = _sample_mermaid_schedule()
    output = micro_schedule_to_mermaid(schedule, graph)
    self.assertIn("VPR", output)

def test_gantt_shows_fragment_content_in_vpr(self):
    from micro_op_report import micro_schedule_to_mermaid
    schedule, graph = _sample_mermaid_schedule()
    output = micro_schedule_to_mermaid(schedule, graph, max_tiles=1)
    # Should show tensor name in VPR bar
    self.assertTrue("Q" in output or "K" in output)
```

**Step 2: Run test to verify it fails**

These may pass partially since the current Gantt shows REG Groups section. Update to check specifically for VPR range format.

**Step 3: Write the implementation**

Rewrite `micro_schedule_to_mermaid` to use VPR ranges as section rows instead of named REG groups. Each VPR range section (e.g., `VPR 0-7`) shows which fragment occupies those VPRs over time. Spill events appear as red `crit` bars.

Keep the VMEM Slots section as-is.

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS (update any broken existing tests)

**Step 5: Commit**

```
git add micro_op_report.py test_micro_op_report.py
git commit -m "feat(tpu-perf-model): rewrite Mermaid Gantt to VPR-centric view"
```

---

### Task 12: Update Mermaid flowchart with VPR labels

**Files:**
- Modify: `micro_op_report.py:344-430` (the `micro_schedule_to_mermaid_flowchart` function)
- Test: `test_micro_op_report.py`

**Step 1: Write the failing test**

Add to `test_micro_op_report.py`:

```python
def test_flowchart_shows_vpr_numbers(self):
    from micro_op_report import micro_schedule_to_mermaid_flowchart
    schedule, graph = _sample_mermaid_schedule()
    output = micro_schedule_to_mermaid_flowchart(schedule, graph, max_tiles=1)
    # REG nodes should show VPR[x..y] format
    self.assertIn("VPR[", output)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_micro_op_report.py -q`
Expected: FAIL — current labels use `REG reg_name:` format

**Step 3: Write the implementation**

In the fragment node label generation section of `micro_schedule_to_mermaid_flowchart`, change the REG level label from:

```python
label = f"REG {reg}: {frag.tensor_name}{shape_str}"
```

To:

```python
vpr_count = frag.vpr_count if frag.vpr_count > 0 else "?"
label = f"REG VPR[{vpr_count}]: {frag.tensor_name}{shape_str}"
```

For proper VPR range display, use the same VPR allocation tracking from Task 10 to compute `vpr_start` and `vpr_end` for each fragment.

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 5: Commit**

```
git add micro_op_report.py test_micro_op_report.py
git commit -m "feat(tpu-perf-model): add VPR numbers to flowchart fragment labels"
```

---

### Task 13: Update SKILL.md documentation

**Files:**
- Modify: `SKILL.md`

**Step 1: Update the SKILL.md**

Add VPR pressure analysis guidance to the relevant sections:

1. In the Hardware Quick Reference table — already has VPR info, no change needed
2. In the Fusion Rules table — add note about VPR pressure per fusion pattern
3. In "Interpret Results" — add VPR pressure interpretation guidance:

```markdown
### VPR Pressure Analysis

| Observation | Action |
|-------------|--------|
| Peak VPR > 24/32 (75%) | Approaching register limit, consider smaller tiles |
| Peak VPR = 32/32 | At register limit, no room for fusion |
| Spill count > 0 | Register spills detected — reduce tile size or unfuse ops |
| VPR per tile > 16 | Large tiles — verify MXU utilization justifies the VPR cost |
```

4. In Required Output Sections — mention VPR Register Map

**Step 2: Run all tests to verify nothing broke**

Run: `python -m pytest test_*.py -q`
Expected: PASS

**Step 3: Commit**

```
git add SKILL.md
git commit -m "docs(tpu-perf-model): add VPR pressure analysis guidance to SKILL.md"
```

---

### Task 14: Final integration test

**Files:**
- Modify: `test_integration.py`

**Step 1: Write integration test**

Add to `test_integration.py`:

```python
def test_cli_micro_mode_json_has_vpr_pressure(self):
    scripts_dir = os.path.dirname(__file__)
    example_path = os.path.join(scripts_dir, "examples", "flash_attention.json")
    result = subprocess.run(
        [
            "python", os.path.join(scripts_dir, "cli.py"),
            "--steps", example_path,
            "--analysis-level", "micro",
            "--format", "json",
        ],
        capture_output=True, text=True, cwd=scripts_dir,
    )
    self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
    data = json.loads(result.stdout)
    self.assertIn("vpr_pressure", data)
    self.assertLessEqual(data["vpr_pressure"]["peak_vpr_count"], 32)
```

**Step 2: Run test to verify it passes**

Run: `python -m pytest test_integration.py -q`
Expected: PASS

**Step 3: Run full test suite**

Run: `python -m pytest test_*.py -q`
Expected: PASS (all tests)

**Step 4: Commit**

```
git add test_integration.py
git commit -m "test(tpu-perf-model): add VPR pressure integration test"
```
