# TPU Performance Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a TPU v7x performance modeling tool (Register ↔ VMEM ↔ HBM data flow simulator) as a Claude Code plugin with Python scripts.

**Architecture:** AI decomposes math formulas into `ComputeStep` JSON; Python scripts simulate instruction-level pipeline scheduling across three storage tiers, find optimal tiling, and compare theoretical vs measured performance.

**Tech Stack:** Python 3.10+ stdlib only (dataclasses, json, math, argparse). No third-party dependencies.

**Design doc:** `docs/plans/2026-04-15-tpu-perf-model-design.md`

---

### Task 1: Plugin Scaffolding

**Files:**
- Create: `plugins/tpu-perf-model/.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`

**Step 1: Create plugin.json**

```json
{
  "name": "tpu-perf-model",
  "description": "Theoretical TPU v7x performance modeling via Register/VMEM/HBM data flow simulation",
  "version": "1.0.0"
}
```

**Step 2: Register in marketplace.json**

Add to the `plugins` array:
```json
{
  "name": "tpu-perf-model",
  "source": "./plugins/tpu-perf-model",
  "description": "Theoretical TPU v7x performance modeling via Register/VMEM/HBM data flow simulation",
  "version": "1.0.0",
  "license": "Apache-2.0",
  "keywords": ["tpu", "performance", "modeling", "roofline", "pallas"],
  "category": "performance"
}
```

**Step 3: Create empty SKILL.md placeholder**

Create `plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md`:
```markdown
---
name: tpu-perf-model
description: Use when analyzing theoretical TPU v7x performance for a mathematical formula or comparing kernel performance against theoretical bounds. Trigger when the user asks about TPU performance modeling, roofline analysis, data flow optimization, or tiling strategy.
---

# TPU Performance Model

(Placeholder — will be filled in Task 9)
```

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/ .claude-plugin/marketplace.json
git commit -m "feat: scaffold tpu-perf-model plugin"
```

---

### Task 2: Hardware Parameters Module

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/hw_params.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py`

**Step 1: Write the test**

```python
#!/usr/bin/env python3
"""Tests for hw_params module."""
import unittest


class TestHWParams(unittest.TestCase):
    def test_v7x_hbm_capacity_bytes(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.hbm_capacity_bytes, 192 * 1024**3)

    def test_v7x_vmem_capacity_bytes(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.vmem_capacity_bytes, 64 * 1024**2)

    def test_v7x_vpr_count(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.vpr_count, 32)

    def test_v7x_vpr_size_bytes(self):
        from hw_params import TPU_V7X
        # 8 * 128 * 4 bytes (32bit) = 4096 bytes
        self.assertEqual(TPU_V7X.vpr_size_bytes, 8 * 128 * 4)

    def test_v7x_hbm_bandwidth(self):
        from hw_params import TPU_V7X
        self.assertAlmostEqual(TPU_V7X.hbm_bw_bytes_per_sec, 3690e9)

    def test_v7x_mxu_peak_flops(self):
        from hw_params import TPU_V7X
        self.assertAlmostEqual(TPU_V7X.mxu_peak_flops, 2307e12)

    def test_dtype_bytes(self):
        from hw_params import dtype_bytes
        self.assertEqual(dtype_bytes("bf16"), 2)
        self.assertEqual(dtype_bytes("f32"), 4)
        self.assertEqual(dtype_bytes("int8"), 1)

    def test_alignment(self):
        from hw_params import TPU_V7X
        self.assertEqual(TPU_V7X.alignment, 128)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_hw_params.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hw_params'`

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""TPU v7x hardware parameters.

All values are for a single TPU v7x chip.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TPUParams:
    """Hardware parameters for a TPU generation."""
    name: str

    # HBM
    hbm_capacity_bytes: int
    hbm_bw_bytes_per_sec: float

    # VMEM
    vmem_capacity_bytes: int

    # Registers
    spr_count: int        # scalar registers, 32bit each
    vpr_count: int        # vector registers
    vpr_size_bytes: int   # bytes per VPR: 8 * 128 * 4 = 4096

    # Compute units
    mxu_peak_flops: float   # BF16 peak FLOPS
    mxu_count: int           # number of MXUs

    # Alignment
    alignment: int  # block dimension alignment requirement

    @property
    def vpr_total_bytes(self) -> int:
        return self.vpr_count * self.vpr_size_bytes

    @property
    def vpr_lane_count(self) -> int:
        """Number of elements per VPR in BF16."""
        return self.vpr_size_bytes // 2  # 2048 bf16 elements

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at which compute = memory bound (FLOPs/byte)."""
        return self.mxu_peak_flops / self.hbm_bw_bytes_per_sec


TPU_V7X = TPUParams(
    name="v7x",
    hbm_capacity_bytes=192 * 1024**3,        # 192 GB
    hbm_bw_bytes_per_sec=3690e9,             # 3690 GB/s
    vmem_capacity_bytes=64 * 1024**2,        # 64 MiB
    spr_count=4096,
    vpr_count=32,
    vpr_size_bytes=8 * 128 * 4,              # 4096 bytes per VPR
    mxu_peak_flops=2307e12,                  # 2307 TFLOPS BF16
    mxu_count=2,
    alignment=128,
)


DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "int8": 1,
    "int32": 4,
}


def dtype_bytes(dtype: str) -> int:
    """Return byte size for a dtype string."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype: {dtype}. Supported: {list(DTYPE_BYTES.keys())}")
    return DTYPE_BYTES[dtype]
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_hw_params.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/hw_params.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_hw_params.py
git commit -m "feat(tpu-perf-model): add TPU v7x hardware parameters module"
```

---

### Task 3: ComputeStep Data Structures

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/compute_step.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_compute_step.py`

**Step 1: Write the test**

```python
#!/usr/bin/env python3
"""Tests for compute_step module."""
import json
import unittest


class TestTensorRef(unittest.TestCase):
    def test_size_bytes_bf16(self):
        from compute_step import TensorRef
        t = TensorRef(name="A", shape=(4096, 128), dtype="bf16")
        self.assertEqual(t.size_bytes, 4096 * 128 * 2)

    def test_size_bytes_f32(self):
        from compute_step import TensorRef
        t = TensorRef(name="B", shape=(128, 4096), dtype="f32")
        self.assertEqual(t.size_bytes, 128 * 4096 * 4)

    def test_numel(self):
        from compute_step import TensorRef
        t = TensorRef(name="C", shape=(32, 64, 128), dtype="bf16")
        self.assertEqual(t.numel, 32 * 64 * 128)


class TestComputeStep(unittest.TestCase):
    def test_eval_flops_matmul(self):
        from compute_step import ComputeStep, TensorRef
        step = ComputeStep(
            name="qk_matmul",
            op_type="matmul",
            inputs=[
                TensorRef("Q", (4096, 128), "bf16"),
                TensorRef("K", (128, 4096), "bf16"),
            ],
            outputs=[TensorRef("S", (4096, 4096), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 4096, "N": 4096, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        self.assertEqual(step.eval_flops(), 2 * 4096 * 4096 * 128)

    def test_total_input_bytes(self):
        from compute_step import ComputeStep, TensorRef
        step = ComputeStep(
            name="scale",
            op_type="elementwise",
            inputs=[TensorRef("S", (4096, 4096), "bf16")],
            outputs=[TensorRef("S_scaled", (4096, 4096), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 4096, "N": 4096},
            compute_unit="VPU",
            fusable_with_prev=True,
        )
        self.assertEqual(step.total_input_bytes, 4096 * 4096 * 2)
        self.assertEqual(step.total_output_bytes, 4096 * 4096 * 2)

    def test_from_json(self):
        from compute_step import ComputeStep, TensorRef
        data = {
            "name": "add",
            "op_type": "elementwise",
            "inputs": [{"name": "A", "shape": [1024], "dtype": "bf16"}],
            "outputs": [{"name": "B", "shape": [1024], "dtype": "bf16"}],
            "flops_formula": "N",
            "flops_vars": {"N": 1024},
            "compute_unit": "VPU",
            "fusable_with_prev": False,
        }
        step = ComputeStep.from_dict(data)
        self.assertEqual(step.name, "add")
        self.assertEqual(step.eval_flops(), 1024)

    def test_load_steps_from_json_string(self):
        from compute_step import load_steps
        json_str = json.dumps([
            {
                "name": "op1",
                "op_type": "elementwise",
                "inputs": [{"name": "X", "shape": [256], "dtype": "bf16"}],
                "outputs": [{"name": "Y", "shape": [256], "dtype": "bf16"}],
                "flops_formula": "N",
                "flops_vars": {"N": 256},
                "compute_unit": "VPU",
                "fusable_with_prev": False,
            }
        ])
        steps = load_steps(json_str)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].name, "op1")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_compute_step.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""ComputeStep and TensorRef data structures for TPU performance modeling."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from hw_params import dtype_bytes


@dataclass
class TensorRef:
    """Reference to a tensor with name, shape, and dtype."""
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        return self.numel * dtype_bytes(self.dtype)

    @staticmethod
    def from_dict(d: dict) -> TensorRef:
        return TensorRef(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass
class ComputeStep:
    """A single compute operation in the performance model."""
    name: str
    op_type: str            # "matmul" | "reduce" | "elementwise" | "softmax"
    inputs: list[TensorRef]
    outputs: list[TensorRef]
    flops_formula: str      # e.g. "2*M*N*K"
    flops_vars: dict[str, int]  # variable bindings for the formula
    compute_unit: str       # "MXU" | "VPU"
    fusable_with_prev: bool

    def eval_flops(self) -> int:
        """Evaluate the FLOPs formula with bound variables."""
        # Safe eval: only allow math operations and bound variables
        allowed = {k: v for k, v in self.flops_vars.items()}
        allowed.update({"math": math, "ceil": math.ceil, "log2": math.log2})
        return int(eval(self.flops_formula, {"__builtins__": {}}, allowed))

    @property
    def total_input_bytes(self) -> int:
        return sum(t.size_bytes for t in self.inputs)

    @property
    def total_output_bytes(self) -> int:
        return sum(t.size_bytes for t in self.outputs)

    @property
    def total_io_bytes(self) -> int:
        return self.total_input_bytes + self.total_output_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte of HBM I/O."""
        return self.eval_flops() / self.total_io_bytes

    @staticmethod
    def from_dict(d: dict) -> ComputeStep:
        return ComputeStep(
            name=d["name"],
            op_type=d["op_type"],
            inputs=[TensorRef.from_dict(t) for t in d["inputs"]],
            outputs=[TensorRef.from_dict(t) for t in d["outputs"]],
            flops_formula=d["flops_formula"],
            flops_vars=d["flops_vars"],
            compute_unit=d["compute_unit"],
            fusable_with_prev=d.get("fusable_with_prev", False),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "flops_formula": self.flops_formula,
            "flops_vars": self.flops_vars,
            "compute_unit": self.compute_unit,
            "fusable_with_prev": self.fusable_with_prev,
        }


def load_steps(json_str: str) -> list[ComputeStep]:
    """Load a list of ComputeSteps from a JSON string."""
    data = json.loads(json_str)
    return [ComputeStep.from_dict(d) for d in data]


def load_steps_from_file(path: str) -> list[ComputeStep]:
    """Load ComputeSteps from a JSON file."""
    with open(path) as f:
        return load_steps(f.read())
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_compute_step.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/compute_step.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_compute_step.py
git commit -m "feat(tpu-perf-model): add ComputeStep and TensorRef data structures"
```

---

### Task 4: Pipeline Simulator

This is the core module. It takes a list of `ComputeStep` and produces a `PipelineSchedule` with per-step timing and bottleneck analysis.

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/pipeline_simulator.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_pipeline_simulator.py`

**Step 1: Write the test**

```python
#!/usr/bin/env python3
"""Tests for pipeline_simulator module."""
import unittest


class TestMicroOpTiming(unittest.TestCase):
    """Test individual micro-operation duration calculations."""

    def test_dma_load_time(self):
        from pipeline_simulator import calc_dma_time_ns
        from hw_params import TPU_V7X
        # 1 MB load at 3690 GB/s
        bytes_ = 1 * 1024**2
        t_ns = calc_dma_time_ns(bytes_, TPU_V7X)
        expected_ns = bytes_ / TPU_V7X.hbm_bw_bytes_per_sec * 1e9
        self.assertAlmostEqual(t_ns, expected_ns, places=1)

    def test_mxu_compute_time(self):
        from pipeline_simulator import calc_mxu_time_ns
        from hw_params import TPU_V7X
        # 2*M*N*K flops for matmul [128,128]x[128,128]
        flops = 2 * 128 * 128 * 128
        t_ns = calc_mxu_time_ns(flops, TPU_V7X)
        expected_ns = flops / TPU_V7X.mxu_peak_flops * 1e9
        self.assertAlmostEqual(t_ns, expected_ns, places=1)


class TestSingleStepSimulation(unittest.TestCase):
    """Test single-step pipeline analysis."""

    def test_matmul_step_result(self):
        from pipeline_simulator import simulate_step
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        step = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[
                TensorRef("A", (1024, 512), "bf16"),
                TensorRef("B", (512, 1024), "bf16"),
            ],
            outputs=[TensorRef("C", (1024, 1024), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 1024, "N": 1024, "K": 512},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        result = simulate_step(step, TPU_V7X)
        self.assertGreater(result.t_compute_ns, 0)
        self.assertGreater(result.t_hbm_ns, 0)
        self.assertIn(result.bottleneck, ("HBM_BW", "COMPUTE"))
        self.assertAlmostEqual(
            result.t_step_ns,
            max(result.t_hbm_ns, result.t_compute_ns),
        )

    def test_elementwise_is_memory_bound(self):
        from pipeline_simulator import simulate_step
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        step = ComputeStep(
            name="add",
            op_type="elementwise",
            inputs=[TensorRef("A", (4096, 4096), "bf16")],
            outputs=[TensorRef("B", (4096, 4096), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 4096, "N": 4096},
            compute_unit="VPU",
            fusable_with_prev=False,
        )
        result = simulate_step(step, TPU_V7X)
        # Elementwise add: 1 FLOP/element, 4 bytes/element → AI < ridge point
        self.assertEqual(result.bottleneck, "HBM_BW")


class TestPipelineSchedule(unittest.TestCase):
    """Test multi-step pipeline with double buffering."""

    def test_double_buffer_overlaps(self):
        from pipeline_simulator import simulate_step
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        step = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[
                TensorRef("A", (4096, 128), "bf16"),
                TensorRef("B", (128, 4096), "bf16"),
            ],
            outputs=[TensorRef("C", (4096, 4096), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 4096, "N": 4096, "K": 128},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        result = simulate_step(step, TPU_V7X)
        # With double buffering, total time < (t_hbm + t_compute) * num_tiles
        naive_time = result.t_hbm_ns + result.t_compute_ns
        self.assertLess(result.t_step_ns, naive_time * 0.99)

    def test_fusion_saves_hbm(self):
        from pipeline_simulator import simulate_steps
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        matmul = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[
                TensorRef("A", (1024, 512), "bf16"),
                TensorRef("B", (512, 1024), "bf16"),
            ],
            outputs=[TensorRef("C", (1024, 1024), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 1024, "N": 1024, "K": 512},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        scale = ComputeStep(
            name="scale",
            op_type="elementwise",
            inputs=[TensorRef("C", (1024, 1024), "bf16")],
            outputs=[TensorRef("C_scaled", (1024, 1024), "bf16")],
            flops_formula="M*N",
            flops_vars={"M": 1024, "N": 1024},
            compute_unit="VPU",
            fusable_with_prev=True,
        )
        report = simulate_steps([matmul, scale], TPU_V7X)
        # Fusion should reduce total HBM bytes
        self.assertGreater(report.fusion_savings_bytes, 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_pipeline_simulator.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Instruction-level pipeline simulator for TPU v7x.

Models data flow across HBM ↔ VMEM ↔ Register with double-buffering
and fusion analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from hw_params import TPUParams, TPU_V7X
from compute_step import ComputeStep


def calc_dma_time_ns(bytes_: int, hw: TPUParams) -> float:
    """Time in nanoseconds for HBM ↔ VMEM DMA transfer."""
    return bytes_ / hw.hbm_bw_bytes_per_sec * 1e9


def calc_mxu_time_ns(flops: int, hw: TPUParams) -> float:
    """Time in nanoseconds for MXU computation."""
    return flops / hw.mxu_peak_flops * 1e9


def calc_vpu_time_ns(flops: int, hw: TPUParams) -> float:
    """Time in nanoseconds for VPU computation.

    VPU peak is approximated as MXU_peak / 100 for elementwise ops.
    This is a rough estimate — VPU throughput depends heavily on the operation.
    """
    vpu_peak = hw.mxu_peak_flops / 100  # rough approximation
    return flops / vpu_peak * 1e9


@dataclass
class TileConfig:
    """Tile dimensions for a compute step."""
    block_dims: dict[str, int]  # e.g. {"M": 128, "N": 256, "K": 128}
    num_tiles: int
    tile_input_bytes: int
    tile_output_bytes: int
    double_buffer: bool
    vmem_usage_bytes: int  # total VMEM needed (including buffers)


@dataclass
class StepResult:
    """Performance analysis result for a single ComputeStep."""
    name: str
    op_type: str
    compute_unit: str
    flops: int
    hbm_bytes: int              # total HBM I/O (after fusion savings)
    t_hbm_ns: float             # HBM transfer time
    t_compute_ns: float         # compute time
    t_step_ns: float            # effective step time (with pipeline overlap)
    bottleneck: str             # "HBM_BW" | "COMPUTE"
    arithmetic_intensity: float # FLOPs / byte
    tile_config: TileConfig | None
    fused_with_prev: bool
    fusion_hbm_savings_bytes: int


@dataclass
class PipelineReport:
    """Full pipeline analysis for a list of ComputeSteps."""
    steps: list[StepResult]
    total_time_ns: float
    total_flops: int
    total_hbm_bytes: int
    fusion_savings_bytes: int
    overall_arithmetic_intensity: float
    overall_bottleneck: str     # "HBM_BW" | "COMPUTE"
    efficiency_vs_peak: float   # achieved / peak


def _find_tile_config(step: ComputeStep, hw: TPUParams) -> TileConfig:
    """Find a reasonable tile configuration for a step.

    For matmul [M,K]x[K,N]: tiles along M, N, K dimensions.
    For elementwise/reduce: tile along the largest dimension.
    """
    align = hw.alignment
    vmem_cap = hw.vmem_capacity_bytes

    if step.op_type == "matmul":
        vars_ = step.flops_vars
        M, N, K = vars_.get("M", 1), vars_.get("N", 1), vars_.get("K", 1)
        from hw_params import dtype_bytes
        elem_bytes = dtype_bytes(step.inputs[0].dtype)

        # Search for largest tile that fits in VMEM with double buffering
        # Try block sizes from large to small
        best = None
        for bm in range(min(M, 2048), align - 1, -align):
            for bn in range(min(N, 2048), align - 1, -align):
                bk = min(K, 512)
                bk = max(align, (bk // align) * align)

                tile_in = (bm * bk + bk * bn) * elem_bytes
                tile_out = bm * bn * elem_bytes
                # Double buffer: 2x input + 1x output accumulator
                vmem_needed = 2 * tile_in + tile_out
                if vmem_needed <= vmem_cap:
                    num_tiles = (
                        math.ceil(M / bm) *
                        math.ceil(N / bn) *
                        math.ceil(K / bk)
                    )
                    best = TileConfig(
                        block_dims={"M": bm, "N": bn, "K": bk},
                        num_tiles=num_tiles,
                        tile_input_bytes=tile_in,
                        tile_output_bytes=tile_out,
                        double_buffer=True,
                        vmem_usage_bytes=vmem_needed,
                    )
                    break  # found largest bn for this bm
            if best is not None:
                break
        if best is None:
            # Fallback: minimum tile
            bm = bn = bk = align
            tile_in = (bm * bk + bk * bn) * elem_bytes
            tile_out = bm * bn * elem_bytes
            best = TileConfig(
                block_dims={"M": bm, "N": bn, "K": bk},
                num_tiles=math.ceil(M/bm) * math.ceil(N/bn) * math.ceil(K/bk),
                tile_input_bytes=tile_in,
                tile_output_bytes=tile_out,
                double_buffer=tile_in * 2 + tile_out <= vmem_cap,
                vmem_usage_bytes=(2 if tile_in*2+tile_out <= vmem_cap else 1)*tile_in + tile_out,
            )
        return best
    else:
        # Elementwise / reduce: single-dimension tiling
        total_bytes = step.total_io_bytes
        from hw_params import dtype_bytes
        elem_bytes = dtype_bytes(step.inputs[0].dtype)
        numel = step.inputs[0].numel

        # Tile size that fits in VMEM with double buffering
        max_elements = vmem_cap // (2 * 2 * elem_bytes)  # 2x buffer, in+out
        tile_elements = min(numel, max_elements)
        tile_elements = (tile_elements // align) * align
        tile_elements = max(align, tile_elements)
        num_tiles = math.ceil(numel / tile_elements)

        tile_in = tile_elements * elem_bytes
        tile_out = tile_elements * elem_bytes
        return TileConfig(
            block_dims={"N": tile_elements},
            num_tiles=num_tiles,
            tile_input_bytes=tile_in,
            tile_output_bytes=tile_out,
            double_buffer=True,
            vmem_usage_bytes=2 * tile_in + tile_out,
        )


def simulate_step(
    step: ComputeStep,
    hw: TPUParams,
    fused_with_prev: bool = False,
    prev_output_bytes: int = 0,
) -> StepResult:
    """Simulate a single ComputeStep and return timing analysis."""
    flops = step.eval_flops()

    # HBM bytes: inputs + outputs, minus fusion savings
    hbm_bytes = step.total_io_bytes
    fusion_savings = 0
    if fused_with_prev and prev_output_bytes > 0:
        # Skip reading fused input from HBM + skip writing it in prev step
        fusion_savings = prev_output_bytes * 2  # save one read + one write
        hbm_bytes -= prev_output_bytes  # don't read fused input from HBM

    # Compute time
    if step.compute_unit == "MXU":
        t_compute = calc_mxu_time_ns(flops, hw)
    else:
        t_compute = calc_vpu_time_ns(flops, hw)

    # Find tile config
    tile_config = _find_tile_config(step, hw)

    # Per-tile timings with double buffering
    t_dma_per_tile = calc_dma_time_ns(
        tile_config.tile_input_bytes + tile_config.tile_output_bytes, hw
    )
    t_compute_per_tile = t_compute / tile_config.num_tiles

    if tile_config.double_buffer and tile_config.num_tiles > 1:
        # Double buffering: DMA overlaps with compute in steady state
        t_steady = max(t_dma_per_tile, t_compute_per_tile)
        # Startup: first tile DMA + first compute; drain: last store
        t_step = (
            t_dma_per_tile +  # startup: load first tile
            (tile_config.num_tiles - 1) * t_steady +  # steady state
            t_compute_per_tile  # drain: last compute
        )
    else:
        # Single buffer: no overlap
        t_step = tile_config.num_tiles * (t_dma_per_tile + t_compute_per_tile)

    t_hbm_total = calc_dma_time_ns(hbm_bytes, hw)
    ai = flops / hbm_bytes if hbm_bytes > 0 else float("inf")
    bottleneck = "COMPUTE" if ai > hw.ridge_point else "HBM_BW"

    return StepResult(
        name=step.name,
        op_type=step.op_type,
        compute_unit=step.compute_unit,
        flops=flops,
        hbm_bytes=hbm_bytes,
        t_hbm_ns=t_hbm_total,
        t_compute_ns=t_compute,
        t_step_ns=t_step,
        bottleneck=bottleneck,
        arithmetic_intensity=ai,
        tile_config=tile_config,
        fused_with_prev=fused_with_prev,
        fusion_hbm_savings_bytes=fusion_savings,
    )


def simulate_steps(steps: list[ComputeStep], hw: TPUParams) -> PipelineReport:
    """Simulate a sequence of ComputeSteps with fusion analysis."""
    results = []
    total_fusion_savings = 0

    for i, step in enumerate(steps):
        fused = step.fusable_with_prev and i > 0
        prev_out_bytes = steps[i - 1].total_output_bytes if fused else 0

        result = simulate_step(step, hw, fused_with_prev=fused, prev_output_bytes=prev_out_bytes)
        results.append(result)
        total_fusion_savings += result.fusion_hbm_savings_bytes

    total_time = sum(r.t_step_ns for r in results)
    total_flops = sum(r.flops for r in results)
    total_hbm = sum(r.hbm_bytes for r in results)
    overall_ai = total_flops / total_hbm if total_hbm > 0 else float("inf")
    overall_bottleneck = "COMPUTE" if overall_ai > hw.ridge_point else "HBM_BW"
    peak_time = total_flops / hw.mxu_peak_flops * 1e9
    efficiency = peak_time / total_time if total_time > 0 else 0

    return PipelineReport(
        steps=results,
        total_time_ns=total_time,
        total_flops=total_flops,
        total_hbm_bytes=total_hbm,
        fusion_savings_bytes=total_fusion_savings,
        overall_arithmetic_intensity=overall_ai,
        overall_bottleneck=overall_bottleneck,
        efficiency_vs_peak=efficiency,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_pipeline_simulator.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/pipeline_simulator.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_pipeline_simulator.py
git commit -m "feat(tpu-perf-model): add instruction-level pipeline simulator"
```

---

### Task 5: Tiling Optimizer

Dedicated tiling search that finds the optimal block shape to balance DMA and compute time.

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/tiling_optimizer.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_tiling_optimizer.py`

**Step 1: Write the test**

```python
#!/usr/bin/env python3
"""Tests for tiling_optimizer module."""
import unittest


class TestTilingOptimizer(unittest.TestCase):
    def test_matmul_tiling_fits_vmem(self):
        from tiling_optimizer import find_optimal_tiling
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        step = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[
                TensorRef("A", (4096, 4096), "bf16"),
                TensorRef("B", (4096, 4096), "bf16"),
            ],
            outputs=[TensorRef("C", (4096, 4096), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 4096, "N": 4096, "K": 4096},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        result = find_optimal_tiling(step, TPU_V7X)
        # Must fit in VMEM
        self.assertLessEqual(result.vmem_usage_bytes, TPU_V7X.vmem_capacity_bytes)
        # Block dims must be aligned
        for dim_val in result.block_dims.values():
            self.assertEqual(dim_val % TPU_V7X.alignment, 0)

    def test_tiling_prefers_double_buffer(self):
        from tiling_optimizer import find_optimal_tiling
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        step = ComputeStep(
            name="small_matmul",
            op_type="matmul",
            inputs=[
                TensorRef("A", (512, 256), "bf16"),
                TensorRef("B", (256, 512), "bf16"),
            ],
            outputs=[TensorRef("C", (512, 512), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 512, "N": 512, "K": 256},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        result = find_optimal_tiling(step, TPU_V7X)
        self.assertTrue(result.double_buffer)

    def test_tiling_report_includes_pipeline_balance(self):
        from tiling_optimizer import find_optimal_tiling_with_analysis
        from compute_step import ComputeStep, TensorRef
        from hw_params import TPU_V7X

        step = ComputeStep(
            name="matmul",
            op_type="matmul",
            inputs=[
                TensorRef("A", (2048, 1024), "bf16"),
                TensorRef("B", (1024, 2048), "bf16"),
            ],
            outputs=[TensorRef("C", (2048, 2048), "bf16")],
            flops_formula="2*M*N*K",
            flops_vars={"M": 2048, "N": 2048, "K": 1024},
            compute_unit="MXU",
            fusable_with_prev=False,
        )
        analysis = find_optimal_tiling_with_analysis(step, TPU_V7X)
        self.assertIn("dma_time_per_tile_ns", analysis)
        self.assertIn("compute_time_per_tile_ns", analysis)
        self.assertIn("pipeline_balance_ratio", analysis)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_tiling_optimizer.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Tiling optimizer: search for optimal block shape under VMEM/VPR constraints.

Objective: balance DMA time ≈ compute time (pipeline equilibrium).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from hw_params import TPUParams, TPU_V7X, dtype_bytes
from compute_step import ComputeStep
from pipeline_simulator import TileConfig, calc_dma_time_ns, calc_mxu_time_ns, calc_vpu_time_ns


def _candidate_dims(max_val: int, alignment: int) -> list[int]:
    """Generate candidate tile dimensions from alignment to max_val."""
    dims = []
    d = alignment
    while d <= max_val:
        dims.append(d)
        d += alignment
    return dims


def find_optimal_tiling(step: ComputeStep, hw: TPUParams) -> TileConfig:
    """Find optimal tile configuration minimizing pipeline time."""
    align = hw.alignment
    vmem_cap = hw.vmem_capacity_bytes
    elem_bytes = dtype_bytes(step.inputs[0].dtype)

    if step.op_type == "matmul":
        vars_ = step.flops_vars
        M, N, K = vars_.get("M", 1), vars_.get("N", 1), vars_.get("K", 1)

        best_config = None
        best_time = float("inf")

        for bm in _candidate_dims(min(M, 2048), align):
            for bn in _candidate_dims(min(N, 2048), align):
                for bk in _candidate_dims(min(K, 1024), align):
                    tile_in = (bm * bk + bk * bn) * elem_bytes
                    tile_out = bm * bn * elem_bytes

                    # Try double buffer first
                    vmem_db = 2 * tile_in + tile_out
                    double_buf = vmem_db <= vmem_cap
                    vmem_used = vmem_db if double_buf else tile_in + tile_out

                    if vmem_used > vmem_cap:
                        continue

                    num_tiles = (
                        math.ceil(M / bm) *
                        math.ceil(N / bn) *
                        math.ceil(K / bk)
                    )

                    # Per-tile times
                    t_dma = calc_dma_time_ns(tile_in + tile_out, hw)
                    tile_flops = 2 * bm * bn * bk
                    t_comp = calc_mxu_time_ns(tile_flops, hw)

                    if double_buf and num_tiles > 1:
                        t_steady = max(t_dma, t_comp)
                        t_total = t_dma + (num_tiles - 1) * t_steady + t_comp
                    else:
                        t_total = num_tiles * (t_dma + t_comp)

                    if t_total < best_time:
                        best_time = t_total
                        best_config = TileConfig(
                            block_dims={"M": bm, "N": bn, "K": bk},
                            num_tiles=num_tiles,
                            tile_input_bytes=tile_in,
                            tile_output_bytes=tile_out,
                            double_buffer=double_buf,
                            vmem_usage_bytes=vmem_used,
                        )
        if best_config is None:
            # Fallback to minimum
            bm = bn = bk = align
            tile_in = (bm*bk + bk*bn) * elem_bytes
            tile_out = bm * bn * elem_bytes
            best_config = TileConfig(
                block_dims={"M": bm, "N": bn, "K": bk},
                num_tiles=math.ceil(M/bm)*math.ceil(N/bn)*math.ceil(K/bk),
                tile_input_bytes=tile_in,
                tile_output_bytes=tile_out,
                double_buffer=False,
                vmem_usage_bytes=tile_in + tile_out,
            )
        return best_config
    else:
        # Elementwise/reduce: simple 1D tiling
        numel = step.inputs[0].numel
        max_elements = vmem_cap // (2 * 2 * elem_bytes)
        tile_elements = min(numel, max_elements)
        tile_elements = max(align, (tile_elements // align) * align)
        num_tiles = math.ceil(numel / tile_elements)
        tile_in = tile_elements * elem_bytes
        tile_out = tile_elements * elem_bytes
        return TileConfig(
            block_dims={"N": tile_elements},
            num_tiles=num_tiles,
            tile_input_bytes=tile_in,
            tile_output_bytes=tile_out,
            double_buffer=True,
            vmem_usage_bytes=2 * tile_in + tile_out,
        )


def find_optimal_tiling_with_analysis(step: ComputeStep, hw: TPUParams) -> dict:
    """Find optimal tiling and return detailed analysis."""
    config = find_optimal_tiling(step, hw)
    t_dma = calc_dma_time_ns(config.tile_input_bytes + config.tile_output_bytes, hw)

    if step.compute_unit == "MXU" and step.op_type == "matmul":
        bd = config.block_dims
        tile_flops = 2 * bd["M"] * bd["N"] * bd["K"]
        t_comp = calc_mxu_time_ns(tile_flops, hw)
    else:
        total_flops = step.eval_flops()
        tile_flops = total_flops // config.num_tiles
        t_comp = calc_vpu_time_ns(tile_flops, hw)

    balance = t_dma / t_comp if t_comp > 0 else float("inf")

    return {
        "tile_config": config,
        "dma_time_per_tile_ns": t_dma,
        "compute_time_per_tile_ns": t_comp,
        "pipeline_balance_ratio": balance,  # 1.0 = perfect balance
        "bottleneck_per_tile": "DMA" if t_dma > t_comp else "COMPUTE",
    }
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_tiling_optimizer.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/tiling_optimizer.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_tiling_optimizer.py
git commit -m "feat(tpu-perf-model): add tiling optimizer with pipeline balance search"
```

---

### Task 6: Gap Analyzer

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/gap_analyzer.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_gap_analyzer.py`

**Step 1: Write the test**

```python
#!/usr/bin/env python3
"""Tests for gap_analyzer module."""
import unittest


class TestGapAnalyzer(unittest.TestCase):
    def test_gap_calculation(self):
        from gap_analyzer import GapEntry, compute_gap
        gap = compute_gap(
            metric="hbm_bytes",
            theoretical=1_000_000,
            measured=1_500_000,
        )
        self.assertAlmostEqual(gap.gap_pct, 50.0)
        self.assertIn("excess", gap.diagnosis.lower())

    def test_no_gap(self):
        from gap_analyzer import compute_gap
        gap = compute_gap("mxu_util", theoretical=95.0, measured=94.0)
        self.assertAlmostEqual(gap.gap_pct, -1.05, places=1)

    def test_analyze_eval_result(self):
        from gap_analyzer import analyze_eval_result
        from pipeline_simulator import PipelineReport, StepResult, TileConfig

        theoretical = PipelineReport(
            steps=[
                StepResult(
                    name="matmul", op_type="matmul", compute_unit="MXU",
                    flops=2_000_000, hbm_bytes=100_000,
                    t_hbm_ns=27.1, t_compute_ns=0.87,
                    t_step_ns=27.1, bottleneck="HBM_BW",
                    arithmetic_intensity=20.0, tile_config=None,
                    fused_with_prev=False, fusion_hbm_savings_bytes=0,
                )
            ],
            total_time_ns=27.1, total_flops=2_000_000,
            total_hbm_bytes=100_000, fusion_savings_bytes=0,
            overall_arithmetic_intensity=20.0,
            overall_bottleneck="HBM_BW", efficiency_vs_peak=0.032,
        )

        eval_result = {
            "total_time_us": 0.05,  # 50 ns
            "metadata": {
                "hw_utilization": {
                    "hbm_bandwidth_bytes": 180_000,
                    "mxu_utilization_pct": 45.0,
                    "vmem_utilization_pct": 30.0,
                },
                "profile": {
                    "vector_spills": 5,
                    "vector_fills": 3,
                },
            },
        }

        report = analyze_eval_result(theoretical, eval_result)
        self.assertGreater(len(report.gaps), 0)
        self.assertGreater(len(report.top_opportunities), 0)
        self.assertGreater(report.achievable_speedup, 1.0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_gap_analyzer.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Gap analyzer: compare theoretical model vs measured eval_result.json."""
from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline_simulator import PipelineReport


@dataclass
class GapEntry:
    metric: str
    theoretical: float
    measured: float
    gap_pct: float         # (measured - theoretical) / theoretical * 100
    diagnosis: str


@dataclass
class ComparisonReport:
    gaps: list[GapEntry]
    top_opportunities: list[str]
    achievable_speedup: float
    theoretical_time_ns: float
    measured_time_ns: float


def compute_gap(metric: str, theoretical: float, measured: float) -> GapEntry:
    """Compute gap between theoretical and measured values."""
    if theoretical == 0:
        gap_pct = 0.0 if measured == 0 else float("inf")
    else:
        gap_pct = (measured - theoretical) / abs(theoretical) * 100

    if metric == "hbm_bytes":
        if gap_pct > 10:
            diagnosis = f"Excess HBM transfer: {gap_pct:.1f}% more than theoretical. Check for missing fusion opportunities or redundant loads."
        elif gap_pct < -10:
            diagnosis = f"Less HBM transfer than expected ({gap_pct:.1f}%). Compiler may have optimized better than model predicts."
        else:
            diagnosis = "HBM transfer close to theoretical."
    elif metric == "mxu_util":
        if measured < 50:
            diagnosis = f"Low MXU utilization ({measured:.1f}%). Likely tile too small, poor alignment, or excessive padding."
        elif measured < 80:
            diagnosis = f"Moderate MXU utilization ({measured:.1f}%). Consider larger tiles or better K-dimension blocking."
        else:
            diagnosis = f"Good MXU utilization ({measured:.1f}%)."
    elif metric == "vector_spills":
        if measured > 0:
            diagnosis = f"VPR spills detected ({int(measured)}). Register pressure too high — consider reducing fusion or smaller tiles."
        else:
            diagnosis = "No VPR spills. Register pressure is healthy."
    elif metric == "total_time":
        if gap_pct > 20:
            diagnosis = f"Measured time {gap_pct:.1f}% above theoretical. Significant optimization headroom."
        elif gap_pct > 5:
            diagnosis = f"Measured time {gap_pct:.1f}% above theoretical. Moderate optimization opportunity."
        else:
            diagnosis = "Close to theoretical optimum."
    else:
        diagnosis = f"Gap: {gap_pct:.1f}%"

    return GapEntry(
        metric=metric,
        theoretical=theoretical,
        measured=measured,
        gap_pct=gap_pct,
        diagnosis=diagnosis,
    )


def analyze_eval_result(
    theoretical: PipelineReport,
    eval_result: dict,
) -> ComparisonReport:
    """Compare theoretical pipeline model against measured eval_result.json."""
    gaps = []

    # Extract measured values
    hw_util = eval_result.get("metadata", {}).get("hw_utilization", {})
    profile = eval_result.get("metadata", {}).get("profile", {})

    # HBM bytes
    measured_hbm = hw_util.get("hbm_bandwidth_bytes", 0)
    if measured_hbm > 0:
        gaps.append(compute_gap("hbm_bytes", theoretical.total_hbm_bytes, measured_hbm))

    # MXU utilization
    measured_mxu = hw_util.get("mxu_utilization_pct", 0)
    if measured_mxu > 0:
        theoretical_mxu = theoretical.efficiency_vs_peak * 100
        gaps.append(compute_gap("mxu_util", theoretical_mxu, measured_mxu))

    # VPR spills
    spills = profile.get("vector_spills", 0)
    gaps.append(compute_gap("vector_spills", 0, spills))

    # Total time
    measured_time_us = eval_result.get("total_time_us", 0)
    measured_time_ns = measured_time_us * 1000
    if measured_time_ns > 0:
        gaps.append(compute_gap("total_time", theoretical.total_time_ns, measured_time_ns))

    # Rank opportunities by gap magnitude
    ranked = sorted(gaps, key=lambda g: abs(g.gap_pct), reverse=True)
    opportunities = [g.diagnosis for g in ranked if abs(g.gap_pct) > 5]

    # Achievable speedup
    speedup = measured_time_ns / theoretical.total_time_ns if theoretical.total_time_ns > 0 else 1.0

    return ComparisonReport(
        gaps=gaps,
        top_opportunities=opportunities,
        achievable_speedup=speedup,
        theoretical_time_ns=theoretical.total_time_ns,
        measured_time_ns=measured_time_ns,
    )


def load_eval_result(path: str) -> dict:
    """Load eval_result.json from disk."""
    with open(path) as f:
        return json.load(f)
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_gap_analyzer.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/gap_analyzer.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_gap_analyzer.py
git commit -m "feat(tpu-perf-model): add gap analyzer for theoretical vs measured comparison"
```

---

### Task 7: Report Formatter

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/report.py`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_report.py`

**Step 1: Write the test**

```python
#!/usr/bin/env python3
"""Tests for report module."""
import json
import unittest


class TestReportJSON(unittest.TestCase):
    def test_pipeline_report_to_json(self):
        from report import pipeline_report_to_json
        from pipeline_simulator import PipelineReport, StepResult

        report = PipelineReport(
            steps=[
                StepResult(
                    name="matmul", op_type="matmul", compute_unit="MXU",
                    flops=2_000_000, hbm_bytes=100_000,
                    t_hbm_ns=27.1, t_compute_ns=0.87,
                    t_step_ns=27.1, bottleneck="HBM_BW",
                    arithmetic_intensity=20.0, tile_config=None,
                    fused_with_prev=False, fusion_hbm_savings_bytes=0,
                )
            ],
            total_time_ns=27.1, total_flops=2_000_000,
            total_hbm_bytes=100_000, fusion_savings_bytes=0,
            overall_arithmetic_intensity=20.0,
            overall_bottleneck="HBM_BW", efficiency_vs_peak=0.032,
        )
        j = pipeline_report_to_json(report)
        data = json.loads(j)
        self.assertIn("steps", data)
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["overall_bottleneck"], "HBM_BW")


class TestReportText(unittest.TestCase):
    def test_pipeline_report_to_text(self):
        from report import pipeline_report_to_text
        from pipeline_simulator import PipelineReport, StepResult

        report = PipelineReport(
            steps=[
                StepResult(
                    name="matmul", op_type="matmul", compute_unit="MXU",
                    flops=2_000_000, hbm_bytes=100_000,
                    t_hbm_ns=27.1, t_compute_ns=0.87,
                    t_step_ns=27.1, bottleneck="HBM_BW",
                    arithmetic_intensity=20.0, tile_config=None,
                    fused_with_prev=False, fusion_hbm_savings_bytes=0,
                )
            ],
            total_time_ns=27.1, total_flops=2_000_000,
            total_hbm_bytes=100_000, fusion_savings_bytes=0,
            overall_arithmetic_intensity=20.0,
            overall_bottleneck="HBM_BW", efficiency_vs_peak=0.032,
        )
        text = pipeline_report_to_text(report)
        self.assertIn("matmul", text)
        self.assertIn("HBM_BW", text)
        self.assertIn("bottleneck", text.lower())


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_report.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""Output formatting for TPU performance model reports."""
from __future__ import annotations

import json
from dataclasses import asdict

from pipeline_simulator import PipelineReport, StepResult, TileConfig
from gap_analyzer import ComparisonReport


def _format_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"


def _format_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.2f} ns"


def _step_to_dict(s: StepResult) -> dict:
    d = {
        "name": s.name,
        "op_type": s.op_type,
        "compute_unit": s.compute_unit,
        "flops": s.flops,
        "hbm_bytes": s.hbm_bytes,
        "t_hbm_ns": s.t_hbm_ns,
        "t_compute_ns": s.t_compute_ns,
        "t_step_ns": s.t_step_ns,
        "bottleneck": s.bottleneck,
        "arithmetic_intensity": s.arithmetic_intensity,
        "fused_with_prev": s.fused_with_prev,
        "fusion_hbm_savings_bytes": s.fusion_hbm_savings_bytes,
    }
    if s.tile_config:
        d["tile_config"] = {
            "block_dims": s.tile_config.block_dims,
            "num_tiles": s.tile_config.num_tiles,
            "double_buffer": s.tile_config.double_buffer,
            "vmem_usage_bytes": s.tile_config.vmem_usage_bytes,
        }
    return d


def pipeline_report_to_json(report: PipelineReport) -> str:
    """Convert PipelineReport to JSON string."""
    data = {
        "steps": [_step_to_dict(s) for s in report.steps],
        "summary": {
            "total_time_ns": report.total_time_ns,
            "total_flops": report.total_flops,
            "total_hbm_bytes": report.total_hbm_bytes,
            "fusion_savings_bytes": report.fusion_savings_bytes,
            "overall_arithmetic_intensity": report.overall_arithmetic_intensity,
            "overall_bottleneck": report.overall_bottleneck,
            "efficiency_vs_peak": report.efficiency_vs_peak,
        },
    }
    return json.dumps(data, indent=2)


def pipeline_report_to_text(report: PipelineReport) -> str:
    """Convert PipelineReport to human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("TPU v7x Performance Model Report")
    lines.append("=" * 70)

    for i, s in enumerate(report.steps):
        lines.append(f"\n--- Step {i+1}: {s.name} ({s.op_type}, {s.compute_unit}) ---")
        lines.append(f"  FLOPs:               {s.flops:,}")
        lines.append(f"  HBM I/O:             {_format_bytes(s.hbm_bytes)}")
        lines.append(f"  T(HBM):              {_format_ns(s.t_hbm_ns)}")
        lines.append(f"  T(compute):          {_format_ns(s.t_compute_ns)}")
        lines.append(f"  T(step):             {_format_ns(s.t_step_ns)}")
        lines.append(f"  Bottleneck:          {s.bottleneck}")
        lines.append(f"  Arithmetic Intensity: {s.arithmetic_intensity:.2f} FLOPs/byte")
        if s.fused_with_prev:
            lines.append(f"  Fused with previous: YES (saved {_format_bytes(s.fusion_hbm_savings_bytes)})")
        if s.tile_config:
            tc = s.tile_config
            lines.append(f"  Tile config:         {tc.block_dims}")
            lines.append(f"  Num tiles:           {tc.num_tiles}")
            lines.append(f"  Double buffer:       {'YES' if tc.double_buffer else 'NO'}")
            lines.append(f"  VMEM usage:          {_format_bytes(tc.vmem_usage_bytes)}")

    lines.append(f"\n{'=' * 70}")
    lines.append("Summary")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Total time:          {_format_ns(report.total_time_ns)}")
    lines.append(f"  Total FLOPs:         {report.total_flops:,}")
    lines.append(f"  Total HBM I/O:       {_format_bytes(report.total_hbm_bytes)}")
    lines.append(f"  Fusion savings:      {_format_bytes(report.fusion_savings_bytes)}")
    lines.append(f"  Arithmetic Intensity: {report.overall_arithmetic_intensity:.2f} FLOPs/byte")
    lines.append(f"  Overall bottleneck:  {report.overall_bottleneck}")
    lines.append(f"  Efficiency vs peak:  {report.efficiency_vs_peak * 100:.1f}%")
    lines.append("")

    return "\n".join(lines)


def comparison_report_to_text(report: ComparisonReport) -> str:
    """Convert ComparisonReport to human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("Gap Analysis: Theoretical vs Measured")
    lines.append("=" * 70)
    lines.append(f"  Theoretical time:    {_format_ns(report.theoretical_time_ns)}")
    lines.append(f"  Measured time:       {_format_ns(report.measured_time_ns)}")
    lines.append(f"  Achievable speedup:  {report.achievable_speedup:.2f}x")

    lines.append(f"\n--- Gaps ---")
    for g in report.gaps:
        lines.append(f"  [{g.metric}] theoretical={g.theoretical:.1f} measured={g.measured:.1f} gap={g.gap_pct:+.1f}%")
        lines.append(f"    -> {g.diagnosis}")

    if report.top_opportunities:
        lines.append(f"\n--- Top Optimization Opportunities ---")
        for i, opp in enumerate(report.top_opportunities, 1):
            lines.append(f"  {i}. {opp}")

    lines.append("")
    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_report.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/report.py \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_report.py
git commit -m "feat(tpu-perf-model): add report formatter (JSON + text)"
```

---

### Task 8: CLI Entry Point

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py`

**Step 1: Write implementation**

```python
#!/usr/bin/env python3
"""CLI entry point for TPU performance model.

Usage:
    python cli.py --steps steps.json [--eval eval_result.json] [--format text|json]
"""
import argparse
import json
import sys

from compute_step import load_steps, load_steps_from_file
from hw_params import TPU_V7X
from pipeline_simulator import simulate_steps
from tiling_optimizer import find_optimal_tiling_with_analysis
from gap_analyzer import analyze_eval_result, load_eval_result
from report import pipeline_report_to_json, pipeline_report_to_text
from report import comparison_report_to_text


def main():
    parser = argparse.ArgumentParser(description="TPU v7x Performance Model")
    parser.add_argument("--steps", required=True, help="Path to ComputeSteps JSON file")
    parser.add_argument("--eval", help="Path to eval_result.json for gap analysis")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--tiling", action="store_true", help="Show detailed tiling analysis")
    args = parser.parse_args()

    # Load steps
    steps = load_steps_from_file(args.steps)

    # Run pipeline simulation
    report = simulate_steps(steps, TPU_V7X)

    # Output pipeline report
    if args.format == "json":
        print(pipeline_report_to_json(report))
    else:
        print(pipeline_report_to_text(report))

    # Tiling analysis
    if args.tiling:
        print("\n" + "=" * 70)
        print("Detailed Tiling Analysis")
        print("=" * 70)
        for step in steps:
            analysis = find_optimal_tiling_with_analysis(step, TPU_V7X)
            tc = analysis["tile_config"]
            print(f"\n  {step.name}:")
            print(f"    Optimal tile:           {tc.block_dims}")
            print(f"    DMA time/tile:          {analysis['dma_time_per_tile_ns']:.2f} ns")
            print(f"    Compute time/tile:      {analysis['compute_time_per_tile_ns']:.2f} ns")
            print(f"    Pipeline balance ratio: {analysis['pipeline_balance_ratio']:.2f} (1.0 = perfect)")
            print(f"    Per-tile bottleneck:    {analysis['bottleneck_per_tile']}")

    # Gap analysis
    if args.eval:
        eval_data = load_eval_result(args.eval)
        comparison = analyze_eval_result(report, eval_data)
        print()
        print(comparison_report_to_text(comparison))


if __name__ == "__main__":
    main()
```

**Step 2: Verify CLI works**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python cli.py --help`
Expected: Prints usage help without errors

**Step 3: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/cli.py
git commit -m "feat(tpu-perf-model): add CLI entry point"
```

---

### Task 9: Write SKILL.md

The skill guides AI through formula decomposition, script invocation, and result interpretation.

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md`

**Step 1: Write the skill**

Replace the placeholder with the full skill content. The SKILL.md should contain:

1. **Frontmatter:** name, description
2. **Overview:** What the tool does, when to use it
3. **Phase 1 — Formula Decomposition:** Guide AI to decompose user's formula into ComputeStep JSON, with a reference table of FLOPs formulas per op type
4. **Phase 2 — Run Simulation:** How to invoke `cli.py` and what flags to use
5. **Phase 3 — Interpret Results:** How to read the output (bottleneck diagnosis, tiling recommendations, pipeline efficiency)
6. **Phase 4 — Gap Analysis (optional):** How to compare with eval_result.json

Key content for the FLOPs reference table:

| Op Type | FLOPs Formula | Compute Unit | Example |
|---------|---------------|--------------|---------|
| matmul [M,K]×[K,N] | 2×M×N×K | MXU | QK^T |
| elementwise (unary) | N (elements) | VPU | exp, scale |
| elementwise (binary) | N (elements) | VPU | add, mul |
| reduce (sum/max) | N (elements) | VPU | softmax reduce |
| softmax [M,N] | 5×M×N | VPU | max+sub+exp+sum+div |

Key content for fusion rules:
- matmul → elementwise: usually fusable (low VPR pressure)
- matmul → reduce: sometimes fusable (check VPR count)
- elementwise → elementwise: always fusable
- matmul → matmul: almost never fusable (extreme VPR pressure)

**Step 2: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/SKILL.md
git commit -m "feat(tpu-perf-model): write SKILL.md with formula decomposition guide"
```

---

### Task 10: Integration Test — Flash Attention Example

Create an example ComputeSteps JSON for flash attention and verify the full pipeline works end-to-end.

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/examples/flash_attention.json`
- Create: `plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py`

**Step 1: Create example JSON**

```json
[
  {
    "name": "qk_matmul",
    "op_type": "matmul",
    "inputs": [
      {"name": "Q", "shape": [4096, 128], "dtype": "bf16"},
      {"name": "K_T", "shape": [128, 4096], "dtype": "bf16"}
    ],
    "outputs": [
      {"name": "S", "shape": [4096, 4096], "dtype": "bf16"}
    ],
    "flops_formula": "2*M*N*K",
    "flops_vars": {"M": 4096, "N": 4096, "K": 128},
    "compute_unit": "MXU",
    "fusable_with_prev": false
  },
  {
    "name": "scale",
    "op_type": "elementwise",
    "inputs": [
      {"name": "S", "shape": [4096, 4096], "dtype": "bf16"}
    ],
    "outputs": [
      {"name": "S_scaled", "shape": [4096, 4096], "dtype": "bf16"}
    ],
    "flops_formula": "M*N",
    "flops_vars": {"M": 4096, "N": 4096},
    "compute_unit": "VPU",
    "fusable_with_prev": true
  },
  {
    "name": "softmax",
    "op_type": "softmax",
    "inputs": [
      {"name": "S_scaled", "shape": [4096, 4096], "dtype": "bf16"}
    ],
    "outputs": [
      {"name": "P", "shape": [4096, 4096], "dtype": "bf16"}
    ],
    "flops_formula": "5*M*N",
    "flops_vars": {"M": 4096, "N": 4096},
    "compute_unit": "VPU",
    "fusable_with_prev": true
  },
  {
    "name": "sv_matmul",
    "op_type": "matmul",
    "inputs": [
      {"name": "P", "shape": [4096, 4096], "dtype": "bf16"},
      {"name": "V", "shape": [4096, 128], "dtype": "bf16"}
    ],
    "outputs": [
      {"name": "O", "shape": [4096, 128], "dtype": "bf16"}
    ],
    "flops_formula": "2*M*N*K",
    "flops_vars": {"M": 4096, "N": 128, "K": 4096},
    "compute_unit": "MXU",
    "fusable_with_prev": false
  }
]
```

**Step 2: Write integration test**

```python
#!/usr/bin/env python3
"""Integration test: full pipeline with flash attention example."""
import json
import os
import unittest


class TestFlashAttentionE2E(unittest.TestCase):
    def test_full_pipeline(self):
        from compute_step import load_steps_from_file
        from pipeline_simulator import simulate_steps
        from hw_params import TPU_V7X
        from report import pipeline_report_to_json, pipeline_report_to_text

        example_path = os.path.join(os.path.dirname(__file__), "examples", "flash_attention.json")
        steps = load_steps_from_file(example_path)
        self.assertEqual(len(steps), 4)

        report = simulate_steps(steps, TPU_V7X)

        # Basic sanity checks
        self.assertEqual(len(report.steps), 4)
        self.assertGreater(report.total_time_ns, 0)
        self.assertGreater(report.total_flops, 0)
        self.assertGreater(report.fusion_savings_bytes, 0)  # scale is fused

        # Fusion: scale and softmax should be fused
        self.assertTrue(report.steps[1].fused_with_prev)   # scale fused with qk_matmul
        self.assertTrue(report.steps[2].fused_with_prev)   # softmax fused with scale

        # Both output formats should work
        json_out = pipeline_report_to_json(report)
        data = json.loads(json_out)
        self.assertIn("steps", data)

        text_out = pipeline_report_to_text(report)
        self.assertIn("qk_matmul", text_out)
        self.assertIn("sv_matmul", text_out)

    def test_cli_runs(self):
        """Test that CLI runs without errors."""
        import subprocess
        scripts_dir = os.path.dirname(__file__)
        example_path = os.path.join(scripts_dir, "examples", "flash_attention.json")
        result = subprocess.run(
            ["python", os.path.join(scripts_dir, "cli.py"),
             "--steps", example_path, "--format", "json", "--tiling"],
            capture_output=True, text=True, cwd=scripts_dir,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        # Should produce valid JSON (at least the first part)
        lines = result.stdout.strip().split("\n")
        # Find JSON portion (before tiling analysis)
        json_lines = []
        brace_count = 0
        for line in lines:
            json_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0 and json_lines:
                break
        json_str = "\n".join(json_lines)
        data = json.loads(json_str)
        self.assertIn("steps", data)


if __name__ == "__main__":
    unittest.main()
```

**Step 3: Run integration test**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest test_integration.py -v`
Expected: All 2 tests PASS

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-perf-model/scripts/examples/ \
        plugins/tpu-perf-model/skills/tpu-perf-model/scripts/test_integration.py
git commit -m "feat(tpu-perf-model): add flash attention example and integration test"
```

---

### Task 11: Run All Tests

**Step 1: Run full test suite**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python -m pytest -v`
Expected: All tests pass (approximately 16 tests across 6 test files)

**Step 2: Verify CLI end-to-end**

Run: `cd plugins/tpu-perf-model/skills/tpu-perf-model/scripts && python cli.py --steps examples/flash_attention.json --tiling`
Expected: Prints full text report with 4 steps, tiling analysis, fusion savings > 0

**Step 3: Verify JSON validity of plugin files**

Run: `python -c "import json; json.load(open('plugins/tpu-perf-model/.claude-plugin/plugin.json')); json.load(open('.claude-plugin/marketplace.json')); print('OK')"`
Expected: "OK"

**Step 4: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "fix(tpu-perf-model): address test failures"
```
