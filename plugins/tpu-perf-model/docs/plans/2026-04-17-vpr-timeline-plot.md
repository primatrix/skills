# VPR Timeline Plot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add matplotlib-based VPR timeline visualization to tpu-pipeline-scheduler — a 2D heatmap (X=time, Y=VPR) colored by hardware unit and access type (write/read/live), with dependency arc arrows overlaid and Gantt strips on top.

**Architecture:** New `pipeline_plot.py` module with lazy matplotlib import. Receives pre-computed `ScheduleResult`, `DependencyGraph`, and `list[PipelineOp]` — no new data computation needed. CLI gets `--plot`/`--plot-output` flags.

**Tech Stack:** matplotlib (only dependency), imported lazily so non-plot CLI usage stays dependency-free.

---

### Task 1: Create `pipeline_plot.py` with `build_vpr_activity()` data function

**Files:**
- Create: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_plot.py`
- Test: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_plot.py`

The core data structure maps each VPR to a list of time intervals tagged with `(unit, access_type)`. This drives the entire plot.

**Step 1: Write the failing test**

Create `test_pipeline_plot.py`:

```python
#!/usr/bin/env python3
"""Tests for pipeline_plot module."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestBuildVPRActivity(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_write_interval(self):
        """Op that writes VPR[0] produces a 'write' interval on VPR 0."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0)]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        self.assertIn(0, activity)
        intervals = activity[0]
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].unit, "VPU")
        self.assertEqual(intervals[0].access, "write")
        self.assertAlmostEqual(intervals[0].start_ns, 0.0)
        self.assertAlmostEqual(intervals[0].end_ns, 100.0)

    def test_read_interval(self):
        """Op that reads VPR[0] produces a 'read' interval on VPR 0."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="w", output_vprs=[0], latency_ns=100.0),
            self._make_op(op_id="r", input_vprs=[0], latency_ns=50.0),
        ]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        intervals = activity[0]
        # Should have write interval from w, then read interval from r
        writes = [i for i in intervals if i.access == "write"]
        reads = [i for i in intervals if i.access == "read"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0].unit, "VPU")

    def test_live_gap_filled(self):
        """VPR that is live between write-end and read-start gets a 'live' interval."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="w", output_vprs=[0], unit="VPU", latency_ns=100.0),
            self._make_op(op_id="stall", unit="MXU", op_kind="MXU", latency_ns=200.0),
            self._make_op(op_id="r", input_vprs=[0], unit="VPU", latency_ns=50.0),
        ]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        lives = [i for i in activity[0] if i.access == "live"]
        self.assertGreater(len(lives), 0, "Should have a live interval in the gap")

    def test_multiple_vprs(self):
        """Activity dict has entries for all touched VPRs."""
        from pipeline_plot import build_vpr_activity
        from pipeline_scheduler import schedule

        ops = [
            self._make_op(op_id="w", output_vprs=[0, 1, 2], latency_ns=100.0),
        ]
        sched = schedule(ops)
        activity = build_vpr_activity(ops, sched)

        self.assertIn(0, activity)
        self.assertIn(1, activity)
        self.assertIn(2, activity)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_plot.py -v`
Expected: FAIL with ImportError (pipeline_plot doesn't exist)

**Step 3: Write minimal implementation**

Create `pipeline_plot.py`:

```python
#!/usr/bin/env python3
"""VPR timeline plot — matplotlib-based visualization of register activity."""

from __future__ import annotations

from dataclasses import dataclass
from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult, ScheduleEntry
from dependency_analyzer import DependencyGraph
from vpr_analyzer import analyze_vpr_liveness


@dataclass
class VPRInterval:
    """A time interval where a VPR is in a specific state."""
    vpr_id: int
    start_ns: float
    end_ns: float
    unit: str        # DMA | MXU | VPU
    access: str      # write | read | live
    op_id: str


def build_vpr_activity(
    ops: list[PipelineOp], sched: ScheduleResult
) -> dict[int, list[VPRInterval]]:
    """Build per-VPR activity intervals from schedule results.

    For each VPR, produces intervals tagged with (unit, access_type):
    - "write": the op's time window when it writes this VPR (output_vprs)
    - "read":  the op's time window when it reads this VPR (input_vprs)
    - "live":  gaps between write-end and last-read-end where VPR holds data
    """
    entries = sched.entries_by_id
    raw: dict[int, list[VPRInterval]] = {}

    for op in ops:
        entry = entries[op.op_id]
        for v in op.output_vprs:
            raw.setdefault(v, []).append(VPRInterval(
                vpr_id=v, start_ns=entry.start_ns, end_ns=entry.end_ns,
                unit=op.unit, access="write", op_id=op.op_id,
            ))
        for v in op.input_vprs:
            raw.setdefault(v, []).append(VPRInterval(
                vpr_id=v, start_ns=entry.start_ns, end_ns=entry.end_ns,
                unit=op.unit, access="read", op_id=op.op_id,
            ))

    # Fill "live" gaps: VPR holds data between active intervals
    occ = analyze_vpr_liveness(ops, sched)
    result: dict[int, list[VPRInterval]] = {}
    for vpr_id, intervals in raw.items():
        intervals.sort(key=lambda i: i.start_ns)
        lv = next((l for l in occ.liveness if l.vpr_id == vpr_id), None)
        if not lv:
            result[vpr_id] = intervals
            continue

        filled: list[VPRInterval] = []
        # Merge overlapping active intervals to find gaps
        active_spans: list[tuple[float, float]] = [
            (i.start_ns, i.end_ns) for i in intervals
        ]
        active_spans.sort()
        merged: list[tuple[float, float]] = []
        for s, e in active_spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Determine the defining unit for live gaps
        def_unit = intervals[0].unit  # unit of first writer

        # Insert live intervals in gaps within liveness range
        prev_end = lv.live_start_ns
        for ms, me in merged:
            if ms > prev_end and ms > lv.live_start_ns:
                gap_start = max(prev_end, lv.live_start_ns)
                if gap_start < ms:
                    filled.append(VPRInterval(
                        vpr_id=vpr_id, start_ns=gap_start, end_ns=ms,
                        unit=def_unit, access="live", op_id="",
                    ))
            prev_end = me
        # Trailing live gap
        if prev_end < lv.live_end_ns:
            filled.append(VPRInterval(
                vpr_id=vpr_id, start_ns=prev_end, end_ns=lv.live_end_ns,
                unit=def_unit, access="live", op_id="",
            ))

        filled.extend(intervals)
        filled.sort(key=lambda i: (i.start_ns, {"write": 0, "read": 1, "live": 2}[i.access]))
        result[vpr_id] = filled

    return result
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_plot.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_plot.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_plot.py
git commit -m "feat(tpu-pipeline-scheduler): add VPR activity data builder"
```

---

### Task 2: Add `plot_vpr_timeline()` rendering function

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_plot.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_plot.py`

This is the main matplotlib rendering function. It produces the 2-panel figure (Gantt strips + VPR heatmap) with dependency arrows and legend.

**Step 1: Write the failing test**

Append to `test_pipeline_plot.py`:

```python
class TestPlotVPRTimeline(unittest.TestCase):
    def _make_op(self, **kwargs):
        from pipeline_ir import PipelineOp
        defaults = dict(
            op_id="op", op_kind="VPU", input_vprs=[], output_vprs=[],
            input_vmem=[], output_vmem=[], latency_ns=10.0, unit="VPU",
        )
        defaults.update(kwargs)
        return PipelineOp(**defaults)

    def test_plot_creates_png(self):
        """plot_vpr_timeline produces a valid PNG file."""
        import tempfile
        from pipeline_plot import plot_vpr_timeline
        from pipeline_scheduler import schedule
        from dependency_analyzer import analyze_dependencies

        ops = [
            self._make_op(op_id="load_q", unit="DMA", op_kind="DMA_LOAD",
                          output_vmem=["q"], latency_ns=200.0),
            self._make_op(op_id="q_reg", unit="VPU", op_kind="VMEM_TO_REG",
                          input_vmem=["q"], output_vprs=[0, 1], latency_ns=10.0),
            self._make_op(op_id="mxu", unit="MXU", op_kind="MXU",
                          input_vprs=[0, 1], output_vprs=[2, 3], latency_ns=500.0),
            self._make_op(op_id="vpu", unit="VPU", op_kind="VPU",
                          input_vprs=[2, 3], output_vprs=[4, 5], latency_ns=100.0),
        ]
        sched = schedule(ops)
        graph = analyze_dependencies(ops)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name

        try:
            plot_vpr_timeline(ops, sched, graph, out_path, title="test_kernel")
            with open(out_path, "rb") as f:
                header = f.read(8)
            # PNG magic bytes
            self.assertEqual(header[:4], b"\x89PNG")
        finally:
            os.unlink(out_path)

    def test_plot_empty_ops(self):
        """plot_vpr_timeline handles empty ops gracefully."""
        import tempfile
        from pipeline_plot import plot_vpr_timeline
        from pipeline_scheduler import ScheduleResult
        from dependency_analyzer import DependencyGraph

        sched = ScheduleResult(entries=[], total_latency_ns=0,
                               critical_path=[], stall_total_ns=0)
        graph = DependencyGraph(ops=[], edges=[])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name

        try:
            plot_vpr_timeline([], sched, graph, out_path)
            self.assertTrue(os.path.exists(out_path))
        finally:
            os.unlink(out_path)
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_plot.py::TestPlotVPRTimeline -v`
Expected: FAIL with AttributeError (`plot_vpr_timeline` not found)

**Step 3: Write the implementation**

Append to `pipeline_plot.py`:

```python
# --- Color scheme ---
_COLORS = {
    ("DMA", "write"): "#1a5276",
    ("DMA", "read"):  "#5dade2",
    ("DMA", "live"):  "#d4e6f1",
    ("MXU", "write"): "#922b21",
    ("MXU", "read"):  "#e74c3c",
    ("MXU", "live"):  "#f5b7b1",
    ("VPU", "write"): "#196f3d",
    ("VPU", "read"):  "#27ae60",
    ("VPU", "live"):  "#d5f5e3",
}

_UNIT_COLORS = {"DMA": "#2980b9", "MXU": "#c0392b", "VPU": "#27ae60"}

_HAZARD_STYLES = {
    "RAW": {"linestyle": "-",  "color": "#333333"},
    "WAR": {"linestyle": "--", "color": "#888888"},
    "WAW": {"linestyle": ":",  "color": "#aaaaaa"},
}


def plot_vpr_timeline(
    ops: list[PipelineOp],
    sched: ScheduleResult,
    graph: DependencyGraph,
    output_path: str,
    title: str = "",
) -> None:
    """Render VPR timeline heatmap with Gantt strips and dependency arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    import matplotlib.patches as mpatches

    if not ops or sched.total_latency_ns == 0:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, "No operations to plot", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    activity = build_vpr_activity(ops, sched)
    used_vprs = sorted(activity.keys())
    n_vprs = len(used_vprs)
    vpr_to_row = {v: i for i, v in enumerate(used_vprs)}
    total_ns = sched.total_latency_ns

    # --- Figure layout: Gantt (top, small) + Heatmap (main) ---
    fig_height = max(4, 1.5 + n_vprs * 0.4)
    fig, (ax_gantt, ax_heat) = plt.subplots(
        2, 1, figsize=(14, fig_height),
        gridspec_kw={"height_ratios": [1, max(3, n_vprs * 0.35)]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # === Gantt strips ===
    unit_order = ["DMA", "MXU", "VPU"]
    for u_idx, unit in enumerate(unit_order):
        for entry in sched.entries:
            if entry.unit == unit:
                ax_gantt.barh(
                    u_idx, entry.end_ns - entry.start_ns, left=entry.start_ns,
                    height=0.7, color=_UNIT_COLORS[unit], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                # Label short ops with op_id
                width = entry.end_ns - entry.start_ns
                if width / total_ns > 0.08:
                    ax_gantt.text(
                        entry.start_ns + width / 2, u_idx,
                        entry.op_id, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )

    ax_gantt.set_yticks(range(len(unit_order)))
    ax_gantt.set_yticklabels(unit_order, fontsize=9, fontweight="bold")
    ax_gantt.set_ylim(-0.5, len(unit_order) - 0.5)
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlim(0, total_ns)
    ax_gantt.tick_params(axis="x", labelbottom=False)
    ax_gantt.set_ylabel("Unit", fontsize=9)
    ax_gantt.spines["top"].set_visible(False)
    ax_gantt.spines["right"].set_visible(False)

    # === VPR Heatmap ===
    for vpr_id, intervals in activity.items():
        row = vpr_to_row[vpr_id]
        for iv in intervals:
            color = _COLORS.get((iv.unit, iv.access), "#eeeeee")
            width = iv.end_ns - iv.start_ns
            if width <= 0:
                continue
            rect = Rectangle(
                (iv.start_ns, row - 0.4), width, 0.8,
                facecolor=color, edgecolor="white", linewidth=0.3,
            )
            ax_heat.add_patch(rect)

    ax_heat.set_yticks(range(n_vprs))
    ax_heat.set_yticklabels([f"VPR[{v}]" for v in used_vprs], fontsize=8,
                            fontfamily="monospace")
    ax_heat.set_ylim(-0.5, n_vprs - 0.5)
    ax_heat.invert_yaxis()
    ax_heat.set_xlim(0, total_ns)
    ax_heat.set_xlabel("Time (ns)", fontsize=10)
    ax_heat.set_ylabel("VPR Register", fontsize=9)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)

    # === Dependency arrows (VPR-type only) ===
    op_map = {op.op_id: op for op in ops}
    entries_by_id = sched.entries_by_id
    for edge in graph.edges:
        if edge.resource_type != "VPR":
            continue
        from_entry = entries_by_id.get(edge.from_op)
        to_entry = entries_by_id.get(edge.to_op)
        if not from_entry or not to_entry:
            continue
        from_op = op_map[edge.from_op]
        to_op = op_map[edge.to_op]

        # Find the VPR that creates this dependency
        vpr_num = int(edge.resource_id.replace("VPR[", "").replace("]", ""))
        if vpr_num not in vpr_to_row:
            continue

        # Arrow from end of producer to start of consumer
        from_x = from_entry.end_ns
        to_x = to_entry.start_ns
        from_row = vpr_to_row[vpr_num]

        # Find target VPR row (where consumer reads/writes)
        to_row = from_row  # default same row
        if edge.hazard_type == "RAW":
            # consumer reads this VPR — same row
            to_row = from_row
        elif edge.hazard_type == "WAW":
            to_row = from_row

        style = _HAZARD_STYLES.get(edge.hazard_type, _HAZARD_STYLES["RAW"])
        mid_x = (from_x + to_x) / 2
        curve_height = 0.3

        arrow = FancyArrowPatch(
            (from_x, from_row - 0.45), (to_x, to_row - 0.45),
            connectionstyle=f"arc3,rad=-0.2",
            arrowstyle="->,head_width=3,head_length=3",
            linewidth=1.0, **style,
        )
        ax_heat.add_patch(arrow)

    # === Legend ===
    legend_handles = []
    for unit in unit_order:
        for access, label in [("write", "Write"), ("read", "Read"), ("live", "Live")]:
            c = _COLORS[(unit, access)]
            legend_handles.append(mpatches.Patch(color=c, label=f"{unit} {label}"))
    # Hazard arrow legend
    for hz, style in _HAZARD_STYLES.items():
        legend_handles.append(plt.Line2D(
            [0], [0], color=style["color"], linestyle=style["linestyle"],
            linewidth=1.5, label=f"{hz}",
        ))

    ax_heat.legend(
        handles=legend_handles, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=True,
        ncol=1, title="Legend", title_fontsize=8,
    )

    # === Title ===
    occ = analyze_vpr_liveness(ops, sched)
    title_str = title or "VPR Timeline"
    fig.suptitle(
        f"{title_str}  |  {total_ns:.0f}ns  |  "
        f"Peak VPR: {occ.peak_concurrent}/32  |  "
        f"Stall: {sched.stall_total_ns:.0f}ns",
        fontsize=11, fontweight="bold", y=0.98,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_pipeline_plot.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_plot.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_pipeline_plot.py
git commit -m "feat(tpu-pipeline-scheduler): add VPR timeline matplotlib plot"
```

---

### Task 3: Add `--plot` / `--plot-output` flags to CLI

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_ir_cli.py`
- Modify: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_integration.py`

**Step 1: Write the failing test**

Append to `test_integration.py`:

```python
    def test_cli_plot_output(self):
        import tempfile
        scripts_dir = self._scripts_dir()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            result = subprocess.run(
                [
                    "python", "pipeline_ir_cli.py",
                    "--pipeline", self._example_path(),
                    "--plot",
                    "--plot-output", out_path,
                ],
                capture_output=True, text=True, cwd=scripts_dir,
            )
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            with open(out_path, "rb") as f:
                header = f.read(4)
            self.assertEqual(header, b"\x89PNG")
        finally:
            os.unlink(out_path)

    def test_cli_plot_default_name(self):
        """--plot without --plot-output uses <spec_name>_vpr_timeline.png"""
        scripts_dir = self._scripts_dir()
        expected_name = "flash_attention_tile_vpr_timeline.png"
        expected_path = os.path.join(scripts_dir, expected_name)
        try:
            result = subprocess.run(
                [
                    "python", "pipeline_ir_cli.py",
                    "--pipeline", self._example_path(),
                    "--plot",
                ],
                capture_output=True, text=True, cwd=scripts_dir,
            )
            self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
            self.assertTrue(os.path.exists(expected_path),
                            f"Expected {expected_path} to be created")
        finally:
            if os.path.exists(expected_path):
                os.unlink(expected_path)
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_integration.py::TestPipelineSchedulerE2E::test_cli_plot_output -v`
Expected: FAIL (unrecognized argument --plot)

**Step 3: Modify `pipeline_ir_cli.py`**

Add to the argument parser (after `--mermaid`):

```python
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate VPR timeline plot as PNG image",
    )
    parser.add_argument(
        "--plot-output",
        help="Output path for plot (default: <spec_name>_vpr_timeline.png)",
    )
```

Add at the end of `main()`, before `print(...)`:

```python
    if args.plot:
        from pipeline_plot import plot_vpr_timeline
        plot_path = args.plot_output or f"{spec.name}_vpr_timeline.png"
        plot_vpr_timeline(spec.ops, sched, graph, plot_path, title=spec.name)
        print(f"Plot saved to: {plot_path}")
```

**Step 4: Run test to verify it passes**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest test_integration.py -v`
Expected: All tests PASS (including the 2 new ones)

**Step 5: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/pipeline_ir_cli.py \
       plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts/test_integration.py
git commit -m "feat(tpu-pipeline-scheduler): add --plot CLI flag for VPR timeline PNG"
```

---

### Task 4: Update SKILL.md documentation

**Files:**
- Modify: `plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/SKILL.md`

**Step 1: Add plot CLI examples**

In the `## CLI Usage` section, append:

```markdown
# VPR timeline plot (PNG)
python scripts/pipeline_ir_cli.py --pipeline kernel.json --plot

# Custom output path
python scripts/pipeline_ir_cli.py --pipeline kernel.json --plot --plot-output my_chart.png
```

**Step 2: Add `--plot` and `--plot-output` to CLI Options table**

Append to the table:

```markdown
| `--plot` | flag | Generate VPR timeline heatmap as PNG image |
| `--plot-output` | path | Output path for plot (default: `<name>_vpr_timeline.png`) |
```

**Step 3: Add new output section**

After `### 4. Reorder Suggestion`, add:

```markdown
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
```

**Step 4: Commit**

```bash
git add plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/SKILL.md
git commit -m "docs(tpu-pipeline-scheduler): add --plot usage to SKILL.md"
```

---

### Task 5: Run full test suite and validate with example

**Step 1: Run all existing tests**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python -m pytest -v`
Expected: All tests PASS

**Step 2: Generate example plot with flash_attention_tile.json**

Run: `cd plugins/tpu-perf-model/skills/tpu-pipeline-scheduler/scripts && python pipeline_ir_cli.py --pipeline examples/flash_attention_tile.json --plot --plot-output /tmp/flash_attention_vpr_timeline.png`

**Step 3: Visually verify the plot**

Open: `/tmp/flash_attention_vpr_timeline.png`
Verify:
- Top band shows DMA/MXU/VPU Gantt strips
- Main area shows VPR[0] through VPR[23] rows
- Colors reflect 3-state × 3-unit scheme (deep=write, mid=read, light=live)
- Dependency arrows connect related VPR rows
- Title shows "flash_attention_tile | 1840ns | Peak VPR: 16/32 | ..."
- Legend on right side

**Step 4: Commit any fixes**

If visual review found issues, fix and re-commit.
