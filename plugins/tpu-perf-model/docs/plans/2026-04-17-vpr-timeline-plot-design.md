# VPR Timeline Plot Design

## Goal

Add matplotlib-based visualization to tpu-pipeline-scheduler: a 2D heatmap with X=time, Y=VPR register, cells colored by hardware unit and access type, with dependency arrows overlaid.

## Color Scheme (3-state × 3-unit)

| State | DMA | MXU | VPU |
|-------|-----|-----|-----|
| write (deep) | #1a5276 | #922b21 | #196f3d |
| read (mid) | #5dade2 | #e74c3c | #27ae60 |
| live (light) | #d4e6f1 | #f5b7b1 | #d5f5e3 |

Priority: write > read > live > empty.

## Dependency Arrows

Arc arrows between producer/consumer VPR rows using FancyArrowPatch. RAW=solid, WAR=dashed, WAW=dotted.

## Layout

- Top band: Gantt strips (DMA/MXU/VPU) sharing X-axis with heatmap
- Main area: VPR timeline heatmap
- Right: legend (color matrix + arrow types)
- Title: kernel name, hw, peak VPR, total latency

## File Changes

| File | Change |
|------|--------|
| `scripts/pipeline_plot.py` | New — matplotlib rendering |
| `scripts/pipeline_ir_cli.py` | Add `--plot` / `--plot-output` flags |
| `SKILL.md` | Document `--plot` usage |

## CLI

```bash
python pipeline_ir_cli.py --pipeline kernel.json --plot
python pipeline_ir_cli.py --pipeline kernel.json --plot --plot-output out.png
```
