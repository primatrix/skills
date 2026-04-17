#!/usr/bin/env python3
"""Generate a self-contained HTML file with interactive pipeline animation."""

from __future__ import annotations

import json
from pipeline_ir import PipelineOp
from pipeline_scheduler import ScheduleResult
from pipeline_plot import build_vpr_activity
from dependency_analyzer import DependencyGraph


def generate_animation(
    ops: list[PipelineOp],
    sched: ScheduleResult,
    graph: DependencyGraph,
    output_path: str,
    title: str = "",
) -> None:
    """Write a self-contained HTML animation of the pipeline schedule."""
    data = _build_data(ops, sched, graph, title)
    html = _HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(data))
    with open(output_path, "w") as f:
        f.write(html)


def _build_data(
    ops: list[PipelineOp],
    sched: ScheduleResult,
    graph: DependencyGraph,
    title: str,
) -> dict:
    entries = []
    for e in sched.entries:
        phases = [
            {"phase_type": p.phase_type, "start_ns": p.start_ns,
             "end_ns": p.end_ns, "unit_slot": p.unit_slot}
            for p in e.phases
        ]
        entries.append({
            "op_id": e.op_id, "start_ns": e.start_ns, "end_ns": e.end_ns,
            "unit": e.unit, "stall_ns": e.stall_ns,
            "wait_reason": e.wait_reason, "phases": phases,
        })

    op_info = {}
    for op in ops:
        op_info[op.op_id] = {
            "op_kind": op.op_kind, "unit": op.unit,
            "label": op.label or op.op_id,
            "pseudocode": op.pseudocode,
            "input_vprs": op.input_vprs, "output_vprs": op.output_vprs,
            "weight_vprs": op.weight_vprs, "data_vprs": op.data_vprs,
            "latency_ns": op.latency_ns,
        }

    edges = [
        {"from_op": e.from_op, "to_op": e.to_op,
         "hazard_type": e.hazard_type, "resource_id": e.resource_id}
        for e in graph.edges
    ]

    vpr_activity: dict[str, list[dict]] = {}
    if ops and sched.entries:
        activity = build_vpr_activity(ops, sched)
        for vpr_id, intervals in activity.items():
            vpr_activity[str(vpr_id)] = [
                {"start_ns": iv.start_ns, "end_ns": iv.end_ns,
                 "unit": iv.unit, "access": iv.access, "op_id": iv.op_id}
                for iv in intervals
            ]

    used_vprs = sorted(int(k) for k in vpr_activity.keys()) if vpr_activity else []

    # Compute peak VPR from activity
    peak_vpr = 0
    if vpr_activity:
        events: list[tuple[float, int]] = []
        for intervals in vpr_activity.values():
            writes = [iv for iv in intervals if iv["access"] == "write"]
            if writes:
                start = min(iv["start_ns"] for iv in writes)
                end = max(iv["end_ns"] for iv in intervals)
                events.append((start, 1))
                events.append((end, -1))
        events.sort(key=lambda x: (x[0], -x[1]))
        cur = 0
        for _, delta in events:
            cur += delta
            peak_vpr = max(peak_vpr, cur)

    return {
        "title": title,
        "total_latency_ns": sched.total_latency_ns,
        "stall_total_ns": sched.stall_total_ns,
        "peak_vpr": peak_vpr,
        "entries": entries,
        "op_info": op_info,
        "edges": edges,
        "vpr_activity": vpr_activity,
        "used_vprs": used_vprs,
        "fusion_pairs": list(sched.fusion_pairs),
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline Animation</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0; }
#header { padding: 10px 20px; background: #16213e; border-bottom: 1px solid #333;
           font-size: 14px; display: flex; gap: 20px; align-items: center; }
#header .title { font-weight: bold; font-size: 16px; color: #fff; }
#header .stat { color: #aaa; }
#main { display: grid; grid-template-columns: 1fr 280px;
        height: calc(100vh - 90px); }
#gantt-area { overflow: auto; padding: 10px; }
#pseudocode { background: #0f3460; border-left: 1px solid #333;
              padding: 15px; overflow-y: auto; font-family: 'Courier New', monospace;
              font-size: 13px; line-height: 1.8; }
#pseudocode .line { padding: 2px 8px; border-radius: 3px; transition: background 0.2s; }
#pseudocode .line.active { background: #e74c3c44; color: #fff; font-weight: bold; }
#pseudocode .section-title { color: #5dade2; font-weight: bold;
                              margin-top: 10px; margin-bottom: 4px; }
#controls { display: flex; align-items: center; gap: 15px; padding: 8px 20px;
            background: #16213e; border-top: 1px solid #333; height: 50px; }
#controls button { background: #2980b9; color: #fff; border: none;
                   padding: 6px 16px; border-radius: 4px; cursor: pointer;
                   font-size: 13px; }
#controls button:hover { background: #3498db; }
#scrubber { flex: 1; }
#scrubber input[type=range] { width: 100%; accent-color: #e74c3c; }
#time-display { font-family: monospace; font-size: 13px; min-width: 100px; }
#speed-select { background: #0f3460; color: #e0e0e0; border: 1px solid #555;
                padding: 4px 8px; border-radius: 4px; font-size: 13px; }
.tooltip { position: absolute; background: #16213e; border: 1px solid #555;
           padding: 8px 12px; border-radius: 4px; font-size: 12px;
           pointer-events: none; z-index: 100; max-width: 300px; }
svg text { user-select: none; }
</style>
</head>
<body>
<div id="header">
  <span class="title" id="hdr-title">Pipeline</span>
  <span class="stat" id="hdr-latency"></span>
  <span class="stat" id="hdr-vpr"></span>
  <span class="stat" id="hdr-stall"></span>
</div>
<div id="main">
  <div id="gantt-area"><svg id="svg-canvas"></svg></div>
  <div id="pseudocode" id="pseudo-panel"></div>
</div>
<div id="controls">
  <button id="btn-play">Play</button>
  <div id="scrubber"><input type="range" id="slider" min="0" max="1000" value="0"></div>
  <span id="time-display">t=0ns</span>
  <label>Speed:
    <select id="speed-select">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
  </label>
</div>
<div class="tooltip" id="tooltip" style="display:none"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;

(function() {
  const UNIT_COLORS = {DMA:"#2980b9", MXU_W:"#e74c3c", MXU_D:"#922b21", VPU:"#27ae60"};
  const VPR_COLORS = {
    DMA_write:"#1a5276", DMA_read:"#5dade2", DMA_live:"#d4e6f1",
    MXU_write:"#922b21", MXU_read:"#e74c3c", MXU_live:"#f5b7b1",
    VPU_write:"#196f3d", VPU_read:"#27ae60", VPU_live:"#d5f5e3",
  };

  const totalNs = DATA.total_latency_ns || 1;
  const usedVprs = DATA.used_vprs || [];
  const entries = DATA.entries || [];
  const opInfo = DATA.op_info || {};
  const vprAct = DATA.vpr_activity || {};
  const unitOrder = ["DMA","MXU_W","MXU_D","VPU"];

  // Layout constants
  const LM = 70, RM = 20, TM = 10;
  const ganttH = 30, ganttGap = 4, ganttTop = TM;
  const ganttTotalH = unitOrder.length * (ganttH + ganttGap);
  const heatTop = ganttTop + ganttTotalH + 20;
  const cellH = 22, cellGap = 2;
  const heatTotalH = usedVprs.length * (cellH + cellGap);
  const svgW = 900, chartW = svgW - LM - RM;
  const svgH = heatTop + heatTotalH + 40;

  // Header
  document.getElementById("hdr-title").textContent = DATA.title || "Pipeline";
  document.getElementById("hdr-latency").textContent = "Latency: " + totalNs.toFixed(0) + "ns";
  document.getElementById("hdr-vpr").textContent = "Peak VPR: " + (DATA.peak_vpr||0) + "/32";
  document.getElementById("hdr-stall").textContent = "Stall: " + (DATA.stall_total_ns||0).toFixed(0) + "ns";

  // Pseudocode panel
  const pseudoEl = document.getElementById("pseudocode");
  const pseudoLines = [];
  entries.forEach(e => {
    const info = opInfo[e.op_id];
    if (!info) return;
    const text = info.pseudocode || info.label || e.op_id;
    const div = document.createElement("div");
    div.className = "line";
    div.textContent = text;
    div.dataset.opId = e.op_id;
    pseudoEl.appendChild(div);
    pseudoLines.push({el: div, opId: e.op_id, start: e.start_ns, end: e.end_ns});
  });

  // SVG setup
  const svg = document.getElementById("svg-canvas");
  svg.setAttribute("width", svgW);
  svg.setAttribute("height", svgH);
  svg.setAttribute("viewBox", "0 0 " + svgW + " " + svgH);
  const NS = "http://www.w3.org/2000/svg";

  function nsToX(ns) { return LM + (ns / totalNs) * chartW; }

  // Draw unit labels
  unitOrder.forEach((u, i) => {
    const y = ganttTop + i * (ganttH + ganttGap) + ganttH / 2;
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", LM - 8); t.setAttribute("y", y + 4);
    t.setAttribute("text-anchor", "end"); t.setAttribute("fill", "#aaa");
    t.setAttribute("font-size", "11"); t.textContent = u;
    svg.appendChild(t);
  });

  // Draw VPR labels
  usedVprs.forEach((v, i) => {
    const y = heatTop + i * (cellH + cellGap) + cellH / 2;
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", LM - 8); t.setAttribute("y", y + 4);
    t.setAttribute("text-anchor", "end"); t.setAttribute("fill", "#aaa");
    t.setAttribute("font-size", "10"); t.setAttribute("font-family", "monospace");
    t.textContent = "VPR[" + v + "]";
    svg.appendChild(t);
  });

  // Separator line
  const sep = document.createElementNS(NS, "line");
  sep.setAttribute("x1", LM); sep.setAttribute("x2", svgW - RM);
  sep.setAttribute("y1", heatTop - 10); sep.setAttribute("y2", heatTop - 10);
  sep.setAttribute("stroke", "#444"); sep.setAttribute("stroke-width", "1");
  svg.appendChild(sep);

  // Time axis
  const axisY = heatTop + heatTotalH + 10;
  const axLine = document.createElementNS(NS, "line");
  axLine.setAttribute("x1", LM); axLine.setAttribute("x2", svgW - RM);
  axLine.setAttribute("y1", axisY); axLine.setAttribute("y2", axisY);
  axLine.setAttribute("stroke", "#555"); axLine.setAttribute("stroke-width", "1");
  svg.appendChild(axLine);
  // Tick marks
  const nTicks = Math.min(10, Math.max(2, Math.floor(chartW / 80)));
  for (let i = 0; i <= nTicks; i++) {
    const ns = (totalNs / nTicks) * i;
    const x = nsToX(ns);
    const tick = document.createElementNS(NS, "line");
    tick.setAttribute("x1", x); tick.setAttribute("x2", x);
    tick.setAttribute("y1", axisY); tick.setAttribute("y2", axisY + 5);
    tick.setAttribute("stroke", "#555"); svg.appendChild(tick);
    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", x); lbl.setAttribute("y", axisY + 16);
    lbl.setAttribute("text-anchor", "middle"); lbl.setAttribute("fill", "#888");
    lbl.setAttribute("font-size", "9"); lbl.textContent = ns.toFixed(0) + "ns";
    svg.appendChild(lbl);
  }

  // Create gantt bar elements (initially hidden via clip)
  const ganttBars = [];
  entries.forEach(e => {
    const info = opInfo[e.op_id];
    if (e.unit === "MXU" && e.phases && e.phases.length > 0) {
      e.phases.forEach(ph => {
        const uIdx = unitOrder.indexOf(ph.unit_slot);
        if (uIdx < 0) return;
        const bar = _makeBar(ph.start_ns, ph.end_ns, uIdx, UNIT_COLORS[ph.unit_slot],
                             e.op_id, ph.phase_type);
        ganttBars.push(bar);
      });
    } else {
      let slot = e.unit === "MXU" ? "MXU_W" : e.unit;
      const uIdx = unitOrder.indexOf(slot);
      if (uIdx < 0) return;
      const bar = _makeBar(e.start_ns, e.end_ns, uIdx, UNIT_COLORS[slot],
                           e.op_id, "");
      ganttBars.push(bar);
    }
  });

  function _makeBar(startNs, endNs, uIdx, color, opId, phaseType) {
    const y = ganttTop + uIdx * (ganttH + ganttGap);
    const x = nsToX(startNs);
    const w = Math.max(1, (endNs - startNs) / totalNs * chartW);
    const g = document.createElementNS(NS, "g");
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", w); rect.setAttribute("height", ganttH);
    rect.setAttribute("fill", color); rect.setAttribute("opacity", "0.15");
    rect.setAttribute("rx", "3");
    g.appendChild(rect);
    // Label
    if (w > 40) {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x + w/2); t.setAttribute("y", y + ganttH/2 + 4);
      t.setAttribute("text-anchor", "middle"); t.setAttribute("fill", "#fff");
      t.setAttribute("font-size", "9"); t.setAttribute("font-weight", "bold");
      t.setAttribute("opacity", "0.15");
      t.textContent = opId + (phaseType ? " (" + phaseType + ")" : "");
      g.appendChild(t);
    }
    svg.appendChild(g);
    // Tooltip
    rect.addEventListener("mouseenter", ev => showTooltip(ev,
      opId + (phaseType ? " [" + phaseType + "]" : "") +
      "\n" + startNs.toFixed(0) + "ns - " + endNs.toFixed(0) + "ns"));
    rect.addEventListener("mouseleave", hideTooltip);
    return {g, rect, texts: g.querySelectorAll("text"),
            startNs, endNs, color, opId};
  }

  // Create VPR heatmap cells
  const vprCells = [];
  const vprRowMap = {};
  usedVprs.forEach((v, i) => { vprRowMap[v] = i; });

  Object.keys(vprAct).forEach(vprIdStr => {
    const vprId = parseInt(vprIdStr);
    const rowIdx = vprRowMap[vprId];
    if (rowIdx === undefined) return;
    const intervals = vprAct[vprIdStr];
    intervals.forEach(iv => {
      const colorKey = iv.unit + "_" + iv.access;
      const color = VPR_COLORS[colorKey] || "#555";
      const y = heatTop + rowIdx * (cellH + cellGap);
      const x = nsToX(iv.start_ns);
      const w = Math.max(1, (iv.end_ns - iv.start_ns) / totalNs * chartW);
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", x); rect.setAttribute("y", y);
      rect.setAttribute("width", w); rect.setAttribute("height", cellH);
      rect.setAttribute("fill", color); rect.setAttribute("opacity", "0.15");
      rect.setAttribute("rx", "2");
      svg.appendChild(rect);
      rect.addEventListener("mouseenter", ev => showTooltip(ev,
        "VPR[" + vprId + "] " + iv.access +
        (iv.op_id ? " by " + iv.op_id : "") +
        "\n" + iv.start_ns.toFixed(0) + "ns - " + iv.end_ns.toFixed(0) + "ns"));
      rect.addEventListener("mouseleave", hideTooltip);
      vprCells.push({rect, startNs: iv.start_ns, endNs: iv.end_ns, color});
    });
  });

  // Time cursor
  const cursor = document.createElementNS(NS, "line");
  cursor.setAttribute("x1", LM); cursor.setAttribute("x2", LM);
  cursor.setAttribute("y1", TM); cursor.setAttribute("y2", axisY);
  cursor.setAttribute("stroke", "#e74c3c"); cursor.setAttribute("stroke-width", "1.5");
  cursor.setAttribute("opacity", "0.8");
  svg.appendChild(cursor);

  // Tooltip
  const tooltipEl = document.getElementById("tooltip");
  function showTooltip(ev, text) {
    tooltipEl.style.display = "block";
    tooltipEl.textContent = text;
    tooltipEl.style.left = (ev.clientX + 12) + "px";
    tooltipEl.style.top = (ev.clientY - 10) + "px";
  }
  function hideTooltip() { tooltipEl.style.display = "none"; }

  // Animation state
  let playing = false;
  let currentNs = 0;
  let speed = 1;
  let lastFrameTime = 0;
  // Scale: 1 real-second = totalNs * speed (full playback in ~1s at 1x, adjust)
  const playDurationMs = 3000; // full playback takes 3s at 1x speed

  const btnPlay = document.getElementById("btn-play");
  const slider = document.getElementById("slider");
  const timeDisp = document.getElementById("time-display");
  const speedSel = document.getElementById("speed-select");

  btnPlay.addEventListener("click", () => {
    playing = !playing;
    btnPlay.textContent = playing ? "Pause" : "Play";
    if (playing) { lastFrameTime = performance.now(); requestAnimationFrame(tick); }
  });

  slider.addEventListener("input", () => {
    currentNs = (slider.value / 1000) * totalNs;
    render(currentNs);
  });

  speedSel.addEventListener("change", () => {
    speed = parseFloat(speedSel.value);
  });

  function tick(ts) {
    if (!playing) return;
    const dt = ts - lastFrameTime;
    lastFrameTime = ts;
    currentNs += (dt / playDurationMs) * totalNs * speed;
    if (currentNs > totalNs) { currentNs = totalNs; playing = false; btnPlay.textContent = "Play"; }
    slider.value = (currentNs / totalNs) * 1000;
    render(currentNs);
    if (playing) requestAnimationFrame(tick);
  }

  function render(ns) {
    timeDisp.textContent = "t=" + ns.toFixed(0) + "ns";
    // Cursor
    const cx = nsToX(ns);
    cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx);

    // Gantt bars: reveal up to current time
    ganttBars.forEach(b => {
      if (ns >= b.endNs) {
        b.rect.setAttribute("opacity", "0.85");
        b.texts.forEach(t => t.setAttribute("opacity", "1"));
      } else if (ns >= b.startNs) {
        const frac = (ns - b.startNs) / (b.endNs - b.startNs);
        b.rect.setAttribute("opacity", (0.3 + 0.55 * frac).toFixed(2));
        b.texts.forEach(t => t.setAttribute("opacity", frac.toFixed(2)));
      } else {
        b.rect.setAttribute("opacity", "0.15");
        b.texts.forEach(t => t.setAttribute("opacity", "0.15"));
      }
    });

    // VPR cells
    vprCells.forEach(c => {
      if (ns >= c.endNs) {
        c.rect.setAttribute("opacity", "0.7");
      } else if (ns >= c.startNs) {
        const frac = (ns - c.startNs) / (c.endNs - c.startNs);
        c.rect.setAttribute("opacity", (0.2 + 0.5 * frac).toFixed(2));
      } else {
        c.rect.setAttribute("opacity", "0.15");
      }
    });

    // Pseudocode highlighting
    pseudoLines.forEach(p => {
      if (ns >= p.start && ns < p.end) {
        p.el.classList.add("active");
      } else {
        p.el.classList.remove("active");
      }
    });
  }

  // Initial render
  render(0);
})();
</script>
</body>
</html>"""
