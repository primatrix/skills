#!/usr/bin/env python3
"""XProf MCP Server — wraps XProf HTTP API as Claude Code tool calls.

Requires: pip install "mcp[cli]" httpx

Usage:
    # stdio mode (plugin default):
    XPROF_URL=http://localhost:8080 python3 server.py

    # SSE mode (shared K8s deployment):
    XPROF_URL=http://xprof-service:8080 MCP_TRANSPORT=sse MCP_PORT=8081 python3 server.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

XPROF_URL = os.environ.get("XPROF_URL", "http://localhost:8080")

mcp = FastMCP(
    "xprof",
    instructions=(
        "XProf profiling analysis tools. Use these to query TPU/GPU profiling "
        "data: operator breakdowns, memory profiles, A/B comparisons, and more. "
        "Runs are identified by XProf run names (from xprof_list_runs)."
    ),
)

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(base_url=XPROF_URL, timeout=60.0)
    return _client


def _fetch(path: str, params: dict[str, Any] | None = None) -> Any:
    resp = _get_client().get(path, params=params)
    resp.raise_for_status()
    return resp.json()


def _fetch_tag(run: str, tag: str, **extra_params: str) -> Any:
    params: dict[str, str] = {"run": run, "tag": tag}
    params.update(extra_params)
    return _fetch("/data/plugin/profile/data", params)


def _parse_table(data: dict | list) -> list[dict]:
    """Convert XProf's {cols, rows} Google Charts table to a list of dicts."""
    if isinstance(data, list):
        if not data:
            return []
        data = data[0]
    cols = [c["id"] for c in data.get("cols", [])]
    rows = []
    for row in data.get("rows", []):
        cells = row.get("c", [])
        rows.append({col: (cells[i].get("v") if i < len(cells) and cells[i] else None) for i, col in enumerate(cols)})
    return rows


def _parse_table_props(data: dict | list) -> dict:
    if isinstance(data, list) and data:
        data = data[0]
    return data.get("p", {})


def _flatten_op_tree(node: dict, depth: int = 0, results: list | None = None) -> list[dict]:
    """Flatten the op_profile tree into a sorted list of ops."""
    if results is None:
        results = []
    metrics = node.get("metrics", {})
    name = node.get("name", "")
    if depth >= 2 and name:
        results.append({
            "name": name,
            "self_time_us": metrics.get("rawTime", 0) / 1e6,
            "flop_utilization": metrics.get("flops", 0),
            "occurrences": metrics.get("occurrences", 0),
            "avg_time_us": metrics.get("avgTimePs", 0) / 1e6,
            "raw_flops": metrics.get("rawFlops", 0),
            "bandwidth_utils": metrics.get("bandwidthUtils", []),
        })
    for child in node.get("children", []):
        _flatten_op_tree(child, depth + 1, results)
    return results


_CATEGORY_PREFIXES = [
    ("gla/", "gla"),
    ("moe/", "moe"),
    ("megablox/", "moe"),
    ("mla/", "attention"),
    ("attention/", "attention"),
    ("rmsnorm/", "norm"),
    ("layernorm/", "norm"),
    ("embedding/", "embedding"),
    ("output_logits/", "embedding"),
    ("all_gather", "collective"),
    ("reduce_scatter", "collective"),
    ("all_reduce", "collective"),
    ("collective_permute", "collective"),
]


def _categorize_op(name: str) -> str:
    name_lower = name.lower()
    for prefix, cat in _CATEGORY_PREFIXES:
        if prefix in name_lower:
            return cat
    if "dot_general" in name_lower or "dot" in name_lower:
        return "matmul"
    return "other"


# =====================================================================
# MCP Tools
# =====================================================================


@mcp.tool()
def xprof_list_runs() -> str:
    """List all available profiling runs in XProf.

    Returns run names that can be used with other xprof_* tools.
    """
    runs = _fetch("/runs")
    return json.dumps({"runs": runs, "count": len(runs)}, indent=2)


@mcp.tool()
def xprof_hosts(run: str) -> str:
    """List TPU/GPU hosts for a specific profiling run.

    Required for trace_viewer which needs a host parameter.

    Args:
        run: XProf run name from xprof_list_runs()
    """
    hosts = _fetch("/hosts", params={"run": run})
    return json.dumps({"run": run, "hosts": hosts}, indent=2)


@mcp.tool()
def xprof_overview(run: str) -> str:
    """Get overview page: MXU utilization, device idle time, top operators by time.

    This is the first tool to call when analyzing a profiling run.

    Args:
        run: XProf run name
    """
    data = _fetch_tag(run, "overview_page")
    props = _parse_table_props(data)
    rows = _parse_table(data)

    top_ops = []
    for row in rows[:20]:
        top_ops.append({
            "operation": row.get("operation", ""),
            "category": row.get("category", ""),
            "self_time_pct": round((row.get("selfTimePercent", 0) or 0) * 100, 2),
            "cumulative_time_pct": round((row.get("cumulativeTimePercent", 0) or 0) * 100, 2),
            "flop_rate_gflops": row.get("flopRate", 0),
        })

    return json.dumps(
        {
            "run": run,
            "summary": {
                "mxu_utilization_pct": props.get("mxu_utilization_percent", "N/A"),
                "device_idle_pct": props.get("device_idle_time_percent", "N/A"),
                "device_duty_cycle_pct": props.get("device_duty_cycle_percent", "N/A"),
                "flop_rate_utilization": props.get("flop_rate_utilization_relative_to_roofline", "N/A"),
                "memory_bw_utilization": props.get("memory_bw_utilization_relative_to_hw_limit", "N/A"),
            },
            "top_ops": top_ops,
        },
        indent=2,
    )


@mcp.tool()
def xprof_op_profile(run: str, top: int = 30) -> str:
    """Get HLO-level operator breakdown with time, FLOPS, and bandwidth.

    Shows hierarchical program breakdown. For JAX-level framework ops,
    use xprof_framework_ops() instead.

    Args:
        run: XProf run name
        top: Number of top operators to return (default 30)
    """
    data = _fetch_tag(run, "op_profile")
    if isinstance(data, list) and data:
        data = data[0]
    by_program = data.get("byProgram", data.get("byCategory", {}))
    flat_ops = _flatten_op_tree(by_program)
    flat_ops.sort(key=lambda x: x["self_time_us"], reverse=True)

    for op in flat_ops[:top]:
        op["category"] = _categorize_op(op["name"])

    cat_summary: dict[str, float] = {}
    for op in flat_ops:
        cat = _categorize_op(op["name"])
        cat_summary[cat] = cat_summary.get(cat, 0) + op["self_time_us"]
    total_us = sum(cat_summary.values()) or 1
    categories = [
        {"category": cat, "total_time_us": round(t, 1), "pct": round(t / total_us * 100, 1)}
        for cat, t in sorted(cat_summary.items(), key=lambda x: -x[1])
    ]

    return json.dumps(
        {"run": run, "top_ops": flat_ops[:top], "by_category": categories, "total_ops": len(flat_ops)},
        indent=2,
    )


@mcp.tool()
def xprof_framework_ops(run: str, top: int = 30, category: str = "") -> str:
    """Get JAX framework-level operation statistics.

    Best for understanding which JAX ops dominate: shows occurrences,
    total/avg time, self-time percentages, and FLOP rates per op.
    Supports category filtering (gla, moe, attention, matmul, norm, etc.).

    Args:
        run: XProf run name
        top: Number of top operations to return (default 30)
        category: Optional comma-separated filter (e.g., 'gla,moe')
    """
    data = _fetch_tag(run, "framework_op_stats")
    rows = _parse_table(data)

    cat_filter = [c.strip() for c in category.split(",") if c.strip()] if category else []
    if cat_filter:
        rows = [r for r in rows if _categorize_op(r.get("operation", "")) in cat_filter]

    rows.sort(key=lambda r: r.get("device_total_self_time_percent", 0) or 0, reverse=True)

    ops = []
    for r in rows[:top]:
        op_name = r.get("operation", "")
        ops.append({
            "name": op_name,
            "type": r.get("type", ""),
            "category": _categorize_op(op_name),
            "occurrences": r.get("occurrences"),
            "total_time_us": r.get("total_time"),
            "avg_time_us": r.get("avg_time"),
            "total_self_time_us": r.get("total_self_time"),
            "avg_self_time_us": r.get("avg_self_time"),
            "device_self_time_pct": r.get("device_total_self_time_percent"),
            "device_cumulative_pct": r.get("device_cumulative_total_self_time_percent"),
            "flop_rate": r.get("measured_flop_rate"),
        })

    return json.dumps({"run": run, "ops": ops, "total_ops": len(rows)}, indent=2)


@mcp.tool()
def xprof_memory(run: str) -> str:
    """Get memory profile: peak HBM, capacity, fragmentation per allocator.

    Args:
        run: XProf run name
    """
    data = _fetch_tag(run, "memory_profile")
    if isinstance(data, list) and data:
        data = data[0]
    allocators = data.get("memoryProfilePerAllocator", {})
    result = {"run": run, "allocators": {}}

    def _bytes_to_gib(val: Any) -> float:
        return round(int(val or 0) / (1024**3), 2)

    for alloc_id, alloc_data in allocators.items():
        summary = alloc_data.get("profileSummary", {})
        peak_stats = summary.get("peakStats", {})
        result["allocators"][alloc_id] = {
            "memory_capacity_gib": _bytes_to_gib(summary.get("memoryCapacity")),
            "peak_usage_gib": _bytes_to_gib(summary.get("peakBytesUsageLifetime")),
            "peak_heap_allocated_gib": _bytes_to_gib(peak_stats.get("heapAllocatedBytes")),
            "peak_stack_reserved_gib": _bytes_to_gib(peak_stats.get("stackReservedBytes")),
            "peak_free_gib": _bytes_to_gib(peak_stats.get("freeMemoryBytes")),
            "fragmentation": peak_stats.get("fragmentation", 0),
        }

    return json.dumps(result, indent=2)


@mcp.tool()
def xprof_compare(run_a: str, run_b: str, top: int = 20) -> str:
    """A/B comparison of framework ops between two profiling runs.

    Computes per-operator self-time deltas, sorted by largest absolute change.
    Use to measure optimization impact.

    Args:
        run_a: Baseline run (A)
        run_b: Experiment run (B)
        top: Number of top changed ops (default 20)
    """
    data_a = _fetch_tag(run_a, "framework_op_stats")
    data_b = _fetch_tag(run_b, "framework_op_stats")

    rows_a = _parse_table(data_a)
    rows_b = _parse_table(data_b)

    ops_a = {r.get("operation", ""): r for r in rows_a}
    ops_b = {r.get("operation", ""): r for r in rows_b}
    all_ops = set(ops_a.keys()) | set(ops_b.keys())

    comparisons = []
    for op in all_ops:
        a_time = ops_a.get(op, {}).get("total_self_time") or 0
        b_time = ops_b.get(op, {}).get("total_self_time") or 0
        if a_time == 0 and b_time == 0:
            continue
        delta = b_time - a_time
        delta_pct = (delta / a_time * 100) if a_time != 0 else float("inf")
        comparisons.append({
            "operation": op,
            "category": _categorize_op(op),
            "time_a_us": a_time,
            "time_b_us": b_time,
            "delta_us": round(delta, 1),
            "delta_pct": round(delta_pct, 1),
        })

    comparisons.sort(key=lambda x: abs(x["delta_us"]), reverse=True)

    return json.dumps(
        {
            "run_a": run_a,
            "run_b": run_b,
            "top_changes": comparisons[:top],
            "total_ops_compared": len(comparisons),
        },
        indent=2,
    )


@mcp.tool()
def xprof_trace_url(run: str) -> str:
    """Get browser URLs for XProf UI views (trace, memory, overview, etc.).

    Returns clickable URLs for human inspection.

    Args:
        run: XProf run name
    """
    base = XPROF_URL
    hosts = _fetch("/hosts", params={"run": run})
    host = hosts[0]["hostname"] if hosts else None

    urls = {
        "run": run,
        "overview": f"{base}/?run={run}&tool=overview_page",
        "op_profile": f"{base}/?run={run}&tool=op_profile",
        "memory_viewer": f"{base}/?run={run}&tool=memory_viewer",
        "pod_viewer": f"{base}/?run={run}&tool=pod_viewer",
        "framework_op_stats": f"{base}/?run={run}&tool=framework_op_stats",
    }
    if host:
        urls["trace_viewer"] = f"{base}/?run={run}&tool=trace_viewer&host={host}"
    else:
        urls["trace_viewer"] = f"{base}/?run={run}&tool=trace_viewer"
    urls["host"] = host
    return json.dumps(urls, indent=2)


@mcp.tool()
def xprof_raw(run: str, tag: str, host: str = "", module_name: str = "") -> str:
    """Fetch raw data from any XProf tag endpoint.

    For tags not covered by other tools. Supports all XProf tags:
    overview_page, op_profile, framework_op_stats, kernel_stats,
    memory_profile, memory_viewer, trace_viewer, pod_viewer.

    Args:
        run: XProf run name
        tag: XProf data tag
        host: Host name (required for trace_viewer)
        module_name: Module name (for memory_viewer)
    """
    kwargs: dict[str, str] = {}
    if host:
        kwargs["host"] = host
    if module_name:
        kwargs["module_name"] = module_name
    data = _fetch_tag(run, tag, **kwargs)
    text = json.dumps(data, indent=2)
    if len(text) > 50_000:
        text = text[:50_000] + f"\n... (truncated, total {len(text)} chars)"
    return text


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        port = int(os.environ.get("MCP_PORT", "8081"))
        mcp.run(transport="sse", sse_params={"port": port})
    else:
        mcp.run(transport="stdio")
