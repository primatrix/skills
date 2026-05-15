#!/usr/bin/env python3
"""Standalone xplane.pb analyzer for Claude Code skill.

Loads .xplane.pb files into an in-memory SQLite database and supports
SQL queries and high-level overview metrics extraction.

Derived from MaxKernel/auto_agent/subagents/profiling/offline_tools.py.
"""

import argparse
import gzip
import json
import sqlite3
import sys

def load_xspace(xplane_path):
    from tensorflow.tsl.profiler.protobuf import xplane_pb2

    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
        xspace = xplane_pb2.XSpace()
        xspace.ParseFromString(f.read())
    return xspace


def build_db(xspace):
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE planes (id INTEGER, name TEXT);
        CREATE TABLE lines (id INTEGER, plane_id INTEGER, display_id INTEGER,
                           name TEXT, timestamp_ns INTEGER);
        CREATE TABLE events (plane_id INTEGER, line_id INTEGER, name TEXT,
                           offset_ps INTEGER, duration_ps INTEGER,
                           start_ps INTEGER, end_ps INTEGER);
    """)

    for plane in xspace.planes:
        c.execute("INSERT INTO planes VALUES (?, ?)", (plane.id, plane.name))
        for line in plane.lines:
            c.execute(
                "INSERT INTO lines VALUES (?, ?, ?, ?, ?)",
                (line.id, plane.id, line.display_id, line.name, line.timestamp_ns),
            )
            for event in line.events:
                meta = plane.event_metadata
                name = (
                    meta[event.metadata_id].name
                    if event.metadata_id in meta
                    else str(event.metadata_id)
                )
                start_ps = event.offset_ps
                c.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        plane.id,
                        line.id,
                        name,
                        event.offset_ps,
                        event.duration_ps,
                        start_ps,
                        start_ps + event.duration_ps,
                    ),
                )
    conn.commit()
    return conn


def cmd_overview(args):
    xspace = load_xspace(args.xplane_path)
    metrics = {}
    host_planes, device_planes = [], []

    for plane in xspace.planes:
        lower = plane.name.lower()
        if "device" in lower or "tpu" in lower or "gpu" in lower:
            device_planes.append(plane)
        else:
            host_planes.append(plane)

    metrics["device_count"] = len(device_planes)
    metrics["host_count"] = len(host_planes)

    min_start_ps, max_end_ps = float("inf"), 0
    found_events = False

    for plane in host_planes + device_planes:
        for line in plane.lines:
            for event in line.events:
                found_events = True
                start = event.offset_ps
                end = start + event.duration_ps
                min_start_ps = min(min_start_ps, start)
                max_end_ps = max(max_end_ps, end)

    if found_events:
        total_ps = max_end_ps - min_start_ps
        metrics["total_duration_ms"] = total_ps / 1e9
        metrics["total_duration_ns"] = total_ps / 1000

        if device_planes and total_ps > 0:
            busy_ps = sum(
                e.duration_ps
                for p in device_planes
                for l in p.lines
                for e in l.events
            )
            potential_ps = len(device_planes) * total_ps
            metrics["device_duty_cycle_percent"] = round(
                (busy_ps / potential_ps) * 100, 2
            )
    else:
        metrics["total_duration_ms"] = 0

    step_durations = []
    for plane in host_planes + device_planes:
        for line in plane.lines:
            if "steps" in line.name.lower():
                for event in line.events:
                    step_durations.append(event.duration_ps)

    if step_durations:
        metrics["average_step_time_ms"] = round(
            sum(step_durations) / len(step_durations) / 1e9, 4
        )
        metrics["step_count"] = len(step_durations)

    print(json.dumps(metrics, indent=2))


def cmd_query(args):
    import pandas as pd

    xspace = load_xspace(args.xplane_path)
    conn = build_db(xspace)
    try:
        df = pd.read_sql_query(args.sql, conn)
        print(df.to_markdown(index=False))
    except Exception as e:
        print(f"SQL error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def cmd_schema(_args):
    print(
        """Tables:

planes:
  - id INTEGER
  - name TEXT

lines:
  - id INTEGER
  - plane_id INTEGER  (FK -> planes.id)
  - display_id INTEGER
  - name TEXT
  - timestamp_ns INTEGER

events:
  - plane_id INTEGER  (FK -> planes.id)
  - line_id INTEGER   (FK -> lines.id)
  - name TEXT          (resolved from plane.event_metadata)
  - offset_ps INTEGER (offset from line start, picoseconds)
  - duration_ps INTEGER
  - start_ps INTEGER  (= offset_ps)
  - end_ps INTEGER    (= start_ps + duration_ps)

Note: 1 ms = 1e9 ps, 1 us = 1e6 ps, 1 ns = 1e3 ps"""
    )


def main():
    parser = argparse.ArgumentParser(description="XPlane profile analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_overview = sub.add_parser("overview", help="Show high-level metrics")
    p_overview.add_argument("xplane_path", help="Path to .xplane.pb file")
    p_overview.set_defaults(func=cmd_overview)

    p_query = sub.add_parser("query", help="Run SQL query on xplane data")
    p_query.add_argument("xplane_path", help="Path to .xplane.pb file")
    p_query.add_argument("sql", help="SQL query to execute")
    p_query.set_defaults(func=cmd_query)

    p_schema = sub.add_parser("schema", help="Print database schema")
    p_schema.set_defaults(func=cmd_schema)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
