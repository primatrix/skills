# tpu-perf `memory-profile` Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth skill `memory-profile` under `plugins/tpu-perf/` that reads `xplane.pb`, locates HBM peak occupancy within a chosen step, and emits one JSON object describing the peak moment, every buffer alive at that moment (with shape/tf_op/parent-jit/lifetime attribution), rollups, a sampled timeline, and diagnostics.

**Architecture:** Single CLI entry `memory_profile.py` (no `--mode`) with helpers in `_loader.py`. Reuse `profile-anatomy`'s `xplane_pb2` via `sys.path.insert`. Two-pass sweep: pass 1 over the full trace to find global peak and produce timeline samples; pass 2 inside the chosen step window to snapshot the live buffer set. Tests use stdlib `unittest`: synthetic XSpace builders for unit tests plus the real `dp8_fsdp128` fixture for end-to-end and invariant tests.

**Tech Stack:** Python 3.10+ (`from __future__ import annotations`, dataclasses with `slots=True`), stdlib `argparse`/`json`/`unittest`, `protobuf` runtime, vendored `xplane_pb2` from `profile-anatomy`.

**Spec:** [`docs/superpowers/specs/2026-05-25-tpu-perf-memory-profile-design.md`](../specs/2026-05-25-tpu-perf-memory-profile-design.md).

**Fixture path:** `/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/` — must contain `gke-tpu-4233cc6e-d8q7.xplane.pb` (1× xplane.pb suffices).

---

## File map

Create:
- `plugins/tpu-perf/skills/memory-profile/SKILL.md`
- `plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/__init__.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/_synthetic.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_cli.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_loader.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_step_selection.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_alive_set.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_rollups.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_snapshot.py`
- `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_invariants.py`

Modify:
- `plugins/tpu-perf/.claude-plugin/plugin.json` — bump `version` and refresh `description` to mention the new skill.

No `_proto/` directory under this skill — proto module is reused from `profile-anatomy/scripts/_proto/` via `sys.path.insert`.

---

## Task 1: Skill scaffold + `--help` smoke test

**Files:**
- Create: `plugins/tpu-perf/skills/memory-profile/SKILL.md`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/__init__.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_cli.py`

- [ ] **Step 1: Create empty `__init__.py` for tests package**

```bash
mkdir -p plugins/tpu-perf/skills/memory-profile/scripts/tests
: > plugins/tpu-perf/skills/memory-profile/scripts/tests/__init__.py
```

- [ ] **Step 2: Write the failing CLI tests**

Create `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_cli.py`:

```python
"""Smoke tests for the memory_profile.py CLI surface."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "memory_profile.py"


class TestCLI(unittest.TestCase):
    def test_help_runs(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--step", r.stdout)
        self.assertIn("--step-policy", r.stdout)
        self.assertIn("--all-trace", r.stdout)
        self.assertIn("--top", r.stdout)
        self.assertIn("--persistent-threshold-steps", r.stdout)
        self.assertIn("--include-host-pools", r.stdout)
        self.assertIn("--time-samples", r.stdout)

    def test_no_xplane_returns_absent(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        doc = json.loads(r.stdout)
        self.assertEqual(doc["status"], "absent")
        self.assertEqual(doc["reason"], "no_xplane_pb")
        self.assertEqual(doc["skill"], "memory-profile")
        self.assertEqual(doc["version"], 1)
        self.assertEqual(doc["inputs"]["profile_dir"], "/tmp")

    def test_step_and_all_trace_mutually_exclusive(self):
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "/tmp", "--step", "0", "--all-trace"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not allowed", r.stderr.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_cli -v
```
Expected: every test errors out (script does not exist).

- [ ] **Step 4: Create the minimal CLI scaffold**

Create `plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py`:

```python
"""
Entry point for tpu-perf:memory-profile.

Locates HBM peak occupancy within a chosen step (or across the full
trace) and emits one JSON object on stdout describing the peak moment
plus every buffer alive at that moment, rollups, timeline samples, and
diagnostics. Single-mode skill — no --mode flag. See spec
docs/superpowers/specs/2026-05-25-tpu-perf-memory-profile-design.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


SKILL_NAME = "memory-profile"
SCHEMA_VERSION = 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_profile.py",
        description=(
            "Locate HBM peak occupancy within a step and list every buffer "
            "alive at that moment."
        ),
    )
    p.add_argument("profile_dir", help="Profile directory containing *.xplane.pb")

    step_group = p.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step", type=int, default=None,
        help="Explicit step index on /device:TPU:0 'Steps' line (0-based).",
    )
    step_group.add_argument(
        "--all-trace", action="store_true",
        help="Disable step scoping; analyse the full trace window.",
    )

    p.add_argument(
        "--step-policy", choices=("peak", "last", "first"), default="peak",
        help="Default step picker when --step is not given (default: peak).",
    )
    p.add_argument("--top", type=int, default=30,
                   help="Top-K applied to alive_at_peak.buffers and rollup tables.")
    p.add_argument("--persistent-threshold-steps", type=int, default=2,
                   help="Min crossed step boundaries for lifetime_class=persistent.")
    p.add_argument("--include-host-pools", action="store_true",
                   help="Include allocator pools other than HBM (id != 0).")
    p.add_argument("--time-samples", type=int, default=200,
                   help="Number of equally-spaced timeline samples.")
    return p


def _emit_absent(profile_dir: str, reason: str, **extra) -> dict:
    return {
        "status": "absent",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "reason": reason,
        "inputs": {"profile_dir": profile_dir, **extra},
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile_dir = args.profile_dir
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        json.dump(_emit_absent(profile_dir, "no_xplane_pb"), sys.stdout)
        sys.stdout.write("\n")
        return 0
    # Wired up in Task 6.
    json.dump(_emit_absent(profile_dir, "not_implemented"), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the CLI tests and confirm they pass**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_cli -v
```
Expected: 3 passed.

- [ ] **Step 6: Stub `SKILL.md` so future skill-discovery sees the new skill**

Create `plugins/tpu-perf/skills/memory-profile/SKILL.md`:

```markdown
---
name: memory-profile
description: Use when analyzing TPU pretraining HBM peak occupancy from xplane.pb — locates the peak moment within a step, lists every live buffer at that moment with shape/tf_op/parent-jit attribution, and rolls up by lifetime-class / shape / tf_op / dtype. Reads schema documented by profile-anatomy.
argument-hint: "<profile_dir> [--step N | --step-policy peak|last|first] [--top K]"
---

# Memory Profile

Stub — full content written in Task 12.
```

- [ ] **Step 7: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile
git commit -m "feat(tpu-perf): scaffold memory-profile skill with --help smoke test"
```

---

## Task 2: `_loader.py` — load XSpace and walk allocator events

**Files:**
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/_synthetic.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_loader.py`

- [ ] **Step 1: Write the synthetic XSpace builder helper**

Create `plugins/tpu-perf/skills/memory-profile/scripts/tests/_synthetic.py`:

```python
"""Builders for synthetic xplane_pb2.XSpace fixtures used in unit tests.

Mirrors the helper pattern in compute-breakdown/tests but tailored to
allocator events on /host:CPU."""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROFILE_ANATOMY_PROTO = (
    _HERE.parent.parent.parent / "profile-anatomy" / "scripts" / "_proto"
)
sys.path.insert(0, str(_PROFILE_ANATOMY_PROTO))
import xplane_pb2  # noqa: E402


# id allocators: we manage stat/event metadata ids per-plane, never globally.
class PlaneBuilder:
    def __init__(self, xs: xplane_pb2.XSpace, name: str):
        self.plane = xs.planes.add()
        self.plane.name = name
        self._next_stat_id = 1
        self._next_event_id = 1
        self._stat_ids: dict[str, int] = {}
        self._event_ids: dict[str, int] = {}

    def stat_id(self, name: str) -> int:
        if name not in self._stat_ids:
            sm = self.plane.stat_metadata[self._next_stat_id]
            sm.id = self._next_stat_id
            sm.name = name
            self._stat_ids[name] = self._next_stat_id
            self._next_stat_id += 1
        return self._stat_ids[name]

    def event_id(self, name: str) -> int:
        if name not in self._event_ids:
            em = self.plane.event_metadata[self._next_event_id]
            em.id = self._next_event_id
            em.name = name
            self._event_ids[name] = self._next_event_id
            self._next_event_id += 1
        return self._event_ids[name]

    def add_line(self, name: str, timestamp_ns: int = 0) -> "LineBuilder":
        return LineBuilder(self, name, timestamp_ns)


class LineBuilder:
    def __init__(self, pb: PlaneBuilder, name: str, timestamp_ns: int):
        self.pb = pb
        self.line = pb.plane.lines.add()
        self.line.name = name
        self.line.timestamp_ns = timestamp_ns

    def add_event(self, name: str, *, offset_ps: int, duration_ps: int = 0,
                  stats: dict[str, int | float | str | bytes] | None = None) -> None:
        ev = self.line.events.add()
        ev.metadata_id = self.pb.event_id(name)
        ev.offset_ps = offset_ps
        ev.duration_ps = duration_ps
        for sname, val in (stats or {}).items():
            st = ev.stats.add()
            st.metadata_id = self.pb.stat_id(sname)
            if isinstance(val, bool):  # bool is int subclass; skip
                raise TypeError("bool stats not supported by xplane")
            elif isinstance(val, int):
                st.int64_value = val
            elif isinstance(val, float):
                st.double_value = val
            elif isinstance(val, str):
                st.str_value = val
            elif isinstance(val, bytes):
                st.bytes_value = val
            else:
                raise TypeError(f"unsupported stat type for {sname}: {type(val)}")


def make_alloc_event(line: LineBuilder, *, offset_ps: int, addr: int,
                     requested: int, allocation: int, pool_id: int = 0,
                     bytes_allocated: int, peak_bytes_in_use: int,
                     bytes_reserved: int, bytes_available: int = 0,
                     fragmentation: float = 0.0,
                     shape: str = "", tf_op: str = "",
                     data_type: int = 0) -> None:
    line.add_event(
        "MemoryAllocation", offset_ps=offset_ps,
        stats={
            "addr": addr, "id": pool_id,
            "requested_bytes": requested, "allocation_bytes": allocation,
            "bytes_allocated": bytes_allocated,
            "peak_bytes_in_use": peak_bytes_in_use,
            "bytes_reserved": bytes_reserved,
            "bytes_available": bytes_available,
            "fragmentation": fragmentation,
            "shape": shape, "tf_op": tf_op, "data_type": data_type,
        },
    )


def make_dealloc_event(line: LineBuilder, *, offset_ps: int, addr: int,
                       bytes_allocated: int, peak_bytes_in_use: int,
                       bytes_reserved: int, bytes_available: int = 0,
                       fragmentation: float = 0.0) -> None:
    line.add_event(
        "MemoryDeallocation", offset_ps=offset_ps,
        stats={
            "addr": addr,
            "bytes_allocated": bytes_allocated,
            "peak_bytes_in_use": peak_bytes_in_use,
            "bytes_reserved": bytes_reserved,
            "bytes_available": bytes_available,
            "fragmentation": fragmentation,
        },
    )


def make_xspace() -> xplane_pb2.XSpace:
    return xplane_pb2.XSpace()
```

- [ ] **Step 2: Write the failing loader tests**

Create `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_loader.py`:

```python
"""Unit tests for the loader: alloc/dealloc extraction and parent-chain walk."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import load_host_allocator_events  # noqa: E402

from tests._synthetic import (  # noqa: E402
    PlaneBuilder, make_alloc_event, make_dealloc_event, make_xspace,
)


class TestLoader(unittest.TestCase):
    def test_returns_absent_when_host_plane_missing(self):
        xs = make_xspace()
        PlaneBuilder(xs, "/device:TPU:0")
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(events)
        self.assertEqual(reason, "host_plane_absent")

    def test_returns_absent_when_no_alloc_events(self):
        xs = make_xspace()
        PlaneBuilder(xs, "/host:CPU")
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(events)
        self.assertEqual(reason, "no_memory_events")

    def test_extracts_alloc_and_dealloc_with_correct_ts(self):
        xs = make_xspace()
        host = PlaneBuilder(xs, "/host:CPU")
        line = host.add_line("pjrt_tpu_execute/0", timestamp_ns=1_000)
        # offset_ps 2_000_000 → 2_000 ns added to line.timestamp_ns 1_000 → ts_ns 3_000.
        make_alloc_event(
            line, offset_ps=2_000_000, addr=0x1000,
            requested=4096, allocation=4096,
            bytes_allocated=4096, peak_bytes_in_use=4096,
            bytes_reserved=10_000, shape="bf16[128]", tf_op="dot",
        )
        make_dealloc_event(
            line, offset_ps=5_000_000, addr=0x1000,
            bytes_allocated=0, peak_bytes_in_use=4096,
            bytes_reserved=10_000,
        )
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(reason)
        self.assertIsNotNone(events)
        self.assertTrue(events.host_plane_present)
        self.assertEqual(len(events.allocs), 1)
        self.assertEqual(len(events.deallocs), 1)
        a = events.allocs[0]
        self.assertEqual(a.ts_ns, 3_000)
        self.assertEqual(a.addr, 0x1000)
        self.assertEqual(a.pool_id, 0)
        self.assertEqual(a.requested_bytes, 4096)
        self.assertEqual(a.shape, "bf16[128]")
        self.assertEqual(a.tf_op, "dot")
        self.assertEqual(a.line_name, "pjrt_tpu_execute/0")
        d = events.deallocs[0]
        self.assertEqual(d.ts_ns, 6_000)
        self.assertEqual(d.addr, 0x1000)
        self.assertEqual(events.pool_capacity, {0: 10_000})

    def test_parent_chain_built_from_time_containment(self):
        xs = make_xspace()
        host = PlaneBuilder(xs, "/host:CPU")
        line = host.add_line("pjrt_tpu_execute/0", timestamp_ns=0)
        # Outer event covers [0ps, 100_000_000ps]; alloc at offset 5_000_000ps.
        line.add_event("[0] Execute (jit_train_step)",
                       offset_ps=0, duration_ps=100_000_000)
        line.add_event("AllocateOutputBuffersWithInputReuse",
                       offset_ps=1_000_000, duration_ps=10_000_000)
        make_alloc_event(
            line, offset_ps=5_000_000, addr=0x2000,
            requested=128, allocation=128,
            bytes_allocated=128, peak_bytes_in_use=128, bytes_reserved=10_000,
        )
        events, reason = load_host_allocator_events(xs)
        self.assertIsNone(reason)
        chain = events.allocs[0].parent_chain
        self.assertIn("[0] Execute (jit_train_step)", chain)
        self.assertIn("AllocateOutputBuffersWithInputReuse", chain)
        self.assertEqual(chain[0], "[0] Execute (jit_train_step)")
        self.assertEqual(chain[1], "AllocateOutputBuffersWithInputReuse")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests and confirm they fail**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_loader -v
```
Expected: 4 errors — `_loader` module not yet importable.

- [ ] **Step 4: Implement `_loader.py`**

Create `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`:

```python
"""
Loader helpers for tpu-perf:memory-profile.

Reads an xplane.pb file, locates the /host:CPU plane, and turns its
MemoryAllocation / MemoryDeallocation XEvents into flat dataclasses.
Parent chain is derived inside each XLine by sweeping events ordered by
(offset_ps, -duration_ps) with a containment stack.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Optional

# Reuse profile-anatomy's xplane_pb2 — explicit dependency, not vendored.
_HERE = pathlib.Path(__file__).resolve().parent
_PROFILE_ANATOMY_PROTO = (
    _HERE.parent.parent / "profile-anatomy" / "scripts" / "_proto"
)
sys.path.insert(0, str(_PROFILE_ANATOMY_PROTO))
import xplane_pb2  # noqa: E402


# XLA dtype enum int → string. Mirrors xla.proto's PrimitiveType. Truncated to
# the dtypes that show up in TPU captures; unknown ints fall back to f"dt{n}".
_DTYPE_NAMES = {
    0: "INVALID", 1: "PRED",
    2: "s8", 3: "s16", 4: "s32", 5: "s64",
    6: "u8", 7: "u16", 8: "u32", 9: "u64",
    10: "f16", 11: "f32", 12: "f64",
    16: "bf16",
    19: "f8e5m2", 20: "f8e4m3fn", 23: "f8e4m3", 24: "f8e5m2fnuz",
    26: "f8e4m3fnuz", 28: "f8e3m4",
}


def _dtype_str(n: int) -> str:
    return _DTYPE_NAMES.get(n, f"dt{n}")


@dataclasses.dataclass(slots=True)
class AllocEvent:
    ts_ns: int
    addr: int
    pool_id: int
    requested_bytes: int
    allocation_bytes: int
    bytes_allocated: int          # allocator-reported pool occupancy after this event
    peak_bytes_in_use: int        # allocator-reported running peak after this event
    fragmentation: float          # allocator-reported pool fragmentation
    shape: str
    tf_op: str
    data_type: str
    parent_chain: list[str]
    line_name: str


@dataclasses.dataclass(slots=True)
class DeallocEvent:
    ts_ns: int
    addr: int
    bytes_allocated: int
    peak_bytes_in_use: int
    fragmentation: float
    line_name: str


@dataclasses.dataclass(slots=True)
class HostAllocatorEvents:
    allocs: list[AllocEvent]
    deallocs: list[DeallocEvent]
    pool_capacity: dict[int, int]
    host_plane_present: bool
    n_planes: int


def load_xspace(profile_dir: str | pathlib.Path) -> Optional[xplane_pb2.XSpace]:
    """Load the first *.xplane.pb in profile_dir. Returns None if none."""
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        return None
    xs = xplane_pb2.XSpace()
    xs.ParseFromString(pbs[0].read_bytes())
    return xs


def load_host_allocator_events(
    xs: xplane_pb2.XSpace,
) -> tuple[Optional[HostAllocatorEvents], Optional[str]]:
    """Extract MemoryAllocation/Deallocation events from /host:CPU plane.

    Returns (events, None) on success, (None, reason_code) when absent.
    Reason codes: 'host_plane_absent', 'no_memory_events'.
    """
    host = next((p for p in xs.planes if p.name == "/host:CPU"), None)
    if host is None:
        return None, "host_plane_absent"

    sm = {meta.id: meta.name for _, meta in host.stat_metadata.items()}
    em = {meta.id: meta.name for _, meta in host.event_metadata.items()}

    name_to_stat_id = {v: k for k, v in sm.items()}
    alloc_event_ids = {eid for eid, name in em.items() if name == "MemoryAllocation"}
    dealloc_event_ids = {eid for eid, name in em.items() if name == "MemoryDeallocation"}
    if not alloc_event_ids and not dealloc_event_ids:
        return None, "no_memory_events"

    def stat_value(ev, stat_name: str):
        sid = name_to_stat_id.get(stat_name)
        if sid is None:
            return None
        for st in ev.stats:
            if st.metadata_id != sid:
                continue
            v = st.WhichOneof("value")
            return getattr(st, v) if v else None
        return None

    allocs: list[AllocEvent] = []
    deallocs: list[DeallocEvent] = []
    pool_capacity: dict[int, int] = {}

    for line in host.lines:
        # Build parent chain via containment sweep.
        ordered = sorted(
            range(len(line.events)),
            key=lambda i: (line.events[i].offset_ps, -line.events[i].duration_ps),
        )
        # stack: list of (end_offset_ps, name) currently containing the cursor.
        stack: list[tuple[int, str]] = []
        # For each event, compute its parent chain (outermost-first) of all
        # events whose [start, start+dur] strictly contains its start.
        parent_chains: dict[int, list[str]] = {}
        for idx in ordered:
            ev = line.events[idx]
            start = ev.offset_ps
            # Pop entries whose end <= start.
            stack = [(end, n) for (end, n) in stack if end > start]
            parent_chains[idx] = [n for (_end, n) in stack]
            # Push self onto stack (only if it has positive duration; zero-dur
            # events do not contain anything).
            if ev.duration_ps > 0:
                stack.append((start + ev.duration_ps, em.get(ev.metadata_id, "")))

        for idx, ev in enumerate(line.events):
            ev_name = em.get(ev.metadata_id, "")
            if ev.metadata_id in alloc_event_ids:
                pool_id = int(stat_value(ev, "id") or 0)
                br = int(stat_value(ev, "bytes_reserved") or 0)
                if br > pool_capacity.get(pool_id, 0):
                    pool_capacity[pool_id] = br
                allocs.append(AllocEvent(
                    ts_ns=line.timestamp_ns + ev.offset_ps // 1000,
                    addr=int(stat_value(ev, "addr") or 0),
                    pool_id=pool_id,
                    requested_bytes=int(stat_value(ev, "requested_bytes") or 0),
                    allocation_bytes=int(stat_value(ev, "allocation_bytes") or 0),
                    bytes_allocated=int(stat_value(ev, "bytes_allocated") or 0),
                    peak_bytes_in_use=int(stat_value(ev, "peak_bytes_in_use") or 0),
                    fragmentation=float(stat_value(ev, "fragmentation") or 0.0),
                    shape=str(stat_value(ev, "shape") or ""),
                    tf_op=str(stat_value(ev, "tf_op") or ""),
                    data_type=_dtype_str(int(stat_value(ev, "data_type") or 0)),
                    parent_chain=parent_chains.get(idx, []),
                    line_name=line.name,
                ))
            elif ev.metadata_id in dealloc_event_ids:
                br = int(stat_value(ev, "bytes_reserved") or 0)
                # Capacity may surface only on dealloc events on some lines.
                # Track largest seen.
                if br > pool_capacity.get(0, 0):
                    pool_capacity[0] = br
                deallocs.append(DeallocEvent(
                    ts_ns=line.timestamp_ns + ev.offset_ps // 1000,
                    addr=int(stat_value(ev, "addr") or 0),
                    bytes_allocated=int(stat_value(ev, "bytes_allocated") or 0),
                    peak_bytes_in_use=int(stat_value(ev, "peak_bytes_in_use") or 0),
                    fragmentation=float(stat_value(ev, "fragmentation") or 0.0),
                    line_name=line.name,
                ))

    if not allocs and not deallocs:
        return None, "no_memory_events"

    allocs.sort(key=lambda e: e.ts_ns)
    deallocs.sort(key=lambda e: e.ts_ns)
    return HostAllocatorEvents(
        allocs=allocs, deallocs=deallocs,
        pool_capacity=pool_capacity, host_plane_present=True,
        n_planes=len(xs.planes),
    ), None
```

- [ ] **Step 5: Run loader tests and confirm they pass**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_loader -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts
git commit -m "feat(tpu-perf): memory-profile loader extracts alloc/dealloc events with parent chain"
```

---

(Continued in the next chunk — Tasks 3 through 13 follow.)

## Task 3: Step window selection

**Files:**
- Modify: `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_step_selection.py`

- [ ] **Step 1: Write the failing step-selection tests**

Create `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_step_selection.py`:

```python
"""Tests for step window selection (peak/last/first/explicit/all-trace)."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import StepPolicyError, pick_step_window  # noqa: E402

from tests._synthetic import PlaneBuilder, make_xspace  # noqa: E402


def _xspace_with_steps(step_intervals_ns):
    """Build an XSpace with /device:TPU:0 'Steps' line at the given (start, end) ns."""
    xs = make_xspace()
    tpu = PlaneBuilder(xs, "/device:TPU:0")
    line = tpu.add_line("Steps", timestamp_ns=0)
    for i, (start_ns, end_ns) in enumerate(step_intervals_ns):
        # offset_ps = ns * 1000; duration_ps = (end-start) * 1000.
        line.add_event(
            f"step_{i}",
            offset_ps=start_ns * 1000,
            duration_ps=(end_ns - start_ns) * 1000,
        )
    return xs


class TestStepWindow(unittest.TestCase):
    def test_all_trace_yields_unbounded_window(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000)])
        sw = pick_step_window(xs, all_trace=True, policy="peak", explicit=None,
                              peak_ts_ns_hint=500)
        self.assertEqual(sw.source, "all_trace")
        self.assertEqual(sw.range_ns, (0, 2**63 - 1))
        self.assertIsNone(sw.id)

    def test_explicit_step_selects_indexed_event(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=1,
                              peak_ts_ns_hint=None)
        self.assertEqual(sw.id, "step_1")
        self.assertEqual(sw.range_ns, (2_000, 3_000))
        self.assertEqual(sw.source, "steps_line")

    def test_explicit_step_out_of_range_raises(self):
        xs = _xspace_with_steps([(0, 1_000)])
        with self.assertRaises(StepPolicyError):
            pick_step_window(xs, all_trace=False, policy="peak", explicit=5,
                             peak_ts_ns_hint=None)

    def test_policy_last_picks_last_step_event(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000)])
        sw = pick_step_window(xs, all_trace=False, policy="last", explicit=None,
                              peak_ts_ns_hint=None)
        self.assertEqual(sw.id, "step_1")
        self.assertEqual(sw.range_ns, (2_000, 3_000))

    def test_policy_first_picks_first_step_event(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000)])
        sw = pick_step_window(xs, all_trace=False, policy="first", explicit=None,
                              peak_ts_ns_hint=None)
        self.assertEqual(sw.id, "step_0")

    def test_policy_peak_picks_step_containing_hint(self):
        xs = _xspace_with_steps([(0, 1_000), (2_000, 3_000), (4_000, 5_000)])
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=None,
                              peak_ts_ns_hint=2_500)
        self.assertEqual(sw.id, "step_1")

    def test_policy_peak_falls_back_to_execute_event_when_no_steps_line(self):
        xs = make_xspace()
        host = PlaneBuilder(xs, "/host:CPU")
        line = host.add_line("pjrt_tpu_execute/0", timestamp_ns=0)
        line.add_event("[0] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                       offset_ps=0, duration_ps=10_000_000)  # [0ns, 10_000ns]
        line.add_event("[1] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
                       offset_ps=20_000_000, duration_ps=10_000_000)
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=None,
                              peak_ts_ns_hint=22_000)
        self.assertEqual(sw.source, "execute_event")
        self.assertEqual(sw.range_ns, (20_000, 30_000))

    def test_policy_peak_returns_none_when_no_step_data_at_all(self):
        xs = make_xspace()
        PlaneBuilder(xs, "/host:CPU")
        sw = pick_step_window(xs, all_trace=False, policy="peak", explicit=None,
                              peak_ts_ns_hint=42)
        self.assertIsNone(sw)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_step_selection -v
```
Expected: 8 errors — `pick_step_window` / `StepPolicyError` not yet defined.

- [ ] **Step 3: Implement step-window selection in `_loader.py`**

Append to `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`:

```python
class StepPolicyError(ValueError):
    """Raised when the requested step policy cannot be satisfied."""


@dataclasses.dataclass(slots=True)
class StepWindow:
    id: Optional[str]                 # event metadata.name (e.g. "step_3"); None for all-trace
    range_ns: tuple[int, int]
    source: str                       # "steps_line" | "execute_event" | "all_trace"
    policy_used: str                  # "explicit" | "peak" | "last" | "first" | "all_trace"


def _steps_line_intervals(xs: xplane_pb2.XSpace) -> list[tuple[str, int, int]]:
    """Return [(name, start_ns, end_ns)] for /device:TPU:0 'Steps' events."""
    for plane in xs.planes:
        if plane.name != "/device:TPU:0":
            continue
        em = {meta.id: meta.name for _, meta in plane.event_metadata.items()}
        for line in plane.lines:
            if line.name != "Steps":
                continue
            out: list[tuple[str, int, int]] = []
            for ev in line.events:
                start_ns = line.timestamp_ns + ev.offset_ps // 1000
                end_ns = start_ns + ev.duration_ps // 1000
                out.append((em.get(ev.metadata_id, ""), start_ns, end_ns))
            return out
    return []


def _execute_event_intervals(xs: xplane_pb2.XSpace) -> list[tuple[str, int, int]]:
    """Return [(name, start_ns, end_ns)] for outer 'Execute (jit_*)' events on /host:CPU."""
    for plane in xs.planes:
        if plane.name != "/host:CPU":
            continue
        em = {meta.id: meta.name for _, meta in plane.event_metadata.items()}
        out: list[tuple[str, int, int]] = []
        for line in plane.lines:
            for ev in line.events:
                name = em.get(ev.metadata_id, "")
                if "Execute (jit_" not in name:
                    continue
                if ev.duration_ps <= 0:
                    continue
                start_ns = line.timestamp_ns + ev.offset_ps // 1000
                end_ns = start_ns + ev.duration_ps // 1000
                out.append((name, start_ns, end_ns))
        out.sort(key=lambda x: x[1])
        return out
    return []


def pick_step_window(
    xs: xplane_pb2.XSpace, *, all_trace: bool, policy: str,
    explicit: Optional[int], peak_ts_ns_hint: Optional[int],
) -> Optional[StepWindow]:
    if all_trace:
        return StepWindow(id=None, range_ns=(0, (1 << 63) - 1),
                          source="all_trace", policy_used="all_trace")

    steps = _steps_line_intervals(xs)
    if explicit is not None:
        if not steps:
            raise StepPolicyError("explicit --step requires a 'Steps' line on /device:TPU:0")
        if explicit < 0 or explicit >= len(steps):
            raise StepPolicyError(
                f"--step {explicit} out of range; Steps line has {len(steps)} events"
            )
        name, start, end = steps[explicit]
        return StepWindow(id=name, range_ns=(start, end),
                          source="steps_line", policy_used="explicit")

    if steps:
        if policy == "first":
            name, start, end = steps[0]
        elif policy == "last":
            name, start, end = steps[-1]
        elif policy == "peak":
            if peak_ts_ns_hint is None:
                name, start, end = steps[0]
            else:
                hit = next(
                    ((n, s, e) for (n, s, e) in steps if s <= peak_ts_ns_hint <= e),
                    None,
                )
                if hit is None:
                    # Pick the closest step.
                    hit = min(steps, key=lambda t: min(
                        abs(t[1] - peak_ts_ns_hint), abs(t[2] - peak_ts_ns_hint)
                    ))
                name, start, end = hit
        else:
            raise StepPolicyError(f"unknown policy: {policy}")
        return StepWindow(id=name, range_ns=(start, end),
                          source="steps_line", policy_used=policy)

    # Fallback: outer Execute (jit_*) events on /host:CPU.
    execs = _execute_event_intervals(xs)
    if not execs:
        return None
    if policy == "first":
        name, start, end = execs[0]
    elif policy == "last":
        name, start, end = execs[-1]
    elif policy == "peak":
        if peak_ts_ns_hint is None:
            name, start, end = execs[0]
        else:
            hit = next(
                ((n, s, e) for (n, s, e) in execs if s <= peak_ts_ns_hint <= e),
                None,
            )
            if hit is None:
                hit = min(execs, key=lambda t: min(
                    abs(t[1] - peak_ts_ns_hint), abs(t[2] - peak_ts_ns_hint)
                ))
            name, start, end = hit
    else:
        raise StepPolicyError(f"unknown policy: {policy}")
    return StepWindow(id=name, range_ns=(start, end),
                      source="execute_event", policy_used=policy)
```

- [ ] **Step 4: Run step-selection tests**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_step_selection -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts
git commit -m "feat(tpu-perf): memory-profile step window picker (peak/last/first/explicit/all-trace)"
```

---

## Task 4: Two-pass sweep — alive set + global peak + timeline samples

**Files:**
- Modify: `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_alive_set.py`

- [ ] **Step 1: Write the failing sweep tests**

Create `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_alive_set.py`:

```python
"""Tests for the two-pass sweep: timeline samples, global peak, alive snapshot."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import (  # noqa: E402
    AllocEvent, DeallocEvent, HostAllocatorEvents, sweep_first_pass,
    snapshot_at_peak,
)


def _alloc(ts_ns, addr, size, *, shape="bf16[1]", tf_op="op", pool=0,
           bytes_allocated=None, peak=None, fragmentation=0.0,
           parent_chain=None, line="L"):
    return AllocEvent(
        ts_ns=ts_ns, addr=addr, pool_id=pool,
        requested_bytes=size, allocation_bytes=size,
        bytes_allocated=bytes_allocated if bytes_allocated is not None else size,
        peak_bytes_in_use=peak if peak is not None else size,
        fragmentation=fragmentation,
        shape=shape, tf_op=tf_op, data_type="bf16",
        parent_chain=parent_chain or ["[0] Execute (jit_train_step)"],
        line_name=line,
    )


def _dealloc(ts_ns, addr, *, bytes_allocated=0, peak=0, fragmentation=0.0):
    return DeallocEvent(
        ts_ns=ts_ns, addr=addr,
        bytes_allocated=bytes_allocated, peak_bytes_in_use=peak,
        fragmentation=fragmentation, line_name="L",
    )


class TestSweepFirstPass(unittest.TestCase):
    def test_global_peak_is_after_two_allocs_before_dealloc(self):
        events = HostAllocatorEvents(
            allocs=[
                _alloc(100, 0x1, 1000, peak=1000),
                _alloc(200, 0x2, 2000, peak=3000),
            ],
            deallocs=[_dealloc(300, 0x1)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=10)
        self.assertEqual(result.global_peak_ts_ns, 200)
        self.assertEqual(result.global_peak_bytes, 3000)
        self.assertEqual(result.trace_end_live_bytes, 2000)
        self.assertEqual(result.unmatched_dealloc_count, 0)
        self.assertEqual(result.unmatched_alloc_count, 1)  # 0x2 still live

    def test_unmatched_dealloc_counted(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 500)],
            deallocs=[_dealloc(50, 0xFFFF)],  # dealloc with no matching alloc
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=4)
        self.assertEqual(result.unmatched_dealloc_count, 1)

    def test_timeline_has_requested_sample_count(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 1000), _alloc(500, 0x2, 2000, peak=3000)],
            deallocs=[],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=5)
        self.assertEqual(len(result.timeline_samples), 5)
        for s in result.timeline_samples:
            self.assertGreaterEqual(s.bytes_allocated, 0)

    def test_drift_pct_zero_when_allocator_self_consistent(self):
        # If our running sum exactly matches the allocator's bytes_allocated, drift = 0.
        events = HostAllocatorEvents(
            allocs=[
                _alloc(100, 0x1, 1000, bytes_allocated=1000, peak=1000),
                _alloc(200, 0x2, 2000, bytes_allocated=3000, peak=3000),
            ],
            deallocs=[_dealloc(300, 0x1, bytes_allocated=2000, peak=3000)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        result = sweep_first_pass(events, time_samples_n=4)
        self.assertEqual(result.alloc_accounting_drift_pct, 0.0)


class TestSnapshotAtPeak(unittest.TestCase):
    def _events(self):
        return HostAllocatorEvents(
            allocs=[
                _alloc(100, 0x1, 1000, shape="bf16[A]", tf_op="weight",
                       parent_chain=["[0] Execute (jit_train_step)"]),
                _alloc(200, 0x2, 2000, shape="bf16[B]", tf_op="act",
                       parent_chain=["[0] Execute (jit_train_step)"]),
                _alloc(400, 0x3, 500, shape="bf16[C]", tf_op="tmp",
                       parent_chain=["[0] Execute (jit_train_step)"]),
            ],
            deallocs=[_dealloc(450, 0x3)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )

    def test_alive_at_peak_excludes_yet_to_alloc_and_already_freed(self):
        events = self._events()
        snap = snapshot_at_peak(events, peak_ts_ns=300, step_range_ns=(0, 1_000_000),
                                step_boundaries_ns=[(0, 1_000), (1_000, 2_000)],
                                persistent_threshold_steps=2)
        addrs = {b.addr for b in snap.alive}
        self.assertEqual(addrs, {0x1, 0x2})
        self.assertEqual(snap.alive_total_bytes, 3000)
        self.assertEqual(snap.bytes_total, 3000)

    def test_lifetime_class_persistent_when_crosses_threshold(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 1000, shape="bf16[W]", tf_op="weight")],
            deallocs=[],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        # 5 step boundaries between alloc_ts_ns=100 and trace end at 10_000.
        boundaries = [(0, 1_000), (1_000, 2_000), (2_000, 3_000),
                      (3_000, 4_000), (4_000, 10_000)]
        snap = snapshot_at_peak(events, peak_ts_ns=5_000,
                                step_range_ns=(4_000, 10_000),
                                step_boundaries_ns=boundaries,
                                persistent_threshold_steps=2)
        self.assertEqual(snap.alive[0].lifetime_class, "persistent")
        self.assertGreaterEqual(snap.alive[0].crossed_step_boundaries, 2)

    def test_lifetime_class_transient_when_alloc_and_dealloc_in_same_step(self):
        events = HostAllocatorEvents(
            allocs=[_alloc(100, 0x1, 500, shape="bf16[T]", tf_op="tmp")],
            deallocs=[_dealloc(200, 0x1)],
            pool_capacity={0: 10_000},
            host_plane_present=True, n_planes=1,
        )
        snap = snapshot_at_peak(events, peak_ts_ns=150,
                                step_range_ns=(0, 1_000),
                                step_boundaries_ns=[(0, 1_000)],
                                persistent_threshold_steps=2)
        self.assertEqual(snap.alive[0].lifetime_class, "transient")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run sweep tests and confirm they fail**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_alive_set -v
```
Expected: errors — `sweep_first_pass`, `snapshot_at_peak` not defined.

- [ ] **Step 3: Implement the two-pass sweep**

Append to `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`:

```python
@dataclasses.dataclass(slots=True)
class TimelineSample:
    ts_ns: int
    bytes_allocated: int
    live_count: int
    fragmentation: float


@dataclasses.dataclass(slots=True)
class AliveBuffer:
    addr: int
    pool_id: int
    size_bytes: int
    alloc_bytes: int
    shape: str
    tf_op: str
    data_type: str
    alloc_ts_ns: int
    age_ns_at_peak: int
    crossed_step_boundaries: int
    parent_chain: list[str]
    lifetime_class: str
    deallocated: bool


@dataclasses.dataclass(slots=True)
class FirstPassResult:
    global_peak_ts_ns: int
    global_peak_bytes: int
    timeline_samples: list[TimelineSample]
    alloc_accounting_drift_pct: float
    unmatched_dealloc_count: int
    unmatched_alloc_count: int
    trace_end_live_bytes: int
    pool_max_peak_in_use: dict[int, int]
    n_alloc: int
    n_dealloc: int


@dataclasses.dataclass(slots=True)
class PeakSnapshot:
    peak_ts_ns: int
    bytes_total: int
    bytes_by_pool: dict[int, int]
    fragmentation_at_peak: float
    is_global_peak: bool
    alive: list[AliveBuffer]
    alive_total_bytes: int


def _merged_event_stream(events: HostAllocatorEvents):
    """Yield (ts_ns, kind, payload) ordered by ts_ns, allocs before deallocs at ties."""
    a_iter = iter(events.allocs)
    d_iter = iter(events.deallocs)
    a = next(a_iter, None)
    d = next(d_iter, None)
    while a is not None or d is not None:
        if d is None or (a is not None and a.ts_ns <= d.ts_ns):
            yield a.ts_ns, "A", a
            a = next(a_iter, None)
        else:
            yield d.ts_ns, "D", d
            d = next(d_iter, None)


def sweep_first_pass(events: HostAllocatorEvents, *, time_samples_n: int) -> FirstPassResult:
    live: dict[tuple[int, int], AllocEvent] = {}
    bytes_now_by_pool: dict[int, int] = {}
    pool_max_peak: dict[int, int] = dict(events.pool_capacity)
    pool_max_peak.update({pid: 0 for pid in pool_max_peak})  # init counters
    pool_max_peak_in_use: dict[int, int] = {}

    global_peak_bytes = 0
    global_peak_ts_ns = 0

    last_fragmentation = 0.0
    drift_max = 0.0
    drift_seen = False

    unmatched_dealloc_count = 0

    # Linear scan of (ts_ns, bytes_allocated_total, fragmentation, live_count).
    samples: list[tuple[int, int, float, int]] = []

    for ts, kind, payload in _merged_event_stream(events):
        if kind == "A":
            a = payload
            key = (a.pool_id, a.addr)
            live[key] = a
            bytes_now_by_pool[a.pool_id] = bytes_now_by_pool.get(a.pool_id, 0) + a.requested_bytes
            last_fragmentation = a.fragmentation
            if a.peak_bytes_in_use > pool_max_peak_in_use.get(a.pool_id, 0):
                pool_max_peak_in_use[a.pool_id] = a.peak_bytes_in_use
            # Drift between our running sum and allocator's report (single pool only).
            if len(bytes_now_by_pool) == 1:
                ours = bytes_now_by_pool[a.pool_id]
                allocator = a.bytes_allocated
                if allocator > 0:
                    drift_seen = True
                    rel = abs(ours - allocator) / allocator
                    if rel > drift_max:
                        drift_max = rel
        else:  # "D"
            d = payload
            # Dealloc events do not carry pool_id; match by addr in any pool.
            match_key = next((k for k in live if k[1] == d.addr), None)
            if match_key is None:
                unmatched_dealloc_count += 1
                last_fragmentation = d.fragmentation
            else:
                a = live.pop(match_key)
                bytes_now_by_pool[a.pool_id] = bytes_now_by_pool[a.pool_id] - a.requested_bytes
                last_fragmentation = d.fragmentation
                if d.peak_bytes_in_use > pool_max_peak_in_use.get(a.pool_id, 0):
                    pool_max_peak_in_use[a.pool_id] = d.peak_bytes_in_use

        total = sum(bytes_now_by_pool.values())
        if total > global_peak_bytes:
            global_peak_bytes = total
            global_peak_ts_ns = ts
        samples.append((ts, total, last_fragmentation, len(live)))

    # Down-sample to time_samples_n equally-spaced wallclock points.
    if not samples:
        timeline_samples: list[TimelineSample] = []
    else:
        t0 = samples[0][0]
        t1 = samples[-1][0]
        if t1 == t0 or time_samples_n <= 1:
            timeline_samples = [TimelineSample(ts_ns=samples[-1][0],
                                               bytes_allocated=samples[-1][1],
                                               fragmentation=samples[-1][2],
                                               live_count=samples[-1][3])]
        else:
            step = (t1 - t0) / (time_samples_n - 1)
            timeline_samples = []
            i = 0
            for k in range(time_samples_n):
                target = t0 + step * k
                while i + 1 < len(samples) and samples[i + 1][0] <= target:
                    i += 1
                ts_k, b_k, f_k, l_k = samples[i]
                timeline_samples.append(TimelineSample(
                    ts_ns=int(target), bytes_allocated=b_k,
                    fragmentation=f_k, live_count=l_k,
                ))

    return FirstPassResult(
        global_peak_ts_ns=global_peak_ts_ns,
        global_peak_bytes=global_peak_bytes,
        timeline_samples=timeline_samples,
        alloc_accounting_drift_pct=(drift_max * 100.0) if drift_seen else 0.0,
        unmatched_dealloc_count=unmatched_dealloc_count,
        unmatched_alloc_count=len(live),
        trace_end_live_bytes=sum(bytes_now_by_pool.values()),
        pool_max_peak_in_use=pool_max_peak_in_use,
        n_alloc=len(events.allocs),
        n_dealloc=len(events.deallocs),
    )


def _count_step_boundaries_crossed(alloc_ts_ns: int, end_ts_ns: int,
                                   boundaries_ns: list[tuple[int, int]]) -> int:
    if not boundaries_ns:
        return 0
    n = 0
    for _i, (s, _e) in enumerate(boundaries_ns):
        if alloc_ts_ns <= s <= end_ts_ns:
            n += 1
    return n


def snapshot_at_peak(events: HostAllocatorEvents, *, peak_ts_ns: int,
                     step_range_ns: tuple[int, int],
                     step_boundaries_ns: list[tuple[int, int]],
                     persistent_threshold_steps: int) -> PeakSnapshot:
    # Re-run the sweep but stop computing at peak_ts_ns to capture the live set.
    live: dict[tuple[int, int], AllocEvent] = {}
    bytes_now_by_pool: dict[int, int] = {}
    last_fragmentation = 0.0

    # Find addr → dealloc_ts_ns map for lifetime classification.
    dealloc_ts_by_addr: dict[int, int] = {}
    for d in events.deallocs:
        dealloc_ts_by_addr.setdefault(d.addr, d.ts_ns)

    for ts, kind, payload in _merged_event_stream(events):
        if ts > peak_ts_ns:
            break
        if kind == "A":
            a = payload
            live[(a.pool_id, a.addr)] = a
            bytes_now_by_pool[a.pool_id] = bytes_now_by_pool.get(a.pool_id, 0) + a.requested_bytes
            last_fragmentation = a.fragmentation
        else:
            d = payload
            match_key = next((k for k in live if k[1] == d.addr), None)
            if match_key is not None:
                a = live.pop(match_key)
                bytes_now_by_pool[a.pool_id] = bytes_now_by_pool[a.pool_id] - a.requested_bytes
                last_fragmentation = d.fragmentation

    bytes_total = sum(bytes_now_by_pool.values())

    # Trace end ts_ns is the max event ts in either stream — used to compute
    # crossed boundaries when an alloc was never deallocated.
    last_event_ts = max(
        (events.allocs[-1].ts_ns if events.allocs else 0),
        (events.deallocs[-1].ts_ns if events.deallocs else 0),
    )

    alive_buffers: list[AliveBuffer] = []
    for (_pool, addr), a in live.items():
        dealloc_ts = dealloc_ts_by_addr.get(addr)
        end_ts = dealloc_ts if dealloc_ts is not None else last_event_ts
        crossed = _count_step_boundaries_crossed(a.ts_ns, end_ts, step_boundaries_ns)
        deallocated = dealloc_ts is not None and dealloc_ts >= peak_ts_ns
        # Lifetime classification.
        same_step = False
        if dealloc_ts is not None:
            for s, e in step_boundaries_ns:
                if s <= a.ts_ns <= e and s <= dealloc_ts <= e:
                    same_step = True
                    break
        if not deallocated and crossed >= persistent_threshold_steps and dealloc_ts is None:
            cls = "persistent"
        elif same_step and dealloc_ts is not None:
            cls = "transient"
        else:
            cls = "unknown"
        alive_buffers.append(AliveBuffer(
            addr=a.addr, pool_id=a.pool_id,
            size_bytes=a.requested_bytes, alloc_bytes=a.allocation_bytes,
            shape=a.shape or "<no shape>",
            tf_op=a.tf_op or "<no tf_op>",
            data_type=a.data_type,
            alloc_ts_ns=a.ts_ns,
            age_ns_at_peak=peak_ts_ns - a.ts_ns,
            crossed_step_boundaries=crossed,
            parent_chain=list(a.parent_chain),
            lifetime_class=cls,
            deallocated=deallocated,
        ))

    alive_buffers.sort(key=lambda b: b.size_bytes, reverse=True)
    return PeakSnapshot(
        peak_ts_ns=peak_ts_ns,
        bytes_total=bytes_total,
        bytes_by_pool=dict(bytes_now_by_pool),
        fragmentation_at_peak=last_fragmentation,
        is_global_peak=False,  # caller sets this against FirstPassResult.global_peak_ts_ns
        alive=alive_buffers,
        alive_total_bytes=sum(b.size_bytes for b in alive_buffers),
    )
```

- [ ] **Step 4: Run sweep tests and confirm they pass**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_alive_set -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts
git commit -m "feat(tpu-perf): memory-profile two-pass sweep with timeline + alive snapshot"
```

---

## Task 5: Rollups (by_lifetime_class, by_shape, by_tf_op, by_parent_jit, by_dtype)

**Files:**
- Modify: `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_rollups.py`

- [ ] **Step 1: Write the failing rollup tests**

Create `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_rollups.py`:

```python
"""Rollups must partition the alive set with no double-count and no loss."""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _loader import AliveBuffer, build_rollups, pick_parent_jit  # noqa: E402


def _ab(addr, size, *, shape, tf_op, dtype, lifetime,
        parent_chain=("[0] Execute (jit_train_step)",)):
    return AliveBuffer(
        addr=addr, pool_id=0, size_bytes=size, alloc_bytes=size,
        shape=shape, tf_op=tf_op, data_type=dtype,
        alloc_ts_ns=0, age_ns_at_peak=0, crossed_step_boundaries=0,
        parent_chain=list(parent_chain),
        lifetime_class=lifetime, deallocated=False,
    )


class TestPickParentJit(unittest.TestCase):
    def test_picks_first_jit_in_chain(self):
        chain = [
            "[3] CommonPjRtLoadedExecutable::Execute (jit_train_step)",
            "AllocateOutputBuffersWithInputReuse",
            "AllocateRawBuffer",
        ]
        self.assertEqual(pick_parent_jit(chain),
                         "[3] CommonPjRtLoadedExecutable::Execute (jit_train_step)")

    def test_falls_back_to_chain_root_when_no_jit(self):
        chain = ["DeferredTpuAllocator::Allocate", "AllocateRawBuffer"]
        self.assertEqual(pick_parent_jit(chain), "DeferredTpuAllocator::Allocate")

    def test_empty_chain_returns_unknown_marker(self):
        self.assertEqual(pick_parent_jit([]), "<no parent>")


class TestBuildRollups(unittest.TestCase):
    def setUp(self):
        self.alive = [
            _ab(0x1, 1000, shape="bf16[A]", tf_op="opA", dtype="bf16",
                lifetime="persistent"),
            _ab(0x2, 2000, shape="bf16[A]", tf_op="opB", dtype="bf16",
                lifetime="transient"),
            _ab(0x3, 500, shape="f32[B]", tf_op="opA", dtype="f32",
                lifetime="unknown"),
        ]

    def test_each_rollup_sums_to_alive_total(self):
        total = sum(b.size_bytes for b in self.alive)
        ru = build_rollups(self.alive, top_k=10, total_bytes=total)
        for key in ("by_lifetime_class", "by_shape", "by_tf_op",
                    "by_parent_jit", "by_dtype"):
            sub_total = sum(row["total_bytes"] for row in ru[key])
            self.assertEqual(sub_total, total, f"{key}: {sub_total} != {total}")

    def test_by_shape_top_k_truncates_with_tail(self):
        many = [_ab(i, (10 - i) * 100, shape=f"bf16[s{i}]", tf_op="x",
                    dtype="bf16", lifetime="persistent") for i in range(8)]
        total = sum(b.size_bytes for b in many)
        ru = build_rollups(many, top_k=3, total_bytes=total)
        shape = ru["by_shape"]
        # 3 head rows + 1 tail row.
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[-1]["key"], "<tail>")
        # Head rows are sorted by total_bytes desc.
        head_bytes = [r["total_bytes"] for r in shape[:3]]
        self.assertEqual(head_bytes, sorted(head_bytes, reverse=True))

    def test_by_lifetime_class_has_no_top_k_truncation(self):
        ru = build_rollups(self.alive, top_k=1, total_bytes=3500)
        keys = {r["key"] for r in ru["by_lifetime_class"]}
        self.assertEqual(keys, {"persistent", "transient", "unknown"})

    def test_lifetime_mix_sums_to_row_total(self):
        ru = build_rollups(self.alive, top_k=10, total_bytes=3500)
        for row in ru["by_shape"] + ru["by_tf_op"] + ru["by_parent_jit"]:
            mix = row["lifetime_mix"]
            self.assertEqual(
                mix["persistent"] + mix["transient"] + mix["unknown"],
                row["total_bytes"],
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_rollups -v
```
Expected: errors — `build_rollups`, `pick_parent_jit` not defined.

- [ ] **Step 3: Implement rollups**

Append to `plugins/tpu-perf/skills/memory-profile/scripts/_loader.py`:

```python
def pick_parent_jit(parent_chain: list[str]) -> str:
    if not parent_chain:
        return "<no parent>"
    for n in parent_chain:
        if "jit_" in n:
            return n
    return parent_chain[0]


_LIFETIME_KEYS = ("persistent", "transient", "unknown")


def _empty_mix() -> dict[str, int]:
    return {k: 0 for k in _LIFETIME_KEYS}


def _group_with_mix(buffers: list[AliveBuffer], key_fn,
                    *, top_k: int, total_bytes: int,
                    truncate: bool) -> list[dict]:
    groups: dict[str, dict] = {}
    for b in buffers:
        k = key_fn(b)
        g = groups.setdefault(k, {
            "key": k, "n_buffers": 0, "total_bytes": 0,
            "lifetime_mix": _empty_mix(),
        })
        g["n_buffers"] += 1
        g["total_bytes"] += b.size_bytes
        g["lifetime_mix"][b.lifetime_class] += b.size_bytes
    rows = sorted(groups.values(), key=lambda r: r["total_bytes"], reverse=True)
    for r in rows:
        r["pct_of_peak"] = (r["total_bytes"] / total_bytes * 100.0) if total_bytes else 0.0
    if truncate and len(rows) > top_k:
        head = rows[:top_k]
        tail_rows = rows[top_k:]
        tail = {
            "key": "<tail>",
            "n_buffers": sum(r["n_buffers"] for r in tail_rows),
            "total_bytes": sum(r["total_bytes"] for r in tail_rows),
            "lifetime_mix": {
                k: sum(r["lifetime_mix"][k] for r in tail_rows)
                for k in _LIFETIME_KEYS
            },
        }
        tail["pct_of_peak"] = (
            tail["total_bytes"] / total_bytes * 100.0 if total_bytes else 0.0
        )
        return head + [tail]
    return rows


def _group_simple(buffers: list[AliveBuffer], key_fn,
                  *, total_bytes: int) -> list[dict]:
    """Rollup without lifetime_mix or Top-K (used for by_lifetime_class, by_dtype)."""
    groups: dict[str, dict] = {}
    for b in buffers:
        k = key_fn(b)
        g = groups.setdefault(k, {"key": k, "n_buffers": 0, "total_bytes": 0})
        g["n_buffers"] += 1
        g["total_bytes"] += b.size_bytes
    rows = sorted(groups.values(), key=lambda r: r["total_bytes"], reverse=True)
    for r in rows:
        r["pct_of_peak"] = (r["total_bytes"] / total_bytes * 100.0) if total_bytes else 0.0
    return rows


def build_rollups(alive: list[AliveBuffer], *, top_k: int,
                  total_bytes: int) -> dict[str, list[dict]]:
    return {
        "by_lifetime_class": _group_simple(alive, lambda b: b.lifetime_class,
                                           total_bytes=total_bytes),
        "by_shape": _group_with_mix(alive, lambda b: b.shape,
                                    top_k=top_k, total_bytes=total_bytes,
                                    truncate=True),
        "by_tf_op": _group_with_mix(alive, lambda b: b.tf_op,
                                    top_k=top_k, total_bytes=total_bytes,
                                    truncate=True),
        "by_parent_jit": _group_with_mix(alive, lambda b: pick_parent_jit(b.parent_chain),
                                         top_k=top_k, total_bytes=total_bytes,
                                         truncate=True),
        "by_dtype": _group_simple(alive, lambda b: b.data_type,
                                  total_bytes=total_bytes),
    }
```

- [ ] **Step 4: Run rollup tests and confirm they pass**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_rollups -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts
git commit -m "feat(tpu-perf): memory-profile rollups (lifetime/shape/tf_op/parent_jit/dtype)"
```

---

## Task 6: Wire CLI to loader/sweep/rollups, emit final JSON

**Files:**
- Modify: `plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py`

- [ ] **Step 1: Replace the placeholder `main()` with the full pipeline**

Replace the entire body of `plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py` with:

```python
"""
Entry point for tpu-perf:memory-profile.

Locates HBM peak occupancy within a chosen step (or across the full
trace) and emits one JSON object on stdout describing the peak moment
plus every buffer alive at that moment, rollups, timeline samples, and
diagnostics. Single-mode skill — no --mode flag. See spec
docs/superpowers/specs/2026-05-25-tpu-perf-memory-profile-design.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys


SKILL_NAME = "memory-profile"
SCHEMA_VERSION = 1


_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _loader  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_profile.py",
        description=(
            "Locate HBM peak occupancy within a step and list every buffer "
            "alive at that moment."
        ),
    )
    p.add_argument("profile_dir", help="Profile directory containing *.xplane.pb")

    step_group = p.add_mutually_exclusive_group()
    step_group.add_argument("--step", type=int, default=None,
                            help="Explicit step index on /device:TPU:0 'Steps' line (0-based).")
    step_group.add_argument("--all-trace", action="store_true",
                            help="Disable step scoping; analyse the full trace window.")

    p.add_argument("--step-policy", choices=("peak", "last", "first"),
                   default="peak", help="Default step picker (default: peak).")
    p.add_argument("--top", type=int, default=30,
                   help="Top-K applied to alive_at_peak.buffers and rollup tables.")
    p.add_argument("--persistent-threshold-steps", type=int, default=2,
                   help="Min crossed step boundaries for lifetime_class=persistent.")
    p.add_argument("--include-host-pools", action="store_true",
                   help="Include allocator pools other than HBM (id != 0).")
    p.add_argument("--time-samples", type=int, default=200,
                   help="Number of equally-spaced timeline samples.")
    return p


def _emit_absent(profile_dir: str, reason: str, **extra) -> dict:
    return {
        "status": "absent",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "reason": reason,
        "inputs": {"profile_dir": profile_dir, **extra},
    }


def _alive_to_json(b: _loader.AliveBuffer) -> dict:
    return {
        "addr": b.addr, "pool_id": b.pool_id,
        "size_bytes": b.size_bytes, "alloc_bytes": b.alloc_bytes,
        "shape": b.shape, "tf_op": b.tf_op, "data_type": b.data_type,
        "alloc_ts_ns": b.alloc_ts_ns,
        "age_ns_at_peak": b.age_ns_at_peak,
        "crossed_step_boundaries": b.crossed_step_boundaries,
        "parent_chain": b.parent_chain,
        "lifetime_class": b.lifetime_class,
        "deallocated": b.deallocated,
    }


def _step_boundaries_for_classification(xs) -> list[tuple[int, int]]:
    boundaries = _loader._steps_line_intervals(xs)
    if boundaries:
        return [(s, e) for (_n, s, e) in boundaries]
    execs = _loader._execute_event_intervals(xs)
    return [(s, e) for (_n, s, e) in execs]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile_dir = args.profile_dir

    xs = _loader.load_xspace(profile_dir)
    if xs is None:
        json.dump(_emit_absent(profile_dir, "no_xplane_pb"), sys.stdout)
        sys.stdout.write("\n")
        return 0

    events, reason = _loader.load_host_allocator_events(xs)
    if events is None:
        json.dump(_emit_absent(profile_dir, reason or "no_memory_events",
                               n_planes=len(xs.planes)), sys.stdout)
        sys.stdout.write("\n")
        return 0

    if not args.include_host_pools:
        kept_alloc = [a for a in events.allocs if a.pool_id == 0]
        if kept_alloc:
            events = dataclasses.replace(
                events, allocs=kept_alloc,
                pool_capacity={0: events.pool_capacity.get(0, 0)},
            )

    first = _loader.sweep_first_pass(events, time_samples_n=args.time_samples)

    try:
        sw = _loader.pick_step_window(
            xs, all_trace=args.all_trace, policy=args.step_policy,
            explicit=args.step,
            peak_ts_ns_hint=first.global_peak_ts_ns if first.global_peak_bytes else None,
        )
    except _loader.StepPolicyError as e:
        json.dump(_emit_absent(profile_dir, "step_policy_error",
                               error=str(e)), sys.stdout)
        sys.stdout.write("\n")
        return 0

    if sw is None:
        json.dump(_emit_absent(profile_dir, "no_step_data",
                               n_planes=len(xs.planes)), sys.stdout)
        sys.stdout.write("\n")
        return 0

    # Find step-scoped peak: max bytes_now within sw.range_ns.
    s0, s1 = sw.range_ns
    live: dict[tuple[int, int], _loader.AllocEvent] = {}
    bytes_by_pool: dict[int, int] = {}
    peak_bytes = 0
    peak_ts = s0
    last_fragmentation = 0.0
    for ts, kind, payload in _loader._merged_event_stream(events):
        if ts > s1:
            break
        if kind == "A":
            a = payload
            live[(a.pool_id, a.addr)] = a
            bytes_by_pool[a.pool_id] = bytes_by_pool.get(a.pool_id, 0) + a.requested_bytes
            last_fragmentation = a.fragmentation
        else:
            d = payload
            mk = next((k for k in live if k[1] == d.addr), None)
            if mk is not None:
                a = live.pop(mk)
                bytes_by_pool[a.pool_id] -= a.requested_bytes
                last_fragmentation = d.fragmentation
        if ts < s0:
            continue
        total = sum(bytes_by_pool.values())
        if total > peak_bytes:
            peak_bytes = total
            peak_ts = ts
    if peak_bytes == 0:
        peak_ts = first.global_peak_ts_ns
        peak_bytes = first.global_peak_bytes

    boundaries = _step_boundaries_for_classification(xs)
    snap = _loader.snapshot_at_peak(
        events, peak_ts_ns=peak_ts,
        step_range_ns=(s0, s1),
        step_boundaries_ns=boundaries,
        persistent_threshold_steps=args.persistent_threshold_steps,
    )
    snap.is_global_peak = (snap.peak_ts_ns == first.global_peak_ts_ns
                           and snap.bytes_total >= first.global_peak_bytes)

    rollups = _loader.build_rollups(
        snap.alive, top_k=args.top, total_bytes=snap.alive_total_bytes,
    )

    head = snap.alive[: args.top]
    tail_rows = snap.alive[args.top:]
    alive_payload = {
        "n_buffers": len(snap.alive),
        "total_bytes": snap.alive_total_bytes,
        "buffers": [_alive_to_json(b) for b in head],
        "tail": {
            "n_buffers": len(tail_rows),
            "total_bytes": sum(b.size_bytes for b in tail_rows),
        },
    }

    timeline_payload = {
        "samples": [
            {"ts_ns": s.ts_ns, "bytes_allocated": s.bytes_allocated,
             "live_count": s.live_count, "fragmentation": s.fragmentation}
            for s in first.timeline_samples
        ],
        "events_of_interest": [
            {"kind": "global_peak", "ts_ns": first.global_peak_ts_ns,
             "bytes": first.global_peak_bytes},
            {"kind": "step_start", "ts_ns": s0,
             "step_id": sw.id if sw.source != "all_trace" else None},
            {"kind": "step_end", "ts_ns": s1,
             "step_id": sw.id if sw.source != "all_trace" else None},
            {"kind": "step_local_peak", "ts_ns": snap.peak_ts_ns,
             "step_id": sw.id, "bytes": snap.bytes_total},
        ],
        "axis_units": {"ts_ns": "nanoseconds since epoch",
                       "bytes": "bytes (base-2)"},
    }

    pool_id = 0 if 0 in events.pool_capacity else next(iter(events.pool_capacity), 0)
    diagnostics = {
        "alloc_accounting_drift_pct": first.alloc_accounting_drift_pct,
        "unmatched_dealloc_count": first.unmatched_dealloc_count,
        "unmatched_alloc_count": first.unmatched_alloc_count,
        "trace_end_live_bytes": first.trace_end_live_bytes,
        "n_pools_seen": len(events.pool_capacity),
        "pools_summary": [
            {"pool_id": pid,
             "n_alloc": sum(1 for a in events.allocs if a.pool_id == pid),
             "n_dealloc": first.n_dealloc,
             "max_peak_bytes_in_use": first.pool_max_peak_in_use.get(pid, 0)}
            for pid in sorted(events.pool_capacity.keys())
        ],
        "step_line_present": sw.source != "execute_event",
        "shape_missing_count": sum(1 for a in events.allocs if not a.shape),
        "tf_op_missing_count": sum(1 for a in events.allocs if not a.tf_op),
        "warnings": [],
    }
    if first.alloc_accounting_drift_pct > 1.0:
        diagnostics["warnings"].append(
            f"alloc_accounting_drift_pct={first.alloc_accounting_drift_pct:.3f}%"
            " exceeds 1% threshold; results may include alignment/metadata padding"
        )
    if first.unmatched_dealloc_count > 0:
        diagnostics["warnings"].append(
            f"{first.unmatched_dealloc_count} MemoryDeallocation event(s) had no matching alloc"
        )
    if sw.source == "execute_event":
        diagnostics["warnings"].append(
            "/device:TPU:0 'Steps' line absent; using outer Execute (jit_*) event as step window"
        )

    output = {
        "status": "ok",
        "skill": SKILL_NAME,
        "version": SCHEMA_VERSION,
        "inputs": {
            "profile_dir": profile_dir,
            "xplane_pb": str(sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))[0]),
            "n_planes": len(xs.planes),
            "host_plane_present": True,
        },
        "step": {
            "id": sw.id, "policy": sw.policy_used,
            "range_ns": [s0, s1], "source": sw.source,
        },
        "pool": {"id": pool_id,
                 "bytes_reserved": events.pool_capacity.get(pool_id, 0)},
        "peak": {
            "ts_ns": snap.peak_ts_ns,
            "bytes_total": snap.bytes_total,
            "bytes_by_pool": {str(k): v for k, v in snap.bytes_by_pool.items()},
            "fragmentation_at_peak": snap.fragmentation_at_peak,
            "is_global_peak": snap.is_global_peak,
        },
        "alive_at_peak": alive_payload,
        "rollups": rollups,
        "timeline": timeline_payload,
        "diagnostics": diagnostics,
    }
    json.dump(output, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run all tests so far to confirm nothing broke**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest discover tests -v
```
Expected: all tests pass (CLI absent path, loader, step selection, sweep, rollups).

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py
git commit -m "feat(tpu-perf): wire memory-profile CLI to loader/sweep/rollups pipeline"
```

---

### Task 7: End-to-end snapshot test against real fixture

**Files:**
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_snapshot.py`

**Fixture:** `/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`. The xplane.pb at this path was verified during design to contain 3,698 `MemoryAllocation` events on `/host:CPU` and a populated `Steps` line on `/device:TPU:0`.

- [ ] **Step 1: Write the e2e test**

```python
# plugins/tpu-perf/skills/memory-profile/scripts/tests/test_snapshot.py
import json
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
SCRIPT = os.path.join(REPO_ROOT, "plugins", "tpu-perf", "skills", "memory-profile", "scripts", "memory_profile.py")
FIXTURE = "/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128"


def _run(*args):
    proc = subprocess.run(
        [sys.executable, SCRIPT, FIXTURE, *args],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return json.loads(proc.stdout)


@unittest.skipUnless(os.path.isdir(FIXTURE), f"fixture not present at {FIXTURE}")
class TestSnapshot(unittest.TestCase):
    def test_default_peak_step(self):
        out = _run()
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["skill"], "memory-profile")
        self.assertEqual(out["version"], 1)
        self.assertTrue(out["inputs"]["host_plane_present"])
        self.assertEqual(out["step"]["policy"], "peak")
        self.assertGreater(out["peak"]["bytes_total"], 0)
        self.assertEqual(out["peak"]["bytes_by_pool"]["0"], out["peak"]["bytes_total"])
        self.assertEqual(out["pool"]["id"], 0)
        self.assertGreater(out["alive_at_peak"]["n_buffers"], 0)
        self.assertGreater(len(out["rollups"]["by_shape"]), 0)
        self.assertGreater(len(out["rollups"]["by_lifetime_class"]), 0)
        self.assertGreater(len(out["timeline"]["samples"]), 0)
        # Top-K is sorted desc by total_bytes
        sizes = [r["total_bytes"] for r in out["rollups"]["by_shape"] if r.get("total_bytes") is not None]
        # filter out the synthetic 'tail' row if present
        leading = [r for r in out["rollups"]["by_shape"] if r.get("kind") != "tail"]
        leading_sizes = [r["total_bytes"] for r in leading]
        self.assertEqual(leading_sizes, sorted(leading_sizes, reverse=True))

    def test_all_trace(self):
        out = _run("--all-trace")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["step"]["source"], "all_trace")
        self.assertGreater(out["peak"]["bytes_total"], 0)

    def test_step_out_of_range_absent(self):
        out = _run("--step", "9999")
        self.assertEqual(out["status"], "absent")
        self.assertIn("reason", out)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_snapshot -v
```
Expected: all three cases pass. If the fixture is absent on this machine, all three are skipped (still green).

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts/tests/test_snapshot.py
git commit -m "test(tpu-perf): e2e snapshot test for memory-profile against dp8_fsdp128 fixture"
```

---

### Task 8: Invariants test (I1–I9 + I2b)

**Files:**
- Create: `plugins/tpu-perf/skills/memory-profile/scripts/tests/test_invariants.py`

Covers the consistency gates from spec §7. Runs against the real fixture; skips when absent.

- [ ] **Step 1: Write the invariants test**

```python
# plugins/tpu-perf/skills/memory-profile/scripts/tests/test_invariants.py
import json
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
SCRIPT = os.path.join(REPO_ROOT, "plugins", "tpu-perf", "skills", "memory-profile", "scripts", "memory_profile.py")
FIXTURE = "/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128"


def _run(*args):
    proc = subprocess.run(
        [sys.executable, SCRIPT, FIXTURE, *args],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _sum_total_bytes(rows):
    return sum(int(r["total_bytes"]) for r in rows)


@unittest.skipUnless(os.path.isdir(FIXTURE), f"fixture not present at {FIXTURE}")
class TestInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = _run()

    def test_I1_topk_plus_tail_equals_alive_total(self):
        out = self.out
        topk_sum = sum(int(b["size_bytes"]) for b in out["alive_at_peak"]["buffers"])
        tail = int(out["alive_at_peak"]["tail"]["total_bytes"])
        self.assertEqual(topk_sum + tail, int(out["alive_at_peak"]["total_bytes"]),
                         "I1: Σ Top-K size_bytes + tail.total_bytes must equal alive_at_peak.total_bytes")

    def test_I2_alive_total_equals_peak_bytes_total(self):
        out = self.out
        self.assertEqual(int(out["alive_at_peak"]["total_bytes"]),
                         int(out["peak"]["bytes_total"]),
                         "I2: alive_at_peak.total_bytes must equal peak.bytes_total")

    def test_I2b_alloc_accounting_drift_within_one_pct(self):
        out = self.out
        drift = float(out["diagnostics"]["alloc_accounting_drift_pct"])
        if drift > 1.0:
            warns = out["diagnostics"]["warnings"]
            self.assertTrue(any("drift" in w.lower() for w in warns),
                            f"I2b: drift={drift}% > 1% must surface a warning; got warnings={warns}")

    def test_I3_by_shape_partition(self):
        rows = self.out["rollups"]["by_shape"]
        self.assertEqual(_sum_total_bytes(rows), int(self.out["alive_at_peak"]["total_bytes"]),
                         "I3: by_shape rows must partition alive_at_peak.total_bytes")

    def test_I4_other_rollups_partition(self):
        out = self.out
        target = int(out["alive_at_peak"]["total_bytes"])
        for key in ("by_tf_op", "by_parent_jit", "by_lifetime_class", "by_dtype"):
            with self.subTest(rollup=key):
                self.assertEqual(_sum_total_bytes(out["rollups"][key]), target,
                                 f"I4: {key} rows must partition alive_at_peak.total_bytes")

    def test_I5_peak_within_pool_reserved(self):
        out = self.out
        self.assertLessEqual(int(out["peak"]["bytes_total"]),
                             int(out["pool"]["bytes_reserved"]),
                             "I5: peak.bytes_total ≤ pool.bytes_reserved")

    def test_I6_timeline_max_at_least_step_peak(self):
        out = self.out
        max_sample = max(int(s["bytes_allocated"]) for s in out["timeline"]["samples"])
        self.assertGreaterEqual(max_sample, int(out["peak"]["bytes_total"]),
                                "I6: max(timeline.samples.bytes_allocated) ≥ peak.bytes_total")

    def test_I7_buffer_alloc_before_peak(self):
        peak_ts = int(self.out["peak"]["ts_ns"])
        for b in self.out["alive_at_peak"]["buffers"]:
            self.assertLessEqual(int(b["alloc_ts_ns"]), peak_ts,
                                 f"I7: buffer addr={b['addr']} alloc_ts > peak_ts")

    def test_I8_no_unmatched_deallocs(self):
        self.assertEqual(int(self.out["diagnostics"]["unmatched_dealloc_count"]), 0,
                         "I8: unmatched_dealloc_count must be 0")

    def test_I9_peak_within_step_window(self):
        out = self.out
        lo, hi = out["step"]["range_ns"]
        self.assertLessEqual(int(lo), int(out["peak"]["ts_ns"]))
        self.assertLessEqual(int(out["peak"]["ts_ns"]), int(hi))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the invariants test**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest tests.test_invariants -v
```
Expected: all 9 cases pass. If any fails, the failure message names the invariant — fix the producer code in `_loader.py` / `memory_profile.py`, do NOT relax the test.

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/scripts/tests/test_invariants.py
git commit -m "test(tpu-perf): cross-rollup invariants I1-I9 + I2b for memory-profile"
```

---

### Task 9: Write SKILL.md content

**Files:**
- Modify: `plugins/tpu-perf/skills/memory-profile/SKILL.md` (replace stub from Task 1)

Follows the outline in spec §9. Modeled on `compute-breakdown/SKILL.md`.

- [ ] **Step 1: Replace the stub SKILL.md with full content**

```markdown
---
name: memory-profile
description: Use when analyzing TPU pretraining HBM occupancy from xplane.pb — locates the peak HBM moment, lists every buffer alive at that moment with size/shape/tf_op/parent_jit/lifetime_class, and rolls the alive set up by lifetime / shape / tf_op / parent jit / dtype. Reads schema documented by profile-anatomy.
argument-hint: "<profile_dir> [--step N | --step-policy {peak,last,first} | --all-trace] [--top K]"
---

# Memory Profile

Answer "when does HBM peak, and what is alive at that moment" for a TPU
pretraining profile, in a form Claude can read structurally and turn
into optimization recommendations. One Python entry script, single JSON
object on stdout, `status: ok | absent`.

This skill is built on top of `profile-anatomy`, which documents the
XSpace/XPlane/XLine/XEvent/XStat hierarchy. Read that first if you
need to know what an XEvent is, where allocator events live, or how
`XLine.timestamp_ns` and `XEvent.offset_ps` combine into a wall clock.

## When to use

Single purpose: "we want to reduce HBM peak — what is sitting in HBM at
the worst moment, and which call sites own it." If you need static
peak (XLA layout) or HLO-instruction-level attribution, that requires
`xla_dump/` artifacts and is **out of scope** for this skill — see
[Limitations](#limitations).

## Concepts you need first

- **`alive_at_peak`** is the set of buffers with `alloc_ts_ns ≤ peak.ts_ns < dealloc_ts_ns` (or no dealloc seen). The peak is the moment `Σ requested_bytes` of live buffers maximises within the chosen step window. The set is taken from runtime allocator events on `/host:CPU` (`MemoryAllocation` / `MemoryDeallocation`).
- **`lifetime_class`** is a heuristic over each alive buffer:
  - `persistent` ⇐ `crossed_step_boundaries ≥ persistent_threshold_steps` (default 2) **and** never deallocated within the trace.
  - `transient` ⇐ alloc and dealloc both within the same step interval.
  - `unknown` ⇐ otherwise (allocated near a step boundary; or trace truncation hides the dealloc — common, since the fixture has 3,698 allocs vs only 106 deallocs).
  Trace truncation biases `unknown` ↑. Use `crossed_step_boundaries` to separate truly-persistent (weights, optimizer state) from trace-truncated-unknown.
- **Timeline vs peak scope.** `timeline.samples` and `timeline.events_of_interest` span the **full trace** so cross-step trend is visible (plateau = persistent baseline; spike = per-step transient). `peak`, `alive_at_peak`, and `rollups` are scoped to the **chosen step** (default: the step containing the global peak).
- **Pool model.** HBM is `id=0`. Other pools are omitted by default; pass `--include-host-pools` to surface them. The fixture has only `id=0`.
- **Dealloc events do not carry pool `id`.** The skill matches deallocs to allocs by `addr` alone. Single-pool captures (the common case) are unambiguous; if a future capture surfaces multiple pools simultaneously, a `warnings` entry flags any same-addr-in-two-pools ambiguity.

## CLI and examples

```bash
# Default: peak step, Top-30 alive buffers, full-trace timeline
python3 .../memory_profile.py /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

# Whole trace, no step scoping
python3 .../memory_profile.py <profile_dir> --all-trace

# A specific Steps-line index
python3 .../memory_profile.py <profile_dir> --step 3

# Last step on the Steps line
python3 .../memory_profile.py <profile_dir> --step-policy last

# Larger Top-K and finer timeline
python3 .../memory_profile.py <profile_dir> --top 100 --time-samples 500
```

## JSON schema cheat-sheet

```json
{
  "status": "ok",
  "skill": "memory-profile",
  "version": 1,
  "inputs":   { "profile_dir": "...", "xplane_pb": "...", "n_planes": 8, "host_plane_present": true },
  "step":     { "id": 3, "policy": "peak", "range_ns": [lo, hi], "source": "steps_line" },
  "pool":     { "id": 0, "bytes_reserved": ... },
  "peak":     { "ts_ns": ..., "bytes_total": ..., "bytes_by_pool": {"0": ...},
                "fragmentation_at_peak": 0.07, "is_global_peak": true },
  "alive_at_peak": {
    "n_buffers": ..., "total_bytes": ...,
    "buffers": [ /* Top-K records with addr, size_bytes, shape, tf_op,
                    data_type, alloc_ts_ns, age_ns_at_peak,
                    crossed_step_boundaries, parent_chain,
                    lifetime_class, deallocated */ ],
    "tail": { "n_buffers": ..., "total_bytes": ... }
  },
  "rollups": {
    "by_lifetime_class": [...],   /* full set, no Top-K */
    "by_shape":          [...],   /* Top-K by total_bytes desc + tail */
    "by_tf_op":          [...],   /* Top-K + tail; <no tf_op> collapses */
    "by_parent_jit":     [...],   /* Top-K + tail */
    "by_dtype":          [...]    /* full set, no Top-K */
  },
  "timeline": {
    "samples": [ {"ts_ns": ..., "bytes_allocated": ..., "live_count": ..., "fragmentation": ...} ],
    "events_of_interest": [ {"kind": "global_peak"|"step_start"|"step_end"|"step_local_peak", ...} ],
    "axis_units": { "ts_ns": "nanoseconds since epoch", "bytes": "bytes (base-2)" }
  },
  "diagnostics": {
    "alloc_accounting_drift_pct": 0.42,
    "unmatched_dealloc_count": 0, "unmatched_alloc_count": 3592,
    "trace_end_live_bytes": ..., "n_pools_seen": 1,
    "pools_summary": [ {"pool_id": 0, "n_alloc": 3698, "n_dealloc": 106,
                        "max_peak_bytes_in_use": ...} ],
    "step_line_present": true,
    "shape_missing_count": 0, "tf_op_missing_count": 12,
    "warnings": []
  }
}
```

## Invariants (consistency gates)

| # | Invariant | Tolerance |
|---|---|---|
| I1 | `Σ buffers[*].size_bytes + tail.total_bytes == alive_at_peak.total_bytes` | exact |
| I2 | `alive_at_peak.total_bytes == peak.bytes_total` | exact |
| I2b | `\|peak.bytes_total − allocator's bytes_allocated at peak_ts\| / peak.bytes_total ≤ 0.01` | soft; >1% raises a warning. Recorded as `diagnostics.alloc_accounting_drift_pct`. |
| I3 | `Σ rollups.by_shape[*].total_bytes == alive_at_peak.total_bytes` | exact |
| I4 | each of `by_tf_op`, `by_parent_jit`, `by_lifetime_class`, `by_dtype` partitions `alive_at_peak.total_bytes` | exact |
| I5 | `peak.bytes_total ≤ pool.bytes_reserved` | exact |
| I6 | `max(timeline.samples.bytes_allocated) ≥ peak.bytes_total` | exact (timeline is full trace, peak is step-scoped) |
| I7 | every alive buffer has `alloc_ts_ns ≤ peak.ts_ns`; if `deallocated`, dealloc `ts_ns ≥ peak.ts_ns` | exact |
| I8 | `diagnostics.unmatched_dealloc_count == 0` | exact; nonzero is a real bug |
| I9 | `step.range_ns[0] ≤ peak.ts_ns ≤ step.range_ns[1]` (skipped under `--all-trace`) | exact |

## Reading guide

- **"Why is peak this high?"** → `rollups.by_lifetime_class`. `persistent` is the baseline (weights / optimizer state); `transient` is the per-step activation spike. The ratio tells you whether to attack persistent footprint (sharding, lower-precision weights) or activation footprint (rematerialization, smaller microbatch).
- **"Where to cut for biggest win?"** → `rollups.by_tf_op` Top few × `pct_of_peak`. The `parent_jit` rollup answers the same question scoped to one jit boundary.
- **"Is fragmentation severe?"** → `peak.fragmentation_at_peak` (allocator's own score 0..1) and `pool.bytes_reserved − peak.bytes_total` (raw headroom).
- **"Is this step an outlier?"** → `peak.is_global_peak` (chosen step holds the global peak) and `timeline.events_of_interest` step-local peaks across steps.

## Common gotchas

- **`alive_at_peak.tail` cannot be ignored.** `buffers` is Top-K only; `n_buffers` and `total_bytes` are the truth.
- **`step.source == "execute_event"`** means the `Steps` line was missing and the skill fell back to the outermost `Execute (jit_*)` event. In that case `step.id` is a sequential index, NOT the user's training step number.
- **Large `unmatched_alloc_count`** is the expected truncated-trace state, not a bug. Most allocs in a short capture stay live to the end.
- **Single HBM pool reported by default.** Multi-pool captures need `--include-host-pools` to surface host-side or other pools.
- **Dealloc events have no pool `id`.** Match-by-addr is unambiguous when only one pool is in flight (the common case). Same-addr-in-two-pools simultaneously is not produced by XLA's allocator today; the skill flags it via `warnings` if it ever appears.
- **No source-line attribution.** `MemoryAllocation` carries `tf_op` (JAXPR identity) but not `source_stack`. File:line attribution would need HLO buffer assignment, which lives in `xla_dump` and is out of scope.

## Limitations

- **No XLA static peak / no `addr → static slice` cross-check** — both need `xla_dump/`'s `*-memory-usage-report.txt`, which the user's normal capture flow does not produce.
- **No HLO-instruction-level attribution.** `hlo_proto.pb` in current captures lacks `buffer_assignment`. The runtime path uses `tf_op` instead.
- **No per-device split.** Allocator events live on `/host:CPU` and are not split per TPU core.
- **Trace truncation biases `lifetime_class`.** With 3,698 allocs and 106 deallocs in the reference fixture, almost every alive-at-peak buffer is also alive at end-of-trace. The `crossed_step_boundaries` field separates truly-persistent from trace-truncated-unknown.

## Files

- `scripts/memory_profile.py` — main entry script.
- `scripts/_loader.py` — xplane load, plane/line lookup, step window picker, two-pass sweep, rollups.
- `scripts/_proto/` — vendored xplane protobuf bindings (reused from `profile-anatomy/_proto/` via `sys.path.insert`).
- `scripts/tests/` — unit + e2e tests (stdlib `unittest`).
```

- [ ] **Step 2: Verify SKILL.md frontmatter is valid YAML**

Run:
```bash
python3 -c "import yaml,sys; t=open('plugins/tpu-perf/skills/memory-profile/SKILL.md').read(); body=t.split('---',2); print(yaml.safe_load(body[1]))"
```
Expected: prints `{'name': 'memory-profile', 'description': '...', 'argument-hint': '...'}` with no error.

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/memory-profile/SKILL.md
git commit -m "docs(tpu-perf): write SKILL.md for memory-profile skill"
```

---

### Task 10: Register skill in plugin manifest

**Files:**
- Modify: `plugins/tpu-perf/.claude-plugin/plugin.json`

- [ ] **Step 1: Inspect the current manifest**

Run:
```bash
cat plugins/tpu-perf/.claude-plugin/plugin.json
```
Expected: shows current `name`, `description`, `version`, and `skills` list (currently three: `profile-anatomy`, `compute-breakdown`, `comm-analysis`).

- [ ] **Step 2: Bump version and add `memory-profile` to the `skills` list**

Edit the file so it matches this shape (preserve all other fields exactly as currently present — only change `version`, `description`, and the `skills` array):

```json
{
  "name": "tpu-perf",
  "description": "Skills for analyzing TPU pretraining profiles: profile-anatomy (xplane.pb schema reference), compute-breakdown (HLO/MFU/roofline), comm-analysis (collective bandwidth + overlap), memory-profile (HBM peak + alive buffers).",
  "version": "0.3.0",
  "skills": [
    "profile-anatomy",
    "compute-breakdown",
    "comm-analysis",
    "memory-profile"
  ]
}
```

(If the existing manifest uses additional fields like `author`, `repository`, etc., keep them — only `version`, `description`, and `skills` are being changed.)

- [ ] **Step 3: Validate JSON**

Run:
```bash
python3 -c "import json; json.load(open('plugins/tpu-perf/.claude-plugin/plugin.json'))"
```
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add plugins/tpu-perf/.claude-plugin/plugin.json
git commit -m "chore(tpu-perf): register memory-profile skill, bump to 0.3.0"
```

---

### Task 11: Final verification — full test suite + sample CLI

- [ ] **Step 1: Run the full memory-profile test suite**

Run:
```bash
cd plugins/tpu-perf/skills/memory-profile/scripts && python3 -m unittest discover tests -v
```
Expected: all of `test_cli`, `test_loader`, `test_step_selection`, `test_alive_set`, `test_rollups`, `test_snapshot`, `test_invariants` pass (snapshot/invariants skip cleanly if the fixture is absent on this machine).

- [ ] **Step 2: Sample CLI invocation against the fixture**

Run:
```bash
python3 plugins/tpu-perf/skills/memory-profile/scripts/memory_profile.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
  | python3 -c "import json,sys; o=json.load(sys.stdin); print('status=', o['status']); print('peak.bytes_total=', o['peak']['bytes_total']); print('alive n_buffers=', o['alive_at_peak']['n_buffers']); print('rollup keys=', list(o['rollups'].keys())); print('warnings=', o['diagnostics']['warnings'])"
```
Expected: `status= ok`, non-zero `peak.bytes_total`, non-zero `alive n_buffers`, all 5 rollup keys present, `warnings= []` (or a known-explained list).

- [ ] **Step 3: Confirm sibling skills still pass**

Run:
```bash
cd plugins/tpu-perf/skills/compute-breakdown/scripts && python3 -m unittest discover tests -v
cd ../../comm-analysis/scripts && python3 -m unittest discover tests -v 2>/dev/null || true
```
Expected: compute-breakdown tests pass; comm-analysis test discovery either passes or reports no test module (unchanged from baseline). The intent is to confirm we did not break neighbors.

- [ ] **Step 4: Final summary commit (only if any tracked file changed in the previous steps — should be a no-op)**

Run:
```bash
git status
```
Expected: working tree clean. If anything is modified, investigate before committing.

---

## Self-Review

Run after writing all tasks above. Fix any issue inline; do not re-review.

**1. Spec coverage:**
- §0 Goal — Tasks 4–6 produce peak ts + alive set + attribution.
- §1 Scope / non-goals — Task 9 SKILL.md "Limitations" enumerates them.
- §2 Data sources — Task 2 loader reads `MemoryAllocation`/`MemoryDeallocation` on `/host:CPU`; Task 3 reads `Steps` on `/device:TPU:0`.
- §3 Skill layout — Task 1 scaffold + Task 10 plugin.json.
- §4 CLI — Task 6 wires every flag listed in the spec.
- §5 Algorithm — Task 2 (load), Task 3 (step pick), Task 4 (sweep + alive snapshot), Task 5 (rollups), Task 6 (timeline + diagnostics emission).
- §6 JSON schema — Task 6 emits the full envelope; Task 7 asserts shape.
- §6.1 Absent envelope — Task 1 absent path; Task 7 covers `--step out of range`.
- §7 Invariants — Task 8 covers I1–I9 + I2b.
- §8 Tests — Tasks 7, 8, plus per-component tests in Tasks 2–5.
- §9 SKILL.md outline — Task 9.
- §10 Limitations — Task 9 SKILL.md section.
- §11 Open questions — none, per spec.

**2. Placeholder scan:** none of "TBD", "implement later", "similar to Task N", "add appropriate error handling", or "write tests for the above" appear. Every code step contains the actual code.

**3. Type consistency:**
- `HostAllocatorEvents`, `StepWindow`, `TimelineSample`, `AliveBuffer`, `FirstPassResult`, `PeakSnapshot` are the dataclasses, defined in Task 2/3/4 and consumed in Tasks 5/6 with matching field names.
- `pick_step_window`, `sweep_first_pass`, `snapshot_at_peak`, `build_rollups` are the functions, defined in Tasks 3/4/5 and called in Task 6 with matching signatures.
- `lifetime_class` ∈ `{persistent, transient, unknown}` is consistent across spec §5.4, Task 4 sweep, Task 5 rollup, Task 8 invariants, Task 9 SKILL.md.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-25-tpu-perf-memory-profile.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?

