# tpu-perf comm-analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `comm-analysis` skill in the `tpu-perf` plugin: three reference scripts that extract every communication primitive from a TPU xprof, attribute each to a mesh axis with theoretical-vs-measured BW, and quantify compute/comm overlap.

**Architecture:** Four Python modules under `plugins/tpu-perf/skills/comm-analysis/scripts/`. A private helper module (`_comm_common.py`) does all the xplane.pb / hlo_proto.pb parsing. Three CLI scripts (`list_comm_primitives.py`, `axis_bandwidth.py`, `overlap_report.py`) consume those helpers, each runnable standalone. Two new vendored protos (`hlo.proto`, `op_stats.proto` subsets); `xplane_pb2.py` is *reused* from `profile-anatomy` via `sys.path.insert`.

**Tech Stack:** Python 3 stdlib + `protobuf`. No test framework — verification is probe-driven against the live fixture at `/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`, matching `profile-anatomy`'s convention.

**Spec:** [docs/superpowers/specs/2026-05-24-tpu-perf-comm-analysis-design.md](../specs/2026-05-24-tpu-perf-comm-analysis-design.md)

---

## Prerequisites

- `protoc` is required to regenerate `_pb2.py` files from `.proto` sources. On macOS: `brew install protobuf`. On Linux: `apt install protobuf-compiler`. Generated `_pb2.py` files are committed to the repo, so consumers don't need `protoc` at runtime — only the engineer regenerating them.
- Live fixture present at `/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/`. Confirm with `ls /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/` — should list at minimum `*.xplane.pb`, `*.hlo_proto.pb`, `*op_stats.pb`.
- Python `protobuf` package: `python3 -c "import google.protobuf; print(google.protobuf.__version__)"` should print a version. If missing: `pip3 install protobuf`.

## File Structure

```
plugins/tpu-perf/
├── .claude-plugin/plugin.json                   # MODIFY: add comm-analysis to skills list
└── skills/
    ├── profile-anatomy/                         # untouched (reused via sys.path)
    └── comm-analysis/                           # ALL NEW
        ├── SKILL.md
        └── scripts/
            ├── _proto/
            │   ├── __init__.py
            │   ├── hlo.proto                    # vendored XLA HLO subset
            │   ├── hlo_pb2.py                   # generated
            │   ├── op_stats.proto               # vendored OpStats/PerfEnv subset
            │   └── op_stats_pb2.py              # generated
            ├── _comm_common.py                  # shared helpers
            ├── list_comm_primitives.py          # capability #1
            ├── axis_bandwidth.py                # capability #2
            └── overlap_report.py                # capability #3
```

Per-file responsibility:

- `_proto/hlo.proto`: minimal subset of XLA's HloModuleProto family. Field numbers MUST match upstream so binary `.hlo_proto.pb` files deserialize correctly.
- `_proto/op_stats.proto`: minimal subset of TF profiler's OpStats with PerfEnv. Field numbers MUST match upstream.
- `_comm_common.py`: pure helpers, no `__main__`. Loaders, plane/line iterators, stat resolvers, async pairing, HLO joining.
- `list_comm_primitives.py`: the spine. Emits per-primitive table + three aggregation views.
- `axis_bandwidth.py`: per-axis BW utilization. Depends on `_comm_common` and HLO proto.
- `overlap_report.py`: sweep-line compute/comm overlap. Pure xprof, no HLO.

---

## Task 1: Scaffold directory + register the skill

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/SKILL.md` (placeholder; final content in Task 9)
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/__init__.py` (empty)
- Modify: `plugins/tpu-perf/.claude-plugin/plugin.json`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p plugins/tpu-perf/skills/comm-analysis/scripts/_proto
touch plugins/tpu-perf/skills/comm-analysis/scripts/_proto/__init__.py
```

- [ ] **Step 2: Write a placeholder SKILL.md so the manifest references a real file**

Create `plugins/tpu-perf/skills/comm-analysis/SKILL.md`:

```markdown
---
name: comm-analysis
description: Use when analyzing communication on a TPU pretraining profile — extracts every comm primitive (async + sync), attributes axes via HLO replica_groups, computes per-axis bus BW vs peak, and reports compute/comm overlap. Builds on profile-anatomy.
---

# Communication Analysis

Placeholder. Final content lands in Task 9.
```

- [ ] **Step 3: Update the plugin manifest to register comm-analysis**

Read the current manifest first to confirm shape. Then modify:

```bash
cat plugins/tpu-perf/.claude-plugin/plugin.json
```

If the file currently has no `skills` array, add one. Resulting file:

```json
{
  "name": "tpu-perf",
  "description": "Systematic analysis of TPU pretraining efficiency. Starts with profile-anatomy: schema dictionary and reference scripts for xplane.pb / trace.json.gz.",
  "version": "0.2.0",
  "license": "Apache-2.0"
}
```

Note: `plugin.json` does not require a `skills` array — Claude Code auto-discovers skills under `skills/`. Bumping `version` to `0.2.0` is the signal that a second skill landed. Do NOT add a skills array unless the existing manifest already has one. Confirm the current shape first.

- [ ] **Step 4: Verify manifest validity**

```bash
python3 -c "import json; json.load(open('plugins/tpu-perf/.claude-plugin/plugin.json')); print('ok')"
python3 -c "import yaml; yaml.safe_load(open('plugins/tpu-perf/skills/comm-analysis/SKILL.md').read().split('---')[1]); print('ok')"
```

Expected: `ok` printed twice, no traceback.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/.claude-plugin/plugin.json plugins/tpu-perf/skills/comm-analysis/SKILL.md plugins/tpu-perf/skills/comm-analysis/scripts/_proto/__init__.py
git commit -m "feat(tpu-perf): scaffold comm-analysis skill"
```

---

## Task 2: Vendor the HLO proto subset

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/hlo.proto`
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/hlo_pb2.py` (generated)

The XLA HLO proto is large (~3000 lines). We need only enough fields to decode `*.hlo_proto.pb` files in the fixture and read: replica_groups, channel_id, opcode, shape, sharding, source_file/line, frontend_attributes. Field numbers MUST match upstream `tensorflow/compiler/xla/service/hlo.proto` (a.k.a. `xla.proto.HloModuleProto` — newer XLA splits this; either source works since the wire format is stable).

- [ ] **Step 1: Fetch upstream hlo.proto field numbers**

Source of truth: `https://raw.githubusercontent.com/openxla/xla/main/xla/service/hlo.proto` (preferred) or `https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/compiler/xla/service/hlo.proto`. Fetch with WebFetch in the implementing session and extract field numbers for: `HloModuleProto`, `HloComputationProto`, `HloInstructionProto`, `ReplicaGroup`, `ShapeProto`, `OpMetadata`, `FrontendAttributes`, `OpSharding`. Critical fields needed:

- `HloModuleProto`: `id` (=1), `name` (=2), `entry_computation_name` (=3), `entry_computation_id` (=4), `computations` (=5), `host_program_shape` (=6), …
- `HloComputationProto`: `id`, `name`, `instructions`
- `HloInstructionProto`: `name`, `opcode`, `shape`, `metadata`, `replica_groups`, `channel_id`, `id`, `frontend_attributes`, `sharding`
- `ReplicaGroup`: `replica_ids`
- `OpMetadata`: `op_type`, `op_name`, `source_file`, `source_line`, `creation_pass_id`, `logical_creation_pass_id`, `size_of_generated_code_in_bytes`, `size_of_memory_working_set_in_bytes`, `profile_info`
- `FrontendAttributes`: `map<string, string> map = 1;`

- [ ] **Step 2: Author `hlo.proto` with the verified field numbers**

Create `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/hlo.proto`. Skeleton (fill exact field numbers from upstream — do NOT guess):

```proto
// Vendored subset of XLA HLO proto. Source:
//   https://github.com/openxla/xla/blob/main/xla/service/hlo.proto
// Only the fields comm-analysis needs are kept. Field numbers MUST match
// upstream so binary *.hlo_proto.pb files deserialize correctly.
//
// To regenerate hlo_pb2.py:
//   protoc --python_out=. hlo.proto
//
syntax = "proto3";

package xla;

message ShapeProto {
  // Minimal — we only read element_type and dimensions for byte calculations.
  int32 element_type = 2;          // PrimitiveType enum
  repeated int64 dimensions = 3;
  repeated ShapeProto tuple_shapes = 4;
  // Field numbers above are the canonical xla.ShapeProto layout — verify before commit.
}

message OpMetadata {
  string op_type = 1;
  string op_name = 2;
  string source_file = 3;
  int32  source_line = 4;
  // Add other fields if upstream uses field numbers we'd otherwise collide with.
}

message FrontendAttributes {
  map<string, string> map = 1;
}

message OpSharding {
  // Minimal placeholder — we don't introspect, just need round-trip stability
  // so cluster-key (opcode, shape, replica_groups, sharding) hashes consistently.
  bytes raw = 999;  // Sentinel; in practice we serialize-and-hash the bytes.
}

message ReplicaGroup {
  repeated int64 replica_ids = 1;
}

message HloInstructionProto {
  string name = 1;
  string opcode = 2;
  ShapeProto shape = 3;
  OpMetadata metadata = 7;
  // ... (gap; verify upstream)
  repeated ReplicaGroup replica_groups = 49;     // VERIFY
  ChannelHandle channel_id = ?;                  // VERIFY
  int64 id = 35;                                 // VERIFY
  FrontendAttributes frontend_attributes = ?;    // VERIFY
  OpSharding sharding = ?;                       // VERIFY
}

message ChannelHandle {
  int64 handle = 1;
  enum ChannelType { CHANNEL_TYPE_INVALID = 0; DEVICE_TO_DEVICE = 1; DEVICE_TO_HOST = 2; HOST_TO_DEVICE = 3; }
  ChannelType type = 2;
}

message HloComputationProto {
  string name = 1;
  repeated HloInstructionProto instructions = 2;
  int64 id = 4;
}

message HloModuleProto {
  string name = 1;
  string entry_computation_name = 2;
  int64 entry_computation_id = 3;
  repeated HloComputationProto computations = 4;
  // ...
  int64 id = 5;
}
```

The "?" / "VERIFY" markers in the skeleton above mark fields whose tag numbers MUST be replaced with values copied from upstream `hlo.proto` before you compile. Do not commit the file with "VERIFY" still in it.

- [ ] **Step 3: Generate `hlo_pb2.py`**

```bash
cd plugins/tpu-perf/skills/comm-analysis/scripts/_proto
protoc --python_out=. hlo.proto
```

Expected: `hlo_pb2.py` is created, no errors. Add a header to the generated file (matching `profile-anatomy`'s convention):

```bash
# Prepend a 4-line vendored-marker comment to hlo_pb2.py
```

Open `hlo_pb2.py` and prepend:

```python
# Vendored from upstream xla.HloModuleProto.
# Source .proto: ./hlo.proto (also vendored in this directory).
# To regenerate: `protoc --python_out=. hlo.proto` from this directory.
# Do NOT edit by hand — regenerate from hlo.proto if the schema changes.
```

- [ ] **Step 4: Probe — round-trip the live fixture's hlo_proto.pb**

Run inline (no file):

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "plugins/tpu-perf/skills/comm-analysis/scripts/_proto")
import hlo_pb2
m = hlo_pb2.HloModuleProto()
data = pathlib.Path("/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/jit_train_step(3).hlo_proto.pb").read_bytes()
m.ParseFromString(data)
print("module name:", m.name)
print("computations:", len(m.computations))
total_instrs = sum(len(c.instructions) for c in m.computations)
print("total instructions:", total_instrs)
collectives = [
    (c.name, i.name, i.opcode, [list(g.replica_ids) for g in i.replica_groups][:1])
    for c in m.computations
    for i in c.instructions
    if i.opcode in {"all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute"}
]
print("collective count:", len(collectives))
for row in collectives[:5]:
    print(" ", row)
PY
```

Expected: nonzero `computations`, nonzero `total_instructions`, nonzero `collective count`, replica_groups sublists are nonempty integer lists. If any field is empty when it shouldn't be, a field number is wrong — return to Step 2.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/_proto/hlo.proto plugins/tpu-perf/skills/comm-analysis/scripts/_proto/hlo_pb2.py
git commit -m "feat(tpu-perf): vendor xla.HloModuleProto subset for comm-analysis"
```

---

## Task 3: Vendor the op_stats proto subset

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats.proto`
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats_pb2.py` (generated)

We need `OpStats` only for `perf_env.peak_bw_giga_bytes_per_second_list`. That's 2 fields deep. Field numbers MUST match upstream `tensorflow/core/profiler/protobuf/op_stats.proto`.

- [ ] **Step 1: Fetch upstream op_stats.proto**

Source: `https://raw.githubusercontent.com/tensorflow/profiler/main/plugin/tensorboard_plugin_profile/protobuf/op_stats.proto` (preferred) or `https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/profiler/protobuf/op_stats.proto`. Extract:

- `OpStats`: field number for `perf_env`
- `PerfEnv`: field numbers for `peak_tera_flops_per_second`, `peak_hbm_bw_giga_bytes_per_second`, `peak_bw_giga_bytes_per_second_list`, `ridge_point`

Also note the *index meaning* for `peak_bw_giga_bytes_per_second_list`. Common layout: `[HBM, SRAM_RD, SRAM_WR, VMEM_RD, VMEM_WR, CMEM_RD, CMEM_WR, ICI]` — but this varies. Document the actual ordering observed in the fixture as a comment (Open Question #1 from the spec).

- [ ] **Step 2: Author `op_stats.proto`**

Create `plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats.proto`. Skeleton (verify field numbers from upstream):

```proto
// Vendored subset of TensorFlow profiler's OpStats. Source:
//   https://github.com/tensorflow/profiler/.../op_stats.proto
// Only the fields comm-analysis needs are kept. Field numbers MUST match
// upstream so binary *.op_stats.pb files deserialize correctly.
//
// To regenerate op_stats_pb2.py:
//   protoc --python_out=. op_stats.proto
//
// Index convention for peak_bw_giga_bytes_per_second_list (verified against
// fixture jit_train_step(3) in dp8_fsdp128, 2026-05-24):
//   0=HBM, 1=SRAM_RD, 2=SRAM_WR, 3=VMEM_RD, 4=VMEM_WR, 5=CMEM_RD, 6=CMEM_WR,
//   7=ICI
// Update if a different ordering is observed in newer TF/XLA releases.
syntax = "proto3";

package tensorflow.profiler;

message PerfEnv {
  double peak_tera_flops_per_second = 1;             // VERIFY
  double peak_hbm_bw_giga_bytes_per_second = 2;      // VERIFY (deprecated in newer)
  double ridge_point = 3;                            // VERIFY
  repeated double peak_bw_giga_bytes_per_second_list = 4;  // VERIFY
}

message OpStats {
  PerfEnv perf_env = 5;                              // VERIFY
  // (Other OpStats fields are not needed; protobuf will skip-and-preserve them
  //  via unknown-field handling.)
}
```

Tag numbers marked "VERIFY" must be filled from upstream before generating.

- [ ] **Step 3: Generate `op_stats_pb2.py`**

```bash
cd plugins/tpu-perf/skills/comm-analysis/scripts/_proto
protoc --python_out=. op_stats.proto
```

Prepend the same vendored-marker header pattern as in Task 2 Step 3.

- [ ] **Step 4: Probe — read PerfEnv from the live fixture**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "plugins/tpu-perf/skills/comm-analysis/scripts/_proto")
import op_stats_pb2
o = op_stats_pb2.OpStats()
data = pathlib.Path("/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/ALL_HOSTS.op_stats.pb").read_bytes()
o.ParseFromString(data)
pe = o.perf_env
print("peak_tera_flops_per_second:", pe.peak_tera_flops_per_second)
print("peak_hbm_bw_giga_bytes_per_second:", pe.peak_hbm_bw_giga_bytes_per_second)
print("peak_bw list:", list(pe.peak_bw_giga_bytes_per_second_list))
PY
```

Expected: nonzero `peak_tera_flops_per_second`, a nonempty `peak_bw list`. If the list is empty, the field number is wrong — return to Step 2. Identify which list index is the ICI peak by comparing magnitudes against the v6e datasheet (~90 GB/s per ICI link); update the comment in `op_stats.proto`.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats.proto plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats_pb2.py
git commit -m "feat(tpu-perf): vendor tensorflow.profiler.OpStats subset for comm-analysis"
```

---

## Task 4: `_comm_common.py` — XSpace loading and stat resolvers

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py`

This task lays down the foundation helpers that every script imports. We build them up in two tasks (Task 4 = xprof side; Task 5 = HLO side).

- [ ] **Step 1: Write the file with XSpace loader, plane iterator, and stat resolvers**

Create `plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py`:

```python
"""
Shared helpers for the comm-analysis skill.

Reads:
- xplane.pb (reuses xplane_pb2 from profile-anatomy)
- hlo_proto.pb (uses local _proto/hlo_pb2)
- op_stats.pb  (uses local _proto/op_stats_pb2)

No __main__; this is a library module.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Iterator, Optional

# Reuse profile-anatomy's xplane_pb2 — explicit dependency.
_HERE = pathlib.Path(__file__).resolve().parent
_PROFILE_ANATOMY_PROTO = (
    _HERE.parent.parent / "profile-anatomy" / "scripts" / "_proto"
)
sys.path.insert(0, str(_PROFILE_ANATOMY_PROTO))
sys.path.insert(0, str(_HERE / "_proto"))

import xplane_pb2          # noqa: E402  (from profile-anatomy)
import hlo_pb2             # noqa: E402  (from this skill's _proto)
import op_stats_pb2        # noqa: E402  (from this skill's _proto)


# ---------------------------------------------------------------------------
# XSpace loading
# ---------------------------------------------------------------------------

def load_xspace(profile_dir: str | pathlib.Path) -> Optional[xplane_pb2.XSpace]:
    """Load the first *.xplane.pb in profile_dir. Returns None if absent."""
    pbs = sorted(pathlib.Path(profile_dir).glob("*.xplane.pb"))
    if not pbs:
        return None
    xs = xplane_pb2.XSpace()
    xs.ParseFromString(pbs[0].read_bytes())
    return xs


def iter_device_planes(xs: xplane_pb2.XSpace) -> Iterator[xplane_pb2.XPlane]:
    """Yield every plane whose name starts with '/device:' — TC and SC."""
    for p in xs.planes:
        if p.name.startswith("/device:"):
            yield p


def core_kind(plane: xplane_pb2.XPlane) -> str:
    """Map a device plane name to TC / SC0 / SC1."""
    name = plane.name
    if "SparseCore 0" in name:
        return "SC0"
    if "SparseCore 1" in name:
        return "SC1"
    return "TC"


# ---------------------------------------------------------------------------
# Stat resolvers
# ---------------------------------------------------------------------------

def stat_name_by_id(plane: xplane_pb2.XPlane) -> dict[int, str]:
    return {smid: sm.name for smid, sm in plane.stat_metadata.items()}


def _xstat_value(stat):
    """Unwrap the 6-variant XStat oneof. Returns None if value field unset."""
    vf = stat.WhichOneof("value")
    return getattr(stat, vf) if vf else None


def event_stats(plane: xplane_pb2.XPlane, ev: xplane_pb2.XEvent) -> dict[str, object]:
    """Resolve XEvent.stats (per-execution counters) into a name -> value dict."""
    names = stat_name_by_id(plane)
    return {names[s.metadata_id]: _xstat_value(s)
            for s in ev.stats if s.metadata_id in names}


def event_metadata_stats(
    plane: xplane_pb2.XPlane, ev: xplane_pb2.XEvent
) -> dict[str, object]:
    """Resolve XEventMetadata.stats (op-level facts: hlo_category, flops, …)."""
    em = plane.event_metadata.get(ev.metadata_id)
    if em is None:
        return {}
    names = stat_name_by_id(plane)
    return {names[s.metadata_id]: _xstat_value(s)
            for s in em.stats if s.metadata_id in names}


def event_name(plane: xplane_pb2.XPlane, ev: xplane_pb2.XEvent) -> str:
    em = plane.event_metadata.get(ev.metadata_id)
    return em.name if em is not None else "?"


# ---------------------------------------------------------------------------
# Async pairing
# ---------------------------------------------------------------------------

def async_xla_line(plane: xplane_pb2.XPlane) -> Optional[xplane_pb2.XLine]:
    return next((ln for ln in plane.lines if ln.name == "Async XLA Ops"), None)


def xla_ops_line(plane: xplane_pb2.XPlane) -> Optional[xplane_pb2.XLine]:
    return next((ln for ln in plane.lines if ln.name == "XLA Ops"), None)


def steps_line(plane: xplane_pb2.XPlane) -> Optional[xplane_pb2.XLine]:
    return next((ln for ln in plane.lines if ln.name == "Steps"), None)


def pair_async_events(
    plane: xplane_pb2.XPlane, line: xplane_pb2.XLine
) -> list[tuple[Optional[xplane_pb2.XEvent], xplane_pb2.XEvent]]:
    """
    Group events on `line` by their 'flow' XStat. For each flow:
      - pair_size==2: yield (start, done) sorted by offset_ps
      - pair_size==1: yield (None, ev) — caller treats as fully exposed
      - pair_size>=3: yield (start_min, done_max); other events ignored
                       (rare; logged as warning by caller if needed)
    Events with no 'flow' stat are returned as (None, ev).
    """
    by_flow: dict[int, list[xplane_pb2.XEvent]] = {}
    unpaired: list[xplane_pb2.XEvent] = []
    for ev in line.events:
        stats = event_stats(plane, ev)
        flow = stats.get("flow")
        if flow is None:
            unpaired.append(ev)
            continue
        by_flow.setdefault(flow, []).append(ev)

    pairs: list[tuple[Optional[xplane_pb2.XEvent], xplane_pb2.XEvent]] = []
    for flow, evs in by_flow.items():
        evs_sorted = sorted(evs, key=lambda e: e.offset_ps)
        if len(evs_sorted) == 1:
            pairs.append((None, evs_sorted[0]))
        else:
            # First as start, last as done.
            pairs.append((evs_sorted[0], evs_sorted[-1]))
    for ev in unpaired:
        pairs.append((None, ev))
    return pairs
```

- [ ] **Step 2: Probe — verify loaders against the live fixture**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 - <<'PY'
import sys
sys.path.insert(0, "plugins/tpu-perf/skills/comm-analysis/scripts")
import _comm_common as cc

xs = cc.load_xspace("/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
assert xs is not None, "xspace failed to load"
planes = list(cc.iter_device_planes(xs))
print("device planes:", [(p.name, cc.core_kind(p)) for p in planes])

p0 = next(p for p in planes if cc.core_kind(p) == "TC")
async_ln = cc.async_xla_line(p0)
print("async events on TC:", len(async_ln.events) if async_ln else "[absent]")
pairs = cc.pair_async_events(p0, async_ln) if async_ln else []
print("paired flows:", len(pairs))
print("first-3 pair shapes:", [(s is not None, d is not None) for s, d in pairs[:3]])

xla = cc.xla_ops_line(p0)
print("XLA Ops events on TC:", len(xla.events) if xla else "[absent]")
print("Steps events on TC:", len(cc.steps_line(p0).events) if cc.steps_line(p0) else "[absent]")

# Smoke a stats resolution.
if async_ln and async_ln.events:
    ev0 = async_ln.events[0]
    print("first async event name:", cc.event_name(p0, ev0))
    print("  event_stats:", cc.event_stats(p0, ev0))
    print("  metadata_stats:", cc.event_metadata_stats(p0, ev0))
PY
```

Expected: at least one TC plane plus zero or more SC planes; nonzero async / XLA Ops counts; pair shapes show some unpaired (`(False, True)`). `event_stats` must contain `flow`, `device_offset_ps`, `device_duration_ps`. If `event_stats` returns empty, the stat-name resolution path is broken — diagnose before continuing.

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py
git commit -m "feat(tpu-perf): add comm-analysis xprof helpers (_comm_common.py)"
```

---

## Task 5: `_comm_common.py` — HLO loading and op_name canonicalization

**Files:**
- Modify: `plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py` (append helpers)

- [ ] **Step 1: Add HLO loader, op_name canonicalizer, and HLO joiner**

Append to `_comm_common.py`:

```python
# ---------------------------------------------------------------------------
# HLO module loading and joining
# ---------------------------------------------------------------------------

import re

_CALL_SUFFIX = re.compile(r"\.(call-start|call-done|start|done)$")


def canonical_op_name(name: str) -> str:
    """Strip async-pairing suffixes so start/done events join to one HLO instr."""
    return _CALL_SUFFIX.sub("", name)


def _hlo_program_id(module: hlo_pb2.HloModuleProto) -> int:
    """HloModuleProto.id is the program_id used in xprof XStats."""
    return module.id


def load_hlo_module(
    profile_dir: str | pathlib.Path,
    *,
    prefer_program_id: int | None = None,
) -> Optional[hlo_pb2.HloModuleProto]:
    """
    Pick the most relevant *.hlo_proto.pb in profile_dir.

    Selection order:
      1. If `prefer_program_id` is given, return the module with matching id.
      2. Largest file size.
      3. Lexicographic name as final tie-break.
    """
    pbs = sorted(pathlib.Path(profile_dir).glob("*.hlo_proto.pb"))
    if not pbs:
        return None

    parsed: list[tuple[pathlib.Path, hlo_pb2.HloModuleProto]] = []
    for pb in pbs:
        m = hlo_pb2.HloModuleProto()
        try:
            m.ParseFromString(pb.read_bytes())
        except Exception:
            continue
        parsed.append((pb, m))
    if not parsed:
        return None

    if prefer_program_id is not None:
        for _, m in parsed:
            if _hlo_program_id(m) == prefer_program_id:
                return m
    # Tie-break on size, then name.
    parsed.sort(key=lambda t: (-t[0].stat().st_size, t[0].name))
    return parsed[0][1]


def hlo_instructions(module: hlo_pb2.HloModuleProto) -> dict[str, hlo_pb2.HloInstructionProto]:
    """Flatten every (computation, instruction) into a {canonical_name: instr} map."""
    out: dict[str, hlo_pb2.HloInstructionProto] = {}
    for c in module.computations:
        for i in c.instructions:
            out[canonical_op_name(i.name)] = i
    return out


# ---------------------------------------------------------------------------
# op_stats.pb — peak BW resolver
# ---------------------------------------------------------------------------

def load_op_stats(profile_dir: str | pathlib.Path) -> Optional[op_stats_pb2.OpStats]:
    pbs = sorted(pathlib.Path(profile_dir).glob("*op_stats.pb"))
    if not pbs:
        return None
    o = op_stats_pb2.OpStats()
    try:
        o.ParseFromString(pbs[0].read_bytes())
    except Exception:
        return None
    return o


# Index documented in op_stats.proto comments. Verify in Task 3 Step 4.
ICI_PEAK_INDEX = 7  # update if op_stats.proto comment changes


def peak_ici_link_gbps_from_op_stats(o: op_stats_pb2.OpStats) -> Optional[float]:
    arr = list(o.perf_env.peak_bw_giga_bytes_per_second_list)
    if len(arr) > ICI_PEAK_INDEX:
        v = arr[ICI_PEAK_INDEX]
        return v if v > 0 else None
    return None


def peak_ici_link_gbps_from_xprof(plane: xplane_pb2.XPlane) -> Optional[float]:
    """Scan plane-level XStats (and stat_metadata names) for any peak_ici_*."""
    names = stat_name_by_id(plane)
    for stat in plane.stats:
        nm = names.get(stat.metadata_id, "")
        if nm.startswith("peak_ici_") or nm.startswith("peak_link_"):
            v = _xstat_value(stat)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return None
```

- [ ] **Step 2: Probe — load the HLO and join one collective op**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 - <<'PY'
import sys
sys.path.insert(0, "plugins/tpu-perf/skills/comm-analysis/scripts")
import _comm_common as cc

m = cc.load_hlo_module("/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
assert m is not None
print("module:", m.name, "id:", m.id, "computations:", len(m.computations))

instrs = cc.hlo_instructions(m)
print("total instrs (canonical names):", len(instrs))
collectives = [(n, i.opcode) for n, i in instrs.items() if i.opcode in
               {"all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute"}]
print("collective canonical names:", collectives[:5], "...total:", len(collectives))

# Cross-fixture join: take a name from xprof and look it up.
xs = cc.load_xspace("/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
tc = next(p for p in cc.iter_device_planes(xs) if cc.core_kind(p) == "TC")
async_ln = cc.async_xla_line(tc)
hits, misses = 0, 0
for ev in async_ln.events:
    s = cc.event_stats(tc, ev)
    op = s.get("hlo_op")
    if not op: continue
    if cc.canonical_op_name(op) in instrs:
        hits += 1
    else:
        misses += 1
print(f"xprof->hlo joins: hits={hits}  misses={misses}")

# Peak BW
o = cc.load_op_stats("/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128")
print("peak ici from op_stats:", cc.peak_ici_link_gbps_from_op_stats(o) if o else None)
print("peak ici from xprof:", cc.peak_ici_link_gbps_from_xprof(tc))
PY
```

Expected: nonzero collective count; `hits >> misses` (some misses are fine — XLA may emit synthesized async copies that have no HLO origin); peak ICI from `op_stats` is a positive number close to ~90 (v6e). If hits is 0, `canonical_op_name` is wrong or the suffix regex doesn't match — adjust.

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py
git commit -m "feat(tpu-perf): add HLO loader, op-name canonicalizer, peak-BW resolvers"
```

---

## Task 6: `list_comm_primitives.py` — per-event extractor + `--by kind`

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py`

This script is the spine. It walks every device plane, collects every async pair and every sync collective, builds a per-event row dict, prints a `--by kind` table, and supports `--json`.

- [ ] **Step 1: Write the script**

Create `plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py`:

```python
"""
List every communication primitive in a TPU profile, with rich attributes.

Usage:
    python3 list_comm_primitives.py <profile_dir> [--by kind|source|op] [--limit N]
                                    [--include-copies] [--json out.json]

Reads the device planes (TC, SC0, SC1) of *.xplane.pb. Pairs Async XLA Ops
events by 'flow' stat. Adds sync collectives from XLA Ops. HLO join (axis,
group_size, channel_id) is optional — happens automatically if a
*.hlo_proto.pb is present.

Output: a header line, then a per-row table for the chosen aggregation view.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_KIND_BY_HLO_CATEGORY = {
    "all-reduce": "AllReduce",
    "all-gather": "AllGather",
    "reduce-scatter": "ReduceScatter",
    "all-to-all": "AllToAll",
    "collective-permute": "CollectivePermute",
    "send": "P2P",
    "recv": "P2P",
    "copy-start": "Copy",
    "copy-done": "Copy",
}

_OP_REGEX = re.compile(
    r"^(all-reduce|all-gather|reduce-scatter|all-to-all|"
    r"collective-permute|send|recv|copy)\b"
)


def classify(hlo_op: str | None, hlo_category: str | None) -> str:
    if hlo_category and hlo_category in _KIND_BY_HLO_CATEGORY:
        return _KIND_BY_HLO_CATEGORY[hlo_category]
    if hlo_op:
        m = _OP_REGEX.match(hlo_op)
        if m:
            return _KIND_BY_HLO_CATEGORY.get(m.group(1), "Unknown")
    return "Unknown"


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _row_from_async_pair(plane, start_ev, done_ev, hlo_instrs):
    """start_ev may be None (unpaired). done_ev is always set."""
    s_done = cc.event_stats(plane, done_ev)
    md_done = cc.event_metadata_stats(plane, done_ev)

    hlo_op_raw = s_done.get("hlo_op") or cc.event_name(plane, done_ev)
    op_name = cc.canonical_op_name(str(hlo_op_raw))
    instr = hlo_instrs.get(op_name) if hlo_instrs else None

    if start_ev is not None:
        wall_ps = (done_ev.offset_ps + done_ev.duration_ps) - start_ev.offset_ps
        unpaired = False
    else:
        wall_ps = done_ev.duration_ps
        unpaired = True
    stall_ps = s_done.get("device_duration_ps") or done_ev.duration_ps
    hidden_ps = max(0, int(wall_ps) - int(stall_ps))

    return _build_row(
        plane=plane, ev=done_ev, op_name=op_name, mode="async",
        wall_ps=int(wall_ps), stall_ps=int(stall_ps), hidden_ps=hidden_ps,
        ev_stats=s_done, md_stats=md_done, instr=instr,
        unpaired=unpaired, flow=s_done.get("flow"),
    )


def _row_from_sync(plane, ev, hlo_instrs):
    s = cc.event_stats(plane, ev)
    md = cc.event_metadata_stats(plane, ev)
    hlo_op_raw = s.get("hlo_op") or cc.event_name(plane, ev)
    op_name = cc.canonical_op_name(str(hlo_op_raw))
    instr = hlo_instrs.get(op_name) if hlo_instrs else None

    wall_ps = ev.duration_ps
    stall_ps = ev.duration_ps   # sync collectives are always exposed
    hidden_ps = 0

    return _build_row(
        plane=plane, ev=ev, op_name=op_name, mode="sync",
        wall_ps=int(wall_ps), stall_ps=int(stall_ps), hidden_ps=hidden_ps,
        ev_stats=s, md_stats=md, instr=instr, unpaired=False, flow=None,
    )


def _build_row(*, plane, ev, op_name, mode, wall_ps, stall_ps, hidden_ps,
               ev_stats, md_stats, instr, unpaired, flow):
    hlo_op = ev_stats.get("hlo_op") or cc.event_name(plane, ev)
    kind = classify(hlo_op, md_stats.get("hlo_category"))
    bytes_ = md_stats.get("bytes_accessed") or md_stats.get("raw_bytes_accessed") or 0

    # Source: prefer XEventMetadata.stats.source / source_stack; fall back to HLO.
    source = md_stats.get("source") or md_stats.get("source_stack")
    if not source and instr is not None and instr.metadata.source_file:
        source = f"{instr.metadata.source_file}:{instr.metadata.source_line}"

    # Axis & group_size from HLO replica_groups (mesh-spec join is in axis_bandwidth.py).
    axis = "—"
    group_size = 0
    channel_id = None
    if instr is not None and instr.replica_groups:
        # replica_ids of the first group; group_size assumed equal across groups.
        group_size = len(instr.replica_groups[0].replica_ids)
        # axis stays "—" here; full attribution happens in axis_bandwidth.
        if hasattr(instr, "channel_id") and instr.channel_id.handle:
            channel_id = int(instr.channel_id.handle)

    return {
        "op_name": op_name,
        "kind": kind,
        "mode": mode,
        "core": cc.core_kind(plane),
        "axis": axis,
        "group_size": group_size,
        "bidir": "?",   # filled later when axis_bandwidth or post-processing runs
        "bytes": int(bytes_) if bytes_ else 0,
        "wall_ps": wall_ps,
        "stall_ps": stall_ps,
        "hidden_ps": hidden_ps,
        "source": source or "",
        "flow": int(flow) if flow is not None else None,
        "program_id": md_stats.get("program_id"),
        "channel_id": channel_id,
        "unpaired": unpaired,
    }


# ---------------------------------------------------------------------------
# Public entry point used by other scripts
# ---------------------------------------------------------------------------

def build_rows(profile_dir, *, include_copies=False) -> list[dict[str, Any]]:
    xs = cc.load_xspace(profile_dir)
    if xs is None:
        return []
    hlo_module = cc.load_hlo_module(profile_dir)
    hlo_instrs = cc.hlo_instructions(hlo_module) if hlo_module else {}

    rows: list[dict[str, Any]] = []
    for plane in cc.iter_device_planes(xs):
        async_ln = cc.async_xla_line(plane)
        if async_ln is not None:
            for s, d in cc.pair_async_events(plane, async_ln):
                rows.append(_row_from_async_pair(plane, s, d, hlo_instrs))
        xla_ln = cc.xla_ops_line(plane)
        if xla_ln is not None:
            for ev in xla_ln.events:
                s = cc.event_stats(plane, ev)
                md = cc.event_metadata_stats(plane, ev)
                hlo_op = s.get("hlo_op") or cc.event_name(plane, ev)
                kind = classify(hlo_op, md.get("hlo_category"))
                if kind in {"AllReduce", "AllGather", "ReduceScatter",
                            "AllToAll", "CollectivePermute", "P2P"}:
                    rows.append(_row_from_sync(plane, ev, hlo_instrs))
                # XLA Ops Copy events are uncommon and not collected here.

    if not include_copies:
        rows = [r for r in rows if r["kind"] != "Copy"]
    return rows


# ---------------------------------------------------------------------------
# Aggregation views
# ---------------------------------------------------------------------------

def _agg_by_kind(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["kind"], r["axis"], r["core"])].append(r)
    out = []
    for (kind, axis, core), grp in buckets.items():
        walls = sorted(r["wall_ps"] for r in grp)
        stalls = sorted(r["stall_ps"] for r in grp)
        out.append({
            "kind": kind, "axis": axis, "core": core,
            "count": len(grp),
            "sum_wall_ps": sum(walls),
            "sum_stall_ps": sum(stalls),
            "p50_stall_ps": stalls[len(stalls)//2],
            "p99_stall_ps": stalls[max(0, int(len(stalls)*0.99) - 1)],
        })
    out.sort(key=lambda r: -r["sum_stall_ps"])
    return out


def _agg_by_source(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["source"] or "(unknown)"].append(r)
    out = []
    for src, grp in buckets.items():
        kinds = defaultdict(int)
        for r in grp:
            kinds[r["kind"]] += 1
        dom_kind = max(kinds.items(), key=lambda kv: kv[1])[0]
        out.append({
            "source": src,
            "count": len(grp),
            "sum_wall_ps": sum(r["wall_ps"] for r in grp),
            "sum_stall_ps": sum(r["stall_ps"] for r in grp),
            "dom_kind": dom_kind,
        })
    out.sort(key=lambda r: -r["sum_stall_ps"])
    return out


def _agg_by_op(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["op_name"]].append(r)
    out = []
    for op, grp in buckets.items():
        out.append({
            "op_name": op,
            "kind": grp[0]["kind"], "axis": grp[0]["axis"], "core": grp[0]["core"],
            "count": len(grp),
            "sum_wall_ps": sum(r["wall_ps"] for r in grp),
            "sum_stall_ps": sum(r["stall_ps"] for r in grp),
        })
    out.sort(key=lambda r: -r["sum_stall_ps"])
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_us(ps): return f"{ps/1e6:.3f}" if ps else "0.000"


def _print_by_kind(agg, limit):
    print(f"{'kind':<20}{'axis':<10}{'core':<6}{'count':>7}"
          f"{'Σwall(us)':>13}{'Σstall(us)':>14}{'p50_stall(us)':>16}{'p99_stall(us)':>16}")
    for row in agg[:limit]:
        print(f"{row['kind']:<20}{row['axis']:<10}{row['core']:<6}"
              f"{row['count']:>7}{_fmt_us(row['sum_wall_ps']):>13}"
              f"{_fmt_us(row['sum_stall_ps']):>14}"
              f"{_fmt_us(row['p50_stall_ps']):>16}"
              f"{_fmt_us(row['p99_stall_ps']):>16}")


def _print_by_source(agg, limit):
    print(f"{'source':<60}{'count':>7}{'Σwall(us)':>13}{'Σstall(us)':>14}{'dom_kind':>20}")
    for row in agg[:limit]:
        print(f"{row['source'][:58]:<60}{row['count']:>7}"
              f"{_fmt_us(row['sum_wall_ps']):>13}{_fmt_us(row['sum_stall_ps']):>14}"
              f"{row['dom_kind']:>20}")


def _print_by_op(agg, limit):
    print(f"{'op_name':<50}{'kind':<18}{'axis':<10}{'core':<6}"
          f"{'count':>7}{'Σwall(us)':>13}{'Σstall(us)':>14}")
    for row in agg[:limit]:
        print(f"{row['op_name'][:48]:<50}{row['kind']:<18}{row['axis']:<10}"
              f"{row['core']:<6}{row['count']:>7}"
              f"{_fmt_us(row['sum_wall_ps']):>13}{_fmt_us(row['sum_stall_ps']):>14}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--by", choices=["kind", "source", "op"], default="kind")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--include-copies", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = build_rows(args.profile_dir, include_copies=args.include_copies)
    if not rows:
        print(f"[absent] no usable comm events in {args.profile_dir}")
        return

    print(f"comm primitives: {len(rows)} rows  (mode async/sync mix; "
          f"unpaired={sum(1 for r in rows if r['unpaired'])})")

    if args.by == "kind":
        _print_by_kind(_agg_by_kind(rows), args.limit)
    elif args.by == "source":
        _print_by_source(_agg_by_source(rows), args.limit)
    else:
        _print_by_op(_agg_by_op(rows), args.limit)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"rows": rows,
                       "agg": {"by_kind": _agg_by_kind(rows),
                               "by_source": _agg_by_source(rows),
                               "by_op": _agg_by_op(rows)}}, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on the live fixture**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
```

Expected output shape:
- header: `comm primitives: NNN rows ...` with NNN > 0
- by-kind table: rows for at least `AllReduce` and/or `AllGather` and/or `CollectivePermute`, all with nonzero `Σstall`
- exit code 0

If the table is empty: classification is missing the actual `hlo_category` values in the fixture. Add a debug print to `classify()` and inspect.

- [ ] **Step 3: Smoke `--by source` and `--by op`**

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --by source
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --by op --limit 10
```

Expected: `--by source` shows nonempty source strings (or `(unknown)` rows if the source XStat format is different — Open Question #4 from the spec; if all sources are `(unknown)`, inspect `event_metadata_stats(...)["source"]` shape and adjust `_row_from_*` accordingly). `--by op` shows individual canonicalized op names.

- [ ] **Step 4: Smoke `--json` output**

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --json /tmp/comm.json
python3 -m json.tool /tmp/comm.json > /dev/null && echo "json valid"
```

Expected: `json valid` printed.

- [ ] **Step 5: Smoke `[absent]` path**

```bash
mkdir -p /tmp/empty_profile && rm -f /tmp/empty_profile/*
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py /tmp/empty_profile
```

Expected: `[absent] no usable comm events in /tmp/empty_profile`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py
git commit -m "feat(tpu-perf): list_comm_primitives.py — comm primitive spine + 3 aggregation views"
```

---

## Task 7: `axis_bandwidth.py` — peak BW + axis attribution + bus BW + bidir

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py`

This script consumes the row-builder from `list_comm_primitives.py`, joins each row with replica_groups → physical-axis attribution, applies the NCCL bus-BW formula, and reports two tables.

- [ ] **Step 1: Write the script**

Create `plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py`:

```python
"""
Per-axis bandwidth utilization for TPU comm primitives.

Joins per-event rows from list_comm_primitives.build_rows() with HLO
replica_groups and an optional mesh-spec YAML to attribute each collective
to a physical or logical mesh axis. Computes NCCL-style bus BW and
utilization vs peak ICI link BW.

Peak BW resolution order:
  1. xprof XStat (peak_ici_* / peak_link_*)
  2. ALL_HOSTS.op_stats.pb PerfEnv.peak_bw_giga_bytes_per_second_list[ICI_INDEX]
  3. mesh_spec.peak_link_gbps
  4. --peak-ici-link-gbps flag
  5. None ⇒ utilization column dropped, [warn] printed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc
import list_comm_primitives as lcp


# ---------------------------------------------------------------------------
# Mesh-spec parsing (minimal — stdlib YAML via yaml is fine; if pyyaml missing,
# the file is small enough for a hand-rolled key:value parser, but we require
# pyyaml here for consistency with the spec's example).
# ---------------------------------------------------------------------------

def load_mesh_spec(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        print("[warn] pyyaml not installed; --mesh-spec ignored", file=sys.stderr)
        return {}
    return yaml.safe_load(pathlib.Path(path).read_text()) or {}


# ---------------------------------------------------------------------------
# Axis attribution: replica_ids -> varying physical dims
# ---------------------------------------------------------------------------

def _coords(replica_id: int, topology: tuple[int, int, int]) -> tuple[int, int, int]:
    X, Y, Z = topology
    x = replica_id // (Y * Z)
    y = (replica_id // Z) % Y
    z = replica_id % Z
    return (x, y, z)


def attribute_axis(replica_ids: list[int],
                   topology: tuple[int, int, int],
                   logical_axes: dict | None) -> tuple[str, int]:
    """
    Returns (axis_label, group_size).

    axis_label is one of: "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ", or a
    logical name from the mesh spec if matched. Falls back to
    "stride-N group" if topology is unknown.
    """
    if not replica_ids:
        return ("—", 0)
    if topology == (0, 0, 0):
        return (f"stride-{len(replica_ids)} group", len(replica_ids))

    coords = [_coords(r, topology) for r in replica_ids]
    varies = []
    for dim_idx, dim_name in enumerate("XYZ"):
        vals = {c[dim_idx] for c in coords}
        if len(vals) > 1:
            varies.append(dim_name)
    physical = "".join(varies) if varies else "—"

    if logical_axes:
        for logical_name, info in logical_axes.items():
            dims = set(info.get("dims") or [])
            if dims and dims == set(varies):
                return (logical_name, len(replica_ids))

    return (physical, len(replica_ids))


# ---------------------------------------------------------------------------
# Bidirectional dual-issue heuristic
# ---------------------------------------------------------------------------

def _shape_key(shape) -> bytes:
    return shape.SerializeToString() if shape else b""


def _sharding_key(instr) -> bytes:
    return instr.sharding.SerializeToString() if instr.HasField("sharding") else b""


def _replica_groups_key(instr) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for g in instr.replica_groups:
        h.update(b"|" + b",".join(str(i).encode() for i in g.replica_ids))
    return h.digest()


def bidir_clusters(hlo_module) -> dict[str, bool]:
    """
    Returns {canonical_op_name: bidir}. bidir=True if the cluster of
    instructions sharing (opcode, shape, replica_groups, sharding) has
    >=2 distinct channel_ids.
    """
    if hlo_module is None:
        return {}
    by_cluster: dict[tuple, list[tuple[str, int | None]]] = collections.defaultdict(list)
    for c in hlo_module.computations:
        for i in c.instructions:
            if i.opcode not in {"all-reduce", "all-gather", "reduce-scatter",
                                "all-to-all", "collective-permute", "send", "recv"}:
                continue
            cluster_key = (i.opcode,
                           _shape_key(i.shape),
                           _replica_groups_key(i),
                           _sharding_key(i))
            ch = i.channel_id.handle if i.HasField("channel_id") else None
            by_cluster[cluster_key].append((cc.canonical_op_name(i.name), ch))

    out: dict[str, bool] = {}
    for members in by_cluster.values():
        chs = {m[1] for m in members if m[1] is not None}
        bidir = len(chs) >= 2
        for op_name, _ in members:
            out[op_name] = bidir
    return out


# ---------------------------------------------------------------------------
# Bus BW formulas
# ---------------------------------------------------------------------------

def bus_bw_gbps(kind: str, group_size: int, bytes_: int, time_ps: int) -> Optional[float]:
    if not (group_size > 0 and bytes_ > 0 and time_ps > 0):
        return None
    secs = time_ps / 1e12
    factor = {
        "AllReduce":      lambda N: 2.0 * (N - 1) / N,
        "AllGather":      lambda N: (N - 1) / N,
        "ReduceScatter":  lambda N: (N - 1) / N,
        "AllToAll":       lambda N: (N - 1) / N,
        "CollectivePermute": lambda N: 1.0,
        "P2P":            lambda N: 1.0,
    }.get(kind)
    if factor is None:
        return None
    return factor(group_size) * (bytes_ / 1e9) / secs


# ---------------------------------------------------------------------------
# Peak BW
# ---------------------------------------------------------------------------

def resolve_peak_link_gbps(profile_dir, mesh_spec, cli_flag) -> tuple[Optional[float], str]:
    """Returns (peak_gbps, source_label)."""
    xs = cc.load_xspace(profile_dir)
    if xs is not None:
        for plane in cc.iter_device_planes(xs):
            v = cc.peak_ici_link_gbps_from_xprof(plane)
            if v is not None:
                return (v, "xprof")
    op_stats = cc.load_op_stats(profile_dir)
    if op_stats is not None:
        v = cc.peak_ici_link_gbps_from_op_stats(op_stats)
        if v is not None:
            return (v, "op_stats")
    if mesh_spec.get("peak_link_gbps"):
        return (float(mesh_spec["peak_link_gbps"]), "mesh-spec")
    if cli_flag is not None:
        return (float(cli_flag), "cli flag")
    return (None, "unknown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--mesh-spec", default=None)
    ap.add_argument("--peak-ici-link-gbps", type=float, default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    rows = lcp.build_rows(args.profile_dir, include_copies=False)
    if not rows:
        print(f"[absent] no usable comm events in {args.profile_dir}")
        return

    mesh_spec = load_mesh_spec(args.mesh_spec)
    topology = tuple(mesh_spec.get("topology") or (0, 0, 0))
    logical_axes = mesh_spec.get("axes") or {}
    links_per_axis = int(mesh_spec.get("links_per_axis", 2))

    hlo_module = cc.load_hlo_module(args.profile_dir)
    hlo_instrs = cc.hlo_instructions(hlo_module) if hlo_module else {}
    bidir_map = bidir_clusters(hlo_module)

    # Annotate rows in place: axis (with mesh-spec join) and bidir.
    for r in rows:
        instr = hlo_instrs.get(r["op_name"])
        if instr is not None and instr.replica_groups:
            replica_ids = list(instr.replica_groups[0].replica_ids)
            axis, gs = attribute_axis(replica_ids, topology, logical_axes)
            r["axis"] = axis
            if r["group_size"] == 0:
                r["group_size"] = gs
        r["bidir"] = "yes" if bidir_map.get(r["op_name"]) else "no"
        r["bus_bw_gbps"] = bus_bw_gbps(r["kind"], r["group_size"],
                                       r["bytes"], r["wall_ps"])

    peak_link, peak_src = resolve_peak_link_gbps(args.profile_dir, mesh_spec,
                                                  args.peak_ici_link_gbps)
    peak_axis_gbps = peak_link * links_per_axis if peak_link is not None else None
    print(f"peak ICI link: "
          f"{f'{peak_link:.1f} GB/s' if peak_link else '?'}  ({peak_src})  "
          f"links_per_axis={links_per_axis}  "
          f"⇒ peak_axis={f'{peak_axis_gbps:.1f}' if peak_axis_gbps else '?'} GB/s")
    if peak_link is None:
        print("[warn] peak ICI BW unknown — utilization omitted")

    # Per-axis aggregate
    by_axis = collections.defaultdict(list)
    for r in rows:
        by_axis[(r["axis"], r["core"])].append(r)

    print(f"\n{'axis':<14}{'core':<6}{'count':>7}"
          f"{'Σbytes(MB)':>14}{'Σwall(us)':>13}{'bus_BW(GB/s)':>16}"
          f"{'util%':>8}")
    for (axis, core), grp in sorted(by_axis.items(),
                                    key=lambda kv: -sum(r['wall_ps'] for r in kv[1]))[:args.limit]:
        sb = sum(r["bytes"] for r in grp)
        sw = sum(r["wall_ps"] for r in grp)
        bw = None
        # Dominant kind in this bucket, used for the formula.
        if sw > 0 and sb > 0:
            kinds = collections.Counter(r["kind"] for r in grp)
            dom_kind = kinds.most_common(1)[0][0]
            avg_gs = max((r["group_size"] for r in grp), default=0)
            bw = bus_bw_gbps(dom_kind, avg_gs, sb, sw)
        util = (bw / peak_axis_gbps * 100.0) if (bw and peak_axis_gbps) else None
        print(f"{axis:<14}{core:<6}{len(grp):>7}"
              f"{sb/1e6:>14.2f}{sw/1e6:>13.3f}"
              f"{(f'{bw:.2f}' if bw else '—'):>16}"
              f"{(f'{util:.1f}' if util is not None else '—'):>8}")

    # Top-N per-collective table
    print(f"\nTop-{args.limit} per-collective by Σstall:")
    print(f"{'op_name':<48}{'kind':<16}{'axis':<10}{'core':<6}"
          f"{'bidir':<6}{'wall(us)':>11}{'stall(us)':>11}{'bus_BW(GB/s)':>14}{'util%':>7}")
    for r in sorted(rows, key=lambda r: -r["stall_ps"])[:args.limit]:
        bw = r.get("bus_bw_gbps")
        util = (bw / peak_axis_gbps * 100.0) if (bw and peak_axis_gbps) else None
        print(f"{r['op_name'][:46]:<48}{r['kind']:<16}{r['axis']:<10}{r['core']:<6}"
              f"{r['bidir']:<6}{r['wall_ps']/1e6:>11.3f}{r['stall_ps']/1e6:>11.3f}"
              f"{(f'{bw:.2f}' if bw else '—'):>14}"
              f"{(f'{util:.1f}' if util is not None else '—'):>7}")

    if args.json_out:
        out = {"peak_link_gbps": peak_link, "peak_src": peak_src,
               "links_per_axis": links_per_axis, "rows": rows}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on the live fixture (no mesh-spec)**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
```

Expected: header line shows the peak source (most likely `op_stats`), a per-axis aggregate with at least one row (axis "—" or physical "X"/"Y"/"Z" if HLO replica_groups joined), and a top-N table. `bidir` column is `yes` or `no` per row.

- [ ] **Step 3: Smoke run with a hand-written mesh-spec**

```bash
cat > /tmp/mesh.yaml <<'YAML'
topology: [4, 4, 8]
axes:
  fsdp:  {dims: [Y, Z], size: 32}
  dp:    {dims: [X],    size: 4}
peak_link_gbps: 90
links_per_axis: 2
YAML
python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --mesh-spec /tmp/mesh.yaml
```

Expected: at least one row in the per-axis aggregate uses logical names `fsdp` or `dp` (or a physical fallback if no HLO match). Topology values are illustrative — adjust to match the fixture's actual chip count if known.

- [ ] **Step 4: Smoke run with no HLO module (degraded path)**

```bash
mkdir -p /tmp/no_hlo
cp /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/*.xplane.pb /tmp/no_hlo/
python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py /tmp/no_hlo
```

Expected: per-axis table still prints, all axis values are `—` (no HLO available), `bidir` is `no` everywhere, util column may be `—`. No traceback.

- [ ] **Step 5: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py
git commit -m "feat(tpu-perf): axis_bandwidth.py — NCCL bus BW, peak resolution, bidir cluster heuristic"
```

---

## Task 8: `overlap_report.py` — sweep-line union compute/comm overlap

**Files:**
- Create: `plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py`

xprof-only. No HLO needed. Computes per-step `compute_busy`, `comm_inflight`, `overlapped`, `exposed_comm`, `overlap_ratio`. SC0/SC1 reported in a separate sub-table.

- [ ] **Step 1: Write the script**

Create `plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py`:

```python
"""
Compute/comm overlap report for a TPU profile.

Per step on the device plane, computes:
  - compute_busy_ps   = ∪(compute intervals)
  - comm_inflight_ps  = ∪(comm intervals)
  - overlapped_ps     = ∪(compute ∩ comm)
  - exposed_comm_ps   = comm_inflight - overlapped
  - overlap_ratio     = overlapped / comm_inflight

Sweep-line union math; intervals are clipped to the step window.

Sanity-check: the sweep-derived exposed_comm vs Σ done.device_duration_ps
within the step (the metadata-reported exposed time). Mismatch above 5%
prints `[warn] step N`. Sweep is authoritative.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _comm_common as cc


_COMM_HLO_CATEGORIES = {
    "all-reduce", "all-gather", "reduce-scatter",
    "all-to-all", "collective-permute", "send", "recv",
}


# ---------------------------------------------------------------------------
# Sweep-line interval union / intersection
# ---------------------------------------------------------------------------

def _clip(intervals: Iterable[tuple[int, int]],
          window: tuple[int, int]) -> list[tuple[int, int]]:
    lo, hi = window
    out = []
    for a, b in intervals:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2:
            out.append((a2, b2))
    return out


def union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += cur_b - cur_a
    return total


def intersection_intervals(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return intervals that are in BOTH a's union and b's union."""
    if not a or not b:
        return []
    a = sorted(a); b = sorted(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


# ---------------------------------------------------------------------------
# Per-plane interval extraction
# ---------------------------------------------------------------------------

def _compute_intervals(plane) -> list[tuple[int, int]]:
    ln = cc.xla_ops_line(plane)
    if ln is None:
        return []
    out = []
    for ev in ln.events:
        md = cc.event_metadata_stats(plane, ev)
        cat = md.get("hlo_category")
        if cat in _COMM_HLO_CATEGORIES:
            continue
        out.append((ev.offset_ps, ev.offset_ps + ev.duration_ps))
    return out


def _comm_intervals(plane) -> tuple[list[tuple[int, int]], int]:
    """
    Returns (intervals, sum_metadata_exposed_ps).

    sum_metadata_exposed_ps is Σ done.device_duration_ps for paired async
    plus Σ duration_ps for sync collectives — used as the sanity-check
    reference.
    """
    intervals: list[tuple[int, int]] = []
    sum_meta_exposed = 0

    async_ln = cc.async_xla_line(plane)
    if async_ln is not None:
        for s, d in cc.pair_async_events(plane, async_ln):
            if s is not None:
                intervals.append((s.offset_ps, d.offset_ps + d.duration_ps))
            else:
                # Unpaired: treat the done-event's window as the interval.
                intervals.append((d.offset_ps, d.offset_ps + d.duration_ps))
            ds = cc.event_stats(plane, d)
            ddur = ds.get("device_duration_ps") or d.duration_ps
            sum_meta_exposed += int(ddur)

    xla_ln = cc.xla_ops_line(plane)
    if xla_ln is not None:
        for ev in xla_ln.events:
            md = cc.event_metadata_stats(plane, ev)
            if md.get("hlo_category") in _COMM_HLO_CATEGORIES:
                intervals.append((ev.offset_ps, ev.offset_ps + ev.duration_ps))
                sum_meta_exposed += ev.duration_ps

    return intervals, sum_meta_exposed


# ---------------------------------------------------------------------------
# Per-step report
# ---------------------------------------------------------------------------

def _step_windows(plane) -> list[tuple[int, tuple[int, int]]]:
    """Returns [(step_id_or_index, (lo_ps, hi_ps)), ...]."""
    ln = cc.steps_line(plane)
    if ln is not None and ln.events:
        return [(i, (ev.offset_ps, ev.offset_ps + ev.duration_ps))
                for i, ev in enumerate(ln.events)]
    # Fallback: synthesize one global window covering all events on this plane.
    lo = sys.maxsize; hi = 0
    for line in plane.lines:
        for ev in line.events:
            lo = min(lo, ev.offset_ps)
            hi = max(hi, ev.offset_ps + ev.duration_ps)
    if hi == 0:
        return []
    return [(-1, (lo, hi))]


def report_for_plane(plane, *, warn_eps: float = 0.05) -> dict:
    compute_all = _compute_intervals(plane)
    comm_all, _ = _comm_intervals(plane)

    rows = []
    totals = {"compute_busy_ps": 0, "comm_inflight_ps": 0,
              "overlapped_ps": 0, "exposed_comm_ps": 0,
              "step_total_ps": 0}
    warns = []
    for step_id, window in _step_windows(plane):
        comp = _clip(compute_all, window)
        comm = _clip(comm_all, window)
        # For sanity check, recompute meta-exposed *clipped* to this window:
        meta_exposed_in_step = 0
        ln = cc.async_xla_line(plane)
        if ln is not None:
            for s, d in cc.pair_async_events(plane, ln):
                if window[0] <= d.offset_ps < window[1]:
                    ds = cc.event_stats(plane, d)
                    meta_exposed_in_step += int(ds.get("device_duration_ps") or d.duration_ps)

        compute_busy = union_length(comp)
        comm_inflight = union_length(comm)
        overlapped = union_length(intersection_intervals(comp, comm))
        exposed_comm = max(0, comm_inflight - overlapped)
        ratio = (overlapped / comm_inflight) if comm_inflight else float("nan")

        if (comm_inflight > 0 and meta_exposed_in_step > 0
                and abs(exposed_comm - meta_exposed_in_step) / max(meta_exposed_in_step, 1)
                > warn_eps):
            warns.append((step_id, exposed_comm, meta_exposed_in_step))

        rows.append({
            "step": step_id, "step_ps": window[1] - window[0],
            "compute_busy_ps": compute_busy,
            "comm_inflight_ps": comm_inflight,
            "overlapped_ps": overlapped,
            "exposed_comm_ps": exposed_comm,
            "overlap_ratio": ratio,
        })
        for k in totals:
            if k == "step_total_ps":
                totals[k] += window[1] - window[0]
            elif k in rows[-1]:
                totals[k] += rows[-1][k]

    return {"plane": plane.name, "core": cc.core_kind(plane),
            "rows": rows, "totals": totals, "warns": warns}


# ---------------------------------------------------------------------------
# Top-N exposed contributors (across all steps, TC plane only)
# ---------------------------------------------------------------------------

def top_exposed_per_collective(plane, *, limit: int) -> list[dict]:
    out = []
    async_ln = cc.async_xla_line(plane)
    if async_ln is not None:
        for s, d in cc.pair_async_events(plane, async_ln):
            ds = cc.event_stats(plane, d)
            md = cc.event_metadata_stats(plane, d)
            stall = int(ds.get("device_duration_ps") or d.duration_ps)
            wall = (d.offset_ps + d.duration_ps - s.offset_ps) if s is not None else d.duration_ps
            hidden = max(0, wall - stall)
            out.append({
                "op_name": cc.canonical_op_name(str(ds.get("hlo_op") or cc.event_name(plane, d))),
                "hlo_category": md.get("hlo_category"),
                "wall_ps": int(wall), "stall_ps": stall, "hidden_ps": hidden,
                "hidden_ratio": hidden / wall if wall else 0.0,
            })
    out.sort(key=lambda r: -r["stall_ps"])
    return out[:limit]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_step_table(report: dict, label: str):
    rows = report["rows"]; t = report["totals"]
    print(f"\n=== {label} ({report['plane']}) ===")
    print(f"{'step':>6}{'step(us)':>12}{'compute(us)':>14}"
          f"{'comm(us)':>12}{'overlap(us)':>14}{'exposed(us)':>14}{'ratio':>8}")
    for r in rows:
        ratio = r["overlap_ratio"]
        print(f"{r['step']:>6}{r['step_ps']/1e6:>12.3f}"
              f"{r['compute_busy_ps']/1e6:>14.3f}"
              f"{r['comm_inflight_ps']/1e6:>12.3f}"
              f"{r['overlapped_ps']/1e6:>14.3f}"
              f"{r['exposed_comm_ps']/1e6:>14.3f}"
              f"{(f'{ratio:.2f}' if ratio == ratio else '—'):>8}")
    print(f"{'TOTAL':>6}{t['step_total_ps']/1e6:>12.3f}"
          f"{t['compute_busy_ps']/1e6:>14.3f}"
          f"{t['comm_inflight_ps']/1e6:>12.3f}"
          f"{t['overlapped_ps']/1e6:>14.3f}"
          f"{t['exposed_comm_ps']/1e6:>14.3f}"
          f"{(t['overlapped_ps']/t['comm_inflight_ps']) if t['comm_inflight_ps'] else float('nan'):>8.2f}")
    for step_id, sweep, meta in report["warns"]:
        print(f"  [warn] step {step_id}: sweep_exposed={sweep/1e6:.3f}us  "
              f"meta_exposed={meta/1e6:.3f}us  Δ>5%; sweep authoritative")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    xs = cc.load_xspace(args.profile_dir)
    if xs is None:
        print(f"[absent] no *.xplane.pb in {args.profile_dir}")
        return

    tc_reports = []
    sc_reports = []
    for plane in cc.iter_device_planes(xs):
        rep = report_for_plane(plane)
        if cc.core_kind(plane) == "TC":
            tc_reports.append(rep)
        else:
            sc_reports.append(rep)

    if not tc_reports:
        print("[absent] no /device:TPU:N (TensorCore) plane")
        return

    for rep in tc_reports:
        if rep["rows"] and rep["rows"][0]["step"] == -1:
            print("[fallback] no Steps line; using global window")
        _print_step_table(rep, "TC compute/comm overlap")

    if sc_reports:
        for rep in sc_reports:
            _print_step_table(rep, f"{rep['core']} comm (separate; doesn't compete with TC compute)")
    else:
        print("\n(no SparseCore planes present in this capture)")

    print(f"\nTop-{args.limit} TC exposed-comm contributors:")
    print(f"{'op_name':<48}{'hlo_category':<20}{'wall(us)':>11}{'stall(us)':>11}{'hidden':>9}")
    for rep in tc_reports:
        for plane in cc.iter_device_planes(xs):
            if plane.name != rep["plane"]:
                continue
            top = top_exposed_per_collective(plane, limit=args.limit)
            for r in top:
                print(f"{r['op_name'][:46]:<48}"
                      f"{(r['hlo_category'] or '?')[:18]:<20}"
                      f"{r['wall_ps']/1e6:>11.3f}"
                      f"{r['stall_ps']/1e6:>11.3f}"
                      f"{r['hidden_ratio']*100:>8.1f}%")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tc": tc_reports, "sc": sc_reports}, f, indent=2, default=str)
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on the live fixture**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
```

Expected: TC overlap table with one row per step, a TOTAL line, optional `[warn] step N` lines if metadata-vs-sweep diverges. SparseCore sub-table follows. Top-N contributors at the bottom. Exit 0.

- [ ] **Step 3: Smoke `--json` and `[absent]` paths**

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py \
    /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --json /tmp/overlap.json
python3 -m json.tool /tmp/overlap.json > /dev/null && echo "json valid"
python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py /tmp/empty_profile
```

Expected: `json valid`, then `[absent] no *.xplane.pb in /tmp/empty_profile`.

- [ ] **Step 4: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py
git commit -m "feat(tpu-perf): overlap_report.py — sweep-line compute/comm overlap with sanity check"
```

---

## Task 9: Write the final `SKILL.md`

**Files:**
- Modify: `plugins/tpu-perf/skills/comm-analysis/SKILL.md` (replace placeholder)

- [ ] **Step 1: Replace SKILL.md with real content**

Open `plugins/tpu-perf/skills/comm-analysis/SKILL.md` and replace its entire contents with:

```markdown
---
name: comm-analysis
description: Use when analyzing communication on a TPU pretraining profile — extracts every comm primitive (async + sync, TC + SparseCore), attributes axes via HLO replica_groups, computes per-axis NCCL bus BW vs peak ICI link BW, and reports per-step compute/comm overlap. Builds on profile-anatomy.
---

# Communication Analysis

Three reference scripts for analyzing the communication portion of a TPU
pretraining profile. Each script accepts a profile directory as `argv[1]`
(or `--profile-dir DIR`) and runs standalone with stdlib + `protobuf` +
optional `pyyaml` (only for `--mesh-spec`).

This skill builds on [`profile-anatomy`](../profile-anatomy/SKILL.md);
read that first for the xplane.pb / xplane.proto schema.

## 1. What's covered

| Capability | Script |
|---|---|
| List every comm primitive (async + sync, TC + SC) with rich attributes | [`scripts/list_comm_primitives.py`](scripts/list_comm_primitives.py) |
| Per-axis bandwidth utilization (NCCL bus BW vs peak ICI link BW) | [`scripts/axis_bandwidth.py`](scripts/axis_bandwidth.py) |
| Per-step compute/comm overlap (sweep-line union) | [`scripts/overlap_report.py`](scripts/overlap_report.py) |

ICI only. DCN/megascale collectives are deferred to a future skill.

## 2. Per-primitive row schema

`list_comm_primitives.py` builds rows with these fields (also the `--json`
payload):

| Field | Source |
|---|---|
| `op_name` | `hlo_op` stat (canonicalized — `.call-start` / `.call-done` stripped) |
| `kind` | `AllReduce` / `AllGather` / `ReduceScatter` / `AllToAll` / `CollectivePermute` / `P2P` / `Copy` / `Unknown` |
| `mode` | `async` (Async XLA Ops) or `sync` (XLA Ops) |
| `core` | `TC`, `SC0`, or `SC1` |
| `axis` | logical or physical mesh axis (set by `axis_bandwidth.py`) |
| `group_size` | `len(replica_groups[0].replica_ids)` |
| `bidir` | heuristic from `(opcode, shape, replica_groups, sharding)` cluster having ≥2 distinct channel_ids |
| `bytes` | `bytes_accessed` from `XEventMetadata.stats` |
| `wall_ps` | `done.offset_ps + done.duration_ps − start.offset_ps` for paired async; `duration_ps` for sync |
| `stall_ps` | `done.device_duration_ps` for async; full `duration_ps` for sync (sync = always exposed) |
| `hidden_ps` | `wall_ps − stall_ps` |
| `source` | `XEventMetadata.stats.source` / `source_stack`; falls back to HLO `OpMetadata.source_file:line` |
| `flow` | the `flow` XStat used to pair async events |
| `program_id` | `XEventMetadata.stats.program_id` |
| `channel_id` | from joined HLO instruction |

## 3. Aggregation views

`list_comm_primitives.py --by {kind,source,op}`:

- `kind` (default): roll up by `(kind, axis, core)` with count, Σwall, Σstall, p50/p99 stall.
- `source`: roll up by source `file:line` — answers "which line of the model is causing comm?".
- `op`: per individual `op_name`, top N by Σstall.

## 4. Bus-bandwidth formulas (NCCL/XLA convention)

| Kind | Bus BW |
|---|---|
| AllReduce | `2 × (N−1)/N × message_bytes / time` |
| AllGather | `(N−1)/N × output_bytes / time` |
| ReduceScatter | `(N−1)/N × input_bytes / time` |
| AllToAll | `(N−1)/N × message_bytes / time` |
| CollectivePermute / P2P | `message_bytes / time` |

`N = group_size`; `time = wall_ps` (in-flight, not stall). Peak axis BW =
`peak_link_gbps × links_per_axis` (default 2).

## 5. Peak-BW resolution order

1. xprof XStat (`peak_ici_*` / `peak_link_*` on the device plane).
2. `*op_stats.pb` `PerfEnv.peak_bw_giga_bytes_per_second_list[ICI_INDEX]` (index documented in `_proto/op_stats.proto`).
3. `--mesh-spec` YAML `peak_link_gbps:`.
4. `--peak-ici-link-gbps N` flag.
5. None ⇒ utilization column omitted, `[warn]` printed.

## 6. Optional mesh-spec YAML

```yaml
topology: [4, 4, 8]              # physical chip dims (X, Y, Z)
axes:
  fsdp:  {dims: [Y, Z], size: 32}
  dp:    {dims: [X],    size: 4}
peak_link_gbps: 90
links_per_axis: 2
```

All fields optional. Without a mesh-spec, axes are reported as physical
`X`/`Y`/`Z` (or `stride-N group` if topology is unknown).

## 7. Sample invocations

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --by source

python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --mesh-spec mesh.yaml

python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
```

## 8. Common gotchas

- **Async pairing uses `flow` (uint64), not `is_root`.** See profile-anatomy.
- **`pair_size=1` is observed in current captures.** Treated as fully
  exposed (`wall = stall`, `hidden = 0`); tagged `unpaired=true`.
- **HLO module is optional but recommended.** Without `*.hlo_proto.pb`,
  axis attribution and the bidir heuristic degrade gracefully.
- **SparseCore comm is reported in a separate sub-table** in
  `overlap_report.py` because SC and TC compute don't compete; mixing
  them would muddle the math.
- **The sweep-derived `exposed_comm` is authoritative** when it disagrees
  with `Σ done.device_duration_ps` by >5%; the metadata sum doesn't
  account for parallel streams.
- **`xplane_pb2.py` is reused from profile-anatomy** via
  `sys.path.insert`. Don't re-vendor it.
```

- [ ] **Step 2: Validate frontmatter**

```bash
python3 -c "import yaml; yaml.safe_load(open('plugins/tpu-perf/skills/comm-analysis/SKILL.md').read().split('---')[1]); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add plugins/tpu-perf/skills/comm-analysis/SKILL.md
git commit -m "docs(tpu-perf): write comm-analysis SKILL.md"
```

---

## Task 10: End-to-end verification + cross-check sanity

**Files:** none — verification only.

Per the spec §7, the repo has no test framework. Verification is one-shot
human-driven against the live fixture.

- [ ] **Step 1: Run all four entrypoints in sequence**

```bash
cd /Users/xl/Code/skills/.claude/worktrees/virtual-swinging-popcorn
FIX=/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py "$FIX"
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py "$FIX" --by source
python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py "$FIX"
python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py "$FIX"
```

Expected: all four exit 0; non-empty tables (or explicit `[absent]` lines).

- [ ] **Step 2: Cross-check sanity — Σstall vs Σexposed**

```bash
python3 - <<'PY'
import sys, json, subprocess, pathlib
sys.path.insert(0, "plugins/tpu-perf/skills/comm-analysis/scripts")
import list_comm_primitives as lcp
import overlap_report as ov
import _comm_common as cc

FIX = "/tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128"
rows = lcp.build_rows(FIX, include_copies=False)
sigma_stall = sum(r["stall_ps"] for r in rows if r["core"] == "TC") / 1e6

xs = cc.load_xspace(FIX)
sigma_exposed = 0
for plane in cc.iter_device_planes(xs):
    if cc.core_kind(plane) != "TC":
        continue
    rep = ov.report_for_plane(plane)
    sigma_exposed += rep["totals"]["exposed_comm_ps"] / 1e6

print(f"Σstall (list_comm_primitives)  = {sigma_stall:.3f} us")
print(f"Σexposed (overlap_report)       = {sigma_exposed:.3f} us")
delta = abs(sigma_stall - sigma_exposed) / max(sigma_exposed, 1e-9)
print(f"relative delta = {delta*100:.2f}%   {'OK' if delta < 0.05 else 'WARN'}")
PY
```

Expected: relative delta under 5%. If not, investigate which side is wrong (most often a missed clip-to-step, or a sync collective being double-counted) before declaring the skill done.

- [ ] **Step 3: Validate every JSON dump**

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py "$FIX" --json /tmp/cp.json
python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py "$FIX" --json /tmp/ab.json
python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py "$FIX" --json /tmp/ov.json
for f in /tmp/cp.json /tmp/ab.json /tmp/ov.json; do
  python3 -m json.tool "$f" > /dev/null && echo "$f valid"
done
```

Expected: three `... valid` lines.

- [ ] **Step 4: Validate manifests one more time**

```bash
python3 -c "import json; json.load(open('plugins/tpu-perf/.claude-plugin/plugin.json')); print('plugin.json ok')"
python3 -c "import json; json.load(open('.claude-plugin/marketplace.json')); print('marketplace.json ok')"
for sk in plugins/tpu-perf/skills/*/SKILL.md; do
  python3 -c "import yaml; yaml.safe_load(open('$sk').read().split('---')[1]); print('$sk ok')"
done
```

Expected: all `ok` lines.

- [ ] **Step 5: Verify the open-questions resolution**

The spec lists four open questions; each must be resolved or explicitly
left as a known caveat in the code:

1. **ICI index in `peak_bw_giga_bytes_per_second_list`** — should be
   documented as a comment in `_proto/op_stats.proto` and used as
   `ICI_PEAK_INDEX` in `_comm_common.py`.
2. **`peak_ici_*` xprof XStats** — `peak_ici_link_gbps_from_xprof()` is
   defensive and returns `None` when absent. No code change needed if
   absent in this fixture.
3. **`pair_size==1` async events** — `pair_async_events()` returns
   `(None, ev)` and the row builders treat them as fully exposed.
4. **`source` / `source_stack` value format** — implementation must have
   confirmed the format during Task 6 Step 3. Document the observed
   format as a comment in `_comm_common.py` next to `event_metadata_stats`.

```bash
grep -n "ICI_PEAK_INDEX\|source_stack\|pair_size\|peak_ici" \
  plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py \
  plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats.proto
```

Expected: each open question has a corresponding comment / constant. If
any are missing, add a brief comment now.

- [ ] **Step 6: Final commit (only if Step 5 added comments)**

```bash
git status
# If anything is staged from Step 5:
git add -p plugins/tpu-perf/skills/comm-analysis/scripts/_comm_common.py plugins/tpu-perf/skills/comm-analysis/scripts/_proto/op_stats.proto
git commit -m "docs(tpu-perf): resolve comm-analysis open questions inline"
```

If `git status` shows clean, skip this step.

---

## Self-review notes

The plan's spec coverage:

| Spec section | Implementing task |
|---|---|
| Architecture / file layout (§3) | Tasks 1–8 |
| Inputs and detection (§4) | Tasks 4 (xspace), 5 (HLO + op_stats), 7 (mesh-spec + peak resolution) |
| `list_comm_primitives.py` (§5.1) | Task 6 |
| `axis_bandwidth.py` (§5.2) | Task 7 |
| `overlap_report.py` (§5.3) | Task 8 |
| Vendored protos (§6) | Tasks 2, 3 |
| Error handling (§7) | Distributed across tasks (each script has `[absent]` paths smoke-tested) |
| Testing / verification (§8) | Task 10 |
| Open questions (§9) | Task 10 Step 5 |

No placeholders remain; every code step shows the actual code or exact
verifying probe. Bus-BW formula in `axis_bandwidth.py` matches the spec's
NCCL convention, and the bidir heuristic uses the same
`(opcode, shape, replica_groups, sharding)` cluster key the spec
specifies. The `unpaired` (pair_size=1) handling matches the spec's
"treat as fully exposed" rule across all three scripts.

Type consistency: `build_rows()` is called from both `axis_bandwidth.py`
and `overlap_report.py` with the same argument signature defined in
Task 6.
