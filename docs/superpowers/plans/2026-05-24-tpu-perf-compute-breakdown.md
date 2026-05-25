# tpu-perf compute-breakdown Skill Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second skill `compute-breakdown` to the existing `tpu-perf` plugin that turns a captured TPU pretraining profile (`*.xplane.pb`) into four actionable compute-efficiency analyses: top time-eaters by source line (`summary`), per-source full table for layer scoping (`by_source`), padding/cast/copy audit (`non_compute`), and v7x roofline shortfall (`roofline`). One skill, one main script, four `--mode` subcommands sharing a single load → step-pick → event-iterate → normalize pipeline.

**Architecture:** A single Python entry script `compute_breakdown.py` exposes four `--mode` subcommands. Stages 1–3 (load XSpace, pick step window, normalize per-event records) are shared helpers; stage 4 (mode-specific projection) dispatches on `--mode`. A small sibling module `_peaks.py` holds the TPU v7x peak table (per-device = per-TensorCore = per-chip ÷ 2) with CLI overrides. The xplane protobuf module is **copied** (not symlinked) from `profile-anatomy/scripts/_proto/` into this skill's `scripts/_proto/` so the skill is self-contained for distribution. Output contract: every invocation prints exactly one top-level JSON object on stdout, with `"status": "ok" | "absent"`.

**Tech Stack:** Python 3 stdlib (`argparse`, `dataclasses`, `gzip`, `hashlib`, `json`, `pathlib`, `re`, `sys`, `unittest`); `protobuf` runtime (already on system, transitive via xprof); upstream `tensorflow.profiler` xplane proto schema, vendored as `xplane_pb2.py`.

---

## Source spec

`docs/superpowers/specs/2026-05-24-tpu-perf-compute-breakdown-design.md` (round-3-approved). When this plan and the spec disagree, **the spec wins** — fix the plan before proceeding.

## Sample profile directory

`/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128/` — the same fixture used by `profile-anatomy`. Contains `gke-tpu-4233cc6e-d8q7.xplane.pb` (~298 MB) and `gke-tpu-4233cc6e-d8q7.trace.json.gz`. Treated as read-only; never copied into the repo.

For absent-input tests, use `/tmp` (no profile files).

## Predecessor

`plugins/tpu-perf/skills/profile-anatomy/` is the schema dictionary. This new skill consumes its schema (XSpace/XPlane/XLine/XEvent/XStat) and reuses its protobuf bindings (copied verbatim into this skill's `scripts/_proto/`). The marketplace already registers `tpu-perf` (during the profile-anatomy plan), so we only update the plugin manifest version.

## File structure (locked)

Every file added or modified by this plan:

```
plugins/tpu-perf/
├── .claude-plugin/
│   └── plugin.json                                                      # MODIFIED (version bump 0.1.0 -> 0.2.0)
└── skills/
    └── compute-breakdown/                                               # NEW directory
        ├── SKILL.md                                                     # NEW
        └── scripts/
            ├── _proto/
            │   ├── __init__.py                                          # NEW (copied verbatim)
            │   ├── xplane.proto                                         # NEW (copied verbatim)
            │   └── xplane_pb2.py                                        # NEW (copied verbatim)
            ├── _peaks.py                                                # NEW (~50 lines)
            ├── compute_breakdown.py                                     # NEW (~400-500 lines)
            └── tests/
                ├── __init__.py                                          # NEW (empty)
                ├── test_pipeline.py                                     # NEW (Stage 1-3 shared helpers)
                ├── test_summary_mode.py                                 # NEW (mode 1)
                ├── test_by_source_mode.py                               # NEW (mode 2)
                ├── test_non_compute_mode.py                             # NEW (mode 3)
                ├── test_roofline_mode.py                                # NEW (mode 4)
                ├── test_peaks.py                                        # NEW (_peaks.py)
                └── test_e2e.py                                          # NEW (cross-mode invariants on real fixture)
```

Total: 13 new files, 1 modified.

**Per-file responsibility:**

| File | Responsibility |
|---|---|
| `compute_breakdown.py` | CLI parser → load XSpace → pick step window → normalize records → dispatch mode → emit one JSON object. ~400-500 lines. |
| `_peaks.py` | Builtin v7x peak table (`bf16`, `fp8`, `fp16`, `fp32`, `hbm_gibps`); `resolve_peaks(args)` honors CLI overrides; emits `peaks_used` block. |
| `_proto/*` | Identical bytes to `profile-anatomy/scripts/_proto/`. The two copies must stay in sync; `xplane.proto` upstream changes update both. |
| `tests/test_pipeline.py` | Step admission, three-tier `agg_key` fallback, `kind` classification, `dtype` parsing, `dtype_uncertain` rule, while-skip + while_total accumulation, unresolved-event handling, `unknown_categories` accumulation. |
| `tests/test_summary_mode.py` | Mode 1: `top_compute_groups` ranking, `tail_compute`, `by_kind_rollup`, `pct_of_compute` / `pct_of_step` denominators, `flops_sum` / `bytes_accessed_sum` null-safe sum, `--top` default and override. |
| `tests/test_by_source_mode.py` | Mode 2: full-table emission (no truncation, no sort), `shapes` cap at 8 + `shapes_truncated` flag, `dtypes` histogram, `--include-data-move` toggle, no-while in groups. |
| `tests/test_non_compute_mode.py` | Mode 3: `by_category` two-layer table, `dtype_change` / `layout_change` regex (match / no-layout / no-match), default `async-done` inclusion + `--no-comm-stalls` flip, `non_compute_pct_of_*` denominators. |
| `tests/test_roofline_mode.py` | Mode 4: per-group MFU/HBM/bound formulas, ridge_point, `dtype_uncertain` propagation (no silent peak swap), `skipped_groups` counters (no_flops / no_bytes / dtype_other / peak_unknown), `weighted_avg_*`, `top_shortfall_groups`, `step_compute_duration_ps == summary.compute_duration_ps`. |
| `tests/test_peaks.py` | v7x builtin values (1153.5 / 2307.0 / 3690.0 / null fp32 / null fp16); `--peak-tflops-fp32` override changes both `peaks_used` and `source`; `unit` string. |
| `tests/test_e2e.py` | On the real `dp8_fsdp128` fixture, exercise §11 cross-mode invariants and sanity bounds. Skipped (`unittest.skipUnless`) when fixture is absent on the runner. |
| `SKILL.md` | YAML frontmatter (`name`, `description`, `argument-hint`); 4-mode usage guide; agg_key concept; v7x per-device peaks; concurrency disclaimer; `dtype_uncertain` interpretation; `null` ≠ "no change" for `dtype_change`/`layout_change`; layer-scoping recipe; common gotchas; file map. |
| `.claude-plugin/plugin.json` | Bump `version` from `0.1.0` to `0.2.0`. Update `description` to mention compute-breakdown alongside profile-anatomy. |

## Per-mode CLI surface (locked)

All four modes share these flags: `<profile_dir>` positional; `--mode {summary|by_source|non_compute|roofline}`; `--device PLANE` (default `/device:TPU:0`); `--step N` (int, default = middle step); `--step-id ID` (string, exact match against Step XEventMetadata.name); `--include-comm`. Mode-specific flags layered on top:

| Mode | Mode-specific flags |
|---|---|
| `summary` | `--top K` (default 50) |
| `by_source` | `--include-data-move` (default false) |
| `non_compute` | `--no-comm-stalls` (default async-done included) |
| `roofline` | `--chip v7x` (default), `--peak-tflops-bf16`, `--peak-tflops-fp8`, `--peak-tflops-fp32`, `--peak-tflops-fp16`, `--peak-hbm-gibps` |

Conflicts: passing both `--step` and `--step-id` → stderr error + exit 1.

**Cross-mode flag scoping (resolution).** All flags are global at the argparse layer (one parser, one `--help`). The mode-specific table above documents *intent*: `--top`, `--include-data-move`, `--no-comm-stalls`, `--include-comm`, and the roofline `--peak-*` / `--chip` flags are silently no-ops when a mode that doesn't consume them is selected. This is a deliberate UX decision (lower friction; matches `kubectl`, `gcloud`); the alternative — mode-aware sub-parsers — was rejected because it would force the user to remember which flag belongs where. The script does NOT error out on irrelevant flags; SKILL.md mentions this as a "no warnings on irrelevant flags" gotcha. The single conflict actively enforced is `--step` ⊕ `--step-id`.

## Test framework

Python stdlib `unittest`. Pattern verified in `plugins/agent-recap/skills/agent-recap/scripts/tests/test_scan_sessions.py`: `import unittest`, `class TestX(unittest.TestCase)`, `self.assertEqual(...)`, `if __name__ == "__main__": unittest.main()`. No pytest, no external deps. The `tests/` directory is package-style (`__init__.py` empty); tests are run as `python3 -m unittest discover -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v`.

**Test fixtures.** Two kinds:
1. **Synthetic XSpace fixtures**, hand-built in test setUp using `xplane_pb2.XSpace()`. Used for unit tests (Stage 1-3, each mode in isolation). Build helpers live in `tests/test_pipeline.py` and are imported by other test modules.
2. **Real fixture**, the `dp8_fsdp128` profile dir on disk. Used only by `test_e2e.py`, gated on `unittest.skipUnless(Path(FIXTURE).is_dir())`.

Synthetic fixtures avoid the 298 MB load cost in unit tests (each mode's tests run in <1 s) and avoid coupling unit tests to a specific profile that might not exist on every machine.

---

## Chunk 1: Scaffolding (manifest + vendored proto + skeleton)

This chunk gets the directory in place, vendors the proto bindings, bumps the plugin manifest, and creates a runnable empty `compute_breakdown.py` skeleton that returns `{"status":"absent","reason":"not_implemented"}` for any input. Every later chunk just adds to this skeleton.

**Files this chunk creates or modifies** (5 created + 1 modified):
- M: `plugins/tpu-perf/.claude-plugin/plugin.json` (version bump)
- N: `plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/{__init__.py, xplane.proto, xplane_pb2.py}` (3 vendored)
- N: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/__init__.py`
- N: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- N: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

`_peaks.py`, the per-mode test files, and `SKILL.md` are deferred to later chunks.

### Task 1: Bump plugin manifest version

**Files:**
- Modify: `plugins/tpu-perf/.claude-plugin/plugin.json`

- [ ] **Step 1: Read the current manifest**

  Run: `cat plugins/tpu-perf/.claude-plugin/plugin.json`
  Expected: `version: "0.1.0"` and a `description` mentioning only profile-anatomy.

- [ ] **Step 2: Edit the manifest**

  Replace the file with:

  ```json
  {
    "name": "tpu-perf",
    "description": "Systematic analysis of TPU pretraining efficiency. Profile schema reference (profile-anatomy) plus compute-efficiency analyses (compute-breakdown: source-line aggregation, layer scoping, non-compute audit, v7x roofline shortfall).",
    "version": "0.2.0",
    "license": "Apache-2.0"
  }
  ```

- [ ] **Step 3: Validate JSON**

  Run: `python3 -m json.tool plugins/tpu-perf/.claude-plugin/plugin.json > /dev/null && echo OK`
  Expected: `OK`.

- [ ] **Step 4: Commit**

  ```bash
  git add plugins/tpu-perf/.claude-plugin/plugin.json
  git commit -m "feat(tpu-perf): bump plugin version to 0.2.0 for compute-breakdown skill"
  ```

### Task 2: Create skill directory and vendor proto bindings

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/__init__.py` (copy verbatim from profile-anatomy)
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane.proto` (copy verbatim)
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane_pb2.py` (copy verbatim)

The spec mandates a *copy* (not symlink) so the skill is self-contained.

- [ ] **Step 1: Create directory tree**

  ```bash
  mkdir -p plugins/tpu-perf/skills/compute-breakdown/scripts/_proto
  mkdir -p plugins/tpu-perf/skills/compute-breakdown/scripts/tests
  ```

- [ ] **Step 2: Copy the three vendored files verbatim**

  ```bash
  cp plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/__init__.py \
     plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/__init__.py
  cp plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto \
     plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane.proto
  cp plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py \
     plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane_pb2.py
  ```

- [ ] **Step 3: Verify byte-identical**

  Run:
  ```bash
  diff -q plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane.proto \
          plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane.proto && \
  diff -q plugins/tpu-perf/skills/profile-anatomy/scripts/_proto/xplane_pb2.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane_pb2.py && \
  echo "vendored proto OK"
  ```
  Expected: `vendored proto OK`.

- [ ] **Step 4: Confirm proto imports cleanly from the new location**

  Run:
  ```bash
  python3 -c "
  import sys
  sys.path.insert(0, 'plugins/tpu-perf/skills/compute-breakdown/scripts/_proto')
  import xplane_pb2
  xs = xplane_pb2.XSpace()
  print('XSpace fields:', sorted(f.name for f in xs.DESCRIPTOR.fields))
  "
  ```
  Expected: `XSpace fields: ['errors', 'hostnames', 'planes', 'warnings']`.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/
  git commit -m "feat(tpu-perf): vendor xplane proto bindings into compute-breakdown skill"
  ```

### Task 3: Create empty test package marker

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/__init__.py` (empty)

- [ ] **Step 1: Create empty file**

  ```bash
  : > plugins/tpu-perf/skills/compute-breakdown/scripts/tests/__init__.py
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/tests/__init__.py
  git commit -m "feat(tpu-perf): add tests package marker for compute-breakdown"
  ```

### Task 4: Stub `compute_breakdown.py` with CLI parser only

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`

The skeleton implements only argument parsing + a single absent-input check. All four modes route to the same stub that emits `{"status": "absent", "reason": "not_implemented", "mode": <mode>}`. Each subsequent task fills in real logic, replacing the stub for that mode.

- [ ] **Step 1: Write the failing test for `--help` exit code**

  Write to `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`:

  ```python
  """Unit tests for the compute_breakdown.py shared pipeline (Stages 1-3) and CLI."""
  import json
  import subprocess
  import sys
  import unittest
  from pathlib import Path

  SCRIPT = Path(__file__).resolve().parent.parent / "compute_breakdown.py"


  class TestCLI(unittest.TestCase):
      def test_help_runs(self):
          r = subprocess.run(
              [sys.executable, str(SCRIPT), "--help"],
              capture_output=True, text=True,
          )
          self.assertEqual(r.returncode, 0, r.stderr)
          self.assertIn("--mode", r.stdout)
          self.assertIn("summary", r.stdout)
          self.assertIn("by_source", r.stdout)
          self.assertIn("non_compute", r.stdout)
          self.assertIn("roofline", r.stdout)

      def test_no_xplane_returns_absent(self):
          r = subprocess.run(
              [sys.executable, str(SCRIPT), "/tmp", "--mode", "summary"],
              capture_output=True, text=True,
          )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["status"], "absent")
          self.assertEqual(doc["reason"], "no_xplane_pb")
          self.assertEqual(doc["mode"], "summary")
          self.assertEqual(doc["profile_dir"], "/tmp")
          self.assertEqual(doc["notes"], [])

      def test_step_and_step_id_mutually_exclusive(self):
          r = subprocess.run(
              [sys.executable, str(SCRIPT), "/tmp",
               "--mode", "summary", "--step", "0", "--step-id", "x"],
              capture_output=True, text=True,
          )
          self.assertEqual(r.returncode, 1)
          self.assertIn("step", r.stderr.lower())


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run the test to verify it fails**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests \
    -p "test_pipeline.py" -v
  ```
  Expected: FAIL or ERROR (no `compute_breakdown.py` exists yet).

- [ ] **Step 3: Write the minimal CLI skeleton**

  Write to `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`:

  ```python
  """
  Entry point for tpu-perf:compute-breakdown.

  Four --mode subcommands share a load -> step-pick -> event-iterate ->
  normalize pipeline (Stages 1-3); only Stage 4 (final projection)
  differs per mode. Output is exactly one top-level JSON object on
  stdout. See spec
  docs/superpowers/specs/2026-05-24-tpu-perf-compute-breakdown-design.md.
  """
  import argparse
  import json
  import pathlib
  import sys

  # Make the vendored protobuf module importable regardless of cwd. Stage 1
  # helpers (added in chunk 2) call xplane_pb2.XSpace().ParseFromString(...).
  _PROTO_DIR = pathlib.Path(__file__).parent / "_proto"
  sys.path.insert(0, str(_PROTO_DIR))
  import xplane_pb2  # noqa: E402  (after sys.path insert, by design)


  def build_parser() -> argparse.ArgumentParser:
      p = argparse.ArgumentParser(
          prog="compute_breakdown.py",
          description="Compute-efficiency analysis of a TPU pretraining profile.",
      )
      p.add_argument("profile_dir", help="Path to a profile directory containing *.xplane.pb")
      p.add_argument("--mode", required=True,
                     choices=["summary", "by_source", "non_compute", "roofline"])
      p.add_argument("--device", default="/device:TPU:0",
                     help="XPlane name to analyze (default: /device:TPU:0)")
      p.add_argument("--step", type=int, default=None,
                     help="0-indexed step to analyze (default: middle step)")
      p.add_argument("--step-id", default=None,
                     help="Exact match against Step XEventMetadata.name")
      p.add_argument("--include-comm", action="store_true",
                     help="Include kind=comm events in the analysis")

      # Mode 1
      p.add_argument("--top", type=int, default=50,
                     help="(summary) top K compute groups to emit (default: 50)")
      # Mode 2
      p.add_argument("--include-data-move", action="store_true",
                     help="(by_source) also emit kind=data_move groups")
      # Mode 3
      p.add_argument("--no-comm-stalls", action="store_true",
                     help="(non_compute) exclude async-done from non-compute table")
      # Mode 4
      p.add_argument("--chip", default="v7x", choices=["v7x"],
                     help="(roofline) chip generation; only v7x supported today")
      p.add_argument("--peak-tflops-bf16", type=float, default=None)
      p.add_argument("--peak-tflops-fp8", type=float, default=None)
      p.add_argument("--peak-tflops-fp32", type=float, default=None)
      p.add_argument("--peak-tflops-fp16", type=float, default=None)
      p.add_argument("--peak-hbm-gibps", type=float, default=None)
      return p


  def _emit(doc: dict) -> None:
      json.dump(doc, sys.stdout)
      sys.stdout.write("\n")


  def _absent(reason: str, mode: str, profile_dir: str) -> dict:
      return {"status": "absent", "reason": reason, "mode": mode,
              "profile_dir": profile_dir, "notes": []}


  def main(argv=None) -> int:
      args = build_parser().parse_args(argv)

      if args.step is not None and args.step_id is not None:
          print("error: cannot pass both --step and --step-id", file=sys.stderr)
          return 1

      profile_dir = pathlib.Path(args.profile_dir)
      pbs = sorted(profile_dir.glob("*.xplane.pb")) if profile_dir.is_dir() else []
      if not pbs:
          _emit(_absent("no_xplane_pb", args.mode, args.profile_dir))
          return 0

      # Stages 1-4 not yet implemented; placeholder for chunk 2+.
      _emit(_absent("not_implemented", args.mode, args.profile_dir))
      return 0


  if __name__ == "__main__":
      raise SystemExit(main())
  ```

- [ ] **Step 4: Run the tests; all pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests \
    -p "test_pipeline.py" -v
  ```
  Expected: `Ran 3 tests in ...s` `OK`.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): scaffold compute_breakdown.py CLI skeleton"
  ```

---

## Chunk 2: Stage 1–3 shared pipeline + synthetic-fixture builders

This chunk fills in the load → step-pick → event-iterate → normalize pipeline. After this chunk, the script can produce a flat list of `EventRecord` objects from any XSpace, but no mode is wired up yet (`--mode summary` etc. still emit `{"status":"absent","reason":"not_implemented"}` until chunk 3).

The order is strict TDD: every algorithm gets a failing test before it gets implementation. Synthetic XSpace fixtures (built in Python with `xplane_pb2.XSpace()`) are introduced first so all later tests can build on them.

### Task 5: Synthetic XSpace builder helpers in tests

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

These builders synthesize an `XSpace` proto with the same shape `compute_breakdown.py` reads (device plane, `Steps` line, `XLA Ops` line, populated `event_metadata` and `stat_metadata`). Used by every later test in this chunk and by mode-specific test files in chunk 3.

- [ ] **Step 1: Add module-level builder helpers below the existing TestCLI class**

  Append to `tests/test_pipeline.py`:

  ```python
  # ----------------------------------------------------------------------
  # Synthetic XSpace builders.
  # ----------------------------------------------------------------------
  import pathlib
  _PROTO_DIR = pathlib.Path(__file__).resolve().parent.parent / "_proto"
  sys.path.insert(0, str(_PROTO_DIR))
  import xplane_pb2  # noqa: E402


  def _add_stat_meta(plane, sm_id: int, name: str) -> None:
      sm = plane.stat_metadata[sm_id]
      sm.id = sm_id
      sm.name = name


  def _add_event_meta(plane, em_id: int, name: str, stats: dict | None = None) -> None:
      """stats: {stat_metadata_id: ('str_value'|'int64_value'|'uint64_value'|'double_value', value)}"""
      em = plane.event_metadata[em_id]
      em.id = em_id
      em.name = name
      for sm_id, (vfield, vval) in (stats or {}).items():
          s = em.stats.add()
          s.metadata_id = sm_id
          setattr(s, vfield, vval)


  def _add_event(line, em_id: int, offset_ps: int, duration_ps: int,
                 per_event_stats: dict | None = None) -> None:
      ev = line.events.add()
      ev.metadata_id = em_id
      ev.offset_ps = offset_ps
      ev.duration_ps = duration_ps
      for sm_id, (vfield, vval) in (per_event_stats or {}).items():
          s = ev.stats.add()
          s.metadata_id = sm_id
          setattr(s, vfield, vval)


  # Stat-metadata IDs used across synthetic fixtures (arbitrary but stable).
  SM_HLO_CATEGORY = 1
  SM_TF_OP = 2
  SM_PROGRAM_ID = 3
  SM_FLOPS = 4
  SM_MODEL_FLOPS = 5
  SM_BYTES_ACCESSED = 6
  SM_RAW_BYTES_ACCESSED = 7
  SM_SHAPE = 8
  SM_SOURCE = 9
  SM_SOURCE_STACK = 10
  SM_FLOW = 11
  SM_DEVICE_DURATION_PS = 12
  SM_DEDUP_NAME = 13


  def make_minimal_xspace(*, device_name: str = "/device:TPU:0",
                           steps: list[tuple[int, int, int]] | None = None) -> "xplane_pb2.XSpace":
      """Build an XSpace with one device plane carrying a `Steps` line and an empty
      `XLA Ops` line. `steps` is [(em_id, offset_ps, duration_ps), ...]."""
      xs = xplane_pb2.XSpace()
      plane = xs.planes.add()
      plane.id = 1
      plane.name = device_name

      _add_stat_meta(plane, SM_HLO_CATEGORY, "hlo_category")
      _add_stat_meta(plane, SM_TF_OP, "tf_op")
      _add_stat_meta(plane, SM_PROGRAM_ID, "program_id")
      _add_stat_meta(plane, SM_FLOPS, "flops")
      _add_stat_meta(plane, SM_MODEL_FLOPS, "model_flops")
      _add_stat_meta(plane, SM_BYTES_ACCESSED, "bytes_accessed")
      _add_stat_meta(plane, SM_RAW_BYTES_ACCESSED, "raw_bytes_accessed")
      _add_stat_meta(plane, SM_SHAPE, "shape_with_layout")
      _add_stat_meta(plane, SM_SOURCE, "source")
      _add_stat_meta(plane, SM_SOURCE_STACK, "source_stack")
      _add_stat_meta(plane, SM_FLOW, "flow")
      _add_stat_meta(plane, SM_DEVICE_DURATION_PS, "device_duration_ps")
      _add_stat_meta(plane, SM_DEDUP_NAME, "deduplicated_name")

      steps_line = plane.lines.add()
      steps_line.id = 100
      steps_line.name = "Steps"
      steps_line.timestamp_ns = 0
      steps_line.duration_ps = 0
      for em_id, off, dur in (steps or []):
          _add_event_meta(plane, em_id, f"step_{em_id}")
          _add_event(steps_line, em_id, off, dur)

      ops_line = plane.lines.add()
      ops_line.id = 101
      ops_line.name = "XLA Ops"
      ops_line.timestamp_ns = 0
      ops_line.duration_ps = 0
      return xs


  def add_hlo_event(xs, *, em_id: int, hlo_op_text: str, offset_ps: int,
                     duration_ps: int, hlo_category: str,
                     tf_op: str | None = None,
                     source_stack: str | None = None,
                     source_inner: str | None = None,
                     flops: int | None = None,
                     bytes_accessed: int | None = None,
                     raw_bytes_accessed: int | None = None,
                     shape_with_layout: str | None = None,
                     program_id: int | None = None,
                     deduplicated_name: str | None = None) -> None:
      """Add one HLO event on the device plane's XLA Ops line. Stats are
      attached to XEventMetadata.stats (op-level) per profile-anatomy
      schema; per-event stats are not used for HLO ops."""
      plane = xs.planes[0]
      meta_stats: dict = {SM_HLO_CATEGORY: ("str_value", hlo_category)}
      if tf_op is not None:
          meta_stats[SM_TF_OP] = ("str_value", tf_op)
      if source_stack is not None:
          meta_stats[SM_SOURCE_STACK] = ("str_value", source_stack)
      if source_inner is not None:
          meta_stats[SM_SOURCE] = ("str_value", source_inner)
      if flops is not None:
          meta_stats[SM_FLOPS] = ("int64_value", flops)
      if bytes_accessed is not None:
          meta_stats[SM_BYTES_ACCESSED] = ("int64_value", bytes_accessed)
      if raw_bytes_accessed is not None:
          meta_stats[SM_RAW_BYTES_ACCESSED] = ("int64_value", raw_bytes_accessed)
      if shape_with_layout is not None:
          meta_stats[SM_SHAPE] = ("str_value", shape_with_layout)
      if program_id is not None:
          meta_stats[SM_PROGRAM_ID] = ("int64_value", program_id)
      if deduplicated_name is not None:
          meta_stats[SM_DEDUP_NAME] = ("str_value", deduplicated_name)
      _add_event_meta(plane, em_id, hlo_op_text, meta_stats)
      ops_line = next(l for l in plane.lines if l.name == "XLA Ops")
      _add_event(ops_line, em_id, offset_ps, duration_ps)


  class TestSyntheticBuilders(unittest.TestCase):
      def test_minimal_xspace_has_device_plane_and_two_lines(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          self.assertEqual(len(xs.planes), 1)
          plane = xs.planes[0]
          self.assertEqual(plane.name, "/device:TPU:0")
          self.assertEqual({l.name for l in plane.lines}, {"Steps", "XLA Ops"})
          steps_line = next(l for l in plane.lines if l.name == "Steps")
          self.assertEqual(len(steps_line.events), 1)
          self.assertEqual(steps_line.events[0].duration_ps, 1_000_000_000)

      def test_add_hlo_event_attaches_meta_stats_not_per_event(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="fusion.0 = bf16[8,8] fusion(...)",
                        offset_ps=10, duration_ps=500, hlo_category="loop fusion",
                        tf_op="jit/Foo", flops=12345, bytes_accessed=678,
                        shape_with_layout="bf16[8,8]{1,0}",
                        source_stack="/x/y.py:5:1\n/x/z.py:9:2")
          plane = xs.planes[0]
          ops_line = next(l for l in plane.lines if l.name == "XLA Ops")
          self.assertEqual(len(ops_line.events), 1)
          ev = ops_line.events[0]
          self.assertEqual(len(ev.stats), 0, "no per-event stats expected")
          em = plane.event_metadata[ev.metadata_id]
          self.assertGreater(len(em.stats), 0, "op-level stats live on event_metadata")
  ```

- [ ] **Step 2: Run the new tests**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests \
    -p "test_pipeline.py" -v
  ```
  Expected: 5 tests pass (3 CLI from Task 4 + 2 builder tests).

- [ ] **Step 3: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "test(tpu-perf): add synthetic XSpace builders for compute-breakdown tests"
  ```

### Task 6: `EventRecord` dataclass + `_extract_meta_stats` helper

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

The dataclass mirrors the per-event record schema in spec §4. The helper resolves a single XEventMetadata's stats into a name-keyed dict using the plane's `stat_metadata` reverse map.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  # ----------------------------------------------------------------------
  # EventRecord dataclass + meta-stat extraction.
  # ----------------------------------------------------------------------
  # _PROTO_DIR (set up in Task 5 Step 1) points at scripts/_proto. Its
  # parent is scripts/, where compute_breakdown.py lives — that directory
  # MUST be on sys.path before the import below or it raises
  # ModuleNotFoundError. Insert it first, then import.
  sys.path.insert(0, str(_PROTO_DIR.parent))
  import compute_breakdown as cb  # noqa: E402  -- after sys.path insert above


  class TestEventRecord(unittest.TestCase):
      def test_event_record_has_all_spec_fields(self):
          # Fields per spec §4
          expected = {
              "duration_ps", "offset_ps", "step_id",
              "hlo_category", "kind",
              "hlo_op", "tf_op",
              "source_stat", "source_stack", "source_inner", "source_stack_hash",
              "agg_key", "agg_key_kind",
              "flops", "model_flops", "bytes_accessed", "raw_bytes_accessed",
              "shape_with_layout", "dtype", "dtype_uncertain",
              "program_id", "deduplicated_name",
          }
          actual = {f.name for f in cb.EventRecord.__dataclass_fields__.values()}
          self.assertEqual(actual, expected)


  class TestExtractMetaStats(unittest.TestCase):
      def test_resolves_stat_names_via_stat_metadata(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="fusion.0",
                        offset_ps=0, duration_ps=100, hlo_category="loop fusion",
                        tf_op="jit/Foo", flops=42, bytes_accessed=99,
                        shape_with_layout="bf16[8]{0}")
          plane = xs.planes[0]
          em = plane.event_metadata[10]
          name_by_id = {smid: sm.name for smid, sm in plane.stat_metadata.items()}
          stats = cb._extract_meta_stats(em, name_by_id)
          self.assertEqual(stats["hlo_category"], "loop fusion")
          self.assertEqual(stats["tf_op"], "jit/Foo")
          self.assertEqual(stats["flops"], 42)
          self.assertEqual(stats["bytes_accessed"], 99)
          self.assertEqual(stats["shape_with_layout"], "bf16[8]{0}")

      def test_returns_empty_dict_for_no_stats(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          plane = xs.planes[0]
          # event_metadata with no stats
          _add_event_meta(plane, 99, "bare-op")
          name_by_id = {smid: sm.name for smid, sm in plane.stat_metadata.items()}
          self.assertEqual(cb._extract_meta_stats(plane.event_metadata[99], name_by_id), {})
  ```

- [ ] **Step 2: Run; expect failure**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests \
    -p "test_pipeline.py" -v
  ```
  Expected: ERROR — `cb.EventRecord` undefined and `cb._extract_meta_stats` undefined.

- [ ] **Step 3: Implement in compute_breakdown.py**

  Edit `compute_breakdown.py`. Add at the top, below the existing imports:

  ```python
  import dataclasses
  import hashlib
  ```

  And insert below the `import xplane_pb2` line (above `build_parser`):

  ```python
  # ----------------------------------------------------------------------
  # Stage 3 per-event normalized record. Field schema per spec §4.
  # ----------------------------------------------------------------------
  @dataclasses.dataclass
  class EventRecord:
      duration_ps: int
      offset_ps: int
      step_id: int
      hlo_category: str
      kind: str                     # 'compute' | 'data_move' | 'comm' | 'other'
      hlo_op: str
      tf_op: str | None
      source_stat: str | None
      source_stack: str | None
      source_inner: str | None
      source_stack_hash: str | None
      agg_key: str
      agg_key_kind: str             # 'stack' | 'tf_op' | 'no_source'
      flops: int | None
      model_flops: int | None
      bytes_accessed: int | None
      raw_bytes_accessed: int | None
      shape_with_layout: str | None
      dtype: str | None
      dtype_uncertain: bool
      program_id: int | None
      deduplicated_name: str | None


  def _extract_meta_stats(event_metadata, stat_name_by_id: dict) -> dict:
      """Resolve the stats list on an XEventMetadata into {name: value}.
      Values use the discriminated `oneof value` (six variants) per
      profile-anatomy schema."""
      out: dict = {}
      for s in event_metadata.stats:
          name = stat_name_by_id.get(s.metadata_id)
          if not name:
              continue
          vf = s.WhichOneof("value")
          if vf is None:
              continue
          out[name] = getattr(s, vf)
      return out
  ```

- [ ] **Step 4: Run; tests pass**

  Run the same `unittest discover` command. Expected: all tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add EventRecord dataclass and meta-stat extractor"
  ```

### Task 7: `kind` classifier and `dtype` parser

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

Implement two pure functions:
- `_classify_kind(hlo_category: str) -> str` per spec §4.4 lookup table
- `_parse_dtype(shape_with_layout: str | None) -> str | None` per spec §4.2 regex

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  class TestClassifyKind(unittest.TestCase):
      def test_compute_categories(self):
          for cat in ["loop fusion", "convolution fusion", "custom fusion",
                      "output fusion", "non-fusion elementwise", "reduce",
                      "reduce-window", "sort", "rng-bit-generator", "custom-call"]:
              self.assertEqual(cb._classify_kind(cat), "compute", cat)

      def test_data_move_categories(self):
          for cat in ["copy-start", "copy-done", "data formatting", "pad",
                      "broadcast", "slice", "dynamic-slice",
                      "dynamic-update-slice", "iota", "convert"]:
              self.assertEqual(cb._classify_kind(cat), "data_move", cat)

      def test_comm_categories(self):
          for cat in ["async-start", "async-done", "all-reduce", "all-gather",
                      "reduce-scatter", "collective-permute"]:
              self.assertEqual(cb._classify_kind(cat), "comm", cat)

      def test_unknown_category_falls_back_to_other(self):
          self.assertEqual(cb._classify_kind("scalar-thing-2031"), "other")
          self.assertEqual(cb._classify_kind(""), "other")


  class TestParseDtype(unittest.TestCase):
      def test_known_dtypes(self):
          self.assertEqual(cb._parse_dtype("bf16[8192,4096]{1,0}"), "bf16")
          self.assertEqual(cb._parse_dtype("f8e4m3fn[1024,4096]{1,0}"), "fp8")
          self.assertEqual(cb._parse_dtype("f8e5m2[64]{0}"), "fp8")
          self.assertEqual(cb._parse_dtype("f32[]"), "fp32")
          self.assertEqual(cb._parse_dtype("f16[16,16]{1,0}"), "fp16")

      def test_other_for_unknown_or_unparseable(self):
          self.assertEqual(cb._parse_dtype("s32[8]{0}"), "other")
          self.assertEqual(cb._parse_dtype("s8[8]{0}"), "other")
          self.assertEqual(cb._parse_dtype("pred[]"), "other")
          self.assertEqual(cb._parse_dtype("(bf16[8],bf16[8])"), "other")
          self.assertEqual(cb._parse_dtype("garbage no bracket"), "other")
          self.assertIsNone(cb._parse_dtype(None))
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert just below `_extract_meta_stats`:

  ```python
  import re

  _COMPUTE_CATS = frozenset({
      "loop fusion", "convolution fusion", "custom fusion", "output fusion",
      "non-fusion elementwise", "reduce", "reduce-window",
      "sort", "rng-bit-generator", "custom-call",
  })
  _DATA_MOVE_CATS = frozenset({
      "copy-start", "copy-done", "data formatting", "pad", "broadcast",
      "slice", "dynamic-slice", "dynamic-update-slice", "iota", "convert",
  })
  _COMM_CATS = frozenset({
      "async-start", "async-done", "all-reduce", "all-gather",
      "reduce-scatter", "collective-permute",
  })

  _DTYPE_PREFIX_RE = re.compile(r"^([a-z][a-z0-9]*)\[")
  _DTYPE_MAP = {
      "bf16": "bf16",
      "f8e4m3fn": "fp8", "f8e5m2": "fp8",
      "f32": "fp32",
      "f16": "fp16",
  }

  def _classify_kind(hlo_category: str) -> str:
      if hlo_category in _COMPUTE_CATS:
          return "compute"
      if hlo_category in _DATA_MOVE_CATS:
          return "data_move"
      if hlo_category in _COMM_CATS:
          return "comm"
      return "other"

  def _parse_dtype(shape_with_layout: str | None) -> str | None:
      if shape_with_layout is None:
          return None
      m = _DTYPE_PREFIX_RE.match(shape_with_layout)
      if not m:
          return "other"
      return _DTYPE_MAP.get(m.group(1), "other")
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add kind classifier and dtype parser"
  ```

### Task 8: `agg_key` three-tier fallback + `_inner_frame` source-stack parser

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

Per spec §4.1:

| Priority | Condition | `agg_key` value | `agg_key_kind` |
|---|---|---|---|
| 1 | `source_stack` non-empty | `"stack:" + sha1(source_stack)[:16]` | `"stack"` |
| 2 | `source_stack` empty, `tf_op` non-empty | `"tfop:" + tf_op` | `"tf_op"` |
| 3 | both absent | `"nosrc:" + hlo_category` | `"no_source"` |

Per spec §4 record-schema note: `source_inner` = "last non-empty line of `source_stack` with trailing `:<col>` suffix stripped to `file:line`"; null when source_stack is null.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  class TestAggKey(unittest.TestCase):
      def test_priority1_stack(self):
          k, kind, h = cb._compute_agg_key(
              source_stack="/x/y.py:5:1\n/x/z.py:9:2",
              tf_op="jit/Foo",
              hlo_category="loop fusion")
          self.assertTrue(k.startswith("stack:"))
          self.assertEqual(kind, "stack")
          self.assertEqual(len(h), 16)
          self.assertEqual(k, f"stack:{h}")

      def test_priority2_tfop(self):
          k, kind, h = cb._compute_agg_key(
              source_stack=None, tf_op="jit/Foo", hlo_category="loop fusion")
          self.assertEqual(k, "tfop:jit/Foo")
          self.assertEqual(kind, "tf_op")
          self.assertIsNone(h)

      def test_priority2_tfop_when_stack_is_empty_string(self):
          # spec §4.1: "source_stack empty" — empty string treated same as None
          k, kind, _ = cb._compute_agg_key(
              source_stack="", tf_op="jit/Bar", hlo_category="reduce")
          self.assertEqual(k, "tfop:jit/Bar")
          self.assertEqual(kind, "tf_op")

      def test_priority3_no_source(self):
          k, kind, h = cb._compute_agg_key(
              source_stack=None, tf_op=None, hlo_category="copy-done")
          self.assertEqual(k, "nosrc:copy-done")
          self.assertEqual(kind, "no_source")
          self.assertIsNone(h)

      def test_priority3_when_tfop_is_empty_string(self):
          k, kind, _ = cb._compute_agg_key(
              source_stack=None, tf_op="", hlo_category="pad")
          self.assertEqual(k, "nosrc:pad")
          self.assertEqual(kind, "no_source")


  class TestInnerFrame(unittest.TestCase):
      def test_strips_column_suffix(self):
          self.assertEqual(cb._inner_frame("/a/b.py:5:1\n/a/c.py:9:2"), "/a/c.py:9")

      def test_keeps_file_line_when_no_column(self):
          self.assertEqual(cb._inner_frame("/a/b.py:5\n/a/c.py:9"), "/a/c.py:9")

      def test_skips_trailing_blank_lines(self):
          self.assertEqual(cb._inner_frame("/a/b.py:5:1\n/a/c.py:9:2\n\n"),
                           "/a/c.py:9")

      def test_single_line(self):
          self.assertEqual(cb._inner_frame("/single.py:42:0"), "/single.py:42")

      def test_returns_none_for_none_or_empty(self):
          self.assertIsNone(cb._inner_frame(None))
          self.assertIsNone(cb._inner_frame(""))
          self.assertIsNone(cb._inner_frame("\n\n"))
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_parse_dtype`:

  ```python
  def _compute_agg_key(*, source_stack: str | None, tf_op: str | None,
                        hlo_category: str) -> tuple[str, str, str | None]:
      """Returns (agg_key, agg_key_kind, source_stack_hash | None).
      Three-tier fallback per spec §4.1."""
      if source_stack:
          h = hashlib.sha1(source_stack.encode("utf-8")).hexdigest()[:16]
          return f"stack:{h}", "stack", h
      if tf_op:
          return f"tfop:{tf_op}", "tf_op", None
      return f"nosrc:{hlo_category}", "no_source", None


  def _inner_frame(source_stack: str | None) -> str | None:
      """Innermost frame of `source_stack`: last non-empty line, stripped
      to `file:line` (drop trailing `:<col>` suffix). Spec §4 record schema."""
      if not source_stack:
          return None
      lines = [ln for ln in source_stack.splitlines() if ln.strip()]
      if not lines:
          return None
      last = lines[-1]
      # Strip trailing :<col> if present (last colon-separated token).
      # Heuristic: file:line:col -> file:line; file:line -> file:line.
      parts = last.rsplit(":", 2)
      if len(parts) == 3:
          return f"{parts[0]}:{parts[1]}"
      return last
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add agg_key three-tier fallback and source_stack inner-frame parser"
  ```

### Task 9: `dtype_uncertain` rule

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

Per spec §4.3, set `dtype_uncertain=true` iff **both**:
1. `hlo_category` ∈ `{"convolution fusion", "custom fusion", "output fusion", "custom-call"}`
2. `dtype` ∈ `{"bf16", "fp32"}`

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  class TestDtypeUncertain(unittest.TestCase):
      _ALLOWED_CATS = ("convolution fusion", "custom fusion", "output fusion", "custom-call")
      _ALLOWED_DTYPES = ("bf16", "fp32")

      def test_true_only_when_both_conditions_hold(self):
          for cat in self._ALLOWED_CATS:
              for dtype in self._ALLOWED_DTYPES:
                  self.assertTrue(cb._is_dtype_uncertain(cat, dtype),
                                  msg=f"{cat}/{dtype}")

      def test_false_when_loop_fusion(self):
          for dtype in self._ALLOWED_DTYPES:
              self.assertFalse(cb._is_dtype_uncertain("loop fusion", dtype))

      def test_false_when_non_fusion_elementwise(self):
          self.assertFalse(cb._is_dtype_uncertain("non-fusion elementwise", "bf16"))

      def test_false_when_dtype_fp8(self):
          for cat in self._ALLOWED_CATS:
              self.assertFalse(cb._is_dtype_uncertain(cat, "fp8"))

      def test_false_when_dtype_other_or_none(self):
          self.assertFalse(cb._is_dtype_uncertain("convolution fusion", "other"))
          self.assertFalse(cb._is_dtype_uncertain("convolution fusion", None))
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_inner_frame`:

  ```python
  _UNCERTAIN_CATS = frozenset({
      "convolution fusion", "custom fusion", "output fusion", "custom-call",
  })
  _UNCERTAIN_DTYPES = frozenset({"bf16", "fp32"})

  def _is_dtype_uncertain(hlo_category: str, dtype: str | None) -> bool:
      """Spec §4.3: True iff category ∈ {fusion family that wraps mixed-precision
      compute} AND dtype ∈ {bf16, fp32}."""
      return hlo_category in _UNCERTAIN_CATS and dtype in _UNCERTAIN_DTYPES
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add dtype_uncertain rule for fusion-family categories"
  ```

### Task 10: Step-window selection (`_pick_step_window`)

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

Per spec §4.5:
- Default: middle step (`sorted_steps[len//2]`).
- `--step N`: pick by 0-indexed position; out-of-range → `ValueError`.
- `--step-id ID`: exact match against `XEventMetadata.name`; zero matches → `ValueError`; multiple matches → pick earliest by `offset_ps` and append note.
- Returns `(step_event, step_id, step_start_ps, step_end_ps, notes)` where `step_id` is the position index in the sorted list.
- If `Steps` line missing/empty → return `(None, -1, t_min, t_max, ["no Steps line; falling back to full-plane window"])` where `t_min, t_max` come from `XLA Ops` line.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  class TestPickStepWindow(unittest.TestCase):
      def _xs_three_steps(self):
          return make_minimal_xspace(steps=[
              (1, 1_000_000, 5_000_000),       # step 0 ends at 6_000_000
              (2, 7_000_000, 5_000_000),       # step 1 ends at 12_000_000
              (3, 13_000_000, 5_000_000),      # step 2 ends at 18_000_000
          ])

      def test_default_picks_middle_step(self):
          xs = self._xs_three_steps()
          plane = xs.planes[0]
          ev, sid, s, e, notes = cb._pick_step_window(plane, step_idx=None, step_id=None)
          self.assertEqual(sid, 1)
          self.assertEqual(s, 7_000_000)
          self.assertEqual(e, 12_000_000)
          self.assertEqual(notes, [])

      def test_step_idx_picks_specific(self):
          xs = self._xs_three_steps()
          ev, sid, s, e, _ = cb._pick_step_window(xs.planes[0], step_idx=0, step_id=None)
          self.assertEqual(sid, 0)
          self.assertEqual(s, 1_000_000)
          self.assertEqual(e, 6_000_000)

      def test_step_idx_out_of_range_raises(self):
          xs = self._xs_three_steps()
          with self.assertRaises(ValueError):
              cb._pick_step_window(xs.planes[0], step_idx=99, step_id=None)
          with self.assertRaises(ValueError):
              cb._pick_step_window(xs.planes[0], step_idx=-1, step_id=None)

      def test_step_id_exact_match(self):
          xs = self._xs_three_steps()
          # Synthetic step names are step_1, step_2, step_3 (em_id mapped)
          ev, sid, s, e, notes = cb._pick_step_window(
              xs.planes[0], step_idx=None, step_id="step_2")
          self.assertEqual(sid, 1)
          self.assertEqual(s, 7_000_000)
          self.assertEqual(notes, [])

      def test_step_id_zero_matches_raises(self):
          xs = self._xs_three_steps()
          with self.assertRaises(ValueError):
              cb._pick_step_window(xs.planes[0], step_idx=None, step_id="nope")

      def test_step_id_multi_match_picks_earliest_with_note(self):
          # Two steps share the same metadata name (we re-use em_id 1)
          xs = make_minimal_xspace(steps=[(1, 5_000_000, 1_000_000),
                                           (1, 1_000_000, 1_000_000),  # earlier
                                           (1, 9_000_000, 1_000_000)])
          ev, sid, s, e, notes = cb._pick_step_window(
              xs.planes[0], step_idx=None, step_id="step_1")
          self.assertEqual(s, 1_000_000)
          self.assertIn("multi-match for step-id; picked first", notes)

      def test_no_steps_line_falls_back_to_full_xla_ops_window(self):
          xs = xplane_pb2.XSpace()
          plane = xs.planes.add()
          plane.name = "/device:TPU:0"
          ops_line = plane.lines.add()
          ops_line.name = "XLA Ops"
          # Two events spanning [10, 30) and [40, 90)
          _add_event_meta(plane, 1, "x")
          _add_event(ops_line, 1, 10, 20)
          _add_event(ops_line, 1, 40, 50)
          ev, sid, s, e, notes = cb._pick_step_window(plane, step_idx=None, step_id=None)
          self.assertIsNone(ev)
          self.assertEqual(sid, -1)
          self.assertEqual(s, 10)
          self.assertEqual(e, 90)
          self.assertIn("no Steps line; falling back to full-plane window", notes)
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_is_dtype_uncertain`:

  ```python
  def _pick_step_window(plane, *, step_idx: int | None, step_id: str | None):
      """Spec §4.5. Returns (step_event_or_None, step_index_int, start_ps,
      end_ps, notes_list)."""
      steps_line = next((l for l in plane.lines if l.name == "Steps"), None)
      notes: list[str] = []
      if steps_line is None or len(steps_line.events) == 0:
          # Fallback: full XLA Ops window.
          ops = next((l for l in plane.lines if l.name == "XLA Ops"), None)
          if ops is None or len(ops.events) == 0:
              return None, -1, 0, 0, ["no Steps line; falling back to full-plane window",
                                       "XLA Ops line empty or missing"]
          starts = [ev.offset_ps for ev in ops.events]
          ends = [ev.offset_ps + ev.duration_ps for ev in ops.events]
          notes.append("no Steps line; falling back to full-plane window")
          return None, -1, min(starts), max(ends), notes

      sorted_steps = sorted(steps_line.events, key=lambda e: e.offset_ps)

      if step_id is not None:
          em_map = plane.event_metadata
          matches = [(i, e) for i, e in enumerate(sorted_steps)
                     if em_map.get(e.metadata_id) is not None
                     and em_map[e.metadata_id].name == step_id]
          if not matches:
              raise ValueError(f"--step-id {step_id!r} matched zero Step events")
          if len(matches) > 1:
              notes.append("multi-match for step-id; picked first")
          idx, ev = matches[0]
      elif step_idx is not None:
          if step_idx < 0 or step_idx >= len(sorted_steps):
              raise ValueError(
                  f"--step {step_idx} out of range [0, {len(sorted_steps)})")
          idx, ev = step_idx, sorted_steps[step_idx]
      else:
          idx = len(sorted_steps) // 2
          ev = sorted_steps[idx]

      return ev, idx, ev.offset_ps, ev.offset_ps + ev.duration_ps, notes
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add step-window selection with default/step-idx/step-id branches"
  ```

### Task 11: Stage 3 event normalizer (`_iter_event_records`)

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

This generator iterates the `XLA Ops` line, admits events whose `offset_ps` is in `[start_ps, end_ps)`, and yields one `EventRecord` per admitted event. Spec §3 stage-3 contract:
- Events with no `event_metadata` entry → drop and increment a counter (recorded out-of-band by the caller, see Task 12).
- Events with `hlo_category == "while"` → don't yield a record; the caller separately accumulates `while_total_ps` and `unknown_categories`.
- All other events → yield a fully populated `EventRecord`.

The returned generator is the *only* output of stage 3; counts and `while_total_ps` are tracked via a sidechannel `_PipelineStats` dataclass that the caller passes in and the generator mutates.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  class TestIterEventRecords(unittest.TestCase):
      def _xs_with_step_and_ops(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          # In window: [0, 1e9)
          add_hlo_event(xs, em_id=10,
                        hlo_op_text="fusion.0 = bf16[8,8] fusion(bf16[8,8] %p)",
                        offset_ps=100, duration_ps=500,
                        hlo_category="loop fusion",
                        tf_op="jit/Foo",
                        flops=1024, bytes_accessed=128,
                        shape_with_layout="bf16[8,8]{1,0}",
                        source_stack="/x/foo.py:5:1\n/x/bar.py:9:2")
          # Out of window (after end)
          add_hlo_event(xs, em_id=11, hlo_op_text="late",
                        offset_ps=2_000_000_000, duration_ps=500,
                        hlo_category="loop fusion")
          # while event in window
          add_hlo_event(xs, em_id=12, hlo_op_text="while.0 = ...",
                        offset_ps=1000, duration_ps=10_000,
                        hlo_category="while")
          # data_move with shape_with_layout
          add_hlo_event(xs, em_id=13, hlo_op_text="copy.0 = bf16[8] copy(bf16[8])",
                        offset_ps=2000, duration_ps=200,
                        hlo_category="data formatting",
                        shape_with_layout="bf16[8]{0}")
          # unknown category
          add_hlo_event(xs, em_id=14, hlo_op_text="weirdo",
                        offset_ps=3000, duration_ps=50,
                        hlo_category="never-seen-category")
          return xs

      def test_yields_records_for_in_window_non_while_events(self):
          xs = self._xs_with_step_and_ops()
          plane = xs.planes[0]
          stats = cb._PipelineStats()
          recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                              step_id=0, stats=stats))
          # 3 records: fusion.0, copy.0, weirdo. (while skipped, late out of window.)
          self.assertEqual(len(recs), 3)
          kinds = sorted(r.kind for r in recs)
          self.assertEqual(kinds, ["compute", "data_move", "other"])

      def test_window_admission_is_start_inclusive_end_exclusive(self):
          xs = make_minimal_xspace(steps=[(1, 0, 100)])
          # Event starts exactly at start_ps -> admit
          add_hlo_event(xs, em_id=10, hlo_op_text="a", offset_ps=0,
                        duration_ps=10, hlo_category="loop fusion")
          # Event starts exactly at end_ps -> exclude
          add_hlo_event(xs, em_id=11, hlo_op_text="b", offset_ps=100,
                        duration_ps=10, hlo_category="loop fusion")
          # Event starts before start_ps -> exclude (even if extends in)
          add_hlo_event(xs, em_id=12, hlo_op_text="c", offset_ps=-50,
                        duration_ps=200, hlo_category="loop fusion")
          plane = xs.planes[0]
          stats = cb._PipelineStats()
          recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=100,
                                              step_id=0, stats=stats))
          ops = sorted(r.hlo_op for r in recs)
          self.assertEqual(ops, ["a"])

      def test_while_skipped_and_accumulated(self):
          xs = self._xs_with_step_and_ops()
          plane = xs.planes[0]
          stats = cb._PipelineStats()
          list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                       step_id=0, stats=stats))
          self.assertEqual(stats.while_total_ps, 10_000)

      def test_unknown_category_counted(self):
          xs = self._xs_with_step_and_ops()
          plane = xs.planes[0]
          stats = cb._PipelineStats()
          list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                       step_id=0, stats=stats))
          self.assertEqual(stats.unknown_categories, {"never-seen-category": 1})

      def test_unresolved_event_metadata_dropped_and_counted(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          plane = xs.planes[0]
          ops = next(l for l in plane.lines if l.name == "XLA Ops")
          # Event references an event_metadata id that doesn't exist
          ev = ops.events.add()
          ev.metadata_id = 9999
          ev.offset_ps = 100
          ev.duration_ps = 50
          stats = cb._PipelineStats()
          recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                              step_id=0, stats=stats))
          self.assertEqual(recs, [])
          self.assertEqual(stats.n_events_unresolved, 1)

      def test_record_field_population(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10,
                        hlo_op_text="fusion.0 = bf16[8,8] fusion(...)",
                        offset_ps=200, duration_ps=300,
                        hlo_category="loop fusion",
                        tf_op="jit/Foo",
                        flops=2048, bytes_accessed=64,
                        raw_bytes_accessed=72, model_flops=1024,
                        shape_with_layout="bf16[8,8]{1,0}",
                        program_id=5,
                        deduplicated_name="dedup.0",
                        source_inner="/x/foo.py:5",
                        source_stack="/x/foo.py:5:1\n/x/bar.py:9:2")
          plane = xs.planes[0]
          stats = cb._PipelineStats()
          recs = list(cb._iter_event_records(plane, start_ps=0, end_ps=1_000_000_000,
                                              step_id=42, stats=stats))
          self.assertEqual(len(recs), 1)
          r = recs[0]
          self.assertEqual(r.duration_ps, 300)
          self.assertEqual(r.offset_ps, 200)
          self.assertEqual(r.step_id, 42)
          self.assertEqual(r.hlo_category, "loop fusion")
          self.assertEqual(r.kind, "compute")
          self.assertEqual(r.tf_op, "jit/Foo")
          self.assertEqual(r.source_stat, "/x/foo.py:5")
          self.assertEqual(r.source_stack, "/x/foo.py:5:1\n/x/bar.py:9:2")
          self.assertEqual(r.source_inner, "/x/bar.py:9")
          self.assertEqual(r.agg_key_kind, "stack")
          self.assertEqual(r.flops, 2048)
          self.assertEqual(r.model_flops, 1024)
          self.assertEqual(r.bytes_accessed, 64)
          self.assertEqual(r.raw_bytes_accessed, 72)
          self.assertEqual(r.shape_with_layout, "bf16[8,8]{1,0}")
          self.assertEqual(r.dtype, "bf16")
          self.assertFalse(r.dtype_uncertain)
          self.assertEqual(r.program_id, 5)
          self.assertEqual(r.deduplicated_name, "dedup.0")
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_pick_step_window`:

  ```python
  @dataclasses.dataclass
  class _PipelineStats:
      """Sidechannel from _iter_event_records to the caller."""
      n_events_unresolved: int = 0
      while_total_ps: int = 0
      unknown_categories: dict = dataclasses.field(default_factory=dict)


  def _iter_event_records(plane, *, start_ps: int, end_ps: int, step_id: int,
                           stats: _PipelineStats):
      """Yield one EventRecord per admitted XLA-Ops event in [start_ps, end_ps).
      Mutates `stats` for events that don't yield a record (unresolved, while,
      unknown category)."""
      ops_line = next((l for l in plane.lines if l.name == "XLA Ops"), None)
      if ops_line is None:
          return
      stat_name_by_id = {smid: sm.name for smid, sm in plane.stat_metadata.items()}

      for ev in ops_line.events:
          if not (start_ps <= ev.offset_ps < end_ps):
              continue
          em = plane.event_metadata.get(ev.metadata_id)
          if em is None:
              stats.n_events_unresolved += 1
              continue
          mstats = _extract_meta_stats(em, stat_name_by_id)
          hlo_cat = mstats.get("hlo_category", "")
          if hlo_cat == "while":
              stats.while_total_ps += ev.duration_ps
              continue

          kind = _classify_kind(hlo_cat)
          if kind == "other":
              # Track unrecognized categories so the spec maintainer can update §4.4.
              stats.unknown_categories[hlo_cat] = stats.unknown_categories.get(hlo_cat, 0) + 1

          source_stack = mstats.get("source_stack")
          tf_op = mstats.get("tf_op")
          shape = mstats.get("shape_with_layout")
          dtype = _parse_dtype(shape) if shape else None
          agg_key, agg_kind, stack_hash = _compute_agg_key(
              source_stack=source_stack, tf_op=tf_op, hlo_category=hlo_cat)

          yield EventRecord(
              duration_ps=ev.duration_ps,
              offset_ps=ev.offset_ps,
              step_id=step_id,
              hlo_category=hlo_cat,
              kind=kind,
              hlo_op=em.name,
              tf_op=tf_op,
              source_stat=mstats.get("source"),
              source_stack=source_stack,
              source_inner=_inner_frame(source_stack),
              source_stack_hash=stack_hash,
              agg_key=agg_key,
              agg_key_kind=agg_kind,
              flops=mstats.get("flops"),
              model_flops=mstats.get("model_flops"),
              bytes_accessed=mstats.get("bytes_accessed"),
              raw_bytes_accessed=mstats.get("raw_bytes_accessed"),
              shape_with_layout=shape,
              dtype=dtype,
              dtype_uncertain=_is_dtype_uncertain(hlo_cat, dtype),
              program_id=mstats.get("program_id"),
              deduplicated_name=mstats.get("deduplicated_name"),
          )
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add Stage 3 event-record iterator with while/unknown/unresolved sidechannel"
  ```

### Task 12: Top-level orchestrator `_load_and_normalize`

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py`

Composes Stages 1, 2, 3: opens the xplane.pb, picks the device plane, runs `_pick_step_window`, runs `_iter_event_records`, returns `(records, ctx)` where `ctx` carries everything needed by mode dispatchers (step_id, window, step_duration_ps, notes, _PipelineStats).

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_pipeline.py`:

  ```python
  class TestLoadAndNormalize(unittest.TestCase):
      def _write_xspace(self, xs, tmpdir):
          path = pathlib.Path(tmpdir) / "synthetic.xplane.pb"
          path.write_bytes(xs.SerializeToString())
          return path

      def test_loads_picks_step_yields_records(self):
          import tempfile
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000),
                                           (2, 1_000_000_000, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="x", offset_ps=500,
                        duration_ps=100, hlo_category="loop fusion",
                        tf_op="jit/Foo", flops=10, bytes_accessed=2,
                        shape_with_layout="bf16[1]{0}")
          add_hlo_event(xs, em_id=11, hlo_op_text="y",
                        offset_ps=1_000_000_500, duration_ps=200,
                        hlo_category="loop fusion")
          with tempfile.TemporaryDirectory() as tmp:
              self._write_xspace(xs, tmp)
              records, ctx = cb._load_and_normalize(
                  profile_dir=tmp, device="/device:TPU:0",
                  step_idx=None, step_id=None)
          # default = middle step (idx 1) -> window [1e9, 2e9)
          self.assertEqual(ctx["step_id"], 1)
          self.assertEqual(ctx["step_window_ps"], [1_000_000_000, 2_000_000_000])
          self.assertEqual(ctx["step_duration_ps"], 1_000_000_000)
          self.assertEqual(len(records), 1)
          self.assertEqual(records[0].hlo_op, "y")

      def test_device_not_found_returns_none(self):
          import tempfile
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          with tempfile.TemporaryDirectory() as tmp:
              self._write_xspace(xs, tmp)
              records, ctx = cb._load_and_normalize(
                  profile_dir=tmp, device="/device:TPU:99",
                  step_idx=None, step_id=None)
          self.assertIsNone(records)
          self.assertEqual(ctx["status"], "absent")
          self.assertEqual(ctx["reason"], "device_not_found")

      def test_no_xla_ops_line_returns_absent(self):
          import tempfile
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          # Remove the XLA Ops line
          plane = xs.planes[0]
          for i, l in enumerate(list(plane.lines)):
              if l.name == "XLA Ops":
                  del plane.lines[i]
                  break
          with tempfile.TemporaryDirectory() as tmp:
              self._write_xspace(xs, tmp)
              records, ctx = cb._load_and_normalize(
                  profile_dir=tmp, device="/device:TPU:0",
                  step_idx=None, step_id=None)
          self.assertIsNone(records)
          self.assertEqual(ctx["status"], "absent")
          self.assertEqual(ctx["reason"], "no_xla_ops_line")
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_iter_event_records`:

  ```python
  def _load_and_normalize(*, profile_dir: str, device: str,
                            step_idx: int | None, step_id: str | None):
      """Stages 1+2+3. Returns (records, ctx).

      `records` is a list[EventRecord] when status is ok; None when absent.
      `ctx` always carries `status` ('ok' | 'absent'); on absent it also has
      `reason` and `notes`. On ok it carries: step_id, step_window_ps,
      step_duration_ps, notes (list), pipeline_stats (_PipelineStats),
      profile_dir (str), device (str), xspace_pb_path (str).
      """
      pdir = pathlib.Path(profile_dir)
      pbs = sorted(pdir.glob("*.xplane.pb")) if pdir.is_dir() else []
      if not pbs:
          return None, {"status": "absent", "reason": "no_xplane_pb", "notes": []}

      xs = xplane_pb2.XSpace()
      with open(pbs[0], "rb") as f:
          xs.ParseFromString(f.read())

      plane = next((p for p in xs.planes if p.name == device), None)
      if plane is None:
          have = [p.name for p in xs.planes]
          return None, {"status": "absent", "reason": "device_not_found",
                        "notes": [f"have: {have}"]}

      if not any(l.name == "XLA Ops" for l in plane.lines):
          have = [l.name for l in plane.lines]
          return None, {"status": "absent", "reason": "no_xla_ops_line",
                        "notes": [f"have: {have}"]}

      step_event, sid, s_ps, e_ps, notes = _pick_step_window(
          plane, step_idx=step_idx, step_id=step_id)
      pstats = _PipelineStats()
      records = list(_iter_event_records(
          plane, start_ps=s_ps, end_ps=e_ps, step_id=sid, stats=pstats))

      return records, {
          "status": "ok",
          "step_id": sid,
          "step_window_ps": [s_ps, e_ps],
          "step_duration_ps": e_ps - s_ps,
          "notes": notes,
          "pipeline_stats": pstats,
          "profile_dir": str(pdir),
          "device": device,
          "xspace_pb_path": str(pbs[0]),
      }
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_pipeline.py
  git commit -m "feat(tpu-perf): add load-and-normalize orchestrator (Stages 1+2+3)"
  ```

---

## Chunk 3: Mode 1 — `summary`

This chunk wires `--mode summary` end-to-end: per-`agg_key` aggregation across `kind=compute` records, ranked by `total_dur_ps`; tail rollup; by-kind rollup; cross-kind totals; `unknown_categories`. After this chunk, `python3 compute_breakdown.py <real-fixture> --mode summary` produces the spec §5 JSON shape.

Per spec §5 numerator/denominator:
- `pct_of_compute` denominator = `compute_duration_ps`
- `pct_of_step` denominator = `step_duration_ps` (includes while)
- `flops_sum` / `bytes_accessed_sum` per group: skip null-field events when summing; emit `null` only if every event in the group lacks the field.

### Task 13: `_aggregate_by_key` helper

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py`

A pure function: given `records: list[EventRecord]`, return a `dict[agg_key, GroupAgg]` where `GroupAgg` carries n_executions, total/min/max duration, `flops_sum` (null-safe), `bytes_accessed_sum`, `hlo_categories` histogram, `example_hlo_op`, plus the canonical `source_inner` / `tf_op` / `source_stack` / `agg_key_kind` of the first record in the group.

- [ ] **Step 1: Write the failing tests**

  Write to `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py`:

  ```python
  """Unit tests for mode 1 (summary) projection logic."""
  import json
  import pathlib
  import subprocess
  import sys
  import tempfile
  import unittest

  TESTS_DIR = pathlib.Path(__file__).resolve().parent
  SCRIPTS_DIR = TESTS_DIR.parent
  SCRIPT = SCRIPTS_DIR / "compute_breakdown.py"
  sys.path.insert(0, str(SCRIPTS_DIR))
  sys.path.insert(0, str(SCRIPTS_DIR / "_proto"))

  import compute_breakdown as cb  # noqa: E402
  from test_pipeline import (  # noqa: E402
      make_minimal_xspace, add_hlo_event,
  )


  def _make_record(**overrides):
      """Build an EventRecord with sensible defaults; override any field."""
      defaults = dict(
          duration_ps=100, offset_ps=0, step_id=0,
          hlo_category="loop fusion", kind="compute",
          hlo_op="x", tf_op=None, source_stat=None,
          source_stack=None, source_inner=None, source_stack_hash=None,
          agg_key="tfop:x", agg_key_kind="tf_op",
          flops=None, model_flops=None, bytes_accessed=None,
          raw_bytes_accessed=None, shape_with_layout=None,
          dtype=None, dtype_uncertain=False,
          program_id=None, deduplicated_name=None,
      )
      defaults.update(overrides)
      return cb.EventRecord(**defaults)


  class TestAggregateByKey(unittest.TestCase):
      def test_groups_by_agg_key(self):
          recs = [
              _make_record(agg_key="A", duration_ps=100),
              _make_record(agg_key="A", duration_ps=200),
              _make_record(agg_key="B", duration_ps=50),
          ]
          out = cb._aggregate_by_key(recs)
          self.assertEqual(set(out.keys()), {"A", "B"})
          self.assertEqual(out["A"].n_executions, 2)
          self.assertEqual(out["A"].total_dur_ps, 300)
          self.assertEqual(out["A"].min_dur_ps, 100)
          self.assertEqual(out["A"].max_dur_ps, 200)
          self.assertEqual(out["B"].n_executions, 1)

      def test_flops_sum_null_safe(self):
          recs = [
              _make_record(agg_key="A", flops=100),
              _make_record(agg_key="A", flops=None),
              _make_record(agg_key="A", flops=200),
          ]
          out = cb._aggregate_by_key(recs)
          self.assertEqual(out["A"].flops_sum, 300)

      def test_flops_sum_all_null_emits_none(self):
          recs = [
              _make_record(agg_key="A", flops=None),
              _make_record(agg_key="A", flops=None),
          ]
          out = cb._aggregate_by_key(recs)
          self.assertIsNone(out["A"].flops_sum)

      def test_hlo_categories_histogram(self):
          recs = [
              _make_record(agg_key="A", hlo_category="loop fusion"),
              _make_record(agg_key="A", hlo_category="loop fusion"),
              _make_record(agg_key="A", hlo_category="convolution fusion"),
          ]
          out = cb._aggregate_by_key(recs)
          self.assertEqual(out["A"].hlo_categories,
                           {"loop fusion": 2, "convolution fusion": 1})

      def test_example_hlo_op_is_first_seen(self):
          recs = [
              _make_record(agg_key="A", hlo_op="first"),
              _make_record(agg_key="A", hlo_op="second"),
          ]
          out = cb._aggregate_by_key(recs)
          self.assertEqual(out["A"].example_hlo_op, "first")
  ```

- [ ] **Step 2: Run; expect failure**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests \
    -p "test_summary_mode.py" -v
  ```
  Expected: ERROR — `cb._aggregate_by_key` undefined.

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_load_and_normalize`:

  ```python
  @dataclasses.dataclass
  class _GroupAgg:
      agg_key: str
      agg_key_kind: str
      source_inner: str | None
      source_stack: str | None
      tf_op: str | None
      kind: str                            # canonical kind of the group
      n_executions: int = 0
      total_dur_ps: int = 0
      min_dur_ps: int = 0
      max_dur_ps: int = 0
      _flops_sum: int = 0
      _flops_seen: int = 0                 # how many records contributed flops
      _bytes_sum: int = 0
      _bytes_seen: int = 0
      _model_flops_sum: int = 0
      _model_flops_seen: int = 0
      hlo_categories: dict = dataclasses.field(default_factory=dict)
      shapes: list = dataclasses.field(default_factory=list)
      dtypes: dict = dataclasses.field(default_factory=dict)
      dtype_uncertain: bool = False        # OR of all member records
      first_dtype: str | None = None       # dtype of the FIRST record in
                                            # the group; never overwritten.
                                            # Used by mode 4 (roofline) per
                                            # spec §8.2.
      example_hlo_op: str | None = None

      @property
      def avg_dur_ps(self) -> float:
          return self.total_dur_ps / self.n_executions if self.n_executions else 0.0

      @property
      def flops_sum(self) -> int | None:
          return self._flops_sum if self._flops_seen > 0 else None

      @property
      def bytes_accessed_sum(self) -> int | None:
          return self._bytes_sum if self._bytes_seen > 0 else None

      @property
      def model_flops_sum(self) -> int | None:
          return self._model_flops_sum if self._model_flops_seen > 0 else None


  def _aggregate_by_key(records: list[EventRecord],
                          *, dedupe_shapes_cap: int = 8) -> dict:
      groups: dict[str, _GroupAgg] = {}
      for r in records:
          g = groups.get(r.agg_key)
          if g is None:
              g = _GroupAgg(
                  agg_key=r.agg_key, agg_key_kind=r.agg_key_kind,
                  source_inner=r.source_inner, source_stack=r.source_stack,
                  tf_op=r.tf_op, kind=r.kind,
                  example_hlo_op=r.hlo_op,
                  min_dur_ps=r.duration_ps, max_dur_ps=r.duration_ps,
                  first_dtype=r.dtype,    # spec §8.2: first record wins.
              )
              groups[r.agg_key] = g
          g.n_executions += 1
          g.total_dur_ps += r.duration_ps
          if r.duration_ps < g.min_dur_ps:
              g.min_dur_ps = r.duration_ps
          if r.duration_ps > g.max_dur_ps:
              g.max_dur_ps = r.duration_ps
          if r.flops is not None:
              g._flops_sum += r.flops
              g._flops_seen += 1
          if r.bytes_accessed is not None:
              g._bytes_sum += r.bytes_accessed
              g._bytes_seen += 1
          if r.model_flops is not None:
              g._model_flops_sum += r.model_flops
              g._model_flops_seen += 1
          g.hlo_categories[r.hlo_category] = g.hlo_categories.get(r.hlo_category, 0) + 1
          if r.shape_with_layout and r.shape_with_layout not in g.shapes:
              if len(g.shapes) < dedupe_shapes_cap:
                  g.shapes.append(r.shape_with_layout)
          if r.dtype:
              g.dtypes[r.dtype] = g.dtypes.get(r.dtype, 0) + 1
          if r.dtype_uncertain:
              g.dtype_uncertain = True
      return groups
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py
  git commit -m "feat(tpu-perf): add per-agg_key group aggregator with null-safe flops/bytes sums"
  ```

### Task 14: `_compute_totals` helper

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py`

Returns the `totals` block per spec §5: per-kind durations and counts, while_container_duration_ps, non_while_duration_ps_sum (with the concurrency caveat in its name), while_pct_of_step, n_events_unresolved, unknown_categories.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_summary_mode.py`:

  ```python
  class TestComputeTotals(unittest.TestCase):
      def test_per_kind_aggregation(self):
          recs = [
              _make_record(kind="compute",   duration_ps=100),
              _make_record(kind="compute",   duration_ps=200),
              _make_record(kind="data_move", duration_ps=50),
              _make_record(kind="comm",      duration_ps=30),
              _make_record(kind="other",     duration_ps=5,
                           hlo_category="never-seen"),
          ]
          pstats = cb._PipelineStats(while_total_ps=4242,
                                       unknown_categories={"never-seen": 1},
                                       n_events_unresolved=7)
          totals = cb._compute_totals(recs, pstats=pstats, step_duration_ps=10_000)
          self.assertEqual(totals["n_events_total"], 5)
          self.assertEqual(totals["n_events_compute"], 2)
          self.assertEqual(totals["n_events_data_move"], 1)
          self.assertEqual(totals["n_events_comm"], 1)
          self.assertEqual(totals["n_events_other"], 1)
          self.assertEqual(totals["n_events_unresolved"], 7)
          self.assertEqual(totals["compute_duration_ps"], 300)
          self.assertEqual(totals["data_move_duration_ps"], 50)
          self.assertEqual(totals["comm_duration_ps"], 30)
          self.assertEqual(totals["other_duration_ps"], 5)
          self.assertEqual(totals["while_container_duration_ps"], 4242)
          self.assertEqual(totals["non_while_duration_ps_sum"], 300 + 50 + 30 + 5)
          self.assertEqual(totals["unknown_categories"], {"never-seen": 1})
          self.assertAlmostEqual(totals["while_pct_of_step"], 100.0 * 4242 / 10000)
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_aggregate_by_key`:

  ```python
  def _compute_totals(records: list[EventRecord], *, pstats: _PipelineStats,
                       step_duration_ps: int) -> dict:
      """Spec §5 totals block: per-kind sums, counts, while accounting,
      unknown_categories, unresolved counter."""
      n_by_kind = {"compute": 0, "data_move": 0, "comm": 0, "other": 0}
      d_by_kind = {"compute": 0, "data_move": 0, "comm": 0, "other": 0}
      for r in records:
          n_by_kind[r.kind] += 1
          d_by_kind[r.kind] += r.duration_ps
      non_while_sum = sum(d_by_kind.values())
      while_pct = (100.0 * pstats.while_total_ps / step_duration_ps
                   if step_duration_ps > 0 else 0.0)
      return {
          "n_events_total":         len(records),
          "n_events_compute":       n_by_kind["compute"],
          "n_events_data_move":     n_by_kind["data_move"],
          "n_events_comm":          n_by_kind["comm"],
          "n_events_other":         n_by_kind["other"],
          "n_events_unresolved":    pstats.n_events_unresolved,
          "compute_duration_ps":    d_by_kind["compute"],
          "data_move_duration_ps":  d_by_kind["data_move"],
          "comm_duration_ps":       d_by_kind["comm"],
          "other_duration_ps":      d_by_kind["other"],
          "while_container_duration_ps": pstats.while_total_ps,
          "non_while_duration_ps_sum":   non_while_sum,
          "while_pct_of_step":      round(while_pct, 3),
          "unknown_categories":     dict(pstats.unknown_categories),
      }
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py
  git commit -m "feat(tpu-perf): add per-kind totals aggregator with concurrency-safe naming"
  ```

### Task 15: `_run_summary_mode` projection

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py`

Builds the full mode-1 JSON object: computes `totals` from the unfiltered record list (so `comm` and `data_move` always appear in totals/`by_kind_rollup` regardless of flags), then derives the rankable subset (compute-only by default, compute+comm with `--include-comm`), aggregates compute groups, sorts by `total_dur_ps` desc, slices `top_compute_groups[:K]`, builds `tail_compute` rollup, fills `agg_key_coverage` (count of `agg_key_kind` across the rankable compute subset, NOT the full record list — describes coverage of source attribution for the events that ranked).

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_summary_mode.py`:

  ```python
  class TestRunSummaryMode(unittest.TestCase):
      def _ctx(self, **overrides):
          base = {
              "status": "ok",
              "step_id": 1,
              "step_window_ps": [0, 1_000_000],
              "step_duration_ps": 1_000_000,
              "notes": [],
              "pipeline_stats": cb._PipelineStats(),
              "profile_dir": "/x", "device": "/device:TPU:0",
              "xspace_pb_path": "/x/p.xplane.pb",
          }
          base.update(overrides)
          return base

      def test_top_compute_groups_sorted_descending(self):
          recs = [
              _make_record(agg_key="A", kind="compute", duration_ps=10),
              _make_record(agg_key="B", kind="compute", duration_ps=100),
              _make_record(agg_key="C", kind="compute", duration_ps=50),
          ]
          doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=10)
          ranks = [g["agg_key"] for g in doc["top_compute_groups"]]
          self.assertEqual(ranks, ["B", "C", "A"])
          self.assertEqual(doc["top_compute_groups"][0]["rank"], 1)

      def test_top_truncates_to_K(self):
          recs = [_make_record(agg_key=f"K{i}", kind="compute", duration_ps=i + 1)
                  for i in range(20)]
          doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=5)
          self.assertEqual(len(doc["top_compute_groups"]), 5)
          self.assertEqual(doc["tail_compute"]["n_groups_omitted"], 15)
          # Tail duration = sum of the 15 smallest durations (1..15)
          self.assertEqual(doc["tail_compute"]["dur_ps"], sum(range(1, 16)))

      def test_pct_denominators(self):
          recs = [
              _make_record(agg_key="A", kind="compute",   duration_ps=400),
              _make_record(agg_key="B", kind="data_move", duration_ps=100),
              _make_record(agg_key="C", kind="comm",      duration_ps=300),
          ]
          # comm excluded by default
          doc = cb._run_summary_mode(recs, ctx=self._ctx(step_duration_ps=1000),
                                       include_comm=False, top=10)
          a = doc["top_compute_groups"][0]
          self.assertEqual(a["agg_key"], "A")
          # pct_of_compute denom = compute_duration_ps = 400
          self.assertAlmostEqual(a["pct_of_compute"], 100.0)
          # pct_of_step denom = step_duration_ps = 1000
          self.assertAlmostEqual(a["pct_of_step"], 40.0)

      def test_include_comm_keeps_comm_records_in_totals(self):
          recs = [
              _make_record(agg_key="A", kind="compute", duration_ps=100),
              _make_record(agg_key="B", kind="comm",    duration_ps=200),
          ]
          doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=True, top=10)
          self.assertEqual(doc["totals"]["comm_duration_ps"], 200)

      def test_totals_and_rollup_include_comm_even_when_flag_false(self):
          # Spec §5: `totals` and `by_kind_rollup` always reflect the whole
          # step (cross-kind), regardless of include_comm. The flag only
          # affects which records get ranked into top_compute_groups.
          recs = [
              _make_record(agg_key="A", kind="compute",   duration_ps=100),
              _make_record(agg_key="B", kind="comm",      duration_ps=200),
              _make_record(agg_key="C", kind="data_move", duration_ps=50),
          ]
          doc = cb._run_summary_mode(recs, ctx=self._ctx(),
                                     include_comm=False, top=10)
          self.assertEqual(doc["totals"]["comm_duration_ps"], 200)
          self.assertEqual(doc["totals"]["data_move_duration_ps"], 50)
          self.assertEqual(doc["by_kind_rollup"]["comm"]["dur_ps"], 200)
          self.assertEqual(doc["by_kind_rollup"]["data_move"]["dur_ps"], 50)
          # But comm must NOT appear in top_compute_groups.
          ranked_keys = {g["agg_key"] for g in doc["top_compute_groups"]}
          self.assertNotIn("B", ranked_keys)
          self.assertNotIn("C", ranked_keys)
          self.assertIn("A", ranked_keys)

      def test_agg_key_coverage(self):
          # Coverage counts records that go into the compute ranking, so all
          # records must be kind="compute" for the assertion to hit. The
          # data_move case is exercised separately in mode 3 tests.
          recs = [
              _make_record(agg_key="stack:abc", agg_key_kind="stack",
                           kind="compute", duration_ps=10),
              _make_record(agg_key="stack:abc", agg_key_kind="stack",
                           kind="compute", duration_ps=10),
              _make_record(agg_key="tfop:Foo",  agg_key_kind="tf_op",
                           kind="compute", duration_ps=20),
              _make_record(agg_key="nosrc:pad", agg_key_kind="no_source",
                           kind="compute", duration_ps=5),
          ]
          doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=10)
          self.assertEqual(doc["agg_key_coverage"],
                           {"stack": 2, "tf_op": 1, "no_source": 1})

      def test_top_default_50(self):
          recs = [_make_record(agg_key=f"K{i}", kind="compute", duration_ps=i + 1)
                  for i in range(60)]
          doc = cb._run_summary_mode(recs, ctx=self._ctx(), include_comm=False, top=50)
          self.assertEqual(len(doc["top_compute_groups"]), 50)
  ```

- [ ] **Step 2: Run; expect failure**

- [ ] **Step 3: Implement in compute_breakdown.py**

  Insert after `_compute_totals`:

  ```python
  def _run_summary_mode(records: list[EventRecord], *, ctx: dict,
                          include_comm: bool, top: int) -> dict:
      pstats = ctx["pipeline_stats"]
      step_dur = ctx["step_duration_ps"]
      # Spec §5: `totals` and `by_kind_rollup` are cross-kind summaries of
      # the *whole* step — they reflect the actual breakdown including
      # comm/data_move regardless of the include_comm flag. The flag only
      # controls which records get ranked into `top_compute_groups`.
      # `include_comm` historically allowed comm into the ranking too;
      # we keep that for parity with --include-data-move (mode 2).
      totals = _compute_totals(records, pstats=pstats, step_duration_ps=step_dur)
      if include_comm:
          rankable = [r for r in records if r.kind in ("compute", "comm")]
      else:
          rankable = [r for r in records if r.kind == "compute"]
      compute_records = rankable
      groups = _aggregate_by_key(compute_records)
      ordered = sorted(groups.values(), key=lambda g: -g.total_dur_ps)

      compute_dur = totals["compute_duration_ps"] or 1
      step_dur_safe = step_dur or 1

      def _g_to_dict(g: _GroupAgg, rank: int) -> dict:
          return {
              "rank": rank,
              "agg_key":      g.agg_key,
              "agg_key_kind": g.agg_key_kind,
              "source_inner": g.source_inner,
              "tf_op":        g.tf_op,
              "source_stack": g.source_stack,
              "n_executions": g.n_executions,
              "total_dur_ps": g.total_dur_ps,
              "min_dur_ps":   g.min_dur_ps,
              "max_dur_ps":   g.max_dur_ps,
              "avg_dur_ps":   round(g.avg_dur_ps, 3),
              "pct_of_compute": round(100.0 * g.total_dur_ps / compute_dur, 3),
              "pct_of_step":    round(100.0 * g.total_dur_ps / step_dur_safe, 3),
              "hlo_categories": dict(g.hlo_categories),
              "flops_sum":          g.flops_sum,
              "bytes_accessed_sum": g.bytes_accessed_sum,
              "example_hlo_op":     g.example_hlo_op,
          }

      top_list = [_g_to_dict(g, i + 1) for i, g in enumerate(ordered[:top])]
      tail = ordered[top:]
      tail_dur = sum(g.total_dur_ps for g in tail)

      coverage = {"stack": 0, "tf_op": 0, "no_source": 0}
      for r in compute_records:
          coverage[r.agg_key_kind] = coverage.get(r.agg_key_kind, 0) + 1

      by_kind_rollup: dict = {}
      for kind in ("compute", "data_move", "comm"):
          n = totals[f"n_events_{kind}"]
          d = totals[f"{kind}_duration_ps"]
          by_kind_rollup[kind] = {
              "n": n, "dur_ps": d,
              "pct_of_step": round(100.0 * d / step_dur_safe, 3),
          }

      return {
          "status": "ok",
          "mode": "summary",
          "profile_dir": ctx["profile_dir"],
          "device": ctx["device"],
          "step_id": ctx["step_id"],
          "step_window_ps": ctx["step_window_ps"],
          "step_duration_ps": step_dur,
          "notes": list(ctx["notes"]),
          "totals": totals,
          "agg_key_coverage": coverage,
          "top_compute_groups": top_list,
          "tail_compute": {"n_groups_omitted": len(tail), "dur_ps": tail_dur},
          "by_kind_rollup": by_kind_rollup,
      }
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py
  git commit -m "feat(tpu-perf): implement mode-1 summary projection"
  ```

### Task 16: Wire `--mode summary` into `main()`

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py`

After this task, `python3 compute_breakdown.py <real-fixture> --mode summary` produces a real JSON document (not the `not_implemented` placeholder).

- [ ] **Step 1: Write the failing test**

  Append to `tests/test_summary_mode.py`:

  ```python
  class TestSummaryEndToEnd(unittest.TestCase):
      def test_summary_on_synthetic_xspace_emits_valid_json(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big",
                         flops=1_000_000, bytes_accessed=1024,
                         shape_with_layout="bf16[8]{0}")
          add_hlo_event(xs, em_id=11, hlo_op_text="small = bf16[2] fusion(...)",
                         offset_ps=600, duration_ps=10_000_000,
                         hlo_category="loop fusion", tf_op="jit/Small",
                         flops=50_000, bytes_accessed=64,
                         shape_with_layout="bf16[2]{0}")
          add_hlo_event(xs, em_id=12, hlo_op_text="copy.0",
                         offset_ps=800, duration_ps=5_000_000,
                         hlo_category="data formatting")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              r = subprocess.run(
                  [sys.executable, str(SCRIPT), tmp, "--mode", "summary"],
                  capture_output=True, text=True,
              )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["status"], "ok")
          self.assertEqual(doc["mode"], "summary")
          self.assertEqual(doc["totals"]["n_events_total"], 3)
          self.assertEqual(doc["totals"]["n_events_compute"], 2)
          self.assertEqual(doc["totals"]["n_events_data_move"], 1)
          self.assertGreaterEqual(len(doc["top_compute_groups"]), 1)
          self.assertEqual(doc["top_compute_groups"][0]["agg_key"], "tfop:jit/Big")


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: returns `{"status":"absent","reason":"not_implemented"}` (the chunk-1 placeholder), failing on `assertEqual(doc["status"], "ok")`.

- [ ] **Step 3: Edit `main()` in compute_breakdown.py**

  Replace the chunk-1 placeholder block with mode dispatch:

  ```python
  def main(argv=None) -> int:
      args = build_parser().parse_args(argv)

      if args.step is not None and args.step_id is not None:
          print("error: cannot pass both --step and --step-id", file=sys.stderr)
          return 1

      try:
          records, ctx = _load_and_normalize(
              profile_dir=args.profile_dir,
              device=args.device,
              step_idx=args.step,
              step_id=args.step_id,
          )
      except ValueError as ex:
          print(f"error: {ex}", file=sys.stderr)
          return 1

      if records is None:
          # Absent path. ctx already carries status/reason/notes.
          out = {
              "status": ctx["status"], "reason": ctx["reason"],
              "mode": args.mode, "profile_dir": args.profile_dir,
              "notes": ctx.get("notes", []),
          }
          _emit(out)
          return 0

      if args.mode == "summary":
          _emit(_run_summary_mode(records, ctx=ctx,
                                    include_comm=args.include_comm,
                                    top=args.top))
          return 0

      # Other modes wired up in later chunks.
      _emit({"status": "absent", "reason": "not_implemented",
             "mode": args.mode, "profile_dir": args.profile_dir,
             "notes": []})
      return 0
  ```

- [ ] **Step 4: Run; all summary tests pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: every test passes.

- [ ] **Step 5: Smoke-test against the real fixture**

  Run:
  ```bash
  python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
    --mode summary | python3 -m json.tool > /dev/null && echo OK
  ```
  Expected: `OK`. (Validates that real-fixture output parses as JSON.)

- [ ] **Step 6: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_summary_mode.py
  git commit -m "feat(tpu-perf): wire --mode summary into main(); end-to-end test"
  ```

---

## Chunk 4: Mode 2 — `by_source` (capability 2)

This chunk produces the per-`agg_key` full table that lets Claude do client-side scope filtering (e.g., "everything in `attention.py`"). It is the simplest mode after summary because Stages 1-3 are already in place and `_aggregate_by_key` already does most of the work; mode 2's only new behavior is per-group projection (no sort, no truncation), `--include-data-move` toggle, the shapes cap-at-8 with `shapes_truncated` flag, and the `dtypes` histogram.

**Files this chunk creates or modifies:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py` (add `_run_by_source_mode`, extend `build_parser`, extend `main()` dispatch)

After this chunk, `python3 compute_breakdown.py <fixture> --mode by_source` returns a real JSON document with the full per-`agg_key` table.

### Task 17: `_run_by_source_mode` projection

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`

The projection takes Stage 3 records, filters per `--include-comm` and `--include-data-move`, runs `_aggregate_by_key` with the shapes cap, and emits the per-group rows in **insertion order** (the spec says "not sorted, not truncated"; insertion order is the most predictable choice for tests). Each row carries the per-group fields enumerated in spec §6, including `flops_sum` / `model_flops_sum` / `bytes_accessed_sum` (which may be `null` when no event in the group reported them — `_GroupAgg`'s null-safe properties already handle that), the `shapes` list with cap-at-8 (`shapes_truncated` set when the cap is hit), the `dtypes` histogram, and `dtype_uncertain`.

`totals` here is the same dict that `_compute_totals` returns plus a single extra field `n_groups_total` (count of rows in `groups`). Reuse `_compute_totals` directly so cross-mode equality (spec §11.2) holds by construction.

- [ ] **Step 1: Write the failing tests**

  Create `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py`:

  ```python
  """Mode 2 (by_source) projection."""
  import json
  import pathlib
  import subprocess
  import sys
  import tempfile
  import unittest

  HERE = pathlib.Path(__file__).resolve().parent
  sys.path.insert(0, str(HERE.parent))
  import compute_breakdown as cb  # noqa: E402

  # Reuse synthetic builders / record factory from the pipeline test module.
  from test_pipeline import (  # noqa: E402
      _make_record, make_minimal_xspace, add_hlo_event,
  )
  from test_summary_mode import SCRIPT  # noqa: E402  (reuse main-script path)


  def _ctx(**overrides):
      base = {
          "status": "ok",
          "step_id": 1,
          "step_window_ps": [0, 1_000_000],
          "step_duration_ps": 1_000_000,
          "notes": [],
          "pipeline_stats": cb._PipelineStats(),
          "profile_dir": "/x", "device": "/device:TPU:0",
          "xspace_pb_path": "/x/p.xplane.pb",
      }
      base.update(overrides)
      return base


  class TestRunBySourceMode(unittest.TestCase):
      def test_groups_emitted_in_insertion_order_not_sorted(self):
          # Three groups; durations chosen so a sort would reorder them.
          recs = [
              _make_record(agg_key="A", kind="compute", duration_ps=10),
              _make_record(agg_key="B", kind="compute", duration_ps=100),
              _make_record(agg_key="C", kind="compute", duration_ps=50),
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
          self.assertEqual([g["agg_key"] for g in doc["groups"]], ["A", "B", "C"])
          self.assertEqual(doc["totals"]["n_groups_total"], 3)

      def test_data_move_excluded_by_default(self):
          recs = [
              _make_record(agg_key="C", kind="compute",   duration_ps=10),
              _make_record(agg_key="D", kind="data_move", duration_ps=20),
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
          self.assertEqual([g["agg_key"] for g in doc["groups"]], ["C"])
          # Totals still reflect ALL records (cross-mode invariant).
          self.assertEqual(doc["totals"]["data_move_duration_ps"], 20)
          self.assertEqual(doc["totals"]["compute_duration_ps"], 10)

      def test_data_move_included_when_flag_set(self):
          recs = [
              _make_record(agg_key="C", kind="compute",   duration_ps=10),
              _make_record(agg_key="D", kind="data_move", duration_ps=20),
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=True)
          self.assertEqual({g["agg_key"] for g in doc["groups"]}, {"C", "D"})

      def test_comm_excluded_by_default_and_includable(self):
          recs = [
              _make_record(agg_key="C", kind="compute", duration_ps=10),
              _make_record(agg_key="X", kind="comm",    duration_ps=99),
          ]
          d_off = cb._run_by_source_mode(recs, ctx=_ctx(),
                                           include_comm=False,
                                           include_data_move=False)
          self.assertEqual([g["agg_key"] for g in d_off["groups"]], ["C"])
          d_on = cb._run_by_source_mode(recs, ctx=_ctx(),
                                          include_comm=True,
                                          include_data_move=False)
          self.assertEqual({g["agg_key"] for g in d_on["groups"]}, {"C", "X"})

      def test_shapes_capped_at_eight_and_flag_set(self):
          # 9 distinct shapes for the same agg_key -> cap at 8, flag true.
          recs = [
              _make_record(agg_key="K", kind="compute", duration_ps=1,
                           shape_with_layout=f"bf16[{i}]{{0}}")
              for i in range(9)
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
          g = doc["groups"][0]
          self.assertEqual(len(g["shapes"]), 8)
          self.assertTrue(g["shapes_truncated"])

      def test_shapes_not_truncated_when_under_cap(self):
          recs = [
              _make_record(agg_key="K", kind="compute", duration_ps=1,
                           shape_with_layout=f"bf16[{i}]{{0}}")
              for i in range(3)
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
          g = doc["groups"][0]
          self.assertEqual(len(g["shapes"]), 3)
          self.assertFalse(g["shapes_truncated"])

      def test_dtypes_histogram_and_uncertain_propagation(self):
          recs = [
              _make_record(agg_key="K", kind="compute", duration_ps=1,
                           dtype="bf16"),
              _make_record(agg_key="K", kind="compute", duration_ps=1,
                           dtype="bf16"),
              _make_record(agg_key="K", kind="compute", duration_ps=1,
                           dtype="fp8", dtype_uncertain=True),
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
          g = doc["groups"][0]
          self.assertEqual(g["dtypes"], {"bf16": 2, "fp8": 1})
          self.assertTrue(g["dtype_uncertain"])

      def test_null_flops_when_no_events_reported(self):
          recs = [
              _make_record(agg_key="K", kind="compute", duration_ps=1,
                           flops=None, bytes_accessed=None,
                           model_flops=None),
          ]
          doc = cb._run_by_source_mode(recs, ctx=_ctx(),
                                         include_comm=False,
                                         include_data_move=False)
          g = doc["groups"][0]
          self.assertIsNone(g["flops_sum"])
          self.assertIsNone(g["bytes_accessed_sum"])
          self.assertIsNone(g["model_flops_sum"])

      def test_totals_match_summary_mode(self):
          # Cross-mode invariant: by_source.totals derives from same records
          # without any kind filter on totals computation.
          recs = [
              _make_record(agg_key="A", kind="compute",   duration_ps=400),
              _make_record(agg_key="B", kind="data_move", duration_ps=100),
              _make_record(agg_key="C", kind="comm",      duration_ps=300),
          ]
          ctx = _ctx(step_duration_ps=1000)
          d_summary = cb._run_summary_mode(recs, ctx=ctx,
                                              include_comm=False, top=10)
          d_bysrc = cb._run_by_source_mode(recs, ctx=ctx,
                                              include_comm=False,
                                              include_data_move=False)
          for k in ("compute_duration_ps", "data_move_duration_ps",
                    "comm_duration_ps", "other_duration_ps",
                    "n_events_unresolved", "unknown_categories"):
              self.assertEqual(d_summary["totals"][k], d_bysrc["totals"][k],
                                msg=f"totals[{k}] differs")


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run; expect failure**

  Run:
  ```bash
  python3 -m unittest -v \
    plugins.tpu-perf.skills.compute-breakdown.scripts.tests.test_by_source_mode
  ```
  Expected: every test fails with `AttributeError: module 'compute_breakdown' has no attribute '_run_by_source_mode'`.

- [ ] **Step 3: Implement `_run_by_source_mode`**

  Insert into `compute_breakdown.py` after `_run_summary_mode`:

  ```python
  def _run_by_source_mode(records: list[EventRecord], *, ctx: dict,
                            include_comm: bool,
                            include_data_move: bool) -> dict:
      """Mode 2: full per-agg_key table for client-side scope filtering.

      Not sorted. Not truncated. Claude post-filters by source_stack
      substring or tf_op contains.
      """
      step_dur = ctx["step_duration_ps"]
      pstats = ctx["pipeline_stats"]
      totals = _compute_totals(records, pstats=pstats, step_duration_ps=step_dur)

      # Filter records that contribute to the visible group rows.
      kept_kinds = {"compute"}
      if include_data_move:
          kept_kinds.add("data_move")
      if include_comm:
          kept_kinds.add("comm")
      visible = [r for r in records if r.kind in kept_kinds]

      groups = _aggregate_by_key(visible, dedupe_shapes_cap=8)
      group_rows = []
      for g in groups.values():
          n = g.n_executions
          group_rows.append({
              "agg_key":            g.agg_key,
              "agg_key_kind":       g.agg_key_kind,
              "source_inner":       g.source_inner,
              "source_stack":       g.source_stack,
              "tf_op":              g.tf_op,
              "kind":               g.kind,
              "hlo_categories":     dict(g.hlo_categories),
              "n_executions":       n,
              "total_dur_ps":       g.total_dur_ps,
              "min_dur_ps":         g.min_dur_ps,
              "max_dur_ps":         g.max_dur_ps,
              "avg_dur_ps":         g.total_dur_ps // n if n else 0,
              "flops_sum":          g.flops_sum,
              "model_flops_sum":    g.model_flops_sum,
              "bytes_accessed_sum": g.bytes_accessed_sum,
              "shapes":             list(g.shapes),
              "shapes_truncated":   g.shapes_truncated,
              "dtypes":             dict(g.dtypes),
              "dtype_uncertain":    g.dtype_uncertain,
              "example_hlo_op":     g.example_hlo_op,
          })

      totals_out = dict(totals)
      totals_out["n_groups_total"] = len(group_rows)

      return {
          "status":           "ok",
          "mode":             "by_source",
          "profile_dir":      ctx["profile_dir"],
          "device":           ctx["device"],
          "step_id":          ctx["step_id"],
          "step_window_ps":   ctx["step_window_ps"],
          "step_duration_ps": step_dur,
          "notes":            list(ctx["notes"]),
          "totals":           totals_out,
          "groups":           group_rows,
      }
  ```

  Note: this assumes `_GroupAgg` already exposes a `shapes_truncated` boolean. It does not yet (chunk 3 only stores `shapes` with a cap). Add it now in the same edit:

  ```python
  # In the _GroupAgg dataclass, add:
  shapes_truncated: bool = False
  ```

  And in `_aggregate_by_key`, change the shapes append block to flip the flag when the cap is exceeded:

  ```python
  if r.shape_with_layout and r.shape_with_layout not in g.shapes:
      if len(g.shapes) < dedupe_shapes_cap:
          g.shapes.append(r.shape_with_layout)
      else:
          g.shapes_truncated = True
  ```

  This is a small, additive change — the chunk-3 summary tests don't assert on `shapes_truncated`, so they continue to pass. Re-run them to confirm.

- [ ] **Step 4: Run; tests pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: all by_source tests pass; all previous chunk's tests still pass.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py
  git commit -m "feat(tpu-perf): add by_source projection with shape cap and dtype histogram"
  ```

### Task 18: Add `--include-data-move` CLI flag

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py`

The flag is global at the argparse layer (per the cross-mode flag scoping policy). It is a no-op for modes other than `by_source`.

- [ ] **Step 1: Write the failing test**

  Append to `tests/test_by_source_mode.py`:

  ```python
  class TestBySourceCLI(unittest.TestCase):
      def test_include_data_move_flag_present(self):
          # The flag must parse without error.
          ns = cb.build_parser().parse_args(
              ["/x", "--mode", "by_source", "--include-data-move"]
          )
          self.assertTrue(ns.include_data_move)

      def test_include_data_move_default_false(self):
          ns = cb.build_parser().parse_args(["/x", "--mode", "by_source"])
          self.assertFalse(ns.include_data_move)
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: `AttributeError: 'Namespace' object has no attribute 'include_data_move'`.

- [ ] **Step 3: Edit `build_parser`**

  Add this to `build_parser()` in `compute_breakdown.py`, alongside the existing `--include-comm` flag:

  ```python
  parser.add_argument("--include-data-move", action="store_true",
                       help="(by_source mode) include kind=data_move groups in the table")
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py
  git commit -m "feat(tpu-perf): add --include-data-move flag for by_source mode"
  ```

### Task 19: Wire `--mode by_source` into `main()`

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py`

After this task, `python3 compute_breakdown.py <real-fixture> --mode by_source` returns a real JSON document, not the placeholder.

- [ ] **Step 1: Write the failing test**

  Append to `tests/test_by_source_mode.py`:

  ```python
  class TestBySourceEndToEnd(unittest.TestCase):
      def test_emits_valid_json_with_groups_block(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big",
                         flops=1_000_000, bytes_accessed=1024,
                         shape_with_layout="bf16[8]{0}")
          add_hlo_event(xs, em_id=11, hlo_op_text="copy.0",
                         offset_ps=600, duration_ps=5_000_000,
                         hlo_category="data formatting")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              # default: data_move excluded, so only the bf16 fusion shows.
              r = subprocess.run(
                  [sys.executable, str(SCRIPT), tmp, "--mode", "by_source"],
                  capture_output=True, text=True,
              )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["status"], "ok")
          self.assertEqual(doc["mode"], "by_source")
          self.assertEqual(doc["totals"]["n_events_compute"], 1)
          self.assertEqual(doc["totals"]["n_events_data_move"], 1)
          self.assertEqual(len(doc["groups"]), 1)
          self.assertEqual(doc["groups"][0]["tf_op"], "jit/Big")

      def test_include_data_move_adds_data_move_groups(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big")
          add_hlo_event(xs, em_id=11, hlo_op_text="copy.0",
                         offset_ps=600, duration_ps=5_000_000,
                         hlo_category="data formatting", tf_op="jit/Copy")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              r = subprocess.run(
                  [sys.executable, str(SCRIPT), tmp, "--mode", "by_source",
                   "--include-data-move"],
                  capture_output=True, text=True,
              )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(len(doc["groups"]), 2)
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: returns `{"status":"absent","reason":"not_implemented"}`, failing on `assertEqual(doc["status"], "ok")`.

- [ ] **Step 3: Edit `main()` dispatch**

  Insert after the `summary` branch and before the `not_implemented` fallback:

  ```python
      if args.mode == "by_source":
          _emit(_run_by_source_mode(records, ctx=ctx,
                                       include_comm=args.include_comm,
                                       include_data_move=args.include_data_move))
          return 0
  ```

- [ ] **Step 4: Run; all chunk tests pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: every test in test_pipeline, test_summary_mode, test_by_source_mode passes.

- [ ] **Step 5: Smoke-test against real fixture**

  Run:
  ```bash
  python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
    --mode by_source | python3 -m json.tool > /dev/null && echo OK
  ```
  Expected: `OK`.

- [ ] **Step 6: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_by_source_mode.py
  git commit -m "feat(tpu-perf): wire --mode by_source into main(); end-to-end test"
  ```

---

## Chunk 5: Mode 3 — `non_compute` (capability 3)

This chunk produces the padding/cast/copy/transpose audit. Two layers of output: `by_category` (one row per `hlo_category`, no thresholding) and `by_source_within_category` (full (category, agg_key) breakdown). New behavior beyond what Stages 1-3 give us:
1. The HLO-IR-text regex (`HLO_OP_RE` per spec §7) to extract `dtype_change` / `layout_change` / `shapes_in` / `shapes_out`.
2. Default-on inclusion of `async-done` events as `hlo_category="async-done (comm stall)"`, with `--no-comm-stalls` flip.
3. Two-layer aggregation: per-category rollup *and* per-(category, agg_key) breakdown.

**Files this chunk creates or modifies:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py` (add `HLO_OP_RE`, `_parse_hlo_op_text`, `_run_non_compute_mode`, extend `build_parser`, extend `main()` dispatch)

After this chunk, `python3 compute_breakdown.py <fixture> --mode non_compute` returns a real JSON document with both layers populated.

### Task 20: `HLO_OP_RE` and `_parse_hlo_op_text` helper

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py`

The regex inspects only the **first operand** of an HLO op (per spec §7), which is sufficient for the data-formatting ops mode 3 surfaces (`convert`, `transpose`, `copy`, `pad`, `broadcast`, etc.). The helper returns a 4-tuple `(out_dtype, out_layout, in_dtype, in_layout)` of `str | None`. On no match: all four are `None`.

- [ ] **Step 1: Write the failing tests**

  Create `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py`:

  ```python
  """Mode 3 (non_compute) projection."""
  import json
  import pathlib
  import subprocess
  import sys
  import tempfile
  import unittest

  HERE = pathlib.Path(__file__).resolve().parent
  sys.path.insert(0, str(HERE.parent))
  import compute_breakdown as cb  # noqa: E402

  from test_pipeline import (  # noqa: E402
      _make_record, make_minimal_xspace, add_hlo_event,
  )
  from test_summary_mode import SCRIPT  # noqa: E402


  class TestParseHloOpText(unittest.TestCase):
      def test_dtype_change_only_layout_match(self):
          # convert: input is bf16, output is f32, both layouts {1,0}
          out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
              "%c.0 = f32[8,4]{1,0} convert(bf16[8,4]{1,0} %x.1)"
          )
          self.assertEqual(out_dt, "f32")
          self.assertEqual(in_dt, "bf16")
          self.assertEqual(out_lay, "{1,0}")
          self.assertEqual(in_lay, "{1,0}")

      def test_layout_change_only(self):
          out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
              "%t.0 = bf16[8,4]{0,1} transpose(bf16[8,4]{1,0} %x.1)"
          )
          self.assertEqual(out_dt, "bf16")
          self.assertEqual(in_dt, "bf16")
          self.assertEqual(out_lay, "{0,1}")
          self.assertEqual(in_lay, "{1,0}")

      def test_layout_omitted_returns_none(self):
          # No braces on either operand or output -> layouts are None.
          out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text(
              "%cp.0 = bf16[8,4] copy(bf16[8,4] %x.1)"
          )
          self.assertEqual(out_dt, "bf16")
          self.assertEqual(in_dt, "bf16")
          self.assertIsNone(out_lay)
          self.assertIsNone(in_lay)

      def test_no_match_returns_all_none(self):
          out_dt, out_lay, in_dt, in_lay = cb._parse_hlo_op_text("")
          self.assertIsNone(out_dt)
          self.assertIsNone(in_dt)
          self.assertIsNone(out_lay)
          self.assertIsNone(in_lay)

      def test_no_match_on_garbage(self):
          out_dt, _, in_dt, _ = cb._parse_hlo_op_text("not an HLO op")
          self.assertIsNone(out_dt)
          self.assertIsNone(in_dt)

      def test_lhs_with_or_without_percent(self):
          # The regex must accept both "%name = ..." and "name = ..." forms.
          out_dt1, *_ = cb._parse_hlo_op_text(
              "%foo = bf16[1]{0} copy(bf16[1]{0} %x)"
          )
          out_dt2, *_ = cb._parse_hlo_op_text(
              "foo.bar = bf16[1]{0} copy(bf16[1]{0} %x)"
          )
          self.assertEqual(out_dt1, "bf16")
          self.assertEqual(out_dt2, "bf16")
  ```

- [ ] **Step 2: Run; expect failure**

  Run:
  ```bash
  python3 -m unittest -v \
    plugins.tpu-perf.skills.compute-breakdown.scripts.tests.test_non_compute_mode.TestParseHloOpText
  ```
  Expected: every test fails with `AttributeError: module 'compute_breakdown' has no attribute '_parse_hlo_op_text'`.

- [ ] **Step 3: Implement regex and helper**

  Insert near the top of `compute_breakdown.py`, after the `EventRecord` dataclass:

  ```python
  HLO_OP_RE = re.compile(
      r'^\s*%?[\w.]+\s*=\s*'                        # lhs: "%name =" or "name ="
      r'([a-z][a-z0-9]*)\['                         # group 1: out dtype
      r'([^\]]*)\]'                                 # group 2: out shape
      r'(\{[^}]*\})?'                               # group 3: out layout (optional)
      r'\s+\w[-\w]*\s*\('                           # opcode + "("
      r'\s*([a-z][a-z0-9]*)\['                      # group 4: first-operand dtype
      r'([^\]]*)\]'                                 # group 5: first-operand shape
      r'(\{[^}]*\})?'                               # group 6: first-operand layout (optional)
  )


  def _parse_hlo_op_text(text: str) -> tuple:
      """Extract (out_dtype, out_layout, in_dtype, in_layout) from an HLO IR
      string. Inspects only the first operand. Returns all-None on no match.
      The four-tuple form is the public contract; shape groups (2 and 5)
      are accessible by re-matching `HLO_OP_RE` for callers that want them.
      """
      if not text:
          return (None, None, None, None)
      m = HLO_OP_RE.match(text)
      if not m:
          return (None, None, None, None)
      return (m.group(1), m.group(3), m.group(4), m.group(6))


  def _parse_hlo_op_text_full(text: str) -> tuple:
      """Like `_parse_hlo_op_text` but also returns the out/in shape strings.
      Returns (out_dt, out_shape, out_lay, in_dt, in_shape, in_lay) or
      a six-None tuple on no match. Use this from `_run_non_compute_mode`
      to avoid fragile string splitting on '(' in the raw HLO IR text.
      """
      if not text:
          return (None, None, None, None, None, None)
      m = HLO_OP_RE.match(text)
      if not m:
          return (None, None, None, None, None, None)
      return (m.group(1), m.group(2), m.group(3),
              m.group(4), m.group(5), m.group(6))
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py
  git commit -m "feat(tpu-perf): add HLO-IR regex parser for dtype/layout change"
  ```

### Task 21: `_run_non_compute_mode` projection

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py`

The projection runs on `kind=data_move` records (and `kind=comm` records with `hlo_category="async-done (comm stall)"` when comm-stalls is enabled). It builds two output blocks:

- **`by_category`** — one row per `hlo_category` with execution count, durations (sum/min/max/avg), `n_groups` (distinct `agg_key`s in this category), and `agg_key_coverage` (3-key dict counting events whose agg_key was resolved via `stack` / `tf_op` / `no_source`).
- **`by_source_within_category`** — one row per (category, agg_key) pair with per-source durations, `shapes_in` / `shapes_out` (cap at 4, deduped), `dtype_change`, `layout_change`, and `example_hlo_op`. The `dtype_change` / `layout_change` come from running `_parse_hlo_op_text` on the example HLO op text per spec §7. `null` ≠ "no change".

Per spec §7 the `--no-comm-stalls` flag controls whether async-done is included; it is **on by default** (i.e., the flag flips it OFF). Mode 3 needs to know about async-done events even though they live as `kind=comm` in Stage 3. Approach: in this mode's filter we re-tag a `kind=comm` record's `hlo_category` to `"async-done (comm stall)"` if its existing category indicates an async-done flow event. Implementation detail to lock down: the simplest, robust signal we already have on the record is the original `hlo_category`. The Stage 3 normalizer leaves the original `hlo_category` on the record; the kind classifier (Task 7) marks comm by category. So an async-done event is exactly a record with `kind=="comm"` AND `hlo_category in {"async-done", "all-reduce-done", "all-gather-done", ...}`. To keep this in one place, define a constant `ASYNC_DONE_CATEGORIES` near the kind classifier in chunk 2's task 7 implementation; in mode 3 we check membership via that set.

This task therefore depends on `ASYNC_DONE_CATEGORIES` having been defined in chunk 2 task 7. If it was not, add it now in the same edit:

```python
ASYNC_DONE_CATEGORIES = frozenset({
    "async-done",
    "all-reduce-done",
    "all-gather-done",
    "reduce-scatter-done",
    "collective-permute-done",
    "send-done",
    "recv-done",
})
```

(These are the canonical async-done categories observed in TPU profiles.)

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_non_compute_mode.py`:

  ```python
  def _ctx(**overrides):
      base = {
          "status": "ok",
          "step_id": 1,
          "step_window_ps": [0, 1_000_000],
          "step_duration_ps": 1_000_000,
          "notes": [],
          "pipeline_stats": cb._PipelineStats(),
          "profile_dir": "/x", "device": "/device:TPU:0",
          "xspace_pb_path": "/x/p.xplane.pb",
      }
      base.update(overrides)
      return base


  class TestRunNonComputeMode(unittest.TestCase):
      def test_by_category_aggregates_data_move(self):
          recs = [
              _make_record(agg_key="A", kind="data_move", duration_ps=100,
                           hlo_category="data formatting"),
              _make_record(agg_key="B", kind="data_move", duration_ps=200,
                           hlo_category="data formatting"),
              _make_record(agg_key="C", kind="data_move", duration_ps=50,
                           hlo_category="copy"),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
          rows = {row["hlo_category"]: row for row in doc["by_category"]}
          self.assertEqual(rows["data formatting"]["n_executions"], 2)
          self.assertEqual(rows["data formatting"]["total_dur_ps"], 300)
          self.assertEqual(rows["data formatting"]["min_dur_ps"], 100)
          self.assertEqual(rows["data formatting"]["max_dur_ps"], 200)
          self.assertEqual(rows["data formatting"]["avg_dur_ps"], 150)
          self.assertEqual(rows["copy"]["n_executions"], 1)

      def test_by_source_within_category_per_pair_rows(self):
          recs = [
              _make_record(agg_key="A", kind="data_move", duration_ps=100,
                           hlo_category="data formatting", tf_op="jit/T"),
              _make_record(agg_key="B", kind="data_move", duration_ps=200,
                           hlo_category="data formatting", tf_op="jit/U"),
              _make_record(agg_key="A", kind="data_move", duration_ps=10,
                           hlo_category="data formatting", tf_op="jit/T"),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
          # Two distinct (category, agg_key) pairs: ("data formatting", A) and ("data formatting", B).
          self.assertEqual(len(doc["by_source_within_category"]), 2)
          a_row = next(r for r in doc["by_source_within_category"]
                       if r["agg_key"] == "A")
          self.assertEqual(a_row["n_executions"], 2)
          self.assertEqual(a_row["total_dur_ps"], 110)

      def test_dtype_and_layout_change_from_hlo_op_text(self):
          recs = [
              _make_record(
                  agg_key="A", kind="data_move", duration_ps=10,
                  hlo_category="data formatting",
                  hlo_op="%c.0 = f32[8,4]{1,0} convert(bf16[8,4]{1,0} %x.1)",
                  shape_with_layout="f32[8,4]{1,0}",
              ),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
          row = doc["by_source_within_category"][0]
          self.assertTrue(row["dtype_change"])
          self.assertFalse(row["layout_change"])

      def test_layout_change_null_when_layout_omitted(self):
          # No braces on input or output; layout cannot be decided.
          recs = [
              _make_record(
                  agg_key="A", kind="data_move", duration_ps=10,
                  hlo_category="copy",
                  hlo_op="%cp.0 = bf16[8,4] copy(bf16[8,4] %x.1)",
              ),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
          row = doc["by_source_within_category"][0]
          self.assertFalse(row["dtype_change"])  # bf16 == bf16
          self.assertIsNone(row["layout_change"])  # layouts both omitted

      def test_dtype_and_layout_change_null_on_unparseable(self):
          recs = [
              _make_record(
                  agg_key="A", kind="data_move", duration_ps=10,
                  hlo_category="data formatting",
                  hlo_op="not an HLO op",
              ),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
          row = doc["by_source_within_category"][0]
          self.assertIsNone(row["dtype_change"])
          self.assertIsNone(row["layout_change"])

      def test_async_done_included_by_default(self):
          # comm-stalls flag (default-on) re-tags async-done as data_move
          # for accounting purposes.
          recs = [
              _make_record(agg_key="X", kind="comm", duration_ps=500,
                           hlo_category="all-reduce-done"),
              _make_record(agg_key="A", kind="data_move", duration_ps=10,
                           hlo_category="data formatting"),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(step_duration_ps=10_000),
                                            include_comm=False,
                                            include_comm_stalls=True)
          cats = {row["hlo_category"]: row for row in doc["by_category"]}
          self.assertIn("async-done (comm stall)", cats)
          self.assertEqual(cats["async-done (comm stall)"]["total_dur_ps"], 500)
          self.assertIn(
              "async-done included as comm-stall non-compute time; pass --no-comm-stalls to exclude",
              doc["notes"],
          )
          # Pct of step counts both data_move (10) AND async-done (500) = 510.
          self.assertAlmostEqual(
              doc["totals"]["non_compute_pct_of_step"], 100.0 * 510 / 10_000
          )

      def test_no_comm_stalls_excludes_async_done(self):
          recs = [
              _make_record(agg_key="X", kind="comm", duration_ps=500,
                           hlo_category="all-reduce-done"),
              _make_record(agg_key="A", kind="data_move", duration_ps=10,
                           hlo_category="data formatting"),
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(step_duration_ps=10_000),
                                            include_comm=False,
                                            include_comm_stalls=False)
          cats = {row["hlo_category"] for row in doc["by_category"]}
          self.assertNotIn("async-done (comm stall)", cats)
          # Note absent.
          self.assertNotIn(
              "async-done included as comm-stall non-compute time; pass --no-comm-stalls to exclude",
              doc["notes"],
          )

      def test_totals_match_summary_when_no_comm_stalls(self):
          # Cross-mode invariant: with --no-comm-stalls,
          # non_compute.totals.data_move_duration_ps == summary.totals.data_move_duration_ps
          recs = [
              _make_record(agg_key="A", kind="compute",   duration_ps=400),
              _make_record(agg_key="B", kind="data_move", duration_ps=100),
              _make_record(agg_key="C", kind="comm",      duration_ps=300),
          ]
          d_summary = cb._run_summary_mode(recs, ctx=_ctx(),
                                              include_comm=False, top=10)
          d_nc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                              include_comm=False,
                                              include_comm_stalls=False)
          self.assertEqual(
              d_summary["totals"]["data_move_duration_ps"],
              d_nc["totals"]["data_move_duration_ps"],
          )

      def test_shapes_in_out_capped_at_four(self):
          # Five distinct hlo_op_texts -> shapes_in/out should cap at 4.
          recs = [
              _make_record(
                  agg_key="A", kind="data_move", duration_ps=1,
                  hlo_category="data formatting",
                  hlo_op=f"%t.{i} = bf16[{i},4]{{0,1}} transpose(bf16[4,{i}]{{1,0}} %x)",
              )
              for i in range(5)
          ]
          doc = cb._run_non_compute_mode(recs, ctx=_ctx(),
                                            include_comm=False,
                                            include_comm_stalls=False)
          row = doc["by_source_within_category"][0]
          self.assertLessEqual(len(row["shapes_in"]), 4)
          self.assertLessEqual(len(row["shapes_out"]), 4)
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: `AttributeError: module 'compute_breakdown' has no attribute '_run_non_compute_mode'`.

- [ ] **Step 3: Implement `_run_non_compute_mode`**

  Insert into `compute_breakdown.py` after `_run_by_source_mode`:

  ```python
  _COMM_STALL_CATEGORY = "async-done (comm stall)"
  _COMM_STALL_NOTE = (
      "async-done included as comm-stall non-compute time; "
      "pass --no-comm-stalls to exclude"
  )


  def _run_non_compute_mode(records: list[EventRecord], *, ctx: dict,
                              include_comm: bool,
                              include_comm_stalls: bool) -> dict:
      """Mode 3: padding/cast/copy/transpose audit. Two layers:
      by_category and by_source_within_category."""
      step_dur = ctx["step_duration_ps"]
      pstats = ctx["pipeline_stats"]
      totals = _compute_totals(records, pstats=pstats, step_duration_ps=step_dur)

      # Build the visible record list with comm-stall re-tagging.
      visible: list[EventRecord] = []
      for r in records:
          if r.kind == "data_move":
              visible.append(r)
          elif (include_comm_stalls and r.kind == "comm"
                and r.hlo_category in ASYNC_DONE_CATEGORIES):
              # Shallow-replace hlo_category for accounting; keep rest.
              visible.append(dataclasses.replace(
                  r, hlo_category=_COMM_STALL_CATEGORY
              ))
          elif include_comm and r.kind == "comm":
              visible.append(r)

      # Layer 1: by_category. Aggregate by hlo_category.
      cat_acc: dict[str, dict] = {}
      for r in visible:
          c = cat_acc.get(r.hlo_category)
          if c is None:
              c = {
                  "hlo_category": r.hlo_category,
                  "n_executions": 0, "total_dur_ps": 0,
                  "min_dur_ps": r.duration_ps, "max_dur_ps": r.duration_ps,
                  "agg_keys": set(),
                  "agg_key_coverage": {"stack": 0, "tf_op": 0, "no_source": 0},
              }
              cat_acc[r.hlo_category] = c
          c["n_executions"] += 1
          c["total_dur_ps"] += r.duration_ps
          if r.duration_ps < c["min_dur_ps"]:
              c["min_dur_ps"] = r.duration_ps
          if r.duration_ps > c["max_dur_ps"]:
              c["max_dur_ps"] = r.duration_ps
          c["agg_keys"].add(r.agg_key)
          c["agg_key_coverage"][r.agg_key_kind] = (
              c["agg_key_coverage"].get(r.agg_key_kind, 0) + 1
          )
      by_category = []
      for c in cat_acc.values():
          n = c["n_executions"]
          by_category.append({
              "hlo_category":     c["hlo_category"],
              "n_executions":     n,
              "total_dur_ps":     c["total_dur_ps"],
              "min_dur_ps":       c["min_dur_ps"],
              "max_dur_ps":       c["max_dur_ps"],
              "avg_dur_ps":       c["total_dur_ps"] // n if n else 0,
              "n_groups":         len(c["agg_keys"]),
              "agg_key_coverage": c["agg_key_coverage"],
          })

      # Layer 2: by_source_within_category. Aggregate by (category, agg_key).
      pair_acc: dict[tuple[str, str], dict] = {}
      for r in visible:
          key = (r.hlo_category, r.agg_key)
          p = pair_acc.get(key)
          if p is None:
              p = {
                  "hlo_category":  r.hlo_category,
                  "agg_key":       r.agg_key,
                  "agg_key_kind":  r.agg_key_kind,
                  "source_inner":  r.source_inner,
                  "source_stack":  r.source_stack,
                  "tf_op":         r.tf_op,
                  "n_executions": 0, "total_dur_ps": 0,
                  "min_dur_ps":   r.duration_ps, "max_dur_ps": r.duration_ps,
                  "shapes_in":  [], "shapes_out": [],
                  "example_hlo_op": r.hlo_op,
                  "_dtype_change_seen": False,
                  "_dtype_change_value": False,
                  "_layout_change_seen": False,
                  "_layout_change_value": False,
                  "_layout_change_null": False,
              }
              pair_acc[key] = p
          p["n_executions"] += 1
          p["total_dur_ps"] += r.duration_ps
          if r.duration_ps < p["min_dur_ps"]:
              p["min_dur_ps"] = r.duration_ps
          if r.duration_ps > p["max_dur_ps"]:
              p["max_dur_ps"] = r.duration_ps
          # dtype/layout change from this event's hlo_op text. Use the
          # full-form helper so shapes come from regex group captures, not
          # from string-splitting on '(' (which breaks on ops whose
          # operand list contains nested parentheses or attributes).
          (out_dt, out_shape, out_lay,
           in_dt, in_shape, in_lay) = _parse_hlo_op_text_full(r.hlo_op)
          if out_dt is not None and in_dt is not None:
              p["_dtype_change_seen"] = True
              if out_dt != in_dt:
                  p["_dtype_change_value"] = True
              # shapes_in / shapes_out (cap at 4, dedup). Annotate full
              # "dtype[shape]{layout}" form for readability.
              if out_shape is not None and len(p["shapes_out"]) < 4:
                  s = f"{out_dt}[{out_shape}]" + (out_lay if out_lay else "")
                  if s not in p["shapes_out"]:
                      p["shapes_out"].append(s)
              if in_shape is not None and len(p["shapes_in"]) < 4:
                  s = f"{in_dt}[{in_shape}]" + (in_lay if in_lay else "")
                  if s not in p["shapes_in"]:
                      p["shapes_in"].append(s)
              # Layout-change semantics across multiple events for one
              # (category, agg_key) group: if ANY event in the group shows
              # the layout actually differs (out_lay != in_lay, both
              # non-None), the group's layout_change is True. If at least
              # one event has both layouts present and equal, it's False.
              # If every event has at least one layout missing, fall back
              # to None (we couldn't tell). Matches dtype_change semantics.
              if out_lay is None or in_lay is None:
                  p["_layout_change_null"] = True
              else:
                  p["_layout_change_seen"] = True
                  if out_lay != in_lay:
                      p["_layout_change_value"] = True

      by_source_within_category = []
      for p in pair_acc.values():
          n = p["n_executions"]
          dtype_change = (p["_dtype_change_value"]
                            if p["_dtype_change_seen"] else None)
          if p["_layout_change_seen"]:
              layout_change = p["_layout_change_value"]
          elif p["_layout_change_null"]:
              layout_change = None
          else:
              layout_change = None
          by_source_within_category.append({
              "hlo_category":   p["hlo_category"],
              "agg_key":        p["agg_key"],
              "agg_key_kind":   p["agg_key_kind"],
              "source_inner":   p["source_inner"],
              "source_stack":   p["source_stack"],
              "tf_op":          p["tf_op"],
              "n_executions":   n,
              "total_dur_ps":   p["total_dur_ps"],
              "min_dur_ps":     p["min_dur_ps"],
              "max_dur_ps":     p["max_dur_ps"],
              "avg_dur_ps":     p["total_dur_ps"] // n if n else 0,
              "shapes_in":      p["shapes_in"] or None,
              "shapes_out":     p["shapes_out"] or None,
              "dtype_change":   dtype_change,
              "layout_change":  layout_change,
              "example_hlo_op": p["example_hlo_op"],
          })

      # totals additions: non_compute_pct_of_step / non_compute_pct_of_compute
      non_compute_dur = sum(p["total_dur_ps"] for p in pair_acc.values())
      compute_dur = totals["compute_duration_ps"]
      totals_out = dict(totals)
      totals_out["non_compute_pct_of_step"] = round(
          100.0 * non_compute_dur / step_dur, 3
      ) if step_dur > 0 else 0.0
      totals_out["non_compute_pct_of_compute"] = round(
          100.0 * non_compute_dur / compute_dur, 3
      ) if compute_dur > 0 else 0.0

      notes = list(ctx["notes"])
      if include_comm_stalls and any(
          r.kind == "comm" and r.hlo_category in ASYNC_DONE_CATEGORIES
          for r in records
      ):
          notes.append(_COMM_STALL_NOTE)

      return {
          "status":           "ok",
          "mode":             "non_compute",
          "profile_dir":      ctx["profile_dir"],
          "device":           ctx["device"],
          "step_id":          ctx["step_id"],
          "step_window_ps":   ctx["step_window_ps"],
          "step_duration_ps": step_dur,
          "notes":            notes,
          "totals":           totals_out,
          "by_category":      by_category,
          "by_source_within_category": by_source_within_category,
      }
  ```

  Note: `dataclasses.replace` requires `import dataclasses` at module top — confirm chunk-2 task 6 already added it. If not, add it now.

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py
  git commit -m "feat(tpu-perf): implement non_compute two-layer projection with async-done tagging"
  ```

### Task 22: Add `--no-comm-stalls` flag and wire `--mode non_compute` into `main()`

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py`

The flag is **default-on inclusion** of comm-stalls; the user passes `--no-comm-stalls` to flip it OFF. Implemented as `--no-comm-stalls` with `action="store_false"` and `dest="include_comm_stalls"` so `args.include_comm_stalls` defaults to `True`.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_non_compute_mode.py`:

  ```python
  class TestNonComputeCLIWiring(unittest.TestCase):
      def test_default_includes_comm_stalls(self):
          ns = cb.build_parser().parse_args(["/x", "--mode", "non_compute"])
          self.assertTrue(ns.include_comm_stalls)

      def test_no_comm_stalls_flag_flips_default(self):
          ns = cb.build_parser().parse_args(
              ["/x", "--mode", "non_compute", "--no-comm-stalls"]
          )
          self.assertFalse(ns.include_comm_stalls)


  class TestNonComputeEndToEnd(unittest.TestCase):
      def test_emits_valid_json_with_both_layers(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big")
          add_hlo_event(xs, em_id=11,
                         hlo_op_text="%c.0 = f32[8]{0} convert(bf16[8]{0} %x)",
                         offset_ps=600, duration_ps=5_000_000,
                         hlo_category="data formatting", tf_op="jit/Cast")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              r = subprocess.run(
                  [sys.executable, str(SCRIPT), tmp, "--mode", "non_compute"],
                  capture_output=True, text=True,
              )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["status"], "ok")
          self.assertEqual(doc["mode"], "non_compute")
          self.assertEqual(len(doc["by_category"]), 1)
          self.assertEqual(doc["by_category"][0]["hlo_category"], "data formatting")
          self.assertEqual(len(doc["by_source_within_category"]), 1)
          self.assertTrue(doc["by_source_within_category"][0]["dtype_change"])


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: `AttributeError: 'Namespace' object has no attribute 'include_comm_stalls'`.

- [ ] **Step 3: Edit `build_parser` and `main()`**

  In `build_parser()`, add (alongside existing flags):

  ```python
  parser.add_argument(
      "--no-comm-stalls", dest="include_comm_stalls",
      action="store_false", default=True,
      help="(non_compute mode) exclude async-done events; default is to include them as 'async-done (comm stall)'",
  )
  ```

  In `main()`, add the dispatch branch after `by_source`:

  ```python
      if args.mode == "non_compute":
          _emit(_run_non_compute_mode(records, ctx=ctx,
                                         include_comm=args.include_comm,
                                         include_comm_stalls=args.include_comm_stalls))
          return 0
  ```

- [ ] **Step 4: Run; all chunk tests pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: every test in test_pipeline, test_summary_mode, test_by_source_mode, test_non_compute_mode passes.

- [ ] **Step 5: Smoke-test against real fixture**

  Run:
  ```bash
  python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
    --mode non_compute | python3 -m json.tool > /dev/null && echo OK
  ```
  Expected: `OK`. Optional sanity: pipe through `jq '.totals.non_compute_pct_of_step'` — should be a small positive number.

- [ ] **Step 6: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_non_compute_mode.py
  git commit -m "feat(tpu-perf): wire --mode non_compute into main(); add --no-comm-stalls"
  ```

---

## Chunk 6: Mode 4 — `roofline` (capability 4) + `_peaks.py`

This chunk adds the v7x peak table module and the dtype-aware roofline analysis. Three pieces:
1. `_peaks.py` — sibling module with the v7x builtin table and CLI-override resolver.
2. `_run_roofline_mode` — eligibility filter (per spec §8.2), per-group MFU/HBM_util/bound formulas, weighted_avg aggregates, top-shortfall ranking, `skipped_groups` counters.
3. CLI flag wiring: `--chip`, `--peak-tflops-bf16`, `--peak-tflops-fp8`, `--peak-tflops-fp32`, `--peak-tflops-fp16`, `--peak-hbm-gibps`.

**Files this chunk creates or modifies:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/_peaks.py`
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_roofline_mode.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py` (add `_run_roofline_mode`, extend `build_parser`, extend `main()` dispatch)

After this chunk, `python3 compute_breakdown.py <fixture> --mode roofline` returns a real JSON document with v7x-specific MFU/HBM_util numbers and shortfall analysis.

### Task 23: Create `_peaks.py` with v7x builtin table

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/_peaks.py`
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_peaks.py`

`_peaks.py` exposes:
- `BUILTIN_PEAKS`: dict keyed by chip name, valued by per-dtype TFLOPS / HBM GiB/s.
- `resolve_peaks(chip, *, override_*)`: returns `dict` matching `peaks_used` block of mode 4 output. Resolution: builtin → CLI override (any provided override wins). `source` is `"builtin v7x table"` when no overrides applied; `"cli override"` when at least one override is set.

Per spec §8.1: per-device = per-chip / 2 (v7x chip has 2 TensorCores; `/device:TPU:N` is one TensorCore).

| Spec | per-chip v7x | per-device v7x (÷2) |
|---|---:|---:|
| Peak BF16 (TFLOPS) | 2307 | 1153.5 |
| Peak FP8  (TFLOPS) | 4614 | 2307.0 |
| HBM bandwidth (GiB/s) | 7380 | 3690 |

`fp32` and `fp16` peaks are not officially listed → table values are `None`.

- [ ] **Step 1: Write the failing tests**

  Create `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_peaks.py`:

  ```python
  """v7x peak table and resolver."""
  import pathlib
  import sys
  import unittest

  HERE = pathlib.Path(__file__).resolve().parent
  sys.path.insert(0, str(HERE.parent))
  import _peaks  # noqa: E402


  class TestBuiltinPeaks(unittest.TestCase):
      def test_v7x_per_device_values(self):
          p = _peaks.BUILTIN_PEAKS["v7x"]
          self.assertEqual(p["peak_tflops_bf16"], 1153.5)
          self.assertEqual(p["peak_tflops_fp8"], 2307.0)
          self.assertEqual(p["peak_hbm_gibps"], 3690.0)
          self.assertIsNone(p["peak_tflops_fp32"])
          self.assertIsNone(p["peak_tflops_fp16"])


  class TestResolvePeaks(unittest.TestCase):
      def test_no_overrides_returns_builtin_with_source_tag(self):
          p = _peaks.resolve_peaks("v7x")
          self.assertEqual(p["peak_tflops_bf16"], 1153.5)
          self.assertEqual(p["source"], "builtin v7x table")
          self.assertEqual(p["unit"], "GiB/s (base-1024) per device")
          # ridge_points only present for dtypes whose peak is known.
          self.assertIn("bf16", p["ridge_points"])
          self.assertIn("fp8", p["ridge_points"])
          self.assertNotIn("fp32", p["ridge_points"])

      def test_ridge_point_formula(self):
          p = _peaks.resolve_peaks("v7x")
          # ridge_point = (peak_tflops * 1e12) / (peak_hbm_gibps * 2**30)
          import math
          expected = (1153.5 * 1e12) / (3690.0 * (1024 ** 3))
          self.assertAlmostEqual(p["ridge_points"]["bf16"], expected, places=2)

      def test_overrides_set_source_to_cli_override(self):
          p = _peaks.resolve_peaks("v7x",
                                     override_tflops_bf16=2000.0)
          self.assertEqual(p["peak_tflops_bf16"], 2000.0)
          self.assertEqual(p["source"], "cli override")
          # Other peaks come from builtin still.
          self.assertEqual(p["peak_tflops_fp8"], 2307.0)

      def test_override_fills_null_dtype(self):
          # fp32 has null builtin peak; CLI can fill it in.
          p = _peaks.resolve_peaks("v7x",
                                     override_tflops_fp32=500.0)
          self.assertEqual(p["peak_tflops_fp32"], 500.0)
          self.assertIn("fp32", p["ridge_points"])
          self.assertEqual(p["source"], "cli override")

      def test_unknown_chip_raises(self):
          with self.assertRaises(KeyError):
              _peaks.resolve_peaks("unknown-chip-x")


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run; expect failure**

  Run:
  ```bash
  python3 -m unittest -v \
    plugins.tpu-perf.skills.compute-breakdown.scripts.tests.test_peaks
  ```
  Expected: `ModuleNotFoundError: No module named '_peaks'`.

- [ ] **Step 3: Implement `_peaks.py`**

  Create `plugins/tpu-perf/skills/compute-breakdown/scripts/_peaks.py`:

  ```python
  """v7x TPU peak table and CLI override resolver.

  Per-device == per-TensorCore. v7x chip has 2 TensorCores, so per-device
  values are per-chip / 2. /device:TPU:N is one TensorCore.

  Source: https://docs.cloud.google.com/tpu/docs/tpu7x

  Unit discipline:
    - TFLOPS uses base-10 (1 TFLOPS = 10^12 FLOPS).
    - HBM bandwidth uses base-1024 (1 GiB = 2^30 bytes).
    - The two scaling factors do NOT cancel in formulas like
      ridge_point = (peak_tflops * 1e12) / (peak_hbm_gibps * 2**30).
  """

  BUILTIN_PEAKS = {
      "v7x": {
          # Per device (= per TensorCore, = per chip / 2).
          "peak_tflops_bf16": 1153.5,   # per-chip 2307
          "peak_tflops_fp8":  2307.0,   # per-chip 4614
          "peak_tflops_fp32": None,     # not officially listed
          "peak_tflops_fp16": None,     # not officially listed
          "peak_hbm_gibps":   3690.0,   # per-chip 7380
      },
  }


  def _ridge_point(peak_tflops, peak_hbm_gibps):
      """FLOPs/byte: where compute and memory roofs meet."""
      if peak_tflops is None or peak_hbm_gibps is None:
          return None
      return (peak_tflops * 1e12) / (peak_hbm_gibps * (1024 ** 3))


  def resolve_peaks(chip: str, *,
                      override_tflops_bf16: float | None = None,
                      override_tflops_fp8:  float | None = None,
                      override_tflops_fp32: float | None = None,
                      override_tflops_fp16: float | None = None,
                      override_hbm_gibps:   float | None = None) -> dict:
      """Build the peaks_used block for mode 4 output.

      Resolution: start from BUILTIN_PEAKS[chip]; any non-None override
      replaces the builtin value. If at least one override applied, the
      returned source is 'cli override'; otherwise 'builtin v7x table'.
      """
      base = dict(BUILTIN_PEAKS[chip])  # KeyError on unknown chip.
      overrides = {
          "peak_tflops_bf16": override_tflops_bf16,
          "peak_tflops_fp8":  override_tflops_fp8,
          "peak_tflops_fp32": override_tflops_fp32,
          "peak_tflops_fp16": override_tflops_fp16,
          "peak_hbm_gibps":   override_hbm_gibps,
      }
      any_override = any(v is not None for v in overrides.values())
      for k, v in overrides.items():
          if v is not None:
              base[k] = v

      hbm = base["peak_hbm_gibps"]
      ridge = {}
      for dt in ("bf16", "fp8", "fp32", "fp16"):
          rp = _ridge_point(base[f"peak_tflops_{dt}"], hbm)
          if rp is not None:
              ridge[dt] = round(rp, 2)

      base["unit"] = "GiB/s (base-1024) per device"
      base["ridge_points"] = ridge
      base["source"] = "cli override" if any_override else "builtin v7x table"
      return base
  ```

- [ ] **Step 4: Run; tests pass**

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/_peaks.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_peaks.py
  git commit -m "feat(tpu-perf): add _peaks.py with v7x builtin table and CLI resolver"
  ```

### Task 24: `_run_roofline_mode` projection

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_roofline_mode.py`

The projection runs on `kind=compute` records only by default (data_move excluded; flops typically 0). For each group output by `_aggregate_by_key`:
1. **Eligibility check** (spec §8.2): all of `flops_sum > 0`, `bytes_accessed_sum > 0`, dtype ∈ {bf16, fp8, fp16, fp32}, peak for that dtype is non-null. Failure routes the group to `skipped_groups` with the matching counter.
2. For eligible groups, compute `t_compute_theory_ps`, `t_hbm_theory_ps`, `t_roofline_theory_ps`, `arithmetic_intensity`, `bound`, `mfu`, `hbm_util`, `roofline_util`, `shortfall_ps`, `shortfall_pct`.
3. **Group dtype** for roofline purposes: the dtype of the group is taken as the `dtype` field on the *first* record (already deduped on the group via Stage 3). If all events in the group disagree (`dtype_uncertain=True`) the dtype field is the first observed one but `dtype_uncertain=True` propagates to the output.
4. **Step summary**: `weighted_avg_*` use `total_dur_ps` as weights; `step_compute_duration_ps` is the sum of `total_dur_ps` over **eligible** groups (this is the spec §11 cross-mode invariant: matches `summary.totals.compute_duration_ps` only when no groups are skipped — the test below covers this corner).
5. **`top_shortfall_groups`**: top 10 by `shortfall_ps` desc (carries `agg_key` + 5 fields per spec §8.4).

- [ ] **Step 1: Write the failing tests**

  Create `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_roofline_mode.py`:

  ```python
  """Mode 4 (roofline) projection."""
  import json
  import pathlib
  import subprocess
  import sys
  import tempfile
  import unittest

  HERE = pathlib.Path(__file__).resolve().parent
  sys.path.insert(0, str(HERE.parent))
  import compute_breakdown as cb  # noqa: E402
  import _peaks  # noqa: E402

  from test_pipeline import (  # noqa: E402
      _make_record, make_minimal_xspace, add_hlo_event,
  )
  from test_summary_mode import SCRIPT  # noqa: E402


  def _ctx(**overrides):
      base = {
          "status": "ok",
          "step_id": 1,
          "step_window_ps": [0, 1_000_000],
          "step_duration_ps": 1_000_000,
          "notes": [],
          "pipeline_stats": cb._PipelineStats(),
          "profile_dir": "/x", "device": "/device:TPU:0",
          "xspace_pb_path": "/x/p.xplane.pb",
      }
      base.update(overrides)
      return base


  def _peaks_v7x():
      return _peaks.resolve_peaks("v7x")


  class TestRoofineEligibility(unittest.TestCase):
      def test_group_with_no_flops_skipped(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=None, bytes_accessed=1024, dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertEqual(doc["groups"], [])
          self.assertEqual(doc["skipped_groups"]["n_no_flops"], 1)

      def test_group_with_zero_flops_skipped(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=0, bytes_accessed=1024, dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertEqual(doc["skipped_groups"]["n_no_flops"], 1)

      def test_group_with_no_bytes_skipped(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=1_000_000, bytes_accessed=None, dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertEqual(doc["skipped_groups"]["n_no_bytes"], 1)

      def test_group_with_dtype_other_skipped(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=1_000_000, bytes_accessed=1024,
                                dtype="other")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertEqual(doc["skipped_groups"]["n_dtype_other"], 1)

      def test_group_with_null_dtype_skipped(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=1_000_000, bytes_accessed=1024, dtype=None)]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertEqual(doc["skipped_groups"]["n_dtype_other"], 1)

      def test_group_with_fp32_no_override_skipped(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=1_000_000, bytes_accessed=1024, dtype="fp32")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertEqual(doc["skipped_groups"]["n_peak_unknown_for_dtype"], 1)

      def test_group_with_fp32_override_eligible(self):
          recs = [_make_record(agg_key="A", kind="compute", duration_ps=10,
                                flops=1_000_000, bytes_accessed=1024, dtype="fp32")]
          peaks = _peaks.resolve_peaks("v7x", override_tflops_fp32=500.0)
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=peaks)
          self.assertEqual(len(doc["groups"]), 1)


  class TestRooflineFormulas(unittest.TestCase):
      def test_bf16_compute_bound_group(self):
          # High arithmetic intensity (FLOPS/byte) -> compute-bound.
          # 1e15 FLOPS / 1024 bytes ≈ 9.77e11 FLOPs/byte >> ridge_point ~320.
          recs = [_make_record(agg_key="A", kind="compute",
                                duration_ps=1_000_000_000,
                                flops=10**15, bytes_accessed=1024,
                                dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          g = doc["groups"][0]
          self.assertEqual(g["bound"], "compute")
          # t_compute_theory_ps = (1e15 / (1153.5 * 1e12)) * 1e12 picoseconds
          #   = 1e15 / 1153.5 ≈ 8.6699e11 ps ≈ 0.867 sec
          # We just check sign and ordering constraints.
          self.assertGreater(g["t_compute_theory_ps"], g["t_hbm_theory_ps"])
          self.assertEqual(g["t_roofline_theory_ps"], g["t_compute_theory_ps"])
          # mfu = t_compute / total_dur. Both around 0.867 sec ≈ 8.67e11 ps,
          # vs total_dur_ps = 1e9. So mfu can exceed 1 in this artificial test
          # — that's fine; sanity bound is mfu <= 1.05 for real fixtures only.
          self.assertGreater(g["mfu"], 0)
          self.assertGreater(g["roofline_util"], 0)

      def test_bf16_memory_bound_group(self):
          # Low arithmetic intensity -> memory-bound.
          # 1e6 FLOPS / 1e9 bytes = 1e-3 FLOPs/byte << ridge_point.
          recs = [_make_record(agg_key="A", kind="compute",
                                duration_ps=1_000_000_000,
                                flops=10**6, bytes_accessed=10**9,
                                dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          g = doc["groups"][0]
          self.assertEqual(g["bound"], "memory")
          self.assertGreater(g["t_hbm_theory_ps"], g["t_compute_theory_ps"])
          self.assertEqual(g["t_roofline_theory_ps"], g["t_hbm_theory_ps"])

      def test_arithmetic_intensity_value(self):
          recs = [_make_record(agg_key="A", kind="compute",
                                duration_ps=1_000_000_000,
                                flops=2048, bytes_accessed=1024,
                                dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          g = doc["groups"][0]
          self.assertAlmostEqual(g["arithmetic_intensity"], 2.0)

      def test_shortfall_nonneg_for_realistic_inputs(self):
          # Realistic: actual time is much larger than theoretical roofline.
          recs = [_make_record(agg_key="A", kind="compute",
                                duration_ps=1_000_000_000,  # 1 ms
                                flops=10**9, bytes_accessed=10**6,
                                dtype="bf16")]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          g = doc["groups"][0]
          self.assertGreaterEqual(g["shortfall_ps"], 0)


  class TestRooflineDtypeUncertain(unittest.TestCase):
      def test_dtype_uncertain_propagated(self):
          recs = [_make_record(agg_key="A", kind="compute",
                                duration_ps=1_000_000_000,
                                flops=10**9, bytes_accessed=10**6,
                                dtype="bf16", dtype_uncertain=True)]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          self.assertTrue(doc["groups"][0]["dtype_uncertain"])


  class TestRooflineStepSummary(unittest.TestCase):
      def test_top_shortfall_top_10_sorted_desc(self):
          recs = [
              _make_record(agg_key=f"K{i}", kind="compute",
                           duration_ps=10**9 * (i + 1),
                           flops=10**6, bytes_accessed=10**6, dtype="bf16")
              for i in range(15)
          ]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          top = doc["step_summary"]["top_shortfall_groups"]
          self.assertEqual(len(top), 10)
          shortfalls = [t["shortfall_ps"] for t in top]
          self.assertEqual(shortfalls, sorted(shortfalls, reverse=True))

      def test_weighted_avg_uses_total_dur_ps_weights(self):
          # Two groups: A small dur+small FLOPS, B large dur+large FLOPS.
          recs = [
              _make_record(agg_key="A", kind="compute", duration_ps=100,
                           flops=10**6, bytes_accessed=10**6, dtype="bf16"),
              _make_record(agg_key="B", kind="compute", duration_ps=900,
                           flops=10**11, bytes_accessed=10**6, dtype="bf16"),
          ]
          doc = cb._run_roofline_mode(recs, ctx=_ctx(), peaks=_peaks_v7x())
          # B dominates the weighted average because its duration is 9x A.
          # MFU(B) is much higher; weighted_avg_mfu should be closer to MFU(B).
          mfus = {g["agg_key"]: g["mfu"] for g in doc["groups"]}
          weighted = doc["step_summary"]["weighted_avg_mfu"]
          self.assertGreater(weighted, (mfus["A"] + mfus["B"]) / 2)


  class TestRooflineEndToEnd(unittest.TestCase):
      def test_emits_valid_json_with_peaks_used(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10,
                         hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big",
                         flops=10**12, bytes_accessed=10**8,
                         shape_with_layout="bf16[8]{0}")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              r = subprocess.run(
                  [sys.executable, str(SCRIPT), tmp, "--mode", "roofline"],
                  capture_output=True, text=True,
              )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["status"], "ok")
          self.assertEqual(doc["mode"], "roofline")
          self.assertEqual(doc["chip"], "v7x")
          self.assertEqual(doc["peaks_used"]["peak_tflops_bf16"], 1153.5)
          self.assertEqual(doc["peaks_used"]["peak_hbm_gibps"], 3690.0)
          self.assertEqual(doc["peaks_used"]["source"], "builtin v7x table")
          self.assertGreaterEqual(len(doc["groups"]), 1)

      def test_cli_override_changes_peaks_used_source(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10,
                         hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big",
                         flops=10**12, bytes_accessed=10**8,
                         shape_with_layout="bf16[8]{0}")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              r = subprocess.run(
                  [sys.executable, str(SCRIPT), tmp, "--mode", "roofline",
                   "--peak-tflops-bf16", "999.0"],
                  capture_output=True, text=True,
              )
          self.assertEqual(r.returncode, 0, r.stderr)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["peaks_used"]["peak_tflops_bf16"], 999.0)
          self.assertEqual(doc["peaks_used"]["source"], "cli override")


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: `AttributeError: module 'compute_breakdown' has no attribute '_run_roofline_mode'`.

- [ ] **Step 3: Implement `_run_roofline_mode`**

  Insert into `compute_breakdown.py` after `_run_non_compute_mode`, and add `import _peaks` near the existing imports:

  ```python
  def _run_roofline_mode(records: list[EventRecord], *, ctx: dict,
                            peaks: dict) -> dict:
      """Mode 4: dtype-aware roofline analysis on the resolved peaks table."""
      step_dur = ctx["step_duration_ps"]
      pstats = ctx["pipeline_stats"]

      # Compute mode runs on kind=compute only; data_move flops typically 0.
      compute_recs = [r for r in records if r.kind == "compute"]
      groups = _aggregate_by_key(compute_recs, dedupe_shapes_cap=8)

      eligible_rows = []
      skipped = {
          "n_no_flops":               0,
          "n_no_bytes":               0,
          "n_dtype_other":            0,
          "n_peak_unknown_for_dtype": 0,
          "total_dur_ps_skipped":     0,
      }

      for g in groups.values():
          # Group dtype: per spec §8.2, "the dtype field on the first
          # record". `_GroupAgg.first_dtype` is set on first insert in
          # `_aggregate_by_key` (chunk-3 task 13) and never overwritten.
          dt = g.first_dtype

          if g.flops_sum is None or g.flops_sum <= 0:
              skipped["n_no_flops"] += 1
              skipped["total_dur_ps_skipped"] += g.total_dur_ps
              continue
          if g.bytes_accessed_sum is None or g.bytes_accessed_sum <= 0:
              skipped["n_no_bytes"] += 1
              skipped["total_dur_ps_skipped"] += g.total_dur_ps
              continue
          if dt not in ("bf16", "fp8", "fp16", "fp32"):
              skipped["n_dtype_other"] += 1
              skipped["total_dur_ps_skipped"] += g.total_dur_ps
              continue
          peak_tflops = peaks.get(f"peak_tflops_{dt}")
          peak_hbm = peaks.get("peak_hbm_gibps")
          if peak_tflops is None or peak_hbm is None:
              skipped["n_peak_unknown_for_dtype"] += 1
              skipped["total_dur_ps_skipped"] += g.total_dur_ps
              continue

          # Two-step decomposition (avoid combined-constant errors).
          t_compute_seconds = g.flops_sum / (peak_tflops * 1e12)
          t_compute_theory_ps = t_compute_seconds * 1e12
          t_hbm_seconds = g.bytes_accessed_sum / (peak_hbm * (1024 ** 3))
          t_hbm_theory_ps = t_hbm_seconds * 1e12
          t_roofline_theory_ps = max(t_compute_theory_ps, t_hbm_theory_ps)

          arithmetic_intensity = g.flops_sum / g.bytes_accessed_sum
          ridge_point = (peak_tflops * 1e12) / (peak_hbm * (1024 ** 3))
          bound = "compute" if arithmetic_intensity >= ridge_point else "memory"

          mfu = t_compute_theory_ps / g.total_dur_ps if g.total_dur_ps > 0 else 0.0
          hbm_util = t_hbm_theory_ps / g.total_dur_ps if g.total_dur_ps > 0 else 0.0
          roofline_util = (t_roofline_theory_ps / g.total_dur_ps
                              if g.total_dur_ps > 0 else 0.0)
          shortfall_ps = g.total_dur_ps - t_roofline_theory_ps
          shortfall_pct = (1 - roofline_util) * 100

          eligible_rows.append({
              "agg_key":              g.agg_key,
              "agg_key_kind":         g.agg_key_kind,
              "source_inner":         g.source_inner,
              "tf_op":                g.tf_op,
              "hlo_categories":       dict(g.hlo_categories),
              "n_executions":         g.n_executions,
              "total_dur_ps":         g.total_dur_ps,
              "flops_sum":            g.flops_sum,
              "bytes_accessed_sum":   g.bytes_accessed_sum,
              "dtype":                dt,
              "dtype_uncertain":      g.dtype_uncertain,
              "arithmetic_intensity": round(arithmetic_intensity, 4),
              "ridge_point":          round(ridge_point, 2),
              "bound":                bound,
              "t_compute_theory_ps":  int(t_compute_theory_ps),
              "t_hbm_theory_ps":      int(t_hbm_theory_ps),
              "t_roofline_theory_ps": int(t_roofline_theory_ps),
              "mfu":                  round(mfu, 4),
              "hbm_util":             round(hbm_util, 4),
              "roofline_util":        round(roofline_util, 4),
              "shortfall_ps":         int(shortfall_ps),
              "shortfall_pct":        round(shortfall_pct, 2),
          })

      # Step summary: weighted averages by total_dur_ps.
      total_eligible_dur = sum(r["total_dur_ps"] for r in eligible_rows)
      def _weighted(field: str) -> float:
          if total_eligible_dur <= 0:
              return 0.0
          return sum(r[field] * r["total_dur_ps"] for r in eligible_rows) / total_eligible_dur

      top_shortfall = sorted(eligible_rows,
                                key=lambda r: r["shortfall_ps"], reverse=True)[:10]
      top_shortfall_short = [
          {
              "agg_key":      r["agg_key"],
              "source_inner": r["source_inner"],
              "tf_op":        r["tf_op"],
              "total_dur_ps": r["total_dur_ps"],
              "shortfall_ps": r["shortfall_ps"],
              "bound":        r["bound"],
          }
          for r in top_shortfall
      ]

      step_summary = {
          "step_compute_duration_ps":   total_eligible_dur,
          "weighted_avg_mfu":           round(_weighted("mfu"), 4),
          "weighted_avg_hbm_util":      round(_weighted("hbm_util"), 4),
          "weighted_avg_roofline_util": round(_weighted("roofline_util"), 4),
          "step_shortfall_ps":          int(sum(r["shortfall_ps"] for r in eligible_rows)),
          "top_shortfall_groups":       top_shortfall_short,
      }

      return {
          "status":           "ok",
          "mode":             "roofline",
          "profile_dir":      ctx["profile_dir"],
          "device":           ctx["device"],
          "step_id":          ctx["step_id"],
          "step_window_ps":   ctx["step_window_ps"],
          "step_duration_ps": step_dur,
          "notes":            list(ctx["notes"]),
          "chip":             ctx.get("chip", "v7x"),
          "peaks_used":       peaks,
          "step_summary":     step_summary,
          "groups":           eligible_rows,
          "skipped_groups":   skipped,
      }
  ```

- [ ] **Step 4: Run; tests pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: every test in test_pipeline, test_summary_mode, test_by_source_mode, test_non_compute_mode, test_peaks, test_roofline_mode passes.

- [ ] **Step 5: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_roofline_mode.py
  git commit -m "feat(tpu-perf): implement roofline projection with v7x peaks"
  ```

### Task 25: Add roofline CLI flags and wire `--mode roofline` into `main()`

**Files:**
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py`
- Modify: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_roofline_mode.py`

Flags: `--chip` (default `"v7x"`), `--peak-tflops-bf16`, `--peak-tflops-fp8`, `--peak-tflops-fp32`, `--peak-tflops-fp16`, `--peak-hbm-gibps` (all default `None`). Each is global at argparse layer; they are no-ops for modes other than `roofline`.

- [ ] **Step 1: Write the failing tests**

  Append to `tests/test_roofline_mode.py`:

  ```python
  class TestRooflineCLIWiring(unittest.TestCase):
      def test_chip_default_is_v7x(self):
          ns = cb.build_parser().parse_args(["/x", "--mode", "roofline"])
          self.assertEqual(ns.chip, "v7x")

      def test_peak_overrides_default_none(self):
          ns = cb.build_parser().parse_args(["/x", "--mode", "roofline"])
          self.assertIsNone(ns.peak_tflops_bf16)
          self.assertIsNone(ns.peak_tflops_fp8)
          self.assertIsNone(ns.peak_tflops_fp32)
          self.assertIsNone(ns.peak_tflops_fp16)
          self.assertIsNone(ns.peak_hbm_gibps)

      def test_peak_override_parses_float(self):
          ns = cb.build_parser().parse_args(
              ["/x", "--mode", "roofline",
               "--peak-tflops-bf16", "1500.0",
               "--peak-hbm-gibps", "4000.0"]
          )
          self.assertEqual(ns.peak_tflops_bf16, 1500.0)
          self.assertEqual(ns.peak_hbm_gibps, 4000.0)
  ```

- [ ] **Step 2: Run; expect failure**

  Expected: `AttributeError: 'Namespace' object has no attribute 'chip'`.

- [ ] **Step 3: Edit `build_parser` and `main()`**

  In `build_parser()`, add (alongside existing flags):

  ```python
  parser.add_argument("--chip", default="v7x",
                       help="(roofline mode) TPU chip name. Default: v7x")
  parser.add_argument("--peak-tflops-bf16", type=float, default=None,
                       help="(roofline mode) override BF16 peak TFLOPS per device")
  parser.add_argument("--peak-tflops-fp8", type=float, default=None,
                       help="(roofline mode) override FP8 peak TFLOPS per device")
  parser.add_argument("--peak-tflops-fp32", type=float, default=None,
                       help="(roofline mode) override FP32 peak TFLOPS per device")
  parser.add_argument("--peak-tflops-fp16", type=float, default=None,
                       help="(roofline mode) override FP16 peak TFLOPS per device")
  parser.add_argument("--peak-hbm-gibps", type=float, default=None,
                       help="(roofline mode) override HBM bandwidth GiB/s per device")
  ```

  In `main()`, add the dispatch branch after `non_compute`:

  ```python
      if args.mode == "roofline":
          peaks = _peaks.resolve_peaks(
              args.chip,
              override_tflops_bf16=args.peak_tflops_bf16,
              override_tflops_fp8=args.peak_tflops_fp8,
              override_tflops_fp32=args.peak_tflops_fp32,
              override_tflops_fp16=args.peak_tflops_fp16,
              override_hbm_gibps=args.peak_hbm_gibps,
          )
          ctx_with_chip = dict(ctx)
          ctx_with_chip["chip"] = args.chip
          _emit(_run_roofline_mode(records, ctx=ctx_with_chip, peaks=peaks))
          return 0
  ```

  Add `import _peaks` to the imports near the top of the file (alongside `import xplane_pb2`).

- [ ] **Step 4: Run; all chunk tests pass**

- [ ] **Step 5: Smoke-test against real fixture**

  Run:
  ```bash
  python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
    --mode roofline | python3 -m json.tool > /dev/null && echo OK
  ```
  Expected: `OK`.

  Spot-check the v7x peak fidelity:
  ```bash
  python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
    --mode roofline | python3 -c "
  import json,sys
  d = json.load(sys.stdin)
  assert d['peaks_used']['peak_tflops_bf16'] == 1153.5
  assert d['peaks_used']['peak_hbm_gibps'] == 3690.0
  print('OK')"
  ```
  Expected: `OK`.

- [ ] **Step 6: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
          plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_roofline_mode.py
  git commit -m "feat(tpu-perf): wire --mode roofline into main(); CLI override flags"
  ```

---

## Chunk 7: SKILL.md, cross-mode e2e tests, and final verification

This chunk completes the skill:
1. Writes the SKILL.md per spec §9 — concept primer, 4-mode usage, layer-scoping recipe, gotchas.
2. Adds a `test_e2e.py` harness that exercises the spec §11 cross-mode invariants against a single synthetic XSpace.
3. Bumps the marketplace registration so the skill is discoverable.
4. Final verification: real-fixture smoke for all 4 modes, error-path checks, sanity-bound checks.

**Files this chunk creates or modifies:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/SKILL.md`
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_e2e.py`
- Modify: `.claude-plugin/marketplace.json` (only if the skill needs explicit listing — confirm during step 1)

### Task 26: Write `SKILL.md`

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/SKILL.md`

The SKILL.md is the agent-facing primer. It must NOT duplicate the protobuf schema (assume the reader has profile-anatomy available). Required sections (per spec §9):
1. Frontmatter (name, description, argument-hint)
2. 1-paragraph overview
3. "When to use which mode" decision table
4. "Concepts you need first" (agg_key, while-container handling, v7x peak per-device convention, copy-start/done DMA, dtype_uncertain meaning, GiB unit discipline)
5. Per-mode sections (1 each) with invocation + JSON shape pointer + reading guide
6. Layer-scoping recipe (mode 2)
7. Common gotchas (mirroring profile-anatomy's plus new)
8. Files manifest

- [ ] **Step 1: Write the SKILL.md**

  Create `plugins/tpu-perf/skills/compute-breakdown/SKILL.md`:

  ```markdown
  ---
  name: compute-breakdown
  description: Use when analyzing TPU pretraining compute efficiency from xplane.pb — produces source-line-aggregated HLO duration tables, layer-scoped breakdowns, non-compute (padding/cast/copy) audits, and v7x roofline shortfall vs theoretical peak. Reads schema documented by profile-anatomy.
  argument-hint: "<profile_dir> --mode {summary|by_source|non_compute|roofline} [--step N] [--top K]"
  ---

  # Compute Breakdown

  Analyze the compute portion of a TPU pretraining profile. One Python entry script with four `--mode` subcommands sharing a single load → step-pick → event-iterate → normalize pipeline. Always emits a single top-level JSON object on stdout (`status: ok | absent`), so output is consumed structurally — Claude reads the JSON, filters/sums client-side, and reports.

  This skill is built on top of `profile-anatomy`, which documents the XSpace/XPlane/XLine/XEvent/XStat hierarchy. Read that first if you need to know what an XEvent is, where source_stack lives, or how `XEventMetadata.stats` differs from `XEvent.stats`.

  ## When to use which mode

  | Question | Mode |
  |---|---|
  | "Top time-eaters in this profile" | `summary` |
  | "How much time does X layer / module spend" | `by_source`, then filter |
  | "How much time goes to padding/cast/copy/transpose" | `non_compute` |
  | "Are we compute- or memory-bound; what's MFU on v7x" | `roofline` |

  ## Concepts you need first

  - **`agg_key`**: groups events by source location with a 3-tier fallback. Tier 1: SHA-256 hash of `source_stack` (`stack:<8-hex>`). Tier 2: `tf_op` string (`tfop:<value>`). Tier 3: `<no source>:<hlo_category>`. The group's `agg_key_kind` field reports which tier was used.
  - **`while` HLO is a container**, not a leaf op. Its events are excluded from per-event tables; their total duration is reported separately as `while_container_duration_ps`. Do **not** double-count it against `compute_duration_ps`.
  - **TPU concurrency**: events on the device plane can overlap (Scalar Unit, vector core, async scheduler). Per-kind sums therefore can exceed wall-clock step duration. Treat the durations as throughput proxies, not exclusive time.
  - **v7x peaks are per-device** (= per-TensorCore). The v7x chip has 2 TensorCores; `/device:TPU:N` is one of them. Per-chip values are divided by 2: BF16 peak 1153.5 TFLOPS/device, FP8 peak 2307.0 TFLOPS/device, HBM 3690 GiB/s/device.
  - **HBM bandwidth uses GiB/s (base-1024)**. Do not mix with GB (base-10). The roofline formulas keep TFLOPS (10^12) and GiB (2^30) separate; the constants do not cancel.
  - **`copy-start` / `copy-done` carry no source** — XLA-internal DMA, not user-code-driven. Real copy waste shows up in `data formatting` and `broadcast` categories.
  - **`dtype_uncertain=true`** flags a fusion whose inputs may differ in precision from the output (e.g. fp8 inputs, bf16 accumulation). Roofline still computes the per-group MFU using the dominant dtype; flag is propagated so Claude can present a caveat (true peak may be ~2× higher).

  ## Mode 1 — `summary`

  ```bash
  python3 .../compute_breakdown.py <profile_dir> --mode summary [--step N] [--top K] [--include-comm]
  ```

  Top-K compute groups by source line. JSON has `totals` (per-kind durations, while accounting, agg_key coverage), `top_compute_groups` (top K, sorted by total_dur_ps desc), `tail_compute` (rollup of the rest), `by_kind_rollup` (4-row table over compute / data_move / comm / other).

  Reading guide: walk `top_compute_groups` for the biggest time-eaters; check `tail_compute.dur_ps` against the top-K sum to see how concentrated the workload is; check `unknown_categories` and `n_events_unresolved` for spec-coverage gaps.

  ## Mode 2 — `by_source` (layer scoping)

  ```bash
  python3 .../compute_breakdown.py <profile_dir> --mode by_source [--step N] [--include-data-move]
  ```

  Full per-`agg_key` table — not sorted, not truncated. Each group carries its `source_stack`, `tf_op`, `kind`, `hlo_categories`, durations, sums (flops/model_flops/bytes_accessed), `shapes` (cap 8), `dtypes` histogram, `dtype_uncertain`, `example_hlo_op`.

  **Layer-scoping recipe** (the canonical use):
  1. Read the user's code (e.g. `attention.py`) — note the file path and function names.
  2. Run `--mode by_source`.
  3. In the JSON, filter `groups` where `source_stack` contains the file path OR `tf_op` contains the function name (be permissive — JAX adds wrapper prefixes).
  4. Sum `total_dur_ps` over the filtered set.
  5. Report: layer total / `step_duration_ps` (% of step), and layer total / `totals.compute_duration_ps` (% of compute).

  ## Mode 3 — `non_compute`

  ```bash
  python3 .../compute_breakdown.py <profile_dir> --mode non_compute [--step N] [--no-comm-stalls]
  ```

  Two-layer output:
  - `by_category`: one row per `hlo_category` (`data formatting`, `copy`, `convert`, `pad`, `broadcast`, …) with execution count, durations, group count, agg_key coverage.
  - `by_source_within_category`: full (category, agg_key) breakdown with `dtype_change` / `layout_change` (parsed from the HLO IR text), `shapes_in` / `shapes_out` (cap 4), `example_hlo_op`.

  **`dtype_change` / `layout_change` semantics:**
  - `true`: detected from the IR text (e.g. `f32[...] convert(bf16[...] ...)` — dtype changes from bf16 to f32).
  - `false`: detected, no change.
  - `null`: undetectable (HLO IR didn't include both layouts, or text wasn't parseable). **`null` is NOT "no change"** — it means we couldn't decide. Don't claim a layout change is absent when this field is `null`.

  By default `async-done` events are included as `hlo_category="async-done (comm stall)"` (with a `notes` entry telling Claude how to flip it off). Pass `--no-comm-stalls` to exclude them.

  ## Mode 4 — `roofline`

  ```bash
  python3 .../compute_breakdown.py <profile_dir> --mode roofline [--step N]
    [--chip v7x]
    [--peak-tflops-bf16 ...] [--peak-tflops-fp8 ...]
    [--peak-tflops-fp32 ...] [--peak-tflops-fp16 ...]
    [--peak-hbm-gibps ...]
  ```

  v7x peaks are built in (per-device: BF16=1153.5, FP8=2307.0, HBM=3690 GiB/s). FP32/FP16 peaks are not officially listed; pass `--peak-tflops-fp32 ...` to include those groups (otherwise they go to `skipped_groups.n_peak_unknown_for_dtype`).

  Per-group output: `arithmetic_intensity` (FLOPs/byte), `ridge_point` (where compute and memory roofs meet), `bound` ∈ `{compute, memory}`, `t_compute_theory_ps`, `t_hbm_theory_ps`, `t_roofline_theory_ps`, `mfu`, `hbm_util`, `roofline_util`, `shortfall_ps`, `shortfall_pct`.

  Step summary: `weighted_avg_mfu`, `weighted_avg_hbm_util`, `weighted_avg_roofline_util` (weighted by `total_dur_ps`); `top_shortfall_groups` (top 10 by absolute `shortfall_ps`).

  **Reading guide:**
  - High `weighted_avg_mfu` → workload is using compute; gains come from reducing wall-clock (kernel fusion, less padding) not from algorithmic changes.
  - High `weighted_avg_hbm_util` with low `weighted_avg_mfu` → memory-bound; gains come from raising arithmetic intensity (fusion to keep activations in SRAM, larger contraction dims, lower-precision activations).
  - Both low → other bottleneck (scheduling, dependencies, control flow). Look at `summary.totals.while_pct_of_step` and the `non_compute` audit.
  - When a group has `dtype_uncertain=true`, present both the bf16-peak MFU **and a note** that the true peak may be fp8 (~2× higher), making the MFU number an upper bound on under-utilization, not a definitive figure.

  ## Common gotchas

  - **`XEvent.stats` vs `XEventMetadata.stats`**: see profile-anatomy. Op-level fields (`flops`, `bytes_accessed`, `hlo_category`, `shape_with_layout`) live on `XEventMetadata.stats`, not `XEvent.stats`.
  - **`while` HLO is a container**: `while_container_duration_ps` is reported separately. Don't add it to `compute_duration_ps`.
  - **Concurrency caveat**: per-kind durations can sum > step duration. The field is named `non_while_duration_ps_sum` (not `total`) for this reason.
  - **`copy-start` / `copy-done` carry no source** — XLA-internal DMA. Real copy waste appears in `data formatting`.
  - **GiB vs GB**: HBM is GiB/s (base-1024). The peak table block tags `unit: "GiB/s (base-1024) per device"` to make this explicit.
  - **Cross-mode equality**: `summary.totals.compute_duration_ps == by_source.totals.compute_duration_ps` exactly. `summary.totals.data_move_duration_ps == non_compute.totals.data_move_duration_ps` only when mode 3 was invoked with `--no-comm-stalls`.

  ## Files

  - `scripts/compute_breakdown.py` — main entry script.
  - `scripts/_peaks.py` — v7x peak table and CLI override resolver.
  - `scripts/_proto/` — vendored xplane protobuf bindings (copy of profile-anatomy's `_proto/`).
  - `scripts/tests/` — unit + e2e tests (stdlib `unittest`).
  ```

- [ ] **Step 2: Verify SKILL.md frontmatter is valid YAML**

  Run:
  ```bash
  python3 -c "
  import re,sys,pathlib
  p = pathlib.Path('plugins/tpu-perf/skills/compute-breakdown/SKILL.md')
  text = p.read_text()
  m = re.match(r'---\n(.*?)\n---\n', text, re.DOTALL)
  assert m, 'no frontmatter found'
  import yaml  # if pyyaml isn't installed, fall back to manual check
  fm = yaml.safe_load(m.group(1))
  assert fm['name'] == 'compute-breakdown'
  assert 'description' in fm
  print('OK')
  " 2>/dev/null || python3 -c "
  import re,pathlib
  p = pathlib.Path('plugins/tpu-perf/skills/compute-breakdown/SKILL.md')
  text = p.read_text()
  m = re.match(r'---\n(.*?)\n---\n', text, re.DOTALL)
  assert m and 'name: compute-breakdown' in m.group(1)
  print('OK (no yaml lib; structural check only)')
  "
  ```
  Expected: `OK`.

- [ ] **Step 3: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/SKILL.md
  git commit -m "docs(tpu-perf): add SKILL.md for compute-breakdown skill"
  ```

### Task 27: Cross-mode invariants e2e test

**Files:**
- Create: `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_e2e.py`

This test materializes one synthetic XSpace, runs all 4 modes by subprocess (with `--no-comm-stalls` for mode 3), then asserts the spec §11 cross-mode equalities and sanity bounds.

- [ ] **Step 1: Write the failing test**

  Create `plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_e2e.py`:

  ```python
  """Cross-mode invariants e2e test (spec §11)."""
  import json
  import pathlib
  import subprocess
  import sys
  import tempfile
  import unittest

  HERE = pathlib.Path(__file__).resolve().parent
  sys.path.insert(0, str(HERE.parent))

  from test_pipeline import (  # noqa: E402
      make_minimal_xspace, add_hlo_event,
  )
  from test_summary_mode import SCRIPT  # noqa: E402


  def _run(profile_dir, *args):
      cmd = [sys.executable, str(SCRIPT), profile_dir] + list(args)
      r = subprocess.run(cmd, capture_output=True, text=True)
      assert r.returncode == 0, f"exit {r.returncode}: {r.stderr}"
      return json.loads(r.stdout)


  class TestCrossModeInvariants(unittest.TestCase):
      def setUp(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          # One bf16 fusion (compute), one transpose (data_move),
          # one all-reduce-done (comm; included as comm-stall by default).
          add_hlo_event(xs, em_id=10,
                         hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big",
                         flops=10**12, bytes_accessed=10**8,
                         shape_with_layout="bf16[8]{0}")
          add_hlo_event(xs, em_id=11,
                         hlo_op_text="%t.0 = bf16[8]{0} transpose(bf16[8]{0} %x)",
                         offset_ps=600, duration_ps=5_000_000,
                         hlo_category="data formatting", tf_op="jit/T")
          add_hlo_event(xs, em_id=12,
                         hlo_op_text="ar.0", offset_ps=700,
                         duration_ps=2_000_000,
                         hlo_category="all-reduce-done", tf_op="jit/AR")
          self.tmpdir_obj = tempfile.TemporaryDirectory()
          self.tmpdir = self.tmpdir_obj.name
          pathlib.Path(self.tmpdir, "x.xplane.pb").write_bytes(xs.SerializeToString())

      def tearDown(self):
          self.tmpdir_obj.cleanup()

      def test_summary_and_by_source_compute_duration_equal(self):
          d_sum = _run(self.tmpdir, "--mode", "summary")
          d_bs  = _run(self.tmpdir, "--mode", "by_source")
          self.assertEqual(
              d_sum["totals"]["compute_duration_ps"],
              d_bs["totals"]["compute_duration_ps"],
          )

      def test_summary_and_non_compute_data_move_duration_equal_with_no_comm_stalls(self):
          d_sum = _run(self.tmpdir, "--mode", "summary")
          d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
          self.assertEqual(
              d_sum["totals"]["data_move_duration_ps"],
              d_nc["totals"]["data_move_duration_ps"],
          )

      def test_other_duration_equal_across_modes(self):
          d_sum = _run(self.tmpdir, "--mode", "summary")
          d_bs  = _run(self.tmpdir, "--mode", "by_source")
          d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
          self.assertEqual(d_sum["totals"]["other_duration_ps"],
                            d_bs["totals"]["other_duration_ps"])
          self.assertEqual(d_sum["totals"]["other_duration_ps"],
                            d_nc["totals"]["other_duration_ps"])

      def test_n_events_unresolved_equal_across_modes(self):
          d_sum = _run(self.tmpdir, "--mode", "summary")
          d_bs  = _run(self.tmpdir, "--mode", "by_source")
          d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
          self.assertEqual(d_sum["totals"]["n_events_unresolved"],
                            d_bs["totals"]["n_events_unresolved"])
          self.assertEqual(d_sum["totals"]["n_events_unresolved"],
                            d_nc["totals"]["n_events_unresolved"])

      def test_unknown_categories_equal_across_modes(self):
          d_sum = _run(self.tmpdir, "--mode", "summary")
          d_bs  = _run(self.tmpdir, "--mode", "by_source")
          d_nc  = _run(self.tmpdir, "--mode", "non_compute", "--no-comm-stalls")
          self.assertEqual(d_sum["totals"]["unknown_categories"],
                            d_bs["totals"]["unknown_categories"])
          self.assertEqual(d_sum["totals"]["unknown_categories"],
                            d_nc["totals"]["unknown_categories"])

      def test_step_window_equal_across_all_four_modes(self):
          windows = [
              _run(self.tmpdir, "--mode", m)["step_window_ps"]
              for m in ("summary", "by_source", "non_compute", "roofline")
          ]
          self.assertEqual(windows[0], windows[1])
          self.assertEqual(windows[0], windows[2])
          self.assertEqual(windows[0], windows[3])

      def test_roofline_step_compute_equals_summary_compute_when_no_skips(self):
          # When all compute events are roofline-eligible, the two values match.
          d_sum = _run(self.tmpdir, "--mode", "summary")
          d_rl  = _run(self.tmpdir, "--mode", "roofline")
          if d_rl["skipped_groups"]["n_no_flops"] == 0 \
              and d_rl["skipped_groups"]["n_no_bytes"] == 0 \
              and d_rl["skipped_groups"]["n_dtype_other"] == 0 \
              and d_rl["skipped_groups"]["n_peak_unknown_for_dtype"] == 0:
              self.assertEqual(
                  d_rl["step_summary"]["step_compute_duration_ps"],
                  d_sum["totals"]["compute_duration_ps"],
              )


  class TestErrorPaths(unittest.TestCase):
      def test_absent_profile_dir_returns_status_absent(self):
          with tempfile.TemporaryDirectory() as empty:
              cmd = [sys.executable, str(SCRIPT), empty, "--mode", "summary"]
              r = subprocess.run(cmd, capture_output=True, text=True)
          self.assertEqual(r.returncode, 0)
          doc = json.loads(r.stdout)
          self.assertEqual(doc["status"], "absent")
          self.assertEqual(doc["reason"], "no_xplane_pb")

      def test_step_out_of_range_returns_exit_1(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          add_hlo_event(xs, em_id=10, hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=400_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              cmd = [sys.executable, str(SCRIPT), tmp,
                     "--mode", "summary", "--step", "999"]
              r = subprocess.run(cmd, capture_output=True, text=True)
          self.assertEqual(r.returncode, 1)
          self.assertIn("error", r.stderr.lower())


  class TestSanityBounds(unittest.TestCase):
      def test_mfu_and_hbm_util_bounded(self):
          xs = make_minimal_xspace(steps=[(1, 0, 1_000_000_000)])
          # Realistic-ish: 1 sec actual, well below theoretical roofline.
          add_hlo_event(xs, em_id=10,
                         hlo_op_text="big = bf16[8] fusion(...)",
                         offset_ps=100, duration_ps=1_000_000_000_000,
                         hlo_category="loop fusion", tf_op="jit/Big",
                         flops=10**9, bytes_accessed=10**6,
                         shape_with_layout="bf16[8]{0}")
          with tempfile.TemporaryDirectory() as tmp:
              pathlib.Path(tmp, "x.xplane.pb").write_bytes(xs.SerializeToString())
              doc = _run(tmp, "--mode", "roofline")
          for g in doc["groups"]:
              self.assertGreaterEqual(g["mfu"], 0.0)
              self.assertGreaterEqual(g["hbm_util"], 0.0)
              self.assertGreaterEqual(g["shortfall_ps"], 0)
          if doc["groups"]:
              self.assertGreaterEqual(
                  doc["step_summary"]["weighted_avg_roofline_util"], 0.0
              )


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run; expect pass**

  Run:
  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: every test passes (the cross-mode invariants are property assertions on already-implemented code).

- [ ] **Step 3: Commit**

  ```bash
  git add plugins/tpu-perf/skills/compute-breakdown/scripts/tests/test_e2e.py
  git commit -m "test(tpu-perf): add cross-mode invariants e2e test (spec §11)"
  ```

### Task 28: Confirm marketplace registration

**Files:**
- Read: `.claude-plugin/marketplace.json`
- Possibly modify: `.claude-plugin/marketplace.json`

The marketplace already lists `tpu-perf` (registered during the profile-anatomy plan). New skills under an existing plugin **may or may not** require explicit marketplace listing depending on the format used. Check first; modify only if needed.

- [ ] **Step 1: Read the marketplace registration**

  ```bash
  python3 -c "
  import json,pathlib
  m = json.loads(pathlib.Path('.claude-plugin/marketplace.json').read_text())
  for p in m['plugins']:
      if p.get('name') == 'tpu-perf':
          print(json.dumps(p, indent=2))
  "
  ```

  Inspect the output. If the entry lists individual skills under a `skills` array, add `compute-breakdown`. If it only points to the plugin directory (relying on `plugin.json` for discovery), no change is required.

- [ ] **Step 2: Update marketplace.json if needed**

  If a `skills` array exists for `tpu-perf`:
  ```python
  # Edit marketplace.json: add "compute-breakdown" to the skills list,
  # keeping JSON formatting consistent with surrounding entries.
  ```
  Otherwise: skip.

- [ ] **Step 3: Validate the JSON**

  ```bash
  python3 -c "import json,pathlib; json.loads(pathlib.Path('.claude-plugin/marketplace.json').read_text()); print('OK')"
  ```
  Expected: `OK`.

- [ ] **Step 4: Commit (only if marketplace.json changed)**

  ```bash
  git diff --quiet .claude-plugin/marketplace.json || (
    git add .claude-plugin/marketplace.json &&
    git commit -m "chore(tpu-perf): register compute-breakdown skill in marketplace"
  )
  ```

### Task 29: Final verification

**Files:**
- Run all tests
- Smoke-test all 4 modes against the real fixture
- Confirm the `plugin.json` version bump from chunk 1 is still in place

- [ ] **Step 1: Run the full test suite**

  ```bash
  python3 -m unittest discover \
    -s plugins/tpu-perf/skills/compute-breakdown/scripts/tests -v
  ```
  Expected: all tests pass. No `ERROR`, no `FAIL`.

- [ ] **Step 2: Smoke-test all 4 modes against the real fixture**

  ```bash
  for m in summary by_source non_compute roofline; do
    python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
      /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
      --mode $m | python3 -m json.tool > /dev/null && echo "  $m OK"
  done
  ```
  Expected: 4 lines, each ending `OK`.

- [ ] **Step 3: Confirm v7x peak fidelity on the real fixture**

  ```bash
  python3 plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py \
    /Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 \
    --mode roofline | python3 -c "
  import json,sys
  d = json.load(sys.stdin)
  assert d['peaks_used']['peak_tflops_bf16'] == 1153.5, d['peaks_used']
  assert d['peaks_used']['peak_hbm_gibps']   == 3690.0, d['peaks_used']
  assert d['peaks_used']['source'] == 'builtin v7x table'
  print('peak fidelity OK')
  "
  ```
  Expected: `peak fidelity OK`.

- [ ] **Step 4: Run cross-mode equality on the real fixture**

  ```bash
  python3 - <<'PY'
  import json, subprocess, sys
  D = "/Users/xl/tensorboard/tensorboard/plugins/profile/dp8_fsdp128"
  S = "plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py"
  def run(*args):
      r = subprocess.run([sys.executable, S, D] + list(args),
                           capture_output=True, text=True)
      assert r.returncode == 0, r.stderr
      return json.loads(r.stdout)
  d_sum = run("--mode","summary")
  d_bs  = run("--mode","by_source")
  d_nc  = run("--mode","non_compute","--no-comm-stalls")
  assert d_sum["totals"]["compute_duration_ps"] == d_bs["totals"]["compute_duration_ps"]
  assert d_sum["totals"]["data_move_duration_ps"] == d_nc["totals"]["data_move_duration_ps"]
  assert d_sum["step_window_ps"] == d_bs["step_window_ps"] == d_nc["step_window_ps"]
  print("cross-mode invariants OK")
  PY
  ```
  Expected: `cross-mode invariants OK`.

- [ ] **Step 5: Confirm final tree and version**

  ```bash
  python3 -c "
  import json, pathlib
  m = json.loads(pathlib.Path('plugins/tpu-perf/.claude-plugin/plugin.json').read_text())
  assert m['version'] == '0.2.0', m
  must = [
      'plugins/tpu-perf/skills/compute-breakdown/SKILL.md',
      'plugins/tpu-perf/skills/compute-breakdown/scripts/compute_breakdown.py',
      'plugins/tpu-perf/skills/compute-breakdown/scripts/_peaks.py',
      'plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/__init__.py',
      'plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane.proto',
      'plugins/tpu-perf/skills/compute-breakdown/scripts/_proto/xplane_pb2.py',
  ]
  for p in must:
      assert pathlib.Path(p).exists(), p
  print('tree OK')
  "
  ```
  Expected: `tree OK`.

- [ ] **Step 6: Final commit (only if anything changed)**

  ```bash
  git status
  # If any uncommitted changes from final verification fixes exist:
  git add -p
  git commit -m "chore(tpu-perf): final compute-breakdown verification fixes"
  ```

---

## Final notes

- **All 4 capabilities delivered:** mode 1 (top time-eaters by source), mode 2 (layer-scoping table), mode 3 (non-compute audit), mode 4 (v7x roofline shortfall vs theoretical peak).
- **Cross-mode invariants** hold by construction: stages 1-3 are shared; mode-specific stage-4 projections only filter and aggregate, never re-derive totals.
- **JSON contract** is uniform: every invocation emits exactly one top-level JSON object on stdout with `status` ∈ `{ok, absent}`. Absent paths exit 0; user-arg errors exit 1 to stderr.
- **No new system dependencies**: stdlib + `protobuf` runtime (already on the system, transitive via xprof). `_proto/` is vendored from profile-anatomy; the two copies must stay in sync (a maintenance note).

