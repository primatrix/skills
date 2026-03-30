---
name: profile-tpu-kernel
description: Use when the user wants to profile Pallas/JAX kernels on TPU with xprof to get LLO utilization (MXU, Vector ALU, etc.). Covers setting LIBTPU flags, capturing traces, transferring files, and viewing results in TensorBoard.
---

# Profile Pallas/JAX Kernels on TPU with xprof

Capture xprof traces with LLO (Low-Level Optimizer) utilization for Pallas custom call kernels running on TPU. Shows per-hardware-unit breakdown: MXU, Vector ALU, Scalar ALU, Vector Load/Store, etc.

> **Scope**: GKE-based TPU pods (e.g. TPU v7x). Requires a running pod — use the `exec-gke-tpu` skill to provision one first.

## Quick Reference

| LLO Row | What it shows |
|---|---|
| MXU | Matrix Unit utilization (matmuls) |
| Scalar ALU | Scalar arithmetic |
| Vector ALU | Vector arithmetic |
| Vector Load / Store | HBM <-> VMEM data movement |
| Vector Fills / Spills | VMEM spill traffic |
| XLU | Cross-Lane Unit (permutes, reductions) |

## The One Critical Rule

**`LIBTPU_INIT_ARGS` must be set BEFORE `import jax`.** If JAX is imported first, the flags have no effect and traces will lack LLO utilization rows.

The two required flags:
```
--xla_enable_custom_call_region_trace=true
--xla_xprof_register_llo_debug_info=true
```

## Step 1: Create profile_launcher.py

The standard `launcher.py` imports JAX immediately, so profiling needs a separate launcher that sets the flags first. Write this to `/tmp/profile_launcher.py`:

```python
#!/usr/bin/env python3
"""Launcher that sets LIBTPU_INIT_ARGS for xprof LLO tracing before importing JAX."""
import os, sys, runpy

# ---- MUST be BEFORE import jax ----
_xla_flags = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true"
)
existing = os.environ.get("LIBTPU_INIT_ARGS", "")
os.environ["LIBTPU_INIT_ARGS"] = (existing + " " + _xla_flags).strip()
print(f"LIBTPU_INIT_ARGS={os.environ['LIBTPU_INIT_ARGS']}", flush=True)

REPO_ROOT = "/tmp/sglang-jax"
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import jax
jax.distributed.initialize()
proc = jax.process_index()
print(f"[Process {proc}] JAX {jax.__version__}, {jax.device_count()} devices", flush=True)

script_path = os.path.join(REPO_ROOT, sys.argv[1])
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(script_path, run_name="__main__")
```

Copy to both containers:
```bash
for C in ${WL}-1 ${WL}-2; do
  kubectl cp /tmp/profile_launcher.py $POD:/tmp/profile_launcher.py -c $C
done
```

## Step 2: Add --profile Flag to Benchmark Script

The benchmark script needs a `--profile` mode that uses `jax.profiler.trace()` instead of timing. Key pattern:

```python
if profile:
    os.makedirs(profile_dir, exist_ok=True)
    # Warmup (required — first run compiles)
    for _ in range(warmup_iters):
        out = compute()
        jax.block_until_ready(out)
    # Capture trace
    with jax.profiler.trace(profile_dir):
        for step in range(iters):
            with jax.profiler.StepTraceAnnotation("kernel_name", step_num=step):
                out = compute()
                jax.block_until_ready(out)
```

`bench_fused_moe.py` already has `--profile` and `--profile-dir` flags.

## Step 3: Run Profiling on Pod

Both containers must run simultaneously (multi-process JAX requirement):

```bash
PROFILE_CMD="python3 -u /tmp/profile_launcher.py benchmark/moe/bench_fused_moe.py \
  --num-experts 64 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \
  --num-tokens 128 --iters 3 --warmup-iters 1 \
  --imbalance-mode balanced --profile --profile-dir /tmp/profile_output"

# Worker in background
kubectl exec $POD -c ${WL}-2 -- bash -c "$PROFILE_CMD" 2>&1 &
BGPID=$!
# Main in foreground
kubectl exec $POD -c ${WL}-1 -- bash -c "$PROFILE_CMD" 2>&1
kill $BGPID 2>/dev/null; wait $BGPID 2>/dev/null
```

**Verify output** — must see `LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true ...` printed before JAX init.

## Step 4: Transfer Trace Files to Local

Traces are ~90 MB total. **Use GCS** — `kubectl cp` truncates files > ~50 MB.

```bash
# Upload from pod to GCS
kubectl exec $POD -c ${WL}-1 -- bash -c '
TRACE_DIR=$(find /tmp/profile_output -name "*.xplane.pb" -exec dirname {} \;)
gsutil cp ${TRACE_DIR}/*.xplane.pb gs://<bucket>/profile_tmp/
gsutil cp ${TRACE_DIR}/*.trace.json.gz gs://<bucket>/profile_tmp/
'

# Download from GCS to local
gsutil cp gs://<bucket>/profile_tmp/*.xplane.pb ./profile_output/
gsutil cp gs://<bucket>/profile_tmp/*.trace.json.gz ./profile_output/
```

Generated files:
- `*.xplane.pb` (~83 MB) — full XPlane data with LLO utilization
- `*.trace.json.gz` (~10 MB) — pre-converted trace events

## Step 5: View in TensorBoard

**TensorBoard MUST run on Linux** (the pod). The xprof native module (`_pywrap_profiler_plugin`) has no macOS build.

### 5a. Install on Pod (version-sensitive)

```bash
kubectl exec $POD -c ${WL}-1 -- pip install \
  'tensorflow>=2.21' \
  'tensorboard>=2.20' \
  'tensorboard-plugin-profile>=2.22' \
  'xprof>=2.22' \
  'protobuf>=5,<7' \
  'setuptools<81'
```

### 5b. Start TensorBoard on Pod

```bash
kubectl exec $POD -c ${WL}-1 -- bash -c "nohup python3 -c '
from tensorboard import main as tb
import sys
sys.argv = [\"tensorboard\", \"--logdir=/tmp/profile_output/\", \"--port=6006\", \"--bind_all\", \"--load_fast=false\"]
tb.run_main()
tb.main()
' > /tmp/tb.log 2>&1 &"
```

Verify no plugin errors: `kubectl exec $POD -c ${WL}-1 -- grep -i "Failed to load plugin" /tmp/tb.log`

### 5c. Port-Forward and Open

```bash
kubectl port-forward $POD 6006:6006
```

Open **http://localhost:6006/** -> **Profile** tab -> **trace_viewer** tool.

### 5d. Navigate Trace Viewer

| Key | Action |
|-----|--------|
| **W / S** | Zoom in / out |
| **A / D** | Pan left / right |
| **1** | Select mode |
| **2** | Pan mode |
| **3** | Zoom mode |
| **4** | Timing mode |

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| No LLO rows (MXU etc.) | LIBTPU flags not set before JAX import | Use `profile_launcher.py`, verify flag printed in output |
| Profiling hangs after warmup | libtpu version issue | Install `libtpu-nightly>=0.0.38.dev` |
| "No dashboards are active" | TensorBoard can't find data | Check `--logdir` points to parent of `plugins/profile/` dir |
| "plugin has moved" message | Missing tensorboard-plugin-profile | `pip install tensorboard-plugin-profile>=2.22` |
| `_pywrap_profiler_plugin` error | Running TensorBoard on macOS | Run on Linux pod + port-forward |
| `proto.id() > INT_MAX` | Old tensorboard-plugin-profile | Upgrade to `>=2.22` with `tensorflow>=2.21` |
| `pkg_resources` missing | setuptools >= 82 removed it | `pip install 'setuptools<81'` |
| `kubectl cp` file truncated | Large file transfer unreliable | Use GCS as intermediate |
