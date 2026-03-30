---
name: exec-gke-tpu
description: Provision and run code on GKE-based TPU (e.g. TPU v7x) via xpk. Use when the user wants to create a TPU workload on GKE, sync sglang-jax code, install dependencies, or run benchmarks/tests on multi-process TPU pods.
argument-hint: "[create|sync|run|teardown] [args...]"
---

# GKE TPU Execution Skill

This skill handles provisioning GKE-based TPU workloads (e.g. TPU v7x) via `xpk`, syncing sglang-jax code, and running benchmarks or tests on multi-process TPU pods.

> **Scope**: This skill is for GKE/xpk-based TPU (e.g. v7x). For SkyPilot-based TPU (v4/v6e), use the `exec-remote` skill instead.

## Prerequisites

The following tools must be installed locally. Install via:

```bash
# 1. Google Cloud SDK
brew install --cask google-cloud-sdk

# 2. kubectl + auth plugin
gcloud components install kubectl gke-gcloud-auth-plugin beta --quiet

# 3. xpk (must use Python 3.13, NOT 3.14 which has argparse incompatibility)
brew install pipx
pipx install xpk --python python3.13

# 4. Auth
gcloud auth login
gcloud config set project tpu-service-473302
gcloud auth application-default login
```

**PATH setup** (needed in every shell/command):
```bash
export PATH="/Users/$(whoami)/.local/bin:/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/bin:$PATH"
```

## 1. Create Cluster + Workload

### Step 1a: Create Pathways Cluster (one-time, reusable)

```bash
xpk cluster create-pathways \
  --cluster xpk-cluster \
  --num-slices=1 \
  --tpu-type=<TPU_TYPE> \
  --zone=us-central1-c \
  --spot \
  --project tpu-service-473302
```

Common TPU types: `tpu7x-8`

### Step 1b: Create Workload

**CRITICAL: Docker image must match pyproject.toml JAX version and have Python >= 3.12.**

| pyproject.toml JAX version | Docker image tag |
|---|---|
| `jax==0.8.1` | `jax0.8.1-rev1` |
| `jax==0.9.0` | `jax0.9.0-rev1` |

Check available tags:
```bash
gcloud artifacts docker images list us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu \
  --include-tags --format="value(tags)" --project=tpu-service-473302 \
  | tr ',' '\n' | grep -E "^jax" | sort -V
```

Create workload:
```bash
xpk workload create \
  --workload <WORKLOAD_NAME> \
  --num-slices=1 \
  --tpu-type=<TPU_TYPE> \
  --cluster=xpk-cluster \
  --zone=us-central1-c \
  --project=tpu-service-473302 \
  --docker-name='<WORKLOAD_NAME>' \
  --docker-image="us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:<IMAGE_TAG>" \
  --command="sleep infinity"
```

Wait for pod ready:
```bash
# Get pod name
kubectl get pods
# Wait for ready
kubectl wait --for=condition=Ready pod/<POD_NAME> --timeout=300s
```

## 2. Sync Code & Install Dependencies

**IMPORTANT: TPU v7x-8 pods have 2 containers (`<name>-1` and `<name>-2`) with independent filesystems. All setup must be done on BOTH containers.**

### Step 2a: Clone repo + install sglang-jax (on each container)

```bash
for CONTAINER in <WORKLOAD_NAME>-1 <WORKLOAD_NAME>-2; do
  kubectl exec <POD_NAME> -c $CONTAINER -- bash -c '
    cd /tmp && git clone --depth 1 https://github.com/sgl-project/sglang-jax.git
    cd sglang-jax/python && pip install --no-deps -e .
  '
done
```

### Step 2b: Install runtime dependencies (on each container)

```bash
for CONTAINER in <WORKLOAD_NAME>-1 <WORKLOAD_NAME>-2; do
  kubectl exec <POD_NAME> -c $CONTAINER -- pip install \
    pyzmq fastapi orjson uvicorn jinja2 pydantic python-multipart \
    huggingface-hub safetensors transformers tiktoken \
    setproctitle psutil pandas httpx openai aiohttp \
    pybase64 partial_json_parser omegaconf \
    msgpack-python requests typing-extensions
done
```

### Step 2c: Sync local code changes (optional)

To push local modifications to the pod:
```bash
for CONTAINER in <WORKLOAD_NAME>-1 <WORKLOAD_NAME>-2; do
  kubectl cp ./python/sgl_jax <POD_NAME>:/tmp/sglang-jax/python/sgl_jax -c $CONTAINER
  kubectl cp ./benchmark <POD_NAME>:/tmp/sglang-jax/benchmark -c $CONTAINER
done
```

## 3. Run Code on Multi-Process TPU

### Key Architecture Facts

- TPU v7x-8 = 4 chips × 2 cores = **8 JAX devices**
- Pod has **2 containers** (processes), each seeing 4 local devices
- `jax.distributed.initialize()` must be called in **both containers simultaneously**
- The init and workload must run in the **same Python process** (not separate invocations)
- **CRITICAL: ALL processes must execute the SAME jitted computations.** If process 0 runs a sharded `jax.jit` call but process 1 is sleeping, JAX will hang forever waiting for process 1 to participate. Do NOT have the worker process just `sleep()` — it must run the same code path.

### Step 3a: Create a launcher script

Write a Python launcher that handles distributed init + runs the target script.
**Both processes must run the same script — use conditional logic only for print/logging, never for computation:**

```python
#!/usr/bin/env python3
"""Launcher for multi-process TPU workloads.
ALL processes run the SAME code - JAX requires this for sharded computations.
"""
import os, sys

# Set up import paths: repo root for benchmark/, python/ for sgl_jax
sys.path.insert(0, "/tmp/sglang-jax/python")
sys.path.insert(0, "/tmp/sglang-jax")
os.chdir("/tmp/sglang-jax")

import jax
jax.distributed.initialize()
proc = jax.process_index()
print(f"[Process {proc}] ready, {jax.device_count()} devices", flush=True)

# ALL processes must run the same code path for sharded ops
sys.argv = ["script_name", "--arg1", "val1", ...]
import runpy
runpy.run_path("/tmp/sglang-jax/path/to/script.py", run_name="__main__")
```

### Step 3b: Copy launcher to both containers

```bash
for CONTAINER in <WORKLOAD_NAME>-1 <WORKLOAD_NAME>-2; do
  kubectl cp /tmp/launcher.py <POD_NAME>:/tmp/launcher.py -c $CONTAINER
done
```

### Step 3c: Launch on both containers simultaneously

```bash
# Container-2 (worker) in background
kubectl exec <POD_NAME> -c <WORKLOAD_NAME>-2 -- python3 -u /tmp/launcher.py 2>&1 &
BGPID=$!

# Container-1 (main) in foreground
kubectl exec <POD_NAME> -c <WORKLOAD_NAME>-1 -- python3 -u /tmp/launcher.py 2>&1
RC=$?

# Cleanup
kill $BGPID 2>/dev/null
wait $BGPID 2>/dev/null
```

## 4. Teardown

```bash
# Delete workload only
xpk workload delete --workload <WORKLOAD_NAME> \
  --cluster=xpk-cluster --zone=us-central1-c --project=tpu-service-473302

# Delete entire cluster (careful - removes all workloads)
xpk cluster delete --cluster xpk-cluster \
  --zone=us-central1-c --project=tpu-service-473302
```

## 5. Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `SyntaxError: invalid syntax` on `*` unpacking | Python < 3.12 | Use Docker image with Python >= 3.12 (jax0.8.1-rev1+) |
| `BooleanOptionalAction.__init__() got unexpected keyword argument 'type'` | xpk installed with Python 3.14 | Reinstall xpk: `pipx reinstall xpk --python python3.13` |
| JAX TPU init hangs > 60s | Only one container started | Must start both containers simultaneously |
| Sharded computation hangs | Worker process sleeping instead of running same code | ALL processes must execute the same jitted code paths — never have workers just `sleep()` |
| `Shutdown barrier DEADLINE_EXCEEDED` | One process crashed while other is alive | Check error in crashed process logs, fix, and restart both |
| `ModuleNotFoundError` | Dependencies not installed or wrong PYTHONPATH | Ensure both `/tmp/sglang-jax` and `/tmp/sglang-jax/python` in sys.path |
| `gcloud auth` errors | Token expired | Re-run `gcloud auth login` |
| Benchmark hangs during compilation | JAX version mismatch with sglang-jax | Use Docker image JAX version matching pyproject.toml |
