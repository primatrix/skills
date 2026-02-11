---
name: exec-remote
description: Executes Python scripts, tests, or benchmarks on a provisioned remote cluster (GPU or TPU) using SkyPilot. Use this skill when the user asks to run code on GPU, TPU, or any "remote" cluster.
argument-hint: [gpu|tpu] [script-path] [args...]
---

# Remote Execution Skill

This skill handles running code on remote GPU or TPU clusters via SkyPilot.

## 1. Determine Target Device

Identify the target device from the user's request:

| Target | Cluster name file   | Launch script              | UV extra | Env prefix                         |
|--------|---------------------|----------------------------|----------|------------------------------------|
| GPU    | `.cluster_name_gpu` | `skills/exec-remote/scripts/launch_gpu.sh`    | `gpu`    | `export CUDA_VISIBLE_DEVICES=0; `  |
| TPU    | `.cluster_name_tpu` | `skills/exec-remote/scripts/launch_tpu.sh`    | `tpu`    | *(none)*                           |

If the user does not specify a device, ask them which one to use.

## 2. Prerequisites

- The cluster must already be provisioned. Check that the corresponding cluster name file (`.cluster_name_gpu` or `.cluster_name_tpu`) exists and is non-empty in the project root.
- If the file does not exist or is empty, ask the user to provision a cluster first using the appropriate launch script.

## 3. Cluster Management

### Provisioning

```bash
# GPU — common accelerator types: H100:1, A100:1, L4:1
bash skills/exec-remote/scripts/launch_gpu.sh <accelerator_type> <experiment_name>

# TPU — common accelerator types: tpu-v4-8, tpu-v4-16, tpu-v6e-1, tpu-v6e-4
bash skills/exec-remote/scripts/launch_tpu.sh <accelerator_type> <experiment_name>
```

The launch script automatically updates the corresponding `.cluster_name_*` file.

### Teardown

```bash
# GPU
sky down $(cat .cluster_name_gpu) -y

# TPU
sky down $(cat .cluster_name_tpu) -y
```

## 4. Execution Command

### GPU

```bash
sky exec $(cat .cluster_name_gpu) --workdir . "export CUDA_VISIBLE_DEVICES=0; uv run --group dev --extra gpu python <PATH_TO_SCRIPT> [ARGS]"
```

- `export CUDA_VISIBLE_DEVICES=0;` ensures deterministic single-GPU execution. Adjust for multi-GPU jobs.
- `--extra gpu` activates GPU optional dependencies (e.g. `jax[cuda]`).

### TPU

```bash
sky exec $(cat .cluster_name_tpu) --workdir . "uv run --extra tpu python <PATH_TO_SCRIPT> [ARGS]"
```

- `--extra tpu` activates TPU optional dependencies (e.g. `jax[tpu]`).

### Common flags

- `--workdir .` syncs the current local directory to the remote instance before running.
- For pytest, use `python -m pytest <test_path>` instead of calling pytest directly.

## 5. Usage Examples

**Run a benchmark on GPU:**
```bash
sky exec $(cat .cluster_name_gpu) --workdir . "export CUDA_VISIBLE_DEVICES=0; uv run --extra gpu python src/lynx/perf/benchmark_train.py"
```

**Run tests on TPU:**
```bash
sky exec $(cat .cluster_name_tpu) --workdir . "uv run --extra tpu python -m pytest src/lynx/test/"
```

## 6. Operational Notes

- **Logs**: SkyPilot streams `stdout` and `stderr` directly to the terminal.
- **Interruption**: `Ctrl+C` may not kill the remote process; check SkyPilot docs for cleanup if needed.
