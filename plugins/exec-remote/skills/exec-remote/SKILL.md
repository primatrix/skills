---
name: exec-remote
description: Executes Python scripts, tests, or benchmarks on a provisioned remote cluster (GPU or TPU) using SkyPilot. Use this skill when the user asks to run code on GPU, TPU, or any "remote" cluster.
argument-hint: "[gpu|tpu] [script-path] [args...]"
---

# Remote Execution Skill

This skill handles running code on remote GPU or TPU clusters via SkyPilot.

## Defaults

The following defaults apply unless the user explicitly overrides them:

| Parameter      | Default                    |
|----------------|----------------------------|
| PROJECT_ID     | `tpu-service-473302`       |
| CLUSTER_NAME   | `sglang-jax-agent-tests`   |
| ZONE           | `asia-northeast1-b`        |
| NUM_SLICES     | `1`                        |

Use these values directly — do NOT ask the user to confirm or re-enter them unless they specify otherwise.

## 1. Determine Target Device

Identify the target device from the user's request:

| Target | Cluster name file   | Env prefix                         |
|--------|---------------------|------------------------------------|
| GPU    | `.cluster_name_gpu` | `export CUDA_VISIBLE_DEVICES=0; `  |
| TPU    | `.cluster_name_tpu` | *(none)*                           |

If the user does not specify a device, ask them which one to use.

## 2. Prerequisites

- The cluster must already be provisioned. Check that the corresponding cluster name file (`.cluster_name_gpu` or `.cluster_name_tpu`) exists and is non-empty in the project root.
- If the file does not exist or is empty, provision the cluster using the appropriate method (see Section 3).

## 3. Cluster Provisioning

### GPU (Standalone SkyPilot)

GPU clusters are provisioned using the standalone `launch_gpu.sh` script. Locate it in the `scripts/` directory alongside this skill definition.

```bash
# Common accelerator types: H100:1, A100:1, L4:1
bash <absolute_path_to_launch_gpu.sh> <accelerator_type> <experiment_name>
```

The launch script automatically updates `.cluster_name_gpu`.

### TPU

There are two provisioning paths for TPU:

#### Path A: GKE-based (via `deploy-cluster` skill) — Recommended

This path provisions TPU on GKE using the full pipeline: `apply-resource` -> `deploy-cluster` -> `exec-remote`.

Each TPU type gets its own SkyPilot cluster named `<cluster>-<username>-<tpu_type>`, allowing multiple topologies to run in parallel.

1. Use the `deploy-cluster` skill which will:
   - Use default cluster/project/zone unless user overrides
   - Ensure the GKE cluster exists (via `apply-resource`)
   - Configure SkyPilot for GKE
   - Launch a per-TPU-type SkyPilot cluster
   - Save the cluster name to `.cluster_name_tpu`

```
/deploy-cluster
```

Supported TPU types: v6e-1, v6e-4, v6e-8, v6e-16, v6e-32, v6e-64, v6e-128, v6e-256

#### Path B: Standalone SkyPilot TPU VM

For quick, single-node TPU usage without GKE, use the standalone `launch_tpu.sh` script:

```bash
# Common accelerator types: tpu-v4-8, tpu-v4-16, tpu-v6e-1, tpu-v6e-4
bash <absolute_path_to_launch_tpu.sh> <accelerator_type> <experiment_name>
```

The launch script automatically updates `.cluster_name_tpu`.

### Teardown

```bash
# GPU
sky down $(cat .cluster_name_gpu) -y

# TPU (tear down all per-TPU-type clusters)
sky down <CLUSTER_NAME>-<USERNAME>-v6e-1 -y
sky down <CLUSTER_NAME>-<USERNAME>-v6e-4 -y
```

For GKE-based TPU, also remove the GKE cluster via `/apply-resource delete` if no longer needed.

## 4. Execution Command

### GPU

```bash
sky exec $(cat .cluster_name_gpu) --workdir . "export CUDA_VISIBLE_DEVICES=0; uv run --extra gpu python <PATH_TO_SCRIPT> [ARGS]"
```

- `export CUDA_VISIBLE_DEVICES=0;` ensures deterministic single-GPU execution. Adjust for multi-GPU jobs.
- `--extra gpu` activates GPU optional dependencies (e.g. `jax[cuda]`).

### TPU

```bash
sky exec <CLUSTER_NAME>-<USERNAME>-<TPU_TYPE> --workdir . "uv run --extra tpu python <PATH_TO_SCRIPT> [ARGS]"
```

- `--extra tpu` activates TPU optional dependencies (e.g. `jax[tpu]`).
- Use the per-TPU-type cluster name (e.g. `sglang-jax-agent-tests-hongmao-v6e-1`).

### Common flags

- `--workdir .` syncs the current local directory to the remote instance before running.
- For pytest, use `python -m pytest <test_path>` instead of calling pytest directly.

## 5. Usage Examples

**Run a benchmark on GPU:**
```bash
sky exec $(cat .cluster_name_gpu) --workdir . "export CUDA_VISIBLE_DEVICES=0; uv run --extra gpu python src/lynx/perf/benchmark_train.py"
```

**Run tests on TPU (single type):**
```bash
sky exec sglang-jax-agent-tests-hongmao-v6e-4 --workdir . "uv run --extra tpu python -m pytest src/lynx/test/"
```

**Run CI tests on multiple TPU types in parallel:**
```bash
# Deploy both types (sequential — config.yaml is global)
python <deploy-cluster>/scripts/deploy.py sglang-jax-agent-tests v6e-1 asia-northeast1-b
python <deploy-cluster>/scripts/deploy.py sglang-jax-agent-tests v6e-4 asia-northeast1-b

# Execute in parallel
sky exec sglang-jax-agent-tests-hongmao-v6e-1 --workdir . "python test/srt/run_suite.py --suite unit-test-tpu-v6e-1" &
sky exec sglang-jax-agent-tests-hongmao-v6e-4 --workdir . "python test/srt/run_suite.py --suite e2e-test-tpu-v6e-4" &
wait
```

## 6. Operational Notes

- **Logs**: SkyPilot streams `stdout` and `stderr` directly to the terminal.
- **Interruption**: `Ctrl+C` may not kill the remote process; check SkyPilot docs for cleanup if needed.

## 7. GKE TPU Full Pipeline Procedure (Path A)

When the user requests to run code on TPU and no `.cluster_name_tpu` exists (or the user explicitly wants a new cluster), follow this procedure to orchestrate the full pipeline: `apply-resource` -> `deploy-cluster` -> `exec-remote`.

All parameters use defaults unless the user explicitly overrides them — do NOT ask for confirmation.

### 7.1 Collect Parameters

Only ask the user for parameters they haven't specified. Use defaults for everything else:

| Parameter      | Default                           | Notes                           |
|----------------|-----------------------------------|---------------------------------|
| PROJECT_ID     | `tpu-service-473302`              | GCP project ID                  |
| CLUSTER_NAME   | `sglang-jax-agent-tests`          | GKE cluster name                |
| TPU_TYPE       | *(must specify)*                  | e.g. `v6e-4`, `v6e-1`          |
| NUM_SLICES     | `1`                               | Default to 1                    |
| ZONE           | `asia-northeast1-b`               | Must support the chosen TPU type |

### 7.2 Create GKE Cluster (apply-resource)

Check prerequisites, then create the GKE cluster:

```bash
which xpk && which gcloud && which kubectl

xpk cluster create-pathways \
  --cluster $CLUSTER_NAME \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --zone=$ZONE \
  --spot \
  --project=$PROJECT_ID
```

### 7.3 Wait for GKE Cluster Ready

Poll until the cluster status becomes `RUNNING`. Do NOT proceed to deploy SkyPilot while status is `PROVISIONING` or `RECONCILING` — it will fail with SSL errors.

```bash
gcloud container clusters list --project=$PROJECT_ID \
  --filter="name=$CLUSTER_NAME" --format="table(name,location,status)"
```

### 7.4 Deploy SkyPilot on GKE (deploy-cluster)

Run the deploy script for each required TPU type. Each call creates a separate SkyPilot cluster.

```bash
# Deploy each TPU type (must be sequential — config.yaml is global)
# Only tpu_type is required; cluster_name and zone use defaults
python <path-to-deploy-cluster>/scripts/deploy.py v6e-1
python <path-to-deploy-cluster>/scripts/deploy.py v6e-4
```

This creates:
- `$CLUSTER_NAME-$USERNAME-v6e-1` — SkyPilot cluster for v6e-1 tests
- `$CLUSTER_NAME-$USERNAME-v6e-4` — SkyPilot cluster for v6e-4 tests

After completion, verify:

```bash
sky status                  # Both clusters should show as UP
```

### 7.5 Execute User Code (exec-remote)

Determine `num_nodes` from the TPU type (v6e-N where total_chips = N, num_nodes = N / 4, minimum 1):

| TPU type | num_nodes |
|----------|-----------|
| v6e-1    | 1         |
| v6e-4    | 1         |
| v6e-8    | 2         |
| v6e-16   | 4         |
| v6e-32   | 8         |
| v6e-64   | 16        |
| v6e-128  | 32        |
| v6e-256  | 64        |

For single-node types (v6e-1, v6e-4), omit `--num-nodes`. For multi-node types, add `--num-nodes <N>`.

```bash
# Single-node (v6e-1, v6e-4) — use per-TPU-type cluster name
sky exec $CLUSTER_NAME-$USERNAME-v6e-1 --workdir . \
  "uv run --extra tpu python <PATH_TO_SCRIPT> [ARGS]"

# Multi-node (v6e-8+)
sky exec $CLUSTER_NAME-$USERNAME-v6e-8 --num-nodes 2 --workdir . \
  "uv run --extra tpu python <PATH_TO_SCRIPT> [ARGS]"

# Parallel execution across multiple TPU types
sky exec $CLUSTER_NAME-$USERNAME-v6e-1 --workdir . "..." &
sky exec $CLUSTER_NAME-$USERNAME-v6e-4 --workdir . "..." &
wait
```

### 7.6 Cleanup

When the user requests teardown, remove both layers:

```bash
# 1. Remove SkyPilot clusters (one per TPU type)
sky down $CLUSTER_NAME-$USERNAME-v6e-1 -y
sky down $CLUSTER_NAME-$USERNAME-v6e-4 -y

# 2. Remove GKE cluster (only for Path A / GKE-based)
xpk cluster delete \
  --cluster $CLUSTER_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID
```
