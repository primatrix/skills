---
name: deploy-cluster
description: Deploys a SkyPilot-managed TPU cluster on GKE. Automatically ensures the required node pool exists for the requested TPU type, creating one if necessary. Supports switching between TPU types on the same GKE cluster.
argument-hint: "[cluster-name] [tpu-type] [zone]"
---

# Deploy SkyPilot TPU Cluster on GKE

This skill deploys a SkyPilot-managed TPU cluster on an existing GKE cluster. It builds on the `apply-resource` skill which handles GKE cluster creation via xpk.

**Key Feature**: Automatically manages GKE node pools per TPU type. When switching between TPU configurations (e.g., v6e-1 for unit tests, v6e-4 for e2e tests), the script adds the required node pool to the existing cluster instead of creating a new cluster. This saves time and cost.

## Prerequisites

- **SkyPilot**: `pip install skypilot`
  - Check: `sky --help`
- **Google Cloud SDK (gcloud)**: [Install guide](https://cloud.google.com/sdk/docs/install)
  - Run `gcloud auth login` to authenticate
- **Kubectl**: [Install guide](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)

## Required Parameters

- **PROJECT_ID**: GCP project ID (must be provided by the caller or user)
- **TPU_TYPE**: TPU accelerator type (e.g., `v6e-1`, `v6e-4`, `v6e-16`)
- **ZONE**: GCP zone (e.g., `asia-northeast1-b`, `us-east5-a`)
- **CLUSTER_NAME**: Inherited from the GKE cluster created by `apply-resource` -- do NOT ask the user again

If all parameters are already known from an upstream caller (e.g., `exec-remote`), use them directly -- do NOT re-ask. Only prompt interactively when this skill is invoked standalone and the parameters are not yet known.

## Supported TPU Types

Each GKE node exposes 4 TPU chips (`google.com/tpu: 4`), except `v6e-1` which exposes 1 chip.
Therefore: `num_nodes = total_chips / 4`, and every pod always requests 4 chips (1 for v6e-1).

| Type | Topology | Chips/Host | Nodes | Machine Type |
|------|----------|------------|-------|--------------|
| v6e-1  | 1x1  | 1 | 1  | ct6e-standard-1t |
| v6e-4  | 2x2  | 4 | 1  | ct6e-standard-4t |
| v6e-8  | 2x4  | 4 | 2  | ct6e-standard-4t |
| v6e-16 | 4x4  | 4 | 4  | ct6e-standard-4t |
| v6e-32 | 4x8  | 4 | 8  | ct6e-standard-4t |
| v6e-64 | 8x8  | 4 | 16 | ct6e-standard-4t |
| v6e-128 | 8x16 | 4 | 32 | ct6e-standard-4t |
| v6e-256 | 16x16 | 4 | 64 | ct6e-standard-4t |

> **Zone vs Region**: xpk always creates GKE clusters at the **region** level (e.g., `asia-northeast1`), even when given a zone like `asia-northeast1-b`. The deploy script handles this automatically -- you may pass either a zone or a region.

## Deployment Workflow

### Step 1: Ensure GKE Cluster Exists

Use the `apply-resource` skill to create the GKE cluster (or confirm it already exists). This only needs to be done once:

```
/apply-resource create
```

Carry forward the resulting **CLUSTER_NAME**, **TPU_TYPE**, and **ZONE** for Step 2.

### Step 2: Wait for GKE Cluster Ready

Before deploying SkyPilot, ensure the GKE cluster status is `RUNNING`:

```bash
gcloud container clusters list --project=$PROJECT_ID \
  --filter="name=<CLUSTER_NAME>" --format="table(name,location,status)"
```

If status is `RECONCILING` or `PROVISIONING`, wait until it becomes `RUNNING`.

### Step 3: Deploy SkyPilot Cluster

Run the deploy script (located in the `scripts/` directory alongside this skill definition):

```bash
python scripts/deploy.py <CLUSTER_NAME> <TPU_TYPE> <ZONE>
```

This script will:
1. Fetch GKE cluster credentials via `gcloud`
2. **Check if a node pool for the TPU type already exists** (by matching machine type and topology)
3. **Create a new node pool if none matches** (named `tpu-<TPU_TYPE>`, e.g., `tpu-v6e-1`)
4. **Handle topology changes**: If switching TPU types, automatically `sky down` the old cluster
5. Generate `~/.sky/config.yaml` from the template with correct TPU parameters
6. Generate a temporary `setup.yaml` with the correct `num_nodes`
7. Execute `sky launch -c <CLUSTER_NAME> -r <setup.yaml>`
8. Save the cluster name to `.cluster_name_tpu` in the plugin root (for `exec-remote` integration)

### Step 4: Verify

```bash
sky status          # Check cluster status
sky exec <CLUSTER_NAME> 'echo hello'  # Test remote execution
```

## Node Pool Management

The deploy script intelligently manages GKE node pools:

- **Detection**: Before creating a node pool, the script checks all existing pools by matching `machineType` and `tpuTopology`. This detects pools created by xpk, manually, or by previous runs.
- **Creation**: New pools use the naming convention `tpu-<type>` (e.g., `tpu-v6e-1`, `tpu-v6e-4`). Single-host TPUs (v6e-1, v6e-4) omit `--tpu-topology` as GKE infers it from the machine type.
- **Coexistence**: Multiple node pools for different TPU types can coexist on the same cluster. SkyPilot's `nodeSelector` ensures pods land on the correct pool.
- **Spot instances**: Node pools are created with `--spot` and autoscaling (`--min-nodes=0`).

### Example: Running tests on different TPU types

```bash
# First time: create cluster with v6e-4 via apply-resource
/apply-resource create   # cluster=my-cluster, tpu=v6e-4, zone=asia-northeast1-b

# Deploy SkyPilot for v6e-4 tests
python scripts/deploy.py my-cluster v6e-4 asia-northeast1-b
sky exec my-cluster 'python test/srt/run_suite.py --suite e2e-test-tpu-v6e-4'

# Switch to v6e-1 for unit tests (same cluster, new node pool)
python scripts/deploy.py my-cluster v6e-1 asia-northeast1-b
# Script auto-detects topology change, runs sky down, adds tpu-v6e-1 pool, relaunches
sky exec my-cluster 'python test/srt/run_suite.py --suite unit-test-tpu-v6e-1'
```

## What the Script Does

The deploy script (`scripts/deploy.py`) automates:

1. **GKE auth**: Runs `gcloud container clusters get-credentials`
2. **Node pool check**: Lists existing pools, matches by machine type + topology
3. **Node pool creation** (if needed): Runs `gcloud beta container node-pools create` with correct TPU params
4. **Topology change handling**: Detects if `~/.sky/config.yaml` has a different topology, runs `sky down` if so
5. **Config generation**: Reads `config.yaml` template -> replaces placeholders -> writes to `~/.sky/config.yaml`
6. **Setup generation**: Reads `setup.yaml` template -> replaces `<NUM_NODES>` -> writes to temp file
7. **SkyPilot launch**: Runs `sky launch -c <name> -r <setup.yaml>`

## Error Handling

- **Unsupported TPU type**: Lists all supported types
- **Missing tools**: Lists installation instructions for missing prerequisites
- **Existing ~/.sky/config.yaml**: Automatically backs up with timestamp before overwriting
- **GKE auth failure**: Reports error and stops before launching
- **Node pool creation failure**: Reports error with the failed command
- **sky launch failure**: Reports error with the failed command
- **Topology mismatch**: Automatically tears down and relaunches

## Cleanup

To tear down the SkyPilot cluster:
```bash
sky down <CLUSTER_NAME>
```

To also remove the GKE cluster:
```
/apply-resource delete
```

## Useful Resources

- [SkyPilot Documentation](https://docs.skypilot.co/)
- [Planning TPUs in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus)
- [TPU Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
