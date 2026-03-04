---
name: deploy-cluster
description: Deploys a SkyPilot-managed TPU cluster on GKE. First ensures the GKE cluster exists via apply-resource, then configures SkyPilot and launches the cluster.
argument-hint: "[cluster-name] [tpu-type] [zone]"
---

# Deploy SkyPilot TPU Cluster on GKE

This skill deploys a SkyPilot-managed TPU cluster on an existing GKE cluster. It builds on the `apply-resource` skill which handles GKE cluster creation via xpk.

## Prerequisites

- **SkyPilot**: `pip install skypilot`
  - Check: `sky --help`
- **Google Cloud SDK (gcloud)**: [Install guide](https://cloud.google.com/sdk/docs/install)
  - Run `gcloud auth login` to authenticate
- **Kubectl**: [Install guide](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl)

## Fixed Parameters

- **PROJECT_ID**: `tpu-service-473302`

## Supported TPU Types

Each GKE v6e node exposes exactly **4 TPU chips** (`google.com/tpu: 4`) regardless of the slice size.
Therefore: `num_nodes = total_chips / 4`, and every pod always requests 4 chips.
Exception: `v6e-1` is a sub-slice that exposes 1 chip on a single node.

| Type | Topology | Chips/Host | Nodes |
|------|----------|------------|-------|
| v6e-1  | 1x1  | 1 | 1  |
| v6e-4  | 2x2  | 4 | 1  |
| v6e-8  | 2x4  | 4 | 2  |
| v6e-16 | 4x4  | 4 | 4  |
| v6e-32 | 4x8  | 4 | 8  |
| v6e-64 | 8x8  | 4 | 16 |
| v6e-128 | 8x16 | 4 | 32 |
| v6e-256 | 16x16 | 4 | 64 |

> **Zone vs Region**: xpk always creates GKE clusters at the **region** level (e.g., `asia-northeast1`), even when given a zone like `asia-northeast1-b`. The deploy script handles this automatically — you may pass either a zone or a region.

## Deployment Workflow

### Step 1: Ensure GKE Cluster Exists

First, use the `apply-resource` skill to create the GKE cluster (or confirm it already exists):

```
/apply-resource create
```

Collect the following from the cluster creation:
- **CLUSTER_NAME**: The cluster name
- **TPU_TYPE**: The TPU type used (e.g., `v6e-16`)
- **ZONE**: The GCP zone (e.g., `asia-northeast1-b`)

### Step 2: Deploy SkyPilot Cluster

Run the deploy script (located in the `scripts/` directory alongside this skill definition):

```bash
python scripts/deploy.py <CLUSTER_NAME> <TPU_TYPE> <ZONE>
```

This script will:
1. Generate `~/.sky/config.yaml` from the template with correct TPU parameters
2. Generate a temporary `setup.yaml` with the correct `num_nodes`
3. Fetch GKE cluster credentials via `gcloud`
4. Execute `sky launch -c <CLUSTER_NAME> -r <setup.yaml>`

### Step 3: Verify

```bash
sky status          # Check cluster status
sky exec <CLUSTER_NAME> 'echo hello'  # Test remote execution
```

## What the Script Does

The deploy script (`scripts/deploy.py`) automates:

1. **Config generation**: Reads `config.yaml` template → replaces `<ACCELERATOR_TYPE>`, `<TPU_TOPOLOGY>`, `<CHIPS_PER_HOST>` → writes to `~/.sky/config.yaml` (backs up existing file if present)
2. **Setup generation**: Reads `setup.yaml` template → replaces `<NUM_NODES>` → writes to a temporary file
3. **GKE auth**: Runs `gcloud container clusters get-credentials`
4. **SkyPilot launch**: Runs `sky launch -c <name> -r <setup.yaml>`

## Error Handling

- **Unsupported TPU type**: Lists all supported types
- **Missing tools**: Lists installation instructions for missing prerequisites
- **Existing ~/.sky/config.yaml**: Automatically backs up with timestamp before overwriting
- **GKE auth failure**: Reports error and stops before launching
- **sky launch failure**: Reports error with the failed command

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
