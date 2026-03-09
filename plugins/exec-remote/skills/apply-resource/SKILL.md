---
name: apply-resource
description: Manages GKE TPU clusters using xpk. Creates, deletes, and lists TPU Nodepool resources on Google Kubernetes Engine. Multi-user safe - always queries GKE for real-time cluster state.
argument-hint: "[create|delete|list] [interactive-params...]"
---

# GKE TPU Cluster Management Skill

This skill manages TPU clusters on Google Kubernetes Engine (GKE) using [xpk](https://github.com/AI-Hypercomputer/xpk).

**Multi-User Safe**: This skill does NOT use local caching. All cluster information is queried directly from GKE in real-time, ensuring accuracy in multi-user scenarios where clusters may be created, modified, or deleted by other users.

## Prerequisites

Before using this skill, ensure the following tools are installed:

- **Google Cloud SDK (gcloud)**: [Install guide](https://cloud.google.com/sdk/docs/install)
  - Run `gcloud auth login` to authenticate
  - Check: `gcloud auth list`
- **Kubectl**: [Install guide](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl)
  - Install auth plugin: `gke-gcloud-auth-plugin`
  - Check: `kubectl version --client`
- **Xpk**: [Install guide](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md)
  - Check: `xpk --help`

## Fixed Parameters

- **PROJECT_ID**: `tpu-service-473302` (always fixed)

## User-Provided Parameters

When creating a cluster, the following parameters are required:

- **CLUSTER_NAME**: Name of the GKE cluster
- **TPU_TYPE**: TPU accelerator type (e.g., `v6e-16`, `v6e-4`, `v4-8`)
- **NUM_SLICES**: Number of TPU slices to provision
- **ZONE**: GCP zone (e.g., `asia-northeast1-b`, `us-east5-a`)

If these parameters are already known from an upstream caller (e.g., `exec-remote` or `deploy-cluster`), use them directly — do NOT re-ask the user. Only prompt interactively when this skill is invoked standalone and the parameters are not yet known.

## Operations

### 1. Create Cluster

Creates a new GKE cluster with TPU nodepool using Pathways.

**Command:**
```bash
xpk cluster create-pathways \
  --cluster $CLUSTER_NAME \
  --num-slices=$NUM_SLICES \
  --tpu-type=$TPU_TYPE \
  --zone=$ZONE \
  --spot \
  --project=tpu-service-473302
```

**Interactive Flow:**
1. Prompt user for CLUSTER_NAME
2. Prompt user for TPU_TYPE
3. Prompt user for NUM_SLICES
4. Prompt user for ZONE
5. Validate TPU availability in the specified zone
6. Check if cluster name already exists in GKE (prevents conflicts)
7. Execute cluster creation

### 2. Delete Cluster

Deletes an existing GKE cluster.

**Command:**
```bash
xpk cluster delete \
  --cluster $CLUSTER_NAME \
  --zone=$ZONE \
  --project=tpu-service-473302
```

**Interactive Flow:**
1. Query GKE for all live clusters
2. Display cluster list with current status
3. Prompt user to select a cluster
4. Verify cluster still exists (multi-user safety check)
5. Confirm deletion
6. Execute cluster deletion

### 3. List Clusters

Lists all managed clusters.

**Command:**
```bash
xpk cluster list
```

**Real-time Query:**
- Queries GKE directly using `gcloud container clusters list`
- Shows current status, location, and node pool information
- No local cache - always up-to-date

### 4. Describe Cluster

Shows detailed information about a specific cluster.

**Command:**
```bash
xpk cluster describe \
  --cluster $CLUSTER_NAME \
  --zone=$ZONE
```

## TPU Availability Handling

When a specified ZONE doesn't support the requested TPU_TYPE:

1. Automatically fetch available zones from [TPU regions documentation](https://docs.cloud.google.com/tpu/docs/regions-zones)
2. Suggest alternative zones that support the TPU type
3. Allow user to select a different zone or cancel

## Multi-User Safety Features

This skill is designed for multi-user environments:

1. **No Local Caching**: All cluster information is queried from GKE in real-time
2. **Existence Checks**: Before creating, checks if cluster name already exists
3. **Pre-deletion Verification**: Before deleting, verifies cluster still exists
4. **Location Validation**: Automatically uses actual cluster location if different from user input
5. **Concurrent Operation Safe**: Multiple users can operate simultaneously without conflicts

## Parameter Passing

This skill can be invoked in two ways:

1. **Standalone** (`/apply-resource create`): Prompt user for all required parameters interactively.
2. **From pipeline** (called by `deploy-cluster` or `exec-remote`): Parameters (CLUSTER_NAME, TPU_TYPE, NUM_SLICES, ZONE) are already known — use them directly without prompting.

## Error Handling

- **Missing prerequisites**: Check if xpk, gcloud, kubectl are installed
- **Authentication errors**: Verify `gcloud auth list`
- **Zone unavailability**: Automatically suggest alternative zones
- **Quota errors**: Display quota information and suggest solutions
- **Network errors**: Retry with exponential backoff

## Useful Resources

- [TPU Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm?hl=zh-cn#tpu-system-architecture)
- [TPU Regions and Zones](https://docs.cloud.google.com/tpu/docs/regions-zones)
- [XPK Cluster Documentation](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/clusters.md)
