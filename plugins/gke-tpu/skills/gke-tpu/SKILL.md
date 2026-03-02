---
name: gke-tpu
description: "Manage GKE clusters and run TPU workloads on Google Kubernetes Engine. Use this skill when the user wants to: (1) Create or configure a GKE cluster for TPU workloads, (2) Create workload policies for multi-host TPU setups, (3) Create TPU node pools (single-host or multi-host), (4) Create and submit Kubernetes Jobs on TPU nodes, (5) Interact with running pods via kubectl (exec, logs, port-forward), (6) Run services, benchmarks, or tests on GKE TPU pods."
---

# GKE TPU Skill

## Overview

This skill manages GKE clusters for TPU workloads using `gcloud` and `kubectl`. It supports autonomous end-to-end deployment: checking prerequisites, creating missing resources, deploying jobs, and monitoring to completion.

**Default project/region (override with user input):**
```
PROJECT_ID=tpu-service-473302
REGION=us-central1
ZONE=us-central1-c
CLUSTER_NAME=tpu7x-cluster
```

---

## Feature 1: Deploy Workload (End-to-End)

When asked to deploy a TPU job, execute all steps in order. Do not skip steps.

### Step 1 — Resolve Chip Count → Topology

Use this table to determine topology, machine count, and host type from the requested chip count:

| Chips | Host type  | Topology | Machines |
|-------|------------|----------|----------|
| 4     | single     | 2x2x1    | 1        |
| 16    | multi      | 2x2x4    | 4        |
| 32    | multi      | 2x2x8    | 8        |

If the user specifies a topology directly, use it. If they specify chips, look up the topology above.

---

### Step 2 — Check Nodepool Exists

Before creating anything, check whether a compatible nodepool already exists:

```bash
# Check for nodes with the required topology label
kubectl get nodes -l cloud.google.com/gke-tpu-topology=<topology> --no-headers 2>/dev/null

# List all nodepools with their config
gcloud container node-pools list \
  --cluster=${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --format="table(name,config.machineType,autoscaling.totalMaxNodeCount)"
```

**Decision:** If `kubectl get nodes` returns ≥1 node with the right topology label → nodepool is ready, skip Step 3. Otherwise proceed to Step 3.

See `references/gke-setup.md § Checking Existing Resources` for checking workload-policy existence.

---

### Step 3 — Create Nodepool (if missing)

**Single-host (topology = 2x2x1):** Create nodepool directly, no workload-policy needed.

```bash
gcloud container node-pools create tpu7x-4-np-flex \
  --cluster=${CLUSTER_NAME} \
  --machine-type=tpu7x-standard-4t \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --project=${PROJECT_ID} \
  --num-nodes=0 \
  --flex-start \
  --reservation-affinity=none \
  --enable-autoscaling \
  --total-min-nodes=0 \
  --total-max-nodes=4
```

**Multi-host (topology = 2x2x4 or 2x2x8):** Check if workload-policy exists first, create if missing, then create nodepool.

```bash
# Check existing workload policies
gcloud compute resource-policies list \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --filter="selfLink~workload-policy" \
  --format="table(name,groupPlacementPolicy.vmCount)"
```

If policy for this topology is missing, create it (see `references/gke-setup.md § Multi-Host Setup`), then create the nodepool with `--placement-policy`.

Set `--max-nodes` to the machine count for the topology (e.g., 4 for 2x2x4, 8 for 2x2x8).

---

### Step 4 — Generate and Apply Job YAML

**Single-host job:** Base on `assets/single-host-job.yaml`. Set:
- `metadata.name`: job name from user
- `nodeSelector.cloud.google.com/gke-tpu-topology`: topology string (e.g., `2x2x1`)
- `resources.requests/limits.google.com/tpu`: chip count (4 for single-host)
- `command`: workload command

**Multi-host job:** Base on `assets/multi-host-job.yaml`. Set:
- `metadata.name` (Job and Service — must be consistent)
- `spec.parallelism` / `spec.completions`: machine count
- `nodeSelector.cloud.google.com/gke-tpu-topology`: topology string
- `resources.requests/limits.google.com/tpu`: 4 (per machine)
- `spec.template.spec.subdomain`: headless Service name
- `command`: workload command

Apply:
```bash
kubectl apply -f <job-file>.yaml
```

---

### Step 5 — Monitor to Running or Report Failure

After applying, monitor the job. See `references/monitoring.md` for the full monitoring loop and failure detection logic.

**Summary:**
- Poll every 15s for up to ~8 min
- Success: all pods show `Running`
- On failure: report pod events, whether scale-up was triggered, whether max-nodes was hit, and a suggested action

---

## Feature 2: Inference Service

*Details TBD — this capability will be added in a future update.*

---

## Feature 3: Setup Environment on Pods

When asked to set up the environment on running pods, execute steps in order.

### Step 1 — Get Running Pod Names

```bash
kubectl get pods -l job-name=<job-name> --field-selector=status.phase=Running -o name --no-headers
```

If no pods are Running, tell the user to deploy the job first (Feature 1).

### Step 2 — Clone and Install on Each Pod

For each pod, run the full setup sequence (clone → checkout branch → install).
See references/kubectl-guide.md § Setup Environment Across All Pods for exact commands.

Parameters (ask user if not provided):
- job-name: the Kubernetes job name
- branch: git branch to check out (e.g., epic/data-parallelism-rebase)
- repo: default https://github.com/sgl-project/sglang-jax.git

### Step 3 — Verify Installation

After setup, verify on the first pod:

```bash
kubectl exec <first-pod> -- python -c "import sglang_jax; print('OK')"
```

Report success or show the import error.

---

## Other Operations

### Connect to Cluster

Before using `kubectl`, connect to the cluster:

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID}
```

### Pod Interaction

```bash
# Exec into a pod
kubectl exec -it <pod-name> -- /bin/bash

# View logs
kubectl logs <pod-name>
kubectl logs -f <pod-name>   # follow

# Port-forward
kubectl port-forward <pod-name> <local-port>:<remote-port>
```

### Job/Pod Status

```bash
kubectl get jobs
kubectl describe job <job-name>
kubectl get pods -l job-name=<job-name>
kubectl describe pod <pod-name>
```

### Cleanup

```bash
# Delete job and its pods
kubectl delete job <job-name>

# Delete nodepool
gcloud container node-pools delete <pool-name> \
  --cluster=${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID}
```

---

## Resources

- **references/gke-setup.md** — Cluster creation, workload policy, node pool commands, existence checks
- **references/kubectl-guide.md** — kubectl interaction patterns (exec, port-forward, services, tests)
- **references/monitoring.md** — Pod monitoring loop and failure detection logic
- **assets/single-host-job.yaml** — Job template for single-host TPU (≤4 chips)
- **assets/multi-host-job.yaml** — Job + headless Service template for multi-host TPU (>4 chips)
