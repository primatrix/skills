# GKE Setup Reference

## Variables

```bash
CLUSTER_NAME=tpu7x-cluster
CPU_MACHINE_TYPE=n1-standard-8
REGION=us-central1
ZONE=us-central1-c
PROJECT_ID=tpu-service-473302
```

## Cluster Creation

```bash
gcloud container clusters create ${CLUSTER_NAME} \
  --release-channel=rapid \
  --project=${PROJECT_ID} \
  --machine-type=${CPU_MACHINE_TYPE} \
  --location=${REGION}
```

## Single-Host Node Pool

For single-host workloads: 1 TPU machine, up to 4 chips (`tpu7x-standard-4t`).

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

- `--flex-start`: use Flex Start (spot-like) capacity
- `--total-max-nodes=4`: max 4 nodes × 4 chips = 16 chips total capacity

## Multi-Host Setup

Multi-host requires a **workload policy** before creating the node pool.

### Step 1: Create Workload Policy

```bash
WORKLOAD_POLICY_NAME=tpu7x-workload-policy-16
TOPOLOGY=2x2x4

gcloud compute resource-policies create workload-policy ${WORKLOAD_POLICY_NAME} \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=${TOPOLOGY} \
  --project=${PROJECT_ID} \
  --region=${REGION}
```

Topology examples:
- `2x2x4` → 16 chips across 4 machines (4 chips each)
- `2x2x8` → 32 chips across 8 machines

### Step 2: Create Multi-Host Node Pool

```bash
gcloud container node-pools create tpu7x-16-np-flex \
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
  --placement-policy=${WORKLOAD_POLICY_NAME} \
  --max-nodes=4
```

- `--placement-policy`: references the workload policy created above
- `--max-nodes=4`: for 16-chip setup (4 machines × 4 chips)

## Checking Existing Resources

Before creating nodepools or workload policies, check if they already exist.

### Check Existing Node Pools and Their Topologies

```bash
# List all nodepools with machine type and autoscaling config
gcloud container node-pools list \
  --cluster=${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --format="table(name,config.machineType,autoscaling.totalMaxNodeCount)"

# Check for nodes already labeled with a specific topology
kubectl get nodes -l cloud.google.com/gke-tpu-topology=<topology> --no-headers 2>/dev/null
# e.g.: kubectl get nodes -l cloud.google.com/gke-tpu-topology=2x2x4 --no-headers
```

If `kubectl get nodes` returns ≥1 result for the target topology, the nodepool is present and autoscaled nodes are ready — skip nodepool creation.

### Check Existing Workload Policies

```bash
# List all workload policies in the region
gcloud compute resource-policies list \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --filter="selfLink~workload-policy" \
  --format="table(name,groupPlacementPolicy.vmCount)"

# Describe a specific policy
gcloud compute resource-policies describe <policy-name> \
  --project=${PROJECT_ID} \
  --region=${REGION}
```

If a workload policy already exists for the required topology (check `vmCount` matches the machine count), reuse it — pass its name as `--placement-policy` when creating the nodepool.

## Node Pool Management

```bash
# List node pools
gcloud container node-pools list --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID}

# Describe a node pool
gcloud container node-pools describe <pool-name> --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID}

# Delete a node pool
gcloud container node-pools delete <pool-name> --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID}
```

## GCS Fuse Service Account

Jobs that mount GCS buckets require a Kubernetes service account `gcs-account` with Workload Identity configured. If not already set up:

```bash
# Create k8s service account
kubectl create serviceaccount gcs-account

# Bind to GCP service account with GCS access
kubectl annotate serviceaccount gcs-account \
  iam.gke.io/gcp-service-account=<gcp-sa>@${PROJECT_ID}.iam.gserviceaccount.com
```
