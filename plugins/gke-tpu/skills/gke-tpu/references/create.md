# create — Create TPU Workload

Full flow: check node pool → create if needed → create workload → verify running.

## Step 1: Connect to cluster

```bash
gcloud container clusters get-credentials <gke.cluster> --zone=<gke.zone> --project=<gke.project>
```

## Step 2: Check if matching node pool exists

```bash
gcloud container node-pools list \
  --cluster=<gke.cluster> --zone=<gke.zone> --project=<gke.project> \
  --format="table(name,config.machineType,placementPolicy.tpuTopology)"
```

Look for a node pool with matching `tpu.machine_type` and `tpu.topology`. If found, use its name as `<nodepool>`. If not, create one in Step 3.

## Step 3: Create node pool (skip if exists)

### Determine single-host vs multi-host

Calculate total chips from topology (e.g. `4x4` = 16 chips). `chips / tpu.chips_per_node = num_hosts`.
- **1 host** (e.g. `2x2`) → single-host node pool, no placement policy needed
- **>1 hosts** (e.g. `4x4`) → multi-host node pool, needs placement policy

### Single-host node pool

```bash
gcloud container node-pools create <nodepool> \
  --cluster=<gke.cluster> \
  --machine-type=<tpu.machine_type> \
  --location=<gke.zone> \
  --project=<gke.project> \
  --num-nodes=0 \
  --enable-autoscaling \
  --total-min-nodes=0 \
  --total-max-nodes=<tpu.max_nodes>
```

If using a reservation, add:
```
  --reservation-affinity=specific \
  --reservation=<tpu.reservation> \
  --num-nodes=<num_hosts>
```
And remove `--enable-autoscaling`, `--total-min-nodes`, `--total-max-nodes` (reserved nodes use fixed count).

### Multi-host node pool

First create a workload policy:

```bash
gcloud compute resource-policies create workload-policy <nodepool>-policy \
  --type HIGH_THROUGHPUT \
  --accelerator-topology <tpu.topology> \
  --project <gke.project> \
  --region <gke.region>
```

Then create the node pool with the policy:

```bash
gcloud container node-pools create <nodepool> \
  --cluster=<gke.cluster> \
  --machine-type=<tpu.machine_type> \
  --location=<gke.zone> \
  --project=<gke.project> \
  --num-nodes=0 \
  --enable-autoscaling \
  --total-min-nodes=0 \
  --max-nodes=<tpu.max_nodes> \
  --placement-policy=<nodepool>-policy
```

If using a reservation, replace autoscaling flags with:
```
  --reservation-affinity=specific \
  --reservation=<tpu.reservation> \
  --num-nodes=<num_hosts>
```

**Note**: `<gke.region>` is the region part of zone (e.g. `us-east5` from `us-east5-b`).

### Checking reservations

To find available reservations:

```bash
gcloud compute reservations list --project=<gke.project>
gcloud compute reservations describe <reservation-name> --zone=<gke.zone> --project=<gke.project>
```

If `specificReservationRequired: true`, nodes **must** use `--reservation-affinity=specific`.

## Step 4: Create workload

### Single-host → Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: <workload.name>
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: <tpu.accelerator>
    cloud.google.com/gke-tpu-topology: <tpu.topology>
    cloud.google.com/gke-nodepool: <nodepool>
  containers:
  - name: <workload.name>
    image: <workload.docker_image>
    command: ["sleep", "infinity"]
    resources:
      requests:
        google.com/tpu: <tpu.chips_per_node>
      limits:
        google.com/tpu: <tpu.chips_per_node>
    volumeMounts:
    - name: gcs-fuse-csi-ephemeral
      mountPath: <storage.mount_path>
      readOnly: false
    - name: dev-shm
      mountPath: /dev/shm
  serviceAccountName: <workload.service_account>
  volumes:
  - name: dev-shm
    emptyDir:
      medium: Memory
  - name: gke-gcsfuse-cache
    emptyDir:
      medium: Memory
  - name: gcs-fuse-csi-ephemeral
    csi:
      driver: gcsfuse.csi.storage.gke.io
      readOnly: false
      volumeAttributes:
        skipCSIBucketAccessCheck: "true"
        gcsfuseMetadataPrefetchOnMount: "true"
        bucketName: <storage.bucket>
        mountOptions: "<storage.mount_options>"
```

### Multi-host → headless Service + Indexed Job

`parallelism` and `completions` = `num_hosts`.

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: <workload.name>-headless-svc
spec:
  clusterIP: None
  selector:
    job-name: <workload.name>
---
apiVersion: batch/v1
kind: Job
metadata:
  name: <workload.name>
spec:
  completionMode: Indexed
  parallelism: <num_hosts>
  completions: <num_hosts>
  backoffLimit: 0
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      subdomain: <workload.name>-headless-svc
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: <tpu.accelerator>
        cloud.google.com/gke-tpu-topology: <tpu.topology>
        cloud.google.com/gke-nodepool: <nodepool>
      containers:
      - name: <workload.name>
        image: <workload.docker_image>
        command: ["sleep", "infinity"]
        resources:
          requests:
            google.com/tpu: <tpu.chips_per_node>
          limits:
            google.com/tpu: <tpu.chips_per_node>
        volumeMounts:
        - name: gcs-fuse-csi-ephemeral
          mountPath: <storage.mount_path>
          readOnly: false
        - name: dev-shm
          mountPath: /dev/shm
      serviceAccountName: <workload.service_account>
      volumes:
      - name: dev-shm
        emptyDir:
          medium: Memory
      - name: gke-gcsfuse-cache
        emptyDir:
          medium: Memory
      - name: gcs-fuse-csi-ephemeral
        csi:
          driver: gcsfuse.csi.storage.gke.io
          readOnly: false
          volumeAttributes:
            skipCSIBucketAccessCheck: "true"
            gcsfuseMetadataPrefetchOnMount: "true"
            bucketName: <storage.bucket>
            mountOptions: "<storage.mount_options>"
```

## Step 5: Apply, wait, and verify

```bash
kubectl apply -f /tmp/<workload.name>.yaml
```

Wait for all pods to be Running (timeout 5 minutes):

```bash
# For Pod:
kubectl wait --for=condition=Ready pod/<workload.name> --timeout=300s

# For Job:
kubectl wait --for=condition=Ready pod -l job-name=<workload.name> --timeout=300s
```

If pods stay Pending, diagnose:

```bash
kubectl describe pod <POD_NAME> | grep -A 5 "Events:"
```

Common Pending causes:
- **Insufficient google.com/tpu**: node pool is full, need to free resources or increase max_nodes
- **didn't match node affinity/selector**: wrong nodepool/topology/accelerator label
- **No preemption victims**: spot nodes not available

Verify TPU devices once Running:

### Single-host verification

```bash
kubectl exec <POD_NAME> -c <workload.name> -- python3 -c "import jax; print('TPU devices:', jax.device_count())"
```

### Multi-host verification

On multi-host TPU, even `import jax` blocks waiting for all hosts to initialize together. Use two-step verification:

**Step 1: Hardware check (per-pod, non-blocking)**

```bash
kubectl exec <POD_NAME> -c <workload.name> -- ls /dev/vfio/
```

Each pod should show devices `0, 1, 2, 3` (one per chip). This confirms TPU hardware is attached.

**Step 2: JAX cluster check (all pods simultaneously)**

All pods must run `jax.distributed.initialize()` at the same time. Use the `run` command ([references/run.md](run.md)) to execute across all pods:

```python
import jax
jax.distributed.initialize()
print(f"Global devices: {jax.device_count()}, Local devices: {jax.local_device_count()}")
```

Expected output: `Global devices: <total_chips>`, `Local devices: <tpu.chips_per_node>`.

## Cleanup: Delete workload and node pool

When workload is no longer needed, delete in order:

```bash
# 1. Delete workload
kubectl delete job/<workload.name> 2>/dev/null; kubectl delete pod/<workload.name> 2>/dev/null
kubectl delete svc/<workload.name>-headless-svc 2>/dev/null

# 2. Wait for nodes to drain (autoscaler scales to 0)
# Check with:
kubectl get nodes -l cloud.google.com/gke-nodepool=<nodepool>

# 3. Delete node pool (once nodes are gone or if you want immediate cleanup)
gcloud container node-pools delete <nodepool> \
  --cluster=<gke.cluster> --zone=<gke.zone> --project=<gke.project> --quiet

# 4. Delete workload policy if multi-host
gcloud compute resource-policies delete <nodepool>-policy \
  --project=<gke.project> --region=<gke.region> --quiet
```
