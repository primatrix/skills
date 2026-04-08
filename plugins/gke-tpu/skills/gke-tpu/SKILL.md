---
name: gke-tpu
description: Manage GKE-based TPU workloads — create pods/jobs via kubectl, sync code, and run multi-process benchmarks. Use when the user wants to create/manage/run TPU workloads on GKE. Reads config from gke.toml in the current working directory.
---

# GKE TPU Skill

Manage GKE-based TPU workloads via `kubectl`. Config-driven via `gke.toml` in the current working directory (CWD).

## Commands

| Command | Description | Reference |
|---|---|---|
| `create` | Create TPU pod (single-host) or job (multi-host) | [references/create.md](references/create.md) |
| `sync` | Sync code + install deps to all containers | [references/sync.md](references/sync.md) |
| `run` | Execute script on multi-process TPU | [references/run.md](references/run.md) |
| `status` | Check pod/workload status | [references/status.md](references/status.md) |

**Read the relevant reference file for the user's command before executing.**

## Configuration

Read `gke.toml` from the current working directory at the start of every command. This keeps configs isolated per worktree/session. Never hardcode project/cluster/zone/bucket. If `gke.toml` does not exist in CWD, prompt the user to create one.

```toml
[gke]
project = "<your-gcp-project>"
cluster = "<your-cluster-name>"
zone = "<your-zone>"

[tpu]
accelerator = "tpu-v6e-slice"   # nodeSelector accelerator label
topology = "4x4"                # TPU topology (determines chip count)
chips_per_node = 4              # google.com/tpu resource per container
machine_type = "ct6e-standard-4t"  # GKE machine type
max_nodes = 4                   # autoscaling max for node pool
reservation = ""                # optional: reservation name for reserved capacity

[workload]
name = "my-workload"
docker_image = "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1"
service_account = "gcs-account"

[storage]
type = "gcsfuse"                # "gcsfuse" or "pvc"
mount_path = "/inference-models"

# --- gcsfuse-specific (only when type = "gcsfuse") ---
bucket = "inference-model-storage-poc-tpu"
mount_options = "implicit-dirs,file-cache:max-parallel-downloads:256,file-cache:enable-parallel-downloads:true,file-cache:download-chunk-size-mb:128,file-cache:max-size-mb:81920,file-cache:parallel-downloads-per-file:512,metadata-cache:ttl-secs:-1,metadata-cache:stat-cache-max-size-mb:-1,metadata-cache:type-cache-max-size-mb:-1,file-cache:cache-file-for-range-read:true,file-system:kernel-list-cache-ttl-secs:-1,read_ahead_kb=1024"

# --- pvc-specific (only when type = "pvc") ---
# pvc_name = "my-model-pvc"       # name of existing PersistentVolumeClaim
# read_only = false                # mount as read-only (default: false)
# gcsfuse_backed = false           # true if PVC's StorageClass uses GCS Fuse CSI driver
                                   # when true: adds gke-gcsfuse/volumes annotation + gke-gcsfuse-cache volume
                                   # when false: plain PVC mount, no sidecar needed

[repo]
git_url = "https://github.com/sgl-project/sglang-jax.git"
remote_path = "/tmp/sglang-jax"
install_cmd = "pip install -e ."            # run in repo root
# requirements_file = "requirements-tpu.txt"  # optional: extra deps file (relative to repo root)
```

### TPU Topology Reference

See [references/tpu-topologies.md](references/tpu-topologies.md) for supported topologies (v6e and v7x), machine types, and chips-per-node mappings.

**Single-host** (1 VM): use Pod. **Multi-host** (>1 VM): use Indexed Job + headless Service.

## Critical Rules

1. **Single vs multi-host**: Determine from topology. `chips / chips_per_node = hosts`. If hosts > 1, must use Job + headless Service.
2. **Storage**: Check `storage.type`:
   - `gcsfuse`: mount with `gke-gcsfuse/volumes: "true"` annotation and `gke-gcsfuse-cache` emptyDir volume.
   - `pvc`: mount the existing PVC directly. The PVC must already exist in the namespace.
     - If `storage.gcsfuse_backed = true`: the PVC's StorageClass uses GCS Fuse CSI driver under the hood — still needs `gke-gcsfuse/volumes: "true"` annotation and `gke-gcsfuse-cache` emptyDir volume, otherwise mount will fail with "failed to find the sidecar container".
     - If `storage.gcsfuse_backed = false` (default): plain PVC mount, no gcsfuse annotation or cache volume needed.
3. **Simultaneous launch**: For multi-host, `jax.distributed.initialize()` must run in all pods at the same time.
4. **Same code path**: ALL processes must execute the SAME jitted computations.
5. **Docker image must match JAX version** in pyproject.toml.
6. **Reservations**: If `tpu.reservation` is set, use `--reservation-affinity=specific` with fixed node count (no autoscaling).
7. **Multi-host verification**: `import jax` blocks on multi-host TPU. Use `/dev/vfio/` for per-pod hardware check, `run` command for full JAX cluster verification.

## Prerequisites

See [references/prerequisites.md](references/prerequisites.md) for gcloud/kubectl install steps.

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for common issues.
