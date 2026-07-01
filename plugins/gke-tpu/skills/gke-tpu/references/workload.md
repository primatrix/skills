# Workload Rendering

`render-workload` writes one multi-doc manifest to `/tmp/gke-tpu/<workload>/workload.yaml` and outputs JSON with the write path plus `kubectl apply` argv.

All workloads are Kubernetes Jobs:

- `parallelism = hosts`
- `completions = hosts`
- `hosts > 1` adds a headless Service and Job `subdomain`
- single-host Jobs do not get a Service

Modes:

- `batch`: recommended for training, eval, and benchmarks.
- `interactive`: debug shell only; command defaults to `["sleep", "infinity"]`.

Batch targets:

- `script` / `module`: render a launcher ConfigMap and run `python3 -u /opt/gke-tpu/launcher.py`.
- `command`: use the command directly; no launcher.
- Multi-host script/module defaults `jax.distributed.initialize()` on. Set `distributed_init = false` when the framework owns initialization.

Storage:

- `none`: no model/data mount.
- `gcsfuse`: renders the required GCS Fuse annotation, cache volume, and CSI volume.
- `pvc`: mounts an existing PVC. If `gcsfuse_backed = true`, also renders the GCS Fuse annotation and cache volume.

The skill does not sync code after workload creation. Code must already be available through the image, mounted storage, or the batch command.
