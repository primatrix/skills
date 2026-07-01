# Config Schema

Config input is TOML. Resolution order:

1. `--config <path>`
2. `--profile <name>` -> `configs/gke-tpu/<name>.toml`
3. `gke-tpu.toml`
4. `configs/gke-tpu/default.toml`

`init` prints a JSON object containing a template; it does not write the file.

```toml
[gke]
project = "your-gcp-project"
cluster = "your-cluster"
zone = "us-east5-b"

[k8s]
namespace = "default"
context = "" # optional; defaults to gke_<project>_<zone>_<cluster>

[tpu]
accelerator = "tpu-v6e-slice"
topology = "4x4"
chips_per_node = 4
machine_type = "ct6e-standard-4t"
max_nodes = 4
nodepool = "" # optional; derived as <workload>-<topology>-np
reservation = ""

[workload]
name = "my-workload"
image = "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1"
service_account = "gcs-account"
mode = "batch" # batch | interactive

[storage]
type = "none" # none | gcsfuse | pvc
```

Optional storage blocks:

```toml
[storage]
type = "gcsfuse"
mount_path = "/models"

[storage.gcsfuse]
bucket = "model-bucket"
mount_options = "" # optional, passed through when set

[storage]
type = "pvc"
mount_path = "/models"

[storage.pvc]
name = "models-pvc"
read_only = false
gcsfuse_backed = false
```

Batch run targets:

```toml
[run]
target = "script" # script | module | command
script = "benchmarks/foo.py"
args = ["--batch-size", "8"]
distributed_init = true # optional

[run]
target = "command"
command = ["bash", "-lc", "git clone ... && python train.py"]
```

`[repo]`, install commands, requirements files, env, and secrets are intentionally unsupported.
