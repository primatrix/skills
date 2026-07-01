# Nodepool Planning

Use `plan-nodepool` to compute expected nodepool shape and default `gcloud` argv. The script does not call `gcloud`.

Agent responsibilities:

1. Query current nodepools with `gcloud container node-pools list --cluster ... --location ... --project ...`.
2. Compare machine type and topology with JSON `data`.
3. If no suitable nodepool exists, show the plan and require the JSON confirmation token.
4. Execute the JSON `argv`, adjusting only when live `gcloud` output requires it.

Rules:

- `hosts = product(topology dims) / chips_per_node`.
- Missing `[tpu].nodepool` is derived as `<workload>-<topology>-np` and emits a warning.
- `hosts > 1` needs a workload policy before nodepool creation.
- `reservation` uses fixed node count and no autoscaling flags.
- Nodepool creation and workload apply are separate actions with separate confirmations.
