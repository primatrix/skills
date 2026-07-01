---
name: gke-tpu
description: Use when managing GKE TPU workloads, TPU nodepools, TPU topology, reservations, or Kubernetes Jobs on TPU v6e/v7x.
---

# GKE TPU

Use this skill as a small planning/rendering aid for GKE TPU workloads. The agent still owns `gcloud` / `kubectl` execution and live diagnosis.

## Commands

Run `python3 scripts/gke_tpu.py <command> [--config path | --profile name]`. All script output is JSON.

| Command | Purpose |
|---|---|
| `init` | Print a TOML template as JSON; does not write files. |
| `validate` | Check config shape and breaking-change violations. |
| `plan-nodepool` | Compute expected TPU nodepool spec and `gcloud` argv. |
| `render-workload` | Render one multi-doc Job manifest to `/tmp/gke-tpu/<workload>/workload.yaml`. |
| `delete-workload-plan` | Emit resource-name deletes for Job/Service/ConfigMap. |
| `delete-nodepool-plan` | Emit nodepool and workload-policy delete argv. |

Read only the reference needed for the user's action:
- Config schema: [references/config.md](references/config.md)
- Nodepool planning: [references/nodepool.md](references/nodepool.md)
- Workload rendering/apply: [references/workload.md](references/workload.md)
- Cleanup: [references/cleanup.md](references/cleanup.md)
- Topology hints: [references/topologies.md](references/topologies.md)

## Hard Boundaries

- Do not sync code, copy launchers into pods, run commands inside existing pods, or wrap `kubectl status/logs/describe`; use normal Kubernetes knowledge for that.
- Do not manage app env, secrets, repo clone paths, install commands, or requirements files.
- Do not write `.claude`, `.codex`, `~/.agents`, or other agent-private state.
- Do not write repo config unless the user explicitly asks. `render-workload` may write only `/tmp/gke-tpu/...`.
- Every write action is plan-first and needs the action-specific confirmation token from JSON.
- Every `kubectl` command must include explicit `--context <context>` and `-n <namespace>`.

## Execution Pattern

1. Resolve config from `--config`, `--profile`, `gke-tpu.toml`, or `configs/gke-tpu/default.toml`.
2. Run `validate`.
3. For nodepools, run `plan-nodepool`, query actual GKE state yourself, then execute the JSON `argv` only after confirmation.
4. For workloads, run `render-workload`, show the JSON plan, then `kubectl apply` the rendered manifest only after confirmation.
5. Use `delete-workload-plan` / `delete-nodepool-plan` for destructive actions.

## Breaking Change

This skill no longer supports `gke.toml`, `[repo]`, `sync`, `run`, or `status` workflows. Put code in the image, mounted storage, or command-time `git clone`; use batch Jobs for normal runs and interactive Jobs only for debugging.
