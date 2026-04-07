# sync — Sync Code & Install Dependencies

All commands must run on ALL pods/containers (multi-host TPU pods have independent filesystems).

## Two sync modes

| Mode | When to use | How |
|---|---|---|
| `copy` | No git remote, or need to sync uncommitted changes | `kubectl cp` entire codebase (excluding `.git` and `.gitignore`'d files) |
| `git` | Code is pushed to GitHub, pods have network access | Local `git push`, then `git pull` on each pod |

Default to `git` mode. Use `copy` when syncing uncommitted/local-only changes.

## Get pod list

```bash
# Single-host:
PODS=<workload.name>

# Multi-host Job:
PODS=$(kubectl get pods -l job-name=<workload.name> -o jsonpath='{.items[*].metadata.name}')
```

## Mode 1: copy (full codebase)

Creates a clean tarball excluding `.git` and `.gitignore`'d files, then copies to each pod.

```bash
# 1. Create tarball locally (from repo root)
tar czf /tmp/sync.tar.gz --exclude-from=.gitignore --exclude='.git' .

# 2. Copy and extract on each pod
for POD in $PODS; do
  kubectl cp /tmp/sync.tar.gz $POD:<repo.remote_path>/sync.tar.gz -c <workload.name>
  kubectl exec $POD -c <workload.name> -- bash -c \
    'cd <repo.remote_path> && tar xzf sync.tar.gz && rm sync.tar.gz'
done

rm /tmp/sync.tar.gz
```

## Mode 2: git (push + pull)

Ensure local changes are committed and pushed first.

```bash
# 1. Push locally
git push origin <branch>

# 2. Pull on each pod
for POD in $PODS; do
  kubectl exec $POD -c <workload.name> -- bash -c \
    'cd <repo.remote_path> && git pull origin <branch>'
done
```

If the repo hasn't been cloned yet on the pods:

```bash
for POD in $PODS; do
  kubectl exec $POD -c <workload.name> -- bash -c \
    'if [ ! -d "<repo.remote_path>" ]; then git clone --depth 1 <repo.git_url> <repo.remote_path>; fi'
done
```

## Install dependencies (after either mode)

```bash
for POD in $PODS; do
  kubectl exec $POD -c <workload.name> -- bash -c \
    'cd <repo.remote_path> && <repo.install_cmd>'
done
```

If `repo.requirements_file` is set:

```bash
for POD in $PODS; do
  kubectl exec $POD -c <workload.name> -- bash -c \
    'cd <repo.remote_path> && pip install -r <repo.requirements_file>'
done
```
