# kubectl Interaction Guide

## Connect to Cluster

```bash
gcloud container clusters get-credentials tpu7x-cluster \
  --region=us-central1 \
  --project=tpu-service-473302
```

## Exec Into a Pod

```bash
# Interactive shell
kubectl exec -it <pod-name> -- bash

# For multi-host jobs, target a specific pod by index
kubectl exec -it <job-name>-0 -- bash

# Run a single command
kubectl exec <pod-name> -- python -c "import jax; print(jax.devices())"
```

## Logs

```bash
# Current logs
kubectl logs <pod-name>

# Follow logs
kubectl logs -f <pod-name>

# All pods for a job
kubectl logs -l job-name=<job-name>

# Previous container (if restarted)
kubectl logs <pod-name> --previous
```

## Port Forwarding

```bash
# Forward local port to pod port (e.g., for Jupyter or HTTP server)
kubectl port-forward <pod-name> 8888:8888

# Forward to a service
kubectl port-forward svc/<service-name> 8080:80
```

## Running Services

To expose a running service (e.g., inference server on port 8431):

```bash
# Create a ClusterIP service pointing to a pod's label
kubectl expose pod <pod-name> --port=8431 --name=<svc-name>

# Or port-forward for local testing
kubectl port-forward <pod-name> 8431:8431
```

## Copying Files To/From Pods

```bash
# Copy to pod
kubectl cp ./local-file.py <pod-name>:/workspace/local-file.py

# Copy from pod
kubectl cp <pod-name>:/workspace/output.txt ./output.txt
```

## Running Tests

```bash
# Run pytest inside a pod
kubectl exec <pod-name> -- bash -c "cd /workspace && python -m pytest tests/"

# Stream test output
kubectl exec -it <pod-name> -- bash -c "cd /workspace && python -m pytest tests/ -v"
```

## Monitoring

```bash
# Get pods with node info
kubectl get pods -o wide

# Get pods for a specific job
kubectl get pods -l job-name=<job-name>

# Describe pod (useful for scheduling issues)
kubectl describe pod <pod-name>

# Watch pod status
kubectl get pods -w

# Get events (useful for debugging failures)
kubectl get events --sort-by='.lastTimestamp'
```

## Cleanup

```bash
# Delete a job and its pods
kubectl delete job <job-name>

# Delete a service
kubectl delete svc <service-name>

# Delete all completed pods
kubectl delete pods --field-selector=status.phase=Succeeded
```

## Setup Environment Across All Pods

### Single-pod setup (sequential, visible output per pod)

```bash
REPO=https://github.com/sgl-project/sglang-jax.git
BRANCH=epic/data-parallelism-rebase
INSTALL_DIR=/workspace/sglang-jax

kubectl exec <pod-name> -- bash -c "
  git clone -b ${BRANCH} ${REPO} ${INSTALL_DIR} 2>/dev/null || \
    (cd ${INSTALL_DIR} && git fetch && git checkout ${BRANCH} && git pull);
  cd ${INSTALL_DIR} && pip install -e 'python[tpu]'
"
```

- `git clone -b <branch>` — clones directly onto the desired branch
- The `||` fallback handles the case where the directory already exists: fetch + checkout + pull
- `pip install -e 'python[tpu]'` — installs from the `python/` subdirectory with TPU extras

### All pods for a job (loop)

```bash
REPO=https://github.com/sgl-project/sglang-jax.git
BRANCH=epic/data-parallelism-rebase
INSTALL_DIR=/workspace/sglang-jax
JOB_NAME=<job-name>

for POD in $(kubectl get pods -l job-name=${JOB_NAME} --field-selector=status.phase=Running -o name --no-headers | sed 's|pod/||'); do
  echo "=== Setting up ${POD} ==="
  kubectl exec ${POD} -- bash -c "
    git clone -b ${BRANCH} ${REPO} ${INSTALL_DIR} 2>/dev/null || \
      (cd ${INSTALL_DIR} && git fetch && git checkout ${BRANCH} && git pull);
    cd ${INSTALL_DIR} && pip install -e 'python[tpu]'
  "
  echo "=== Done ${POD} ==="
done
```

### Parallel setup across all pods (faster for multi-host)

```bash
for POD in $(kubectl get pods -l job-name=${JOB_NAME} --field-selector=status.phase=Running -o name --no-headers | sed 's|pod/||'); do
  kubectl exec ${POD} -- bash -c "
    git clone -b ${BRANCH} ${REPO} ${INSTALL_DIR} 2>/dev/null || \
      (cd ${INSTALL_DIR} && git fetch && git checkout ${BRANCH} && git pull);
    cd ${INSTALL_DIR} && pip install -e 'python[tpu]'
  " &
done
wait
echo "All pods setup complete"
```

### Verify on all pods

```bash
for POD in $(kubectl get pods -l job-name=${JOB_NAME} -o name --no-headers | sed 's|pod/||'); do
  echo -n "${POD}: "
  kubectl exec ${POD} -- python -c "import sglang_jax; print('OK')"
done
```

## Useful Aliases

```bash
alias k=kubectl
alias kgp='kubectl get pods'
alias kgj='kubectl get jobs'
alias kl='kubectl logs'
alias ke='kubectl exec -it'
```
