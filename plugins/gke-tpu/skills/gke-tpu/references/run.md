# run — Execute on Multi-Process TPU

## Launcher script

Write a Python launcher that handles distributed init + runs the target script. Both processes must run the same script.

```python
#!/usr/bin/env python3
"""Launcher for multi-process TPU workloads."""
import os, sys

sys.path.insert(0, "<repo.remote_path>/<repo.python_subdir>")
sys.path.insert(0, "<repo.remote_path>")
os.chdir("<repo.remote_path>")

import jax
jax.distributed.initialize()
proc = jax.process_index()
print(f"[Process {proc}] ready, {jax.device_count()} devices", flush=True)

sys.argv = ["script_name", "--arg1", "val1", ...]
import runpy
runpy.run_path("<repo.remote_path>/path/to/script.py", run_name="__main__")
```

## Copy and launch

```bash
# Get pod list
PODS=$(kubectl get pods -l job-name=<workload.name> -o jsonpath='{.items[*].metadata.name}')
FIRST_POD=$(echo $PODS | awk '{print $1}')

# Copy to all pods
for POD in $PODS; do
  kubectl cp /tmp/launcher.py $POD:/tmp/launcher.py -c <workload.name>
done

# Launch worker pods in background
for POD in $PODS; do
  if [ "$POD" != "$FIRST_POD" ]; then
    kubectl exec $POD -c <workload.name> -- python3 -u /tmp/launcher.py 2>&1 &
  fi
done

# Launch main pod in foreground
kubectl exec $FIRST_POD -c <workload.name> -- python3 -u /tmp/launcher.py 2>&1

# Cleanup
wait
```
