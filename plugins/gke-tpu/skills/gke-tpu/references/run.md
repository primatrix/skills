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
# Copy to all containers
for CONTAINER in <all containers>; do
  kubectl cp /tmp/launcher.py <POD_NAME>:/tmp/launcher.py -c $CONTAINER
done

# Launch worker containers in background
for WORKER in <all containers except first>; do
  kubectl exec <POD_NAME> -c $WORKER -- python3 -u /tmp/launcher.py 2>&1 &
done

# Launch main container in foreground
kubectl exec <POD_NAME> -c <first container> -- python3 -u /tmp/launcher.py 2>&1

# Cleanup
wait
```
