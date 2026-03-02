# Pod Monitoring Reference

Use this reference for Step 5 of the Deploy Workload workflow. Monitor after `kubectl apply` until all pods are running or a failure is detected.

## Monitoring Loop

Poll every 15 seconds, up to ~8 minutes (≈32 iterations). Run these commands each iteration:

```bash
# 1. Check pod statuses
kubectl get pods -l job-name=<job-name> --no-headers

# 2. Check autoscaler scale-up events
kubectl get events --field-selector reason=FailedScaleUp 2>/dev/null
kubectl get events --sort-by='.lastTimestamp' | grep -E "ScaleUp|TriggeredScale|NotTrigger"

# 3. Count nodes with the required topology (to detect new nodes being added)
kubectl get nodes -l cloud.google.com/gke-tpu-topology=<topology> --no-headers | wc -l
```

## Success Condition

All pods for the job show `Running` status:

```bash
kubectl get pods -l job-name=<job-name> --no-headers
# All rows show "Running" in the STATUS column
```

Report success and provide the job name and pod names.

## Failure Signals

Check failure signals in this order. Stop at the first confirmed signal.

### Signal 1 — Autoscaler Did Not Respond (2 min timeout)

If after 2 minutes no `TriggeredScaleUp` event appears and node count hasn't increased → autoscaler is not responding, likely no available capacity.

```bash
kubectl get events --sort-by='.lastTimestamp' | grep -E "TriggeredScaleUp|ScaleUp"
```

**Likely cause:** No Flex Start capacity in this zone.
**Suggested action:** Try a different zone (`--node-locations`) or retry later.

---

### Signal 2 — Autoscaler Capacity Error

`cluster-autoscaler` event contains `not.enough.capacity` or `max nodes reached`:

```bash
kubectl get events --field-selector reason=FailedScaleUp 2>/dev/null
kubectl get events --sort-by='.lastTimestamp' | grep -i "cluster-autoscaler"
```

**Likely cause:** GCP quota exhausted or `--total-max-nodes` / `--max-nodes` limit hit.
**Suggested action:** Increase `--total-max-nodes` on the nodepool, or check GCP quota for the project/region.

---

### Signal 3 — Pod Scheduling Failure

Pod is stuck in `Pending` with `FailedScheduling` event citing `Insufficient google.com/tpu` and no new nodes are being added after 3 minutes:

```bash
kubectl describe pod <pod-name> | grep -A10 Events
# Look for: "Insufficient google.com/tpu" or "0/N nodes are available"
```

**Likely cause:** Nodepool exists but has no available capacity, or topology mismatch.
**Suggested action:** Verify `nodeSelector` topology matches the nodepool's label. Check nodepool `--total-max-nodes`.

---

### Signal 4 — Workload Crashed After Scheduling

Pod reached `Running` but then transitioned to `Error` or `CrashLoopBackOff` → deployment succeeded but the workload itself failed:

```bash
kubectl get pods -l job-name=<job-name> --no-headers
# STATUS shows "Error" or "CrashLoopBackOff"

kubectl logs <pod-name>   # check workload output for errors
```

**Likely cause:** Application-level error (bad command, missing dependency, OOM).
**Suggested action:** Share pod logs with the user. This is a workload issue, not an infrastructure issue.

---

## Failure Report Format

When reporting a failure, always include:

1. **Pod status summary:**
   ```bash
   kubectl get pods -l job-name=<job-name> --no-headers
   ```

2. **Pod event messages** (for the first stuck/failed pod):
   ```bash
   kubectl describe pod <pod-name> | grep -A10 Events
   ```

3. **Scale-up status:** Was `TriggeredScaleUp` seen? Were new nodes added?

4. **Max-nodes check:** Was a `FailedScaleUp` or `max nodes reached` event present?

5. **Suggested action:** One of:
   - Increase `--total-max-nodes` on the nodepool
   - Try a different zone
   - Check GCP quota for the project/region
   - Investigate workload logs (for `CrashLoopBackOff`)
