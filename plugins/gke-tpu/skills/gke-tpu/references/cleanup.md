# Cleanup

Use delete plans for destructive actions.

`delete-workload-plan` emits resource-name deletes rather than relying on `/tmp` manifests:

- `job/<workload>`
- `svc/<workload>-headless-svc` when `hosts > 1`
- `configmap/<workload>-launcher` when a launcher was rendered

Require the JSON confirmation token before executing deletes.

`delete-nodepool-plan` is stronger:

- Show project, cluster, zone, nodepool, topology, hosts, and reservation.
- Query live GKE state first: nodes and running pods using the nodepool.
- Require `DELETE nodepool <nodepool> from cluster <cluster>`.
- Do not delete nodepools as part of normal workload cleanup.
