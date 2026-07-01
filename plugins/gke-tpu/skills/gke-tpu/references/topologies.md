# Topologies

`hosts = product(topology dimensions) / chips_per_node`.

Common v6e:

| Topology | Chips | Chips/Node | Hosts | Machine Type |
|---|---:|---:|---:|---|
| `2x2` | 4 | 4 | 1 | `ct6e-standard-4t` |
| `2x4` | 8 | 8 | 1 | `ct6e-standard-8t` |
| `2x4` | 8 | 4 | 2 | `ct6e-standard-4t` |
| `4x4` | 16 | 4 | 4 | `ct6e-standard-4t` |
| `4x8` | 32 | 4 | 8 | `ct6e-standard-4t` |
| `8x8` | 64 | 4 | 16 | `ct6e-standard-4t` |

Common v7x:

| Topology | Chips | Chips/Node | Hosts | Machine Type |
|---|---:|---:|---:|---|
| `2x2x1` | 4 | 4 | 1 | `tpu7x-standard-4t` |
| `2x2x2` | 8 | 4 | 2 | `tpu7x-standard-4t` |
| `2x2x4` | 16 | 4 | 4 | `tpu7x-standard-4t` |
| `4x4x4` | 64 | 4 | 16 | `tpu7x-standard-4t` |

Use these as hints, not as a complete product catalog. If GKE rejects a machine/topology pair, trust live `gcloud` output and adjust the config.
