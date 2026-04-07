# TPU Topology Reference

Use this to determine topology, machine type, chips per node, and single-host vs multi-host mode.

**Rule**: `chips / chips_per_node = VMs`. If VMs = 1 → single-host Pod. If VMs > 1 → multi-host Indexed Job + headless Service.

## TPU v6e (Trillium)

Interconnect: **2D torus**. Max pod: 256 chips.

| Topology | Chips | VMs | Machine Type | Chips/VM | Mode |
|----------|-------|-----|--------------|----------|------|
| `1x1` | 1 | 1 | `ct6e-standard-1t` | 1 | single-host Pod |
| `2x2` | 4 | 1 | `ct6e-standard-4t` | 4 | single-host Pod |
| `2x4` | 8 | 1 | `ct6e-standard-8t` | 8 | single-host Pod |
| `2x4` | 8 | 2 | `ct6e-standard-4t` | 4 | multi-host Job |
| `4x4` | 16 | 4 | `ct6e-standard-4t` | 4 | multi-host Job |
| `4x8` | 32 | 8 | `ct6e-standard-4t` | 4 | multi-host Job |
| `8x8` | 64 | 16 | `ct6e-standard-4t` | 4 | multi-host Job |
| `8x16` | 128 | 32 | `ct6e-standard-4t` | 4 | multi-host Job |
| `16x16` | 256 | 64 | `ct6e-standard-4t` | 4 | multi-host Job |

**Notes**:
- `2x4` with `ct6e-standard-8t` (single VM, 8 chips) is optimized for inference.
- `2x4` with `ct6e-standard-4t` (2 VMs, 4 chips each) is GKE API only.
- Multi-host topologies (≥ `4x4`) always use `ct6e-standard-4t`.

### v6e GKE nodeSelector labels

```yaml
cloud.google.com/gke-tpu-accelerator: tpu-v6e-slice
cloud.google.com/gke-tpu-topology: <topology>
```

## TPU v7x (Ironwood)

Interconnect: **3D torus**. Max pod: 9,216 chips. Slices > 64 chips are composed of 4x4x4 "cubes".

| Topology | Chips | VMs | Machine Type | Chips/VM | Mode |
|----------|-------|-----|--------------|----------|------|
| `2x2x1` | 4 | 1 | `tpu7x-standard-4t` | 4 | single-host Pod |
| `2x2x2` | 8 | 2 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `2x2x4` | 16 | 4 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `2x4x4` | 32 | 8 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `4x4x4` | 64 | 16 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `4x4x8` | 128 | 32 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `4x8x8` | 256 | 64 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `8x8x8` | 512 | 128 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `8x8x16` | 1,024 | 256 | `tpu7x-standard-4t` | 4 | multi-host Job |
| `8x16x16` | 2,048 | 512 | `tpu7x-standard-4t` | 4 | multi-host Job |

**Notes**:
- All v7x topologies use `tpu7x-standard-4t` (4 chips/VM, 224 vCPUs, 960 GB RAM).
- Each v7x "chip" has two chiplets — JAX exposes each chip as **2 devices**. A 4-chip VM shows 8 JAX devices.
- Topology is 3D (e.g. `2x2x1`) vs v6e's 2D (e.g. `2x2`).

### v7x GKE nodeSelector labels

```yaml
cloud.google.com/gke-tpu-accelerator: tpu-v7x-slice
cloud.google.com/gke-tpu-topology: <topology>
```

## gke.toml examples

### v6e 16-chip (4x4)

```toml
[tpu]
accelerator = "tpu-v6e-slice"
topology = "4x4"
chips_per_node = 4
machine_type = "ct6e-standard-4t"
max_nodes = 4
```

### v7x 64-chip (4x4x4)

```toml
[tpu]
accelerator = "tpu-v7x-slice"
topology = "4x4x4"
chips_per_node = 4
machine_type = "tpu7x-standard-4t"
max_nodes = 16
```
