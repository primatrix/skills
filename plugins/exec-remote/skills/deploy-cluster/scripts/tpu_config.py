#!/usr/bin/env python3
"""
TPU type to SkyPilot/Kubernetes configuration mapping.

Maps xpk TPU types (e.g., v6e-16) to the corresponding:
- accelerator: GKE nodeSelector label for tpu-accelerator
- topology: GKE nodeSelector label for tpu-topology
- chips_per_host: google.com/tpu resource request per node
- num_nodes: number of nodes for SkyPilot setup.yaml
- cpu/memory: resource requests sized to fit the GCE machine type
- machine_type: GCE machine type for node pool creation

Reference: https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
"""

import sys

TPU_CONFIG_MAP = {
    # v6e configurations.
    # Each GKE node exposes exactly 4 TPU chips (google.com/tpu: 4).
    # num_nodes = total_chips / 4.
    # Exception: v6e-1 is a sub-slice that exposes only 1 chip.
    # cpu/memory: resource requests sized to fit the GCE machine type for each TPU slice.
    # ct6e-standard-1t: 44 vCPUs, 176 GB  |  ct6e-standard-4t: 180 vCPUs, 720 GB
    "v6e-1":   {"accelerator": "tpu-v6e-slice", "topology": "1x1",   "chips_per_host": 1, "num_nodes": 1,  "cpu": "30",  "memory": "150Gi", "machine_type": "ct6e-standard-1t"},
    "v6e-4":   {"accelerator": "tpu-v6e-slice", "topology": "2x2",   "chips_per_host": 4, "num_nodes": 1,  "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
    "v6e-8":   {"accelerator": "tpu-v6e-slice", "topology": "2x4",   "chips_per_host": 4, "num_nodes": 2,  "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
    "v6e-16":  {"accelerator": "tpu-v6e-slice", "topology": "4x4",   "chips_per_host": 4, "num_nodes": 4,  "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
    "v6e-32":  {"accelerator": "tpu-v6e-slice", "topology": "4x8",   "chips_per_host": 4, "num_nodes": 8,  "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
    "v6e-64":  {"accelerator": "tpu-v6e-slice", "topology": "8x8",   "chips_per_host": 4, "num_nodes": 16, "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
    "v6e-128": {"accelerator": "tpu-v6e-slice", "topology": "8x16",  "chips_per_host": 4, "num_nodes": 32, "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
    "v6e-256": {"accelerator": "tpu-v6e-slice", "topology": "16x16", "chips_per_host": 4, "num_nodes": 64, "cpu": "36",  "memory": "500Gi", "machine_type": "ct6e-standard-4t"},
}


def get_tpu_config(tpu_type: str) -> dict:
    """Get SkyPilot/K8s configuration for a given TPU type.

    Args:
        tpu_type: TPU type string (e.g., "v6e-16")

    Returns:
        Dict with keys: accelerator, topology, chips_per_host, num_nodes,
        cpu, memory, machine_type

    Raises:
        ValueError: If tpu_type is not supported
    """
    config = TPU_CONFIG_MAP.get(tpu_type)
    if config is None:
        supported = ", ".join(sorted(TPU_CONFIG_MAP.keys()))
        raise ValueError(
            f"Unsupported TPU type: '{tpu_type}'. "
            f"Supported types: {supported}"
        )
    return config


def list_supported_types() -> list:
    """Return all supported TPU type strings."""
    return sorted(TPU_CONFIG_MAP.keys())


if __name__ == "__main__":
    print("Supported TPU types and configurations:\n")
    print(f"{'Type':<10} {'Accelerator':<18} {'Topology':<10} {'Chips/Host':<12} {'Nodes':<8} {'Machine Type'}")
    print("-" * 85)
    for tpu_type in list_supported_types():
        cfg = TPU_CONFIG_MAP[tpu_type]
        print(f"{tpu_type:<10} {cfg['accelerator']:<18} {cfg['topology']:<10} {cfg['chips_per_host']:<12} {cfg['num_nodes']:<8} {cfg['machine_type']}")
