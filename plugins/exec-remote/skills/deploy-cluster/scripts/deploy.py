#!/usr/bin/env python3
"""
Deploy a SkyPilot-managed TPU cluster on GKE.

Reads config.yaml and setup.yaml templates, replaces placeholders
with real values based on TPU type, and launches via SkyPilot.

Automatically ensures the required GKE node pool exists for the
requested TPU type, creating one if necessary. This allows reusing
the same GKE cluster across different TPU configurations (e.g.,
v6e-1 for unit tests, v6e-4 for e2e tests) without recreating
the entire cluster.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
SKILL_DIR = SCRIPTS_DIR.parent
CONFIG_TEMPLATE = SKILL_DIR / "config.yaml"
SETUP_TEMPLATE = SKILL_DIR / "setup.yaml"
SKY_CONFIG_DIR = Path.home() / ".sky"
SKY_CONFIG_PATH = SKY_CONFIG_DIR / "config.yaml"
CLUSTER_NAME_FILE = Path(".cluster_name_tpu")

DEFAULT_PROJECT = "tpu-service-473302"

sys.path.insert(0, str(SCRIPTS_DIR))
from tpu_config import get_tpu_config, list_supported_types


def check_prerequisites() -> bool:
    """Check that required CLI tools are installed."""
    tools = ["sky", "gcloud", "kubectl"]
    missing = []
    for tool in tools:
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(f"Error: Missing required tools: {', '.join(missing)}")
        print("\nPlease install:")
        if "sky" in missing:
            print("  - sky (SkyPilot): pip install skypilot")
        if "gcloud" in missing:
            print("  - gcloud: https://cloud.google.com/sdk/docs/install")
        if "kubectl" in missing:
            print("  - kubectl: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl")
        return False
    return True


def extract_region(zone: str) -> str:
    """Extract region from zone (e.g., asia-northeast1-b -> asia-northeast1)."""
    if zone.count("-") == 2:
        return "-".join(zone.split("-")[:2])
    return zone


# ---------------------------------------------------------------------------
# Node pool management
# ---------------------------------------------------------------------------

def list_node_pools(cluster_name: str, region: str, project: str) -> list:
    """List all node pools in the GKE cluster."""
    cmd = [
        "gcloud", "container", "node-pools", "list",
        f"--cluster={cluster_name}",
        f"--region={region}",
        f"--project={project}",
        "--format=json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Failed to list node pools")
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def pool_matches_tpu_config(pool: dict, config: dict) -> bool:
    """Check if a GKE node pool matches the requested TPU configuration.

    Matching logic:
    - Machine type must match exactly.
    - For multi-host TPUs (num_nodes > 1), the placement policy topology
      must also match, because multiple types share ct6e-standard-4t.
    - For single-host TPUs, machine type alone is sufficient (ct6e-standard-1t
      is unique to v6e-1; ct6e-standard-4t without topology = v6e-4).
    """
    pool_machine = pool.get("config", {}).get("machineType", "")
    if pool_machine != config["machine_type"]:
        return False

    pool_topology = pool.get("placementPolicy", {}).get("tpuTopology", "")

    if config["num_nodes"] > 1:
        # Multi-host: must match topology exactly
        return pool_topology == config["topology"]
    else:
        # Single-host: reject pools that have a different multi-host topology
        return pool_topology == "" or pool_topology == config["topology"]


def create_node_pool(
    cluster_name: str, pool_name: str, config: dict,
    region: str, project: str,
) -> bool:
    """Create a new GKE TPU node pool.

    For single-host TPUs (v6e-1, v6e-4), --tpu-topology is omitted because
    the topology is implicit in the machine type. For multi-host TPUs (v6e-8+),
    --tpu-topology is required.
    """
    cmd = [
        "gcloud", "beta", "container", "node-pools", "create", pool_name,
        f"--cluster={cluster_name}",
        f"--region={region}",
        f"--project={project}",
        f"--machine-type={config['machine_type']}",
        f"--num-nodes={config['num_nodes']}",
        "--spot",
        "--enable-autoscaling",
        "--min-nodes=0",
        f"--max-nodes={config['num_nodes']}",
    ]

    # Only add --tpu-topology for multi-host configurations
    if config["num_nodes"] > 1:
        cmd.append(f"--tpu-topology={config['topology']}")

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  Error: Failed to create node pool '{pool_name}'")
        return False
    print(f"  Node pool '{pool_name}' created successfully")
    return True


def ensure_node_pool(
    cluster_name: str, tpu_type: str, zone: str,
    project: str = DEFAULT_PROJECT,
) -> bool:
    """Ensure a GKE node pool exists for the specified TPU type.

    Checks existing node pools by matching machine type and topology.
    This detects both pools created by this script (named tpu-<type>)
    and pools created by xpk (arbitrary names). Only creates a new
    pool if no match is found.
    """
    config = get_tpu_config(tpu_type)
    region = extract_region(zone)

    print(f"\nChecking node pools for {tpu_type}...")
    pools = list_node_pools(cluster_name, region, project)

    for pool in pools:
        if pool_matches_tpu_config(pool, config):
            print(f"  Found matching node pool: '{pool['name']}'")
            return True

    # No matching pool found, create one
    pool_name = f"tpu-{tpu_type}"
    print(f"  No matching node pool found. Creating '{pool_name}'...")
    return create_node_pool(cluster_name, pool_name, config, region, project)


# ---------------------------------------------------------------------------
# SkyPilot config generation
# ---------------------------------------------------------------------------

def handle_topology_change(cluster_name: str, new_topology: str):
    """Tear down SkyPilot cluster if topology has changed.

    SkyPilot pods use nodeSelector labels that are topology-specific.
    When switching TPU types, the old cluster must be torn down first
    so that new pods are scheduled on the correct node pool.
    """
    if not SKY_CONFIG_PATH.exists():
        return

    current_topology = None
    content = SKY_CONFIG_PATH.read_text()
    for line in content.split("\n"):
        if "gke-tpu-topology" in line:
            current_topology = line.split(":")[-1].strip()
            break

    if current_topology is None or current_topology == new_topology:
        return

    print(f"\nTopology change detected: {current_topology} -> {new_topology}")
    print(f"Tearing down existing SkyPilot cluster '{cluster_name}'...")
    subprocess.run(["sky", "down", cluster_name, "-y"])


def generate_sky_config(tpu_type: str) -> bool:
    """Generate ~/.sky/config.yaml from template with TPU-specific values."""
    config = get_tpu_config(tpu_type)

    template = CONFIG_TEMPLATE.read_text()
    rendered = (
        template
        .replace("<ACCELERATOR_TYPE>", config["accelerator"])
        .replace("<TPU_TOPOLOGY>", config["topology"])
        .replace("<CHIPS_PER_HOST>", str(config["chips_per_host"]))
        .replace("<CPU_REQUEST>", config["cpu"])
        .replace("<MEMORY_REQUEST>", config["memory"])
    )

    SKY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing config if present
    if SKY_CONFIG_PATH.exists():
        backup_name = f"config.yaml.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = SKY_CONFIG_DIR / backup_name
        shutil.copy2(SKY_CONFIG_PATH, backup_path)
        print(f"Backed up existing config to {backup_path}")

    SKY_CONFIG_PATH.write_text(rendered)
    print(f"Generated {SKY_CONFIG_PATH}")
    print(f"  accelerator: {config['accelerator']}")
    print(f"  topology:    {config['topology']}")
    print(f"  chips/host:  {config['chips_per_host']}")
    return True


def generate_setup_yaml(tpu_type: str) -> str:
    """Generate a temporary setup.yaml with NUM_NODES replaced."""
    config = get_tpu_config(tpu_type)

    template = SETUP_TEMPLATE.read_text()
    rendered = template.replace("<NUM_NODES>", str(config["num_nodes"]))

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="sky_setup_")
    with os.fdopen(fd, "w") as f:
        f.write(rendered)

    print(f"Generated setup.yaml at {tmp_path}")
    print(f"  num_nodes: {config['num_nodes']}")
    return tmp_path


# ---------------------------------------------------------------------------
# GKE credentials & SkyPilot launch
# ---------------------------------------------------------------------------

def get_gke_credentials(
    cluster_name: str, zone: str, project: str = DEFAULT_PROJECT,
) -> bool:
    """Fetch GKE cluster credentials."""
    region = extract_region(zone)
    cmd = [
        "gcloud", "container", "clusters", "get-credentials",
        cluster_name,
        f"--region={region}",
        f"--project={project}",
    ]
    print(f"\nFetching GKE credentials for cluster '{cluster_name}'...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Failed to get credentials for cluster '{cluster_name}'")
        return False
    print("Credentials configured successfully.")
    return True


def launch_sky_cluster(cluster_name: str, setup_yaml_path: str) -> bool:
    """Launch a SkyPilot cluster using the generated setup.yaml."""
    cmd = [
        "sky", "launch",
        "-c", cluster_name,
        "-y",
        "-r",
        setup_yaml_path,
    ]
    print(f"\nLaunching SkyPilot cluster '{cluster_name}'...")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: sky launch failed for cluster '{cluster_name}'")
        return False
    print(f"\nSkyPilot cluster '{cluster_name}' launched successfully!")
    return True


# ---------------------------------------------------------------------------
# Main deploy flow
# ---------------------------------------------------------------------------

def deploy(cluster_name: str, tpu_type: str, zone: str):
    """Full deployment flow.

    1. Check prerequisites
    2. Fetch GKE credentials
    3. Ensure node pool exists for the TPU type (create if missing)
    4. Handle topology change (sky down if switching TPU types)
    5. Generate SkyPilot config
    6. Launch SkyPilot cluster
    """
    config = get_tpu_config(tpu_type)

    print(f"\n{'=' * 60}")
    print(f"  Deploy SkyPilot Cluster")
    print(f"{'=' * 60}")
    print(f"\n  Cluster: {cluster_name}")
    print(f"  TPU:     {tpu_type}")
    print(f"  Zone:    {zone}\n")

    # Step 1: Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Step 2: Get GKE credentials
    if not get_gke_credentials(cluster_name, zone):
        sys.exit(1)

    # Step 3: Ensure node pool exists for the TPU type
    try:
        if not ensure_node_pool(cluster_name, tpu_type, zone):
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 4: Handle topology change (sky down if switching TPU types)
    handle_topology_change(cluster_name, config["topology"])

    # Step 5: Generate ~/.sky/config.yaml
    try:
        if not generate_sky_config(tpu_type):
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 6: Generate setup.yaml
    try:
        setup_path = generate_setup_yaml(tpu_type)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 7: Launch SkyPilot cluster
    try:
        if not launch_sky_cluster(cluster_name, setup_path):
            sys.exit(1)
    finally:
        # Clean up temp file
        if os.path.exists(setup_path):
            os.unlink(setup_path)

    # Save cluster name for exec-remote skill
    CLUSTER_NAME_FILE.write_text(cluster_name)
    print(f"\nSaved cluster name to {CLUSTER_NAME_FILE}")

    print(f"\n{'=' * 60}")
    print(f"  Deployment Complete!")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  sky status          # Check cluster status")
    print(f"  sky exec {cluster_name} 'command'  # Run commands on cluster")
    print(f"  sky down {cluster_name}            # Tear down cluster")


def main():
    if len(sys.argv) < 2:
        print("Usage: deploy.py <cluster_name> <tpu_type> <zone>")
        print(f"\nSupported TPU types: {', '.join(list_supported_types())}")
        print("\nExample:")
        print("  python deploy.py my-cluster v6e-16 asia-northeast1-b")
        sys.exit(1)

    if sys.argv[1] == "--help":
        print("Deploy a SkyPilot-managed TPU cluster on GKE.\n")
        print("Usage: deploy.py <cluster_name> <tpu_type> <zone>\n")
        print("Arguments:")
        print("  cluster_name  Name of the GKE cluster (must already exist)")
        print("  tpu_type      TPU type (e.g., v6e-16)")
        print("  zone          GCP zone (e.g., asia-northeast1-b)\n")
        print(f"Supported TPU types: {', '.join(list_supported_types())}")
        sys.exit(0)

    if len(sys.argv) != 4:
        print("Error: Expected 3 arguments: <cluster_name> <tpu_type> <zone>")
        sys.exit(1)

    cluster_name, tpu_type, zone = sys.argv[1], sys.argv[2], sys.argv[3]
    deploy(cluster_name, tpu_type, zone)


if __name__ == "__main__":
    main()
