#!/usr/bin/env python3
"""
Deploy a SkyPilot-managed TPU cluster on GKE.

Reads config.yaml and setup.yaml templates, replaces placeholders
with real values based on TPU type, and launches via SkyPilot.
"""

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


def generate_sky_config(tpu_type: str) -> bool:
    """Generate ~/.sky/config.yaml from template with TPU-specific values.

    Args:
        tpu_type: TPU type string (e.g., "v6e-16")

    Returns:
        True on success, False on failure
    """
    config = get_tpu_config(tpu_type)

    template = CONFIG_TEMPLATE.read_text()
    rendered = (
        template
        .replace("<ACCELERATOR_TYPE>", config["accelerator"])
        .replace("<TPU_TOPOLOGY>", config["topology"])
        .replace("<CHIPS_PER_HOST>", str(config["chips_per_host"]))
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
    """Generate a temporary setup.yaml with NUM_NODES replaced.

    Args:
        tpu_type: TPU type string (e.g., "v6e-16")

    Returns:
        Path to the generated temporary setup.yaml file
    """
    config = get_tpu_config(tpu_type)

    template = SETUP_TEMPLATE.read_text()
    rendered = template.replace("<NUM_NODES>", str(config["num_nodes"]))

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="sky_setup_")
    with os.fdopen(fd, "w") as f:
        f.write(rendered)

    print(f"Generated setup.yaml at {tmp_path}")
    print(f"  num_nodes: {config['num_nodes']}")
    return tmp_path


def get_gke_credentials(cluster_name: str, zone: str, project: str = "tpu-service-473302") -> bool:
    """Fetch GKE cluster credentials.

    Args:
        cluster_name: GKE cluster name
        zone: GCP zone or region (e.g., asia-northeast1-b or asia-northeast1)
        project: GCP project ID

    Returns:
        True on success, False on failure
    """
    # xpk creates clusters at the region level, so extract region from zone if needed
    # e.g., asia-northeast1-b → asia-northeast1
    if zone.count("-") == 2:  # zone format: region-zone (e.g., asia-northeast1-b)
        location = "-".join(zone.split("-")[:2])
        location_flag = "--region"
    else:  # already a region (e.g., asia-northeast1)
        location = zone
        location_flag = "--region"

    cmd = [
        "gcloud", "container", "clusters", "get-credentials",
        cluster_name,
        f"{location_flag}={location}",
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
    """Launch a SkyPilot cluster using the generated setup.yaml.

    Args:
        cluster_name: Name for the SkyPilot cluster
        setup_yaml_path: Path to the generated setup.yaml

    Returns:
        True on success, False on failure
    """
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


def deploy(cluster_name: str, tpu_type: str, zone: str):
    """Full deployment flow.

    Args:
        cluster_name: GKE/SkyPilot cluster name
        tpu_type: TPU type (e.g., "v6e-16")
        zone: GCP zone
    """
    print(f"\n{'=' * 60}")
    print(f"  Deploy SkyPilot Cluster")
    print(f"{'=' * 60}")
    print(f"\n  Cluster: {cluster_name}")
    print(f"  TPU:     {tpu_type}")
    print(f"  Zone:    {zone}\n")

    # Step 1: Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Step 2: Generate ~/.sky/config.yaml
    try:
        if not generate_sky_config(tpu_type):
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 3: Generate setup.yaml
    try:
        setup_path = generate_setup_yaml(tpu_type)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Step 4: Get GKE credentials
    if not get_gke_credentials(cluster_name, zone):
        sys.exit(1)

    # Step 5: Launch SkyPilot cluster
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
