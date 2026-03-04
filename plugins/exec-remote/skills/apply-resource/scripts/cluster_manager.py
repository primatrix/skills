#!/usr/bin/env python3
"""
GKE TPU Cluster Manager using xpk
Handles cluster creation, deletion, and listing operations.
Multi-user safe: Always queries GKE for real-time cluster state.
"""

import json
import subprocess
import sys
from typing import Dict, List, Optional


class ClusterManager:
    """Manages GKE TPU clusters using xpk.

    This manager does NOT use local caching. All cluster information
    is queried directly from GKE to ensure accuracy in multi-user scenarios.
    """

    PROJECT_ID = "tpu-service-473302"

    def __init__(self):
        """Initialize cluster manager."""
        pass

    def list_gke_clusters(self) -> List[Dict]:
        """List all GKE clusters with detailed information.

        Returns:
            List of cluster dictionaries with name, location, status, etc.
        """
        cmd = [
            "gcloud", "container", "clusters", "list",
            f"--project={self.PROJECT_ID}",
            "--format=json"
        ]

        result = self.run_command(cmd, capture_output=True)

        if result.returncode == 0:
            try:
                clusters = json.loads(result.stdout)
                return clusters
            except json.JSONDecodeError:
                print("Error: Failed to parse cluster list")
                return []
        return []

    def get_cluster_details(self, name: str, location: str) -> Optional[Dict]:
        """Get detailed information about a specific cluster.

        Args:
            name: Cluster name
            location: Cluster location (zone or region)

        Returns:
            Cluster details dictionary or None if not found
        """
        cmd = [
            "gcloud", "container", "clusters", "describe",
            name,
            f"--location={location}",
            f"--project={self.PROJECT_ID}",
            "--format=json"
        ]

        result = self.run_command(cmd, capture_output=True)

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return None
        return None

    def cluster_exists(self, name: str) -> tuple[bool, Optional[str]]:
        """Check if a cluster exists and return its location.

        Args:
            name: Cluster name

        Returns:
            Tuple of (exists, location)
        """
        clusters = self.list_gke_clusters()
        for cluster in clusters:
            if cluster.get("name") == name:
                return True, cluster.get("location")
        return False, None

    def list_clusters(self) -> List[Dict]:
        """List all GKE clusters (alias for list_gke_clusters)."""
        return self.list_gke_clusters()

    def run_command(self, cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a shell command.

        Args:
            cmd: Command and arguments as a list
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess instance
        """
        if not capture_output:
            print(f"Running: {' '.join(cmd)}")

        if capture_output:
            return subprocess.run(cmd, capture_output=True, text=True)
        else:
            return subprocess.run(cmd)

    def check_prerequisites(self) -> bool:
        """Check if required tools are installed."""
        tools = ["xpk", "gcloud", "kubectl"]
        missing = []

        for tool in tools:
            result = subprocess.run(["which", tool], capture_output=True)
            if result.returncode != 0:
                missing.append(tool)

        if missing:
            print(f"Error: Missing required tools: {', '.join(missing)}")
            print("\nPlease install:")
            for tool in missing:
                if tool == "xpk":
                    print("  - xpk: https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md")
                elif tool == "gcloud":
                    print("  - gcloud: https://cloud.google.com/sdk/docs/install")
                elif tool == "kubectl":
                    print("  - kubectl: https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl")
            return False

        return True

    def create_cluster(self, name: str, tpu_type: str, num_slices: int, zone: str, spot: bool = True):
        """Create a new GKE cluster with TPU nodepool.

        Args:
            name: Cluster name
            tpu_type: TPU accelerator type (e.g., v6e-16)
            num_slices: Number of TPU slices
            zone: GCP zone
            spot: Whether to use spot instances
        """
        if not self.check_prerequisites():
            return False

        # Check if cluster already exists
        exists, location = self.cluster_exists(name)
        if exists:
            print(f"Error: Cluster '{name}' already exists in {location}")
            print("Please choose a different name or delete the existing cluster first.")
            return False

        cmd = [
            "xpk", "cluster", "create-pathways",
            "--cluster", name,
            f"--num-slices={num_slices}",
            f"--tpu-type={tpu_type}",
            f"--zone={zone}",
        ]

        if spot:
            cmd.append("--spot")

        print(f"\nCreating cluster '{name}' with {num_slices} slice(s) of {tpu_type} in {zone}...")
        result = self.run_command(cmd)

        if result.returncode == 0:
            print(f"\n✓ Cluster '{name}' created successfully!")
            return True
        else:
            print(f"\n✗ Failed to create cluster '{name}'")
            return False

    def delete_cluster(self, name: str, zone: str):
        """Delete a GKE cluster.

        Args:
            name: Cluster name
            zone: GCP zone
        """
        if not self.check_prerequisites():
            return False

        # Verify cluster exists before attempting deletion
        exists, location = self.cluster_exists(name)
        if not exists:
            print(f"Warning: Cluster '{name}' not found in GKE")
            print("It may have already been deleted by another user.")
            return False

        # Verify the zone matches
        if location != zone and not location.startswith(zone.rsplit('-', 1)[0]):
            print(f"Warning: Cluster '{name}' is in {location}, not {zone}")
            print(f"Using actual location: {location}")
            zone = location

        cmd = [
            "xpk", "cluster", "delete",
            "--cluster", name,
            f"--zone={zone}",
        ]

        print(f"\nDeleting cluster '{name}' in {zone}...")
        result = self.run_command(cmd)

        if result.returncode == 0:
            print(f"\n✓ Cluster '{name}' deleted successfully!")
            return True
        else:
            print(f"\n✗ Failed to delete cluster '{name}'")
            return False

    def describe_cluster(self, name: str, zone: str):
        """Describe a cluster in detail.

        Args:
            name: Cluster name
            zone: GCP zone
        """
        # First check if cluster exists
        exists, location = self.cluster_exists(name)
        if not exists:
            print(f"Error: Cluster '{name}' not found")
            return False

        # Use actual location
        if location != zone:
            print(f"Note: Using actual cluster location: {location}")
            zone = location

        cmd = [
            "xpk", "cluster", "describe",
            "--cluster", name,
            f"--zone={zone}",
        ]

        print(f"\nDescribing cluster '{name}'...")
        self.run_command(cmd)
        return True


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: cluster_manager.py [create|delete|list|describe]")
        sys.exit(1)

    manager = ClusterManager()
    operation = sys.argv[1]

    if operation == "create":
        if len(sys.argv) != 6:
            print("Usage: cluster_manager.py create <name> <tpu_type> <num_slices> <zone>")
            sys.exit(1)
        name, tpu_type, num_slices, zone = sys.argv[2:6]
        manager.create_cluster(name, tpu_type, int(num_slices), zone)

    elif operation == "delete":
        if len(sys.argv) != 4:
            print("Usage: cluster_manager.py delete <name> <zone>")
            sys.exit(1)
        name, zone = sys.argv[2:4]
        manager.delete_cluster(name, zone)

    elif operation == "list":
        clusters = manager.list_clusters()
        if not clusters:
            print("No GKE clusters found.")
        else:
            print(f"\nGKE Clusters (total: {len(clusters)}):")
            for cluster in clusters:
                name = cluster.get("name", "Unknown")
                location = cluster.get("location", "Unknown")
                status = cluster.get("status", "Unknown")
                print(f"\n  Name:     {name}")
                print(f"  Location: {location}")
                print(f"  Status:   {status}")

                # Try to extract TPU nodepool info
                node_pools = cluster.get("nodePools", [])
                tpu_pools = [np for np in node_pools if "tpu" in np.get("name", "").lower()]
                if tpu_pools:
                    print(f"  TPU Pools: {len(tpu_pools)}")
                    for pool in tpu_pools:
                        pool_name = pool.get("name", "Unknown")
                        node_count = pool.get("initialNodeCount", 0)
                        print(f"    - {pool_name}: {node_count} nodes")

    elif operation == "describe":
        if len(sys.argv) != 4:
            print("Usage: cluster_manager.py describe <name> <zone>")
            sys.exit(1)
        name, zone = sys.argv[2:4]
        manager.describe_cluster(name, zone)

    else:
        print(f"Unknown operation: {operation}")
        sys.exit(1)


if __name__ == "__main__":
    main()
