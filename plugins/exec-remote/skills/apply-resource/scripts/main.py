#!/usr/bin/env python3
"""
Interactive GKE TPU Cluster Manager
Main entry point for the apply-resource skill.
"""

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from cluster_manager import ClusterManager
from tpu_availability import TPUAvailabilityChecker


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_section(text: str):
    """Print a formatted section."""
    print(f"\n--- {text} ---\n")


def get_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def get_choice(prompt: str, options: list) -> int:
    """Get user choice from a list of options."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print(f"  0. Cancel")

    while True:
        try:
            choice = int(input("\nEnter your choice: ").strip())
            if 0 <= choice <= len(options):
                return choice
            print(f"Please enter a number between 0 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


def confirm(prompt: str) -> bool:
    """Ask for user confirmation."""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")


def create_cluster_interactive(manager: ClusterManager, checker: TPUAvailabilityChecker):
    """Interactive cluster creation flow."""
    print_header("Create GKE TPU Cluster")

    # Get cluster name
    cluster_name = get_input("Cluster name")
    if not cluster_name:
        print("Error: Cluster name is required")
        return

    # Get TPU type
    print("\nCommon TPU types:")
    print("  - v6e-1, v6e-4, v6e-8, v6e-16")
    print("  - v5e-1, v5e-4, v5e-8, v5e-16")
    print("  - v4-8, v4-16, v4-32")
    tpu_type = get_input("TPU type", "v6e-16")

    # Get number of slices
    num_slices = get_input("Number of slices", "1")
    try:
        num_slices = int(num_slices)
    except ValueError:
        print("Error: Number of slices must be an integer")
        return

    # Get zone with availability check
    while True:
        print("\nCommon zones:")
        print("  - asia-northeast1-b (Tokyo)")
        print("  - us-east5-a (Columbus)")
        print("  - us-west4-a (Las Vegas)")
        zone = get_input("Zone", "asia-northeast1-b")

        # Check availability
        print("\nChecking TPU availability...")
        is_available, error_msg = checker.check_zone_availability(tpu_type, zone)

        if is_available:
            print(f"✓ TPU type '{tpu_type}' is available in zone '{zone}'")
            break
        else:
            print(f"\n✗ {error_msg}")

            # Suggest alternatives
            suggestions = checker.suggest_alternative_zones(tpu_type)
            if suggestions:
                print(f"\nSuggested alternative zones:")
                for i, suggested_zone in enumerate(suggestions, 1):
                    print(f"  {i}. {suggested_zone}")

                if confirm("\nWould you like to choose an alternative zone?"):
                    choice = get_choice("Select a zone", suggestions)
                    if choice == 0:
                        print("Cancelled")
                        return
                    zone = suggestions[choice - 1]
                    break
                elif confirm("Would you like to try a different zone?"):
                    continue
                else:
                    print("Cancelled")
                    return
            else:
                print("\nNo alternative zones found.")
                if not confirm("Would you like to try a different zone?"):
                    print("Cancelled")
                    return

    # Use spot instances
    use_spot = confirm("\nUse spot instances? (cheaper but can be preempted)")

    # Summary
    print_section("Summary")
    print(f"Cluster name:    {cluster_name}")
    print(f"TPU type:        {tpu_type}")
    print(f"Number of slices: {num_slices}")
    print(f"Zone:            {zone}")
    print(f"Spot instances:  {'Yes' if use_spot else 'No'}")
    print(f"Project ID:      {manager.PROJECT_ID}")

    if not confirm("\nProceed with cluster creation?"):
        print("Cancelled")
        return

    # Create cluster
    success = manager.create_cluster(cluster_name, tpu_type, num_slices, zone, use_spot)

    if success:
        print_section("Next Steps")
        print("1. Verify cluster status:")
        print(f"   kubectl get nodes")
        print("\n2. View cluster details:")
        print(f"   xpk cluster describe --cluster {cluster_name} --zone={zone}")


def delete_cluster_interactive(manager: ClusterManager):
    """Interactive cluster deletion flow."""
    print_header("Delete GKE TPU Cluster")

    # Query live GKE clusters
    print("Querying GKE for live clusters...")
    clusters = manager.list_gke_clusters()

    if not clusters:
        print("No GKE clusters found.")
        return

    # Show clusters
    print("\nAvailable clusters:")
    cluster_options = []
    for cluster in clusters:
        name = cluster.get("name", "Unknown")
        location = cluster.get("location", "Unknown")
        status = cluster.get("status", "Unknown")
        option = f"{name} (Location: {location}, Status: {status})"
        cluster_options.append(option)

    choice = get_choice("Select a cluster to delete", cluster_options)

    if choice == 0:
        print("Cancelled")
        return

    selected_cluster = clusters[choice - 1]
    cluster_name = selected_cluster.get("name")
    cluster_location = selected_cluster.get("location")

    # Confirm deletion
    print_section("Cluster Details")
    print(f"Name:     {cluster_name}")
    print(f"Location: {cluster_location}")
    print(f"Status:   {selected_cluster.get('status')}")

    # Show node pools
    node_pools = selected_cluster.get("nodePools", [])
    if node_pools:
        print(f"\nNode Pools: {len(node_pools)}")
        for pool in node_pools:
            pool_name = pool.get("name", "Unknown")
            node_count = pool.get("initialNodeCount", 0)
            print(f"  - {pool_name}: {node_count} nodes")

    if not confirm("\n⚠️  This will permanently delete the cluster. Continue?"):
        print("Cancelled")
        return

    # Delete cluster
    manager.delete_cluster(cluster_name, cluster_location)


def list_clusters_interactive(manager: ClusterManager):
    """Interactive cluster listing."""
    print_header("List GKE TPU Clusters")

    # Query live GKE clusters
    print("Querying GKE for live clusters...")
    clusters = manager.list_gke_clusters()

    if not clusters:
        print("No GKE clusters found.")
        return

    print(f"\nGKE Clusters (total: {len(clusters)}):")
    for i, cluster in enumerate(clusters, 1):
        name = cluster.get("name", "Unknown")
        location = cluster.get("location", "Unknown")
        status = cluster.get("status", "Unknown")
        created_time = cluster.get("createTime", "Unknown")

        print(f"\n{i}. {name}")
        print(f"   Location:    {location}")
        print(f"   Status:      {status}")
        print(f"   Created:     {created_time}")

        # Show node pools
        node_pools = cluster.get("nodePools", [])
        tpu_pools = [np for np in node_pools if "tpu" in np.get("name", "").lower() or "np" in np.get("name", "").lower()]

        if tpu_pools:
            print(f"   TPU Pools:   {len(tpu_pools)}")
            for pool in tpu_pools:
                pool_name = pool.get("name", "Unknown")
                node_count = pool.get("initialNodeCount", 0)
                machine_type = pool.get("config", {}).get("machineType", "Unknown")
                print(f"     - {pool_name}: {node_count} nodes ({machine_type})")


def describe_cluster_interactive(manager: ClusterManager):
    """Interactive cluster description."""
    print_header("Describe GKE TPU Cluster")

    # Query live GKE clusters
    print("Querying GKE for live clusters...")
    clusters = manager.list_gke_clusters()

    if not clusters:
        print("No GKE clusters found.")
        return

    # Show clusters
    cluster_options = []
    for cluster in clusters:
        name = cluster.get("name", "Unknown")
        location = cluster.get("location", "Unknown")
        option = f"{name} ({location})"
        cluster_options.append(option)

    choice = get_choice("Select a cluster to describe", cluster_options)

    if choice == 0:
        print("Cancelled")
        return

    selected_cluster = clusters[choice - 1]
    cluster_name = selected_cluster.get("name")
    cluster_location = selected_cluster.get("location")

    manager.describe_cluster(cluster_name, cluster_location)


def main():
    """Main entry point."""
    manager = ClusterManager()
    checker = TPUAvailabilityChecker()

    if len(sys.argv) < 2:
        print("Usage: main.py [create|delete|list|describe]")
        print("\nOperations:")
        print("  create   - Create a new GKE TPU cluster")
        print("  delete   - Delete an existing cluster")
        print("  list     - List all clusters")
        print("  describe - Describe a cluster in detail")
        sys.exit(1)

    operation = sys.argv[1].lower()

    if operation == "create":
        create_cluster_interactive(manager, checker)
    elif operation == "delete":
        delete_cluster_interactive(manager)
    elif operation == "list":
        list_clusters_interactive(manager)
    elif operation == "describe":
        describe_cluster_interactive(manager)
    else:
        print(f"Unknown operation: {operation}")
        print("Valid operations: create, delete, list, describe")
        sys.exit(1)


if __name__ == "__main__":
    main()
