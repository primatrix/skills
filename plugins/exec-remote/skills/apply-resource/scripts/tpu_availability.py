#!/usr/bin/env python3
"""
TPU Availability Checker
Queries TPU availability across GCP zones and suggests alternatives.
"""

import re
import subprocess
from typing import Dict, List, Optional, Tuple


class TPUAvailabilityChecker:
    """Checks TPU availability in GCP zones."""

    # Common TPU types and their typical zones
    # This is a fallback mapping; we'll try to query dynamically first
    TPU_ZONE_MAP = {
        "v4-8": ["us-central2-b", "us-east5-a"],
        "v4-16": ["us-central2-b", "us-east5-a"],
        "v4-32": ["us-central2-b", "us-east5-a"],
        "v5e-1": ["us-east1-c", "us-west1-c", "us-west4-a"],
        "v5e-4": ["us-east1-c", "us-west1-c", "us-west4-a"],
        "v5e-8": ["us-east1-c", "us-west1-c", "us-west4-a"],
        "v5e-16": ["us-east1-c", "us-west1-c", "us-west4-a"],
        "v5p-8": ["us-east5-a", "us-east5-b"],
        "v5p-16": ["us-east5-a", "us-east5-b"],
        "v6e-1": ["asia-northeast1-b", "us-east5-a", "us-west4-a"],
        "v6e-4": ["asia-northeast1-b", "us-east5-a", "us-west4-a"],
        "v6e-8": ["asia-northeast1-b", "us-east5-a", "us-west4-a"],
        "v6e-16": ["asia-northeast1-b", "us-east5-a", "us-west4-a"],
    }

    def __init__(self):
        """Initialize TPU availability checker."""
        pass

    def normalize_tpu_type(self, tpu_type: str) -> str:
        """Normalize TPU type string.

        Args:
            tpu_type: TPU type (e.g., 'v6e-16', 'ct6e-standard-4t')

        Returns:
            Normalized TPU type (e.g., 'v6e-16')
        """
        # Handle ct6e-standard-4t format
        if "ct" in tpu_type and "standard" in tpu_type:
            # Extract version and topology
            match = re.search(r'ct(\d+[a-z]+)-standard-(\d+)t', tpu_type)
            if match:
                version = match.group(1)
                chips = int(match.group(2))
                return f"v{version}-{chips}"

        # Already in v6e-16 format
        return tpu_type

    def get_available_zones(self, tpu_type: str) -> List[str]:
        """Get available zones for a TPU type.

        Args:
            tpu_type: TPU type (e.g., 'v6e-16')

        Returns:
            List of available zones
        """
        normalized_type = self.normalize_tpu_type(tpu_type)

        # Try to query dynamically using gcloud
        zones = self._query_gcloud_zones(normalized_type)
        if zones:
            return zones

        # Fallback to static mapping
        return self.TPU_ZONE_MAP.get(normalized_type, [])

    def _query_gcloud_zones(self, tpu_type: str) -> List[str]:
        """Query available zones using gcloud.

        Args:
            tpu_type: Normalized TPU type

        Returns:
            List of available zones
        """
        try:
            # Try to list TPU types in all zones
            cmd = [
                "gcloud", "compute", "tpus", "accelerator-types", "list",
                "--format=value(zone,name)",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                zones = []
                for line in result.stdout.split('\n'):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        zone, accel_type = parts[0], parts[1]
                        if tpu_type in accel_type or accel_type in tpu_type:
                            zones.append(zone)
                return list(set(zones))  # Remove duplicates

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

        return []

    def check_zone_availability(self, tpu_type: str, zone: str) -> Tuple[bool, Optional[str]]:
        """Check if a TPU type is available in a specific zone.

        Args:
            tpu_type: TPU type
            zone: GCP zone

        Returns:
            Tuple of (is_available, error_message)
        """
        available_zones = self.get_available_zones(tpu_type)

        if not available_zones:
            return False, f"Could not determine availability for TPU type '{tpu_type}'"

        if zone in available_zones:
            return True, None

        return False, f"TPU type '{tpu_type}' is not available in zone '{zone}'"

    def suggest_alternative_zones(self, tpu_type: str, preferred_region: Optional[str] = None) -> List[str]:
        """Suggest alternative zones for a TPU type.

        Args:
            tpu_type: TPU type
            preferred_region: Preferred region (e.g., 'asia-northeast1', 'us-east5')

        Returns:
            List of suggested zones
        """
        available_zones = self.get_available_zones(tpu_type)

        if not available_zones:
            return []

        # If preferred region is specified, prioritize zones in that region
        if preferred_region:
            region_zones = [z for z in available_zones if z.startswith(preferred_region)]
            if region_zones:
                return region_zones

        return available_zones

    def format_suggestions(self, tpu_type: str, zone: str) -> str:
        """Format zone suggestions as a user-friendly message.

        Args:
            tpu_type: TPU type
            zone: Requested zone

        Returns:
            Formatted suggestion message
        """
        is_available, error_msg = self.check_zone_availability(tpu_type, zone)

        if is_available:
            return f"✓ TPU type '{tpu_type}' is available in zone '{zone}'"

        # Extract region from zone
        region = '-'.join(zone.split('-')[:2]) if '-' in zone else None

        # Get suggestions
        suggestions = self.suggest_alternative_zones(tpu_type, region)

        if not suggestions:
            return f"✗ {error_msg}\n\nNo alternative zones found. Please check the TPU documentation:\nhttps://docs.cloud.google.com/tpu/docs/regions-zones"

        msg = f"✗ {error_msg}\n\nSuggested alternative zones for '{tpu_type}':\n"
        for i, suggested_zone in enumerate(suggestions, 1):
            msg += f"  {i}. {suggested_zone}\n"

        return msg


def main():
    """Main entry point for CLI usage."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: tpu_availability.py <tpu_type> <zone>")
        sys.exit(1)

    checker = TPUAvailabilityChecker()
    tpu_type = sys.argv[1]
    zone = sys.argv[2]

    print(checker.format_suggestions(tpu_type, zone))


if __name__ == "__main__":
    main()
