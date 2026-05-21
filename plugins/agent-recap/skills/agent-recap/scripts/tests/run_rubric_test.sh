#!/usr/bin/env bash
# Manual rubric verification harness.
#
# Usage:
#   bash run_rubric_test.sh
#
# This script does NOT invoke an LLM. It prints the subagent prompt template
# and the fixture paths so a human reviewer can dispatch them via Claude Code's
# Agent tool, collect the returned JSON, and diff against expected.json.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/fixtures/classification"

echo "=== agent-recap rubric verification ==="
echo
echo "Expected classifications:"
cat "$FIX/expected.json"
echo
echo "For each fixture, dispatch one Explore subagent with the prompt template"
echo "from SKILL.md (Stage 2). The subagent should return JSON with a 'type'"
echo "field matching the expected value above."
echo
echo "Fixtures to classify:"
for f in "$FIX"/*.jsonl; do
  basename "$f" .jsonl
  echo "  path: $f"
done
echo
echo "After collecting all 5 results, verify each 'type' matches expected.json."
echo "Mismatches indicate either:"
echo "  (a) classification-rubric.md needs adjustment, or"
echo "  (b) the fixture is ambiguous and should be revised."
