#!/usr/bin/env bash
#
# Test for Issue #181:
#   "修改 beaver-focus：除了 Review 以外也展示分配的 Issue（Project #14）"
#
# Verifies that _project_items_to_records does NOT filter out issues
# lacking the Control-By-Beaver label — all open issues in Project #14
# should be visible regardless of labels.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
focus_sh="$repo_root/plugins/beaver/scripts/beaver-focus.sh"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

[ -f "$focus_sh" ] || { echo "FAIL: $focus_sh missing" >&2; exit 1; }

# ---------- AC1: _project_items_to_records must NOT require Control-By-Beaver label ----------
# The jq filter in _project_items_to_records should not contain a label check
# for "Control-By-Beaver".
if grep -q 'Control-By-Beaver' "$focus_sh"; then
  report_fail "AC1: beaver-focus.sh still references Control-By-Beaver — filter should be removed"
else
  report_pass "AC1: beaver-focus.sh does not filter by Control-By-Beaver label"
fi

# ---------- AC2: _project_items_to_records still requires OPEN state ----------
# We must keep the state == "OPEN" filter — only the label filter is removed.
if grep -qE '\.content\.state\s*==\s*"OPEN"' "$focus_sh"; then
  report_pass "AC2: _project_items_to_records still filters for OPEN state"
else
  report_fail "AC2: _project_items_to_records must still filter for OPEN state"
fi

# ---------- AC3: Unit test with mock data ----------
# Source the function via a subshell trick: extract _project_items_to_records
# and feed it mock JSON that includes an issue WITHOUT the Control-By-Beaver label.
mock_input='{
  "data": {
    "organization": {
      "projectV2": {
        "items": {
          "nodes": [
            {
              "content": {
                "number": 999,
                "title": "Test issue without label",
                "url": "https://github.com/primatrix/projects/issues/999",
                "state": "OPEN",
                "createdAt": "2026-04-01T00:00:00Z",
                "updatedAt": "2026-04-01T00:00:00Z",
                "issueType": {"name": "Task"},
                "repository": {"name": "projects", "nameWithOwner": "primatrix/projects"},
                "labels": {"nodes": []},
                "assignees": {"nodes": [{"login": "testuser"}]},
                "comments": {"nodes": []}
              },
              "status": {"name": "In Progress"},
              "priority": {"name": "p/2-normal"},
              "iter": null
            },
            {
              "content": {
                "number": 1000,
                "title": "Test issue with label",
                "url": "https://github.com/primatrix/projects/issues/1000",
                "state": "OPEN",
                "createdAt": "2026-04-01T00:00:00Z",
                "updatedAt": "2026-04-01T00:00:00Z",
                "issueType": {"name": "Task"},
                "repository": {"name": "projects", "nameWithOwner": "primatrix/projects"},
                "labels": {"nodes": [{"name": "Control-By-Beaver"}]},
                "assignees": {"nodes": [{"login": "testuser"}]},
                "comments": {"nodes": []}
              },
              "status": {"name": "Ready to Develop"},
              "priority": {"name": "p/1-high"},
              "iter": null
            },
            {
              "content": {
                "number": 1001,
                "title": "Closed issue should be excluded",
                "url": "https://github.com/primatrix/projects/issues/1001",
                "state": "CLOSED",
                "createdAt": "2026-04-01T00:00:00Z",
                "updatedAt": "2026-04-01T00:00:00Z",
                "issueType": {"name": "Bug"},
                "repository": {"name": "projects", "nameWithOwner": "primatrix/projects"},
                "labels": {"nodes": []},
                "assignees": {"nodes": [{"login": "testuser"}]},
                "comments": {"nodes": []}
              },
              "status": {"name": "Triage"},
              "priority": null,
              "iter": null
            }
          ]
        }
      }
    }
  }
}'

# Extract the jq filter from _project_items_to_records and run against mock data.
# The jq expression spans from "jq '" to the matching closing "'".
jq_filter=$(awk '
  /^_project_items_to_records\(\)/, /^}/ {
    if (capture) { buf = buf " " $0 }
    if (/jq '\''/) { capture=1; buf=$0; sub(/.*jq '\''/, "", buf) }
    if (capture && /'\''$/) {
      sub(/'\''$/, "", buf)
      print buf
      exit
    }
  }
' "$focus_sh")

result=$(echo "$mock_input" | jq "$jq_filter")

count=$(echo "$result" | jq 'length')
if [ "$count" -eq 2 ]; then
  report_pass "AC3: _project_items_to_records returns both labeled and unlabeled open issues (got $count)"
else
  report_fail "AC3: _project_items_to_records should return 2 open issues (labeled + unlabeled), got $count"
  echo "  Result: $result" >&2
fi

# Check that the closed issue was excluded
closed_count=$(echo "$result" | jq '[.[] | select(.number == 1001)] | length')
if [ "$closed_count" -eq 0 ]; then
  report_pass "AC3.closed: closed issue correctly excluded"
else
  report_fail "AC3.closed: closed issue should be excluded but was included"
fi

# ---------- summary ----------
echo
if [ "$failures" -eq 0 ]; then
  echo "All acceptance assertions passed."
  exit 0
else
  echo "$failures assertion(s) failed."
  exit 1
fi
