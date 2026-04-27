#!/usr/bin/env bash
#
# beaver-lib.sh — Shared library for Beaver commands.
#
# Provides 12 public functions that read/write Project V2 #14 fields and the
# native GitHub Issue Type, plus a `self-test` subcommand that exercises the
# full read/write round-trip on a sandbox issue in primatrix/projects.
#
# May be sourced (`source beaver-lib.sh`) or invoked as a CLI:
#   bash beaver-lib.sh self-test
#   bash beaver-lib.sh <function-name> <args...>
#
# Public API (see RFC-0013 §"Public API"):
#   resolve_item_id <issue_url>
#   get_field_id <field_name>
#   get_option_id <field_name> <option_name>
#   set_status <issue_number> <status>
#   set_size <issue_number> <size>
#   set_type <issue_number> <type_name>
#   get_type <issue_number>
#   get_iteration <issue_number>
#   set_iteration <issue_number> <iteration_title>
#   latest_iteration_for_repo <repo>
#   get_target_date <issue_number>
#   set_target_date <issue_number> <date>

set -euo pipefail

ORG=primatrix
PROJECT_REPO=projects
PROJECT_NUMBER=14

# ---------- internal helpers ----------

# Echo Project V2 node id (cached after first call).
_project_id() {
  if [ -z "${_BEAVER_LIB_PROJECT_ID:-}" ]; then
    _BEAVER_LIB_PROJECT_ID=$(gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) { projectV2(number: $number) { id } }
      }' -f owner="$ORG" -F number="$PROJECT_NUMBER" \
      --jq '.data.organization.projectV2.id')
  fi
  printf '%s' "$_BEAVER_LIB_PROJECT_ID"
}

# Echo node id of an issue in primatrix/projects given its number.
_issue_node_id() {
  local number=$1
  gh api "repos/${ORG}/${PROJECT_REPO}/issues/${number}" --jq '.node_id'
}

# Echo project item id for issue <number> on Project #14, or empty string.
_resolve_item_id_by_number() {
  local number=$1
  gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) { nodes { id project { number } } }
        }
      }
    }' -f owner="$ORG" -f repo="$PROJECT_REPO" -F number="$number" \
    --jq ".data.repository.issue.projectItems.nodes
          | map(select(.project.number == ${PROJECT_NUMBER})) | .[0].id // \"\""
}

# Echo "<projectItemId>" — adds the issue if not yet on Project #14.
_ensure_item_id() {
  local number=$1
  local item_id
  item_id=$(_resolve_item_id_by_number "$number")
  if [ -z "$item_id" ]; then
    local content_id project_id
    content_id=$(_issue_node_id "$number")
    project_id=$(_project_id)
    item_id=$(gh api graphql -f query='
      mutation($projectId: ID!, $contentId: ID!) {
        addProjectV2ItemById(input: { projectId: $projectId, contentId: $contentId }) {
          item { id }
        }
      }' -f projectId="$project_id" -f contentId="$content_id" \
      --jq '.data.addProjectV2ItemById.item.id')
  fi
  printf '%s' "$item_id"
}

# Echo singleSelect option id for a field, or fail.
_set_single_select() {
  local item_id=$1 field_id=$2 option_id=$3
  local project_id; project_id=$(_project_id)
  gh api graphql -f query='
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId
        itemId: $itemId
        fieldId: $fieldId
        value: { singleSelectOptionId: $optionId }
      }) { projectV2Item { id } }
    }' \
    -f projectId="$project_id" \
    -f itemId="$item_id" \
    -f fieldId="$field_id" \
    -f optionId="$option_id" >/dev/null
}

# Echo current single-select value name (or empty) for given field.
_get_single_select_value() {
  local number=$1 field_name=$2
  gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!, $field: String!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) {
            nodes {
              project { number }
              fieldValueByName(name: $field) {
                ... on ProjectV2ItemFieldSingleSelectValue { name }
              }
            }
          }
        }
      }
    }' -f owner="$ORG" -f repo="$PROJECT_REPO" -F number="$number" -f field="$field_name" \
    --jq ".data.repository.issue.projectItems.nodes
          | map(select(.project.number == ${PROJECT_NUMBER})) | .[0].fieldValueByName.name // \"\""
}

# ---------- public API ----------

# resolve_item_id <issue_url>
# Echo Project V2 item id for the given issue URL on Project #14, or empty.
resolve_item_id() {
  local url=$1
  # Parse owner/repo/number from URL like https://github.com/<owner>/<repo>/issues/<n>
  local stripped="${url#https://github.com/}"
  local owner="${stripped%%/*}"; stripped="${stripped#*/}"
  local repo="${stripped%%/*}"; stripped="${stripped#*/}"
  local number="${stripped##*/}"
  gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) { nodes { id project { number } } }
        }
      }
    }' -f owner="$owner" -f repo="$repo" -F number="$number" \
    --jq ".data.repository.issue.projectItems.nodes
          | map(select(.project.number == ${PROJECT_NUMBER})) | .[0].id // \"\""
}

# get_field_id <field_name>
# Echo Project V2 field id on Project #14 for a custom field by name.
get_field_id() {
  local name=$1
  gh api graphql -f query='
    query($owner: String!, $number: Int!, $field: String!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          field(name: $field) {
            ... on ProjectV2FieldCommon { id }
          }
        }
      }
    }' -f owner="$ORG" -F number="$PROJECT_NUMBER" -f field="$name" \
    --jq '.data.organization.projectV2.field.id // ""'
}

# get_option_id <field_name> <option_name>
# Echo singleSelect option id within a field, or empty.
get_option_id() {
  local field=$1 option=$2
  gh api graphql -f query='
    query($owner: String!, $number: Int!, $field: String!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          field(name: $field) {
            ... on ProjectV2SingleSelectField { options { id name } }
          }
        }
      }
    }' -f owner="$ORG" -F number="$PROJECT_NUMBER" -f field="$field" \
    | jq -r --arg opt "$option" \
      '.data.organization.projectV2.field.options // []
       | map(select(.name == $opt)) | .[0].id // ""'
}

# set_status <issue_number> <status_name>
set_status() {
  local number=$1 status=$2
  local item_id field_id option_id
  item_id=$(_ensure_item_id "$number")
  field_id=$(get_field_id "Status")
  option_id=$(get_option_id "Status" "$status")
  if [ -z "$option_id" ]; then
    echo "set_status: unknown Status option: $status" >&2
    return 1
  fi
  _set_single_select "$item_id" "$field_id" "$option_id"
}

# set_size <issue_number> <size_name>
set_size() {
  local number=$1 size=$2
  local item_id field_id option_id
  item_id=$(_ensure_item_id "$number")
  field_id=$(get_field_id "Size")
  option_id=$(get_option_id "Size" "$size")
  if [ -z "$option_id" ]; then
    echo "set_size: unknown Size option: $size" >&2
    return 1
  fi
  _set_single_select "$item_id" "$field_id" "$option_id"
}

# set_type <issue_number> <type_name>
# Sets the *native* GitHub Issue Type via updateIssueIssueType.
# Requires admin:org scope and the issue_types public-preview feature header.
set_type() {
  local number=$1 type_name=$2
  local issue_id type_id
  issue_id=$(_issue_node_id "$number")
  type_id=$(gh api -H "GraphQL-Features: issue_types" graphql -f query='
    query($owner: String!) {
      organization(login: $owner) {
        issueTypes(first: 50) { nodes { id name } }
      }
    }' -f owner="$ORG" \
    | jq -r --arg name "$type_name" \
      '.data.organization.issueTypes.nodes
       | map(select(.name == $name)) | .[0].id // ""')
  if [ -z "$type_id" ]; then
    echo "set_type: unknown Issue Type: $type_name" >&2
    return 1
  fi
  gh api -H "GraphQL-Features: issue_types" graphql -f query='
    mutation($issueId: ID!, $issueTypeId: ID!) {
      updateIssueIssueType(input: {
        issueId: $issueId
        issueTypeId: $issueTypeId
      }) { issue { id issueType { name } } }
    }' -f issueId="$issue_id" -f issueTypeId="$type_id" >/dev/null
}

# get_type <issue_number>
# Echo native Issue Type name (or empty string if not set).
get_type() {
  local number=$1
  gh api -H "GraphQL-Features: issue_types" graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) { issueType { name } }
      }
    }' -f owner="$ORG" -f repo="$PROJECT_REPO" -F number="$number" \
    --jq '.data.repository.issue.issueType.name // ""'
}

# get_iteration <issue_number>
# Echo current Iteration title for the issue (or empty).
get_iteration() {
  local number=$1
  gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) {
            nodes {
              project { number }
              fieldValueByName(name: "Iteration") {
                ... on ProjectV2ItemFieldIterationValue { title }
              }
            }
          }
        }
      }
    }' -f owner="$ORG" -f repo="$PROJECT_REPO" -F number="$number" \
    --jq ".data.repository.issue.projectItems.nodes
          | map(select(.project.number == ${PROJECT_NUMBER}))
          | .[0].fieldValueByName.title // \"\""
}

# set_iteration <issue_number> <iteration_title>
# Title may be a full iteration title or a YYYY-MM prefix.
set_iteration() {
  local number=$1 iteration_title=$2
  local item_id field_id project_id iteration_id
  item_id=$(_ensure_item_id "$number")
  field_id=$(get_field_id "Iteration")
  project_id=$(_project_id)
  iteration_id=$(gh api graphql -f query='
    query($owner: String!, $number: Int!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          field(name: "Iteration") {
            ... on ProjectV2IterationField {
              configuration { iterations { id title } }
            }
          }
        }
      }
    }' -f owner="$ORG" -F number="$PROJECT_NUMBER" \
    | jq -r --arg t "$iteration_title" \
      '.data.organization.projectV2.field.configuration.iterations // []
       | map(select(.title == $t or (.title | startswith($t)))) | .[0].id // ""')
  if [ -z "$iteration_id" ]; then
    echo "set_iteration: no Iteration entry matches: $iteration_title" >&2
    return 1
  fi
  gh api graphql -f query='
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId
        itemId: $itemId
        fieldId: $fieldId
        value: { iterationId: $iterationId }
      }) { projectV2Item { id } }
    }' \
    -f projectId="$project_id" \
    -f itemId="$item_id" \
    -f fieldId="$field_id" \
    -f iterationId="$iteration_id" >/dev/null
}

# get_target_date <issue_number>
# Echo current "Target date" value (YYYY-MM-DD) for the issue, or empty string.
get_target_date() {
  local number=$1
  gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) {
            nodes {
              project { number }
              fieldValueByName(name: "Target date") {
                ... on ProjectV2ItemFieldDateValue { date }
              }
            }
          }
        }
      }
    }' -f owner="$ORG" -f repo="$PROJECT_REPO" -F number="$number" \
    --jq ".data.repository.issue.projectItems.nodes
          | map(select(.project.number == ${PROJECT_NUMBER}))
          | .[0].fieldValueByName.date // \"\""
}

# set_target_date <issue_number> <date>
# Set "Target date" field (YYYY-MM-DD) on Project #14.
set_target_date() {
  local number=$1 date=$2
  local item_id field_id project_id
  item_id=$(_ensure_item_id "$number")
  field_id=$(get_field_id "Target date")
  project_id=$(_project_id)
  gh api graphql -f query='
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $date: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId
        itemId: $itemId
        fieldId: $fieldId
        value: { date: $date }
      }) { projectV2Item { id } }
    }' \
    -f projectId="$project_id" \
    -f itemId="$item_id" \
    -f fieldId="$field_id" \
    -f date="$date" >/dev/null
}

# latest_iteration_for_repo <repo>
# Implements RFC-0013 G011 algorithm:
#   Step A: select Iteration entry where startDate <= today < endDate
#           - exactly 1: return its title
#           - more than 1: error (ambiguous)
#   Step B: if A is empty, select min(startDate) where startDate > today
#           - return its title
#   Step C: if both empty, echo nothing and return non-zero (caller maps to G011 fail)
# Today is evaluated in UTC.
latest_iteration_for_repo() {
  local repo=$1
  local today
  today=$(date -u +%Y-%m-%d)
  local entries
  entries=$(gh api graphql -f query='
    query($owner: String!, $number: Int!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          field(name: "Iteration") {
            ... on ProjectV2IterationField {
              configuration { iterations { title startDate duration } }
            }
          }
        }
      }
    }' -f owner="$ORG" -F number="$PROJECT_NUMBER" \
    --jq '.data.organization.projectV2.field.configuration.iterations // []')

  # Step A: current iterations.
  local current
  current=$(echo "$entries" | jq --arg t "$today" '
    [ .[] | . as $i
      | ($i.startDate) as $s
      | (($i.startDate | strptime("%Y-%m-%d") | mktime) + ($i.duration * 86400)
         | strftime("%Y-%m-%d")) as $e
      | select($s <= $t and $t < $e) ]')
  local current_count
  current_count=$(echo "$current" | jq 'length')
  if [ "$current_count" = "1" ]; then
    echo "$current" | jq -r '.[0].title'
    return 0
  elif [ "$current_count" -gt 1 ]; then
    local titles
    titles=$(echo "$current" | jq -r '[.[].title] | join(", ")')
    echo "Ambiguous current Iteration for ${repo}: ${titles}. Resolve overlap before retrying." >&2
    return 1
  fi

  # Step B: next future iteration.
  local future_title
  future_title=$(echo "$entries" | jq -r --arg t "$today" '
    [ .[] | select(.startDate > $t) ]
    | sort_by(.startDate) | .[0].title // ""')
  if [ -n "$future_title" ]; then
    echo "$future_title"
    return 0
  fi

  # Step C: nothing found.
  echo "No current or future Iteration found on Project #14 for ${repo}. Run /beaver-tracker ${repo} to create this month's Iteration entry." >&2
  return 1
}

# ---------- self-test ----------

# Globals (used by trap-installed cleanup so they survive _self_test return).
_BEAVER_LIB_SANDBOX_NUMBER=""
_BEAVER_LIB_SANDBOX_NODE_ID=""

_self_test_cleanup() {
  if [ -z "$_BEAVER_LIB_SANDBOX_NUMBER" ]; then
    return 0
  fi
  if [ -z "$_BEAVER_LIB_SANDBOX_NODE_ID" ]; then
    _BEAVER_LIB_SANDBOX_NODE_ID=$(_issue_node_id "$_BEAVER_LIB_SANDBOX_NUMBER" 2>/dev/null || echo "")
  fi
  if [ -n "$_BEAVER_LIB_SANDBOX_NODE_ID" ]; then
    echo "self-test: cleanup deleting sandbox issue #${_BEAVER_LIB_SANDBOX_NUMBER}..."
    gh api graphql -f query='
      mutation($issueId: ID!) {
        deleteIssue(input: { issueId: $issueId }) { repository { name } }
      }' -f issueId="$_BEAVER_LIB_SANDBOX_NODE_ID" >/dev/null 2>&1 || \
      echo "self-test: WARN: failed to delete issue #${_BEAVER_LIB_SANDBOX_NUMBER}; please remove manually" >&2
  else
    echo "self-test: WARN: could not resolve node_id for #${_BEAVER_LIB_SANDBOX_NUMBER}; please remove manually" >&2
  fi
}

_self_test() {
  local sandbox_title="[beaver-lib self-test] $(date -u +%Y-%m-%dT%H:%M:%SZ) $$"
  local sandbox_body="Auto-generated sandbox issue for beaver-lib.sh self-test. Will be deleted on success."
  echo "self-test: creating sandbox issue in ${ORG}/${PROJECT_REPO}..."

  local issue_url
  issue_url=$(gh issue create --repo "${ORG}/${PROJECT_REPO}" \
    --title "$sandbox_title" --body "$sandbox_body")
  _BEAVER_LIB_SANDBOX_NUMBER=${issue_url##*/}
  # Install cleanup trap as soon as we know the issue number, so a failure in
  # _issue_node_id below cannot leak the sandbox issue. The cleanup helper
  # resolves node_id on demand if it isn't set yet.
  trap _self_test_cleanup EXIT
  _BEAVER_LIB_SANDBOX_NODE_ID=$(_issue_node_id "$_BEAVER_LIB_SANDBOX_NUMBER")
  echo "self-test: sandbox issue #${_BEAVER_LIB_SANDBOX_NUMBER} created (${issue_url})"

  local number=$_BEAVER_LIB_SANDBOX_NUMBER
  local actual

  # Round-trip 1: Status
  echo "self-test: set_status In Progress"
  set_status "$number" "In Progress"
  actual=$(_get_single_select_value "$number" "Status")
  if [ "$actual" != "In Progress" ]; then
    echo "self-test: FAIL: Status round-trip expected 'In Progress', got '$actual'" >&2
    return 1
  fi

  # Round-trip 2: Size
  echo "self-test: set_size L"
  set_size "$number" "L"
  actual=$(_get_single_select_value "$number" "Size")
  if [ "$actual" != "L" ]; then
    echo "self-test: FAIL: Size round-trip expected 'L', got '$actual'" >&2
    return 1
  fi

  # Round-trip 3: Type
  echo "self-test: set_type Task"
  set_type "$number" "Task"
  actual=$(get_type "$number")
  if [ "$actual" != "Task" ]; then
    echo "self-test: FAIL: Type round-trip expected 'Task', got '$actual'" >&2
    return 1
  fi

  # Close issue (cleanup deletes it).
  echo "self-test: closing sandbox issue"
  gh issue close "$_BEAVER_LIB_SANDBOX_NUMBER" --repo "${ORG}/${PROJECT_REPO}" >/dev/null

  echo "self-test: PASS — Status/Size/Type round-trips equal."
  return 0
}

# ---------- CLI dispatch ----------

# Only dispatch CLI if executed directly (not sourced).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  case "${1:-}" in
    self-test)
      _self_test
      ;;
    resolve_item_id|get_field_id|get_option_id|\
    set_status|set_size|set_type|get_type|\
    get_iteration|set_iteration|latest_iteration_for_repo|\
    get_target_date|set_target_date)
      fn=$1; shift
      "$fn" "$@"
      ;;
    --help|-h|"")
      cat <<'EOF'
Usage: beaver-lib.sh <subcommand> [args]

Subcommands:
  self-test                                Run sandbox round-trip on primatrix/projects
  resolve_item_id <issue_url>
  get_field_id <field_name>
  get_option_id <field_name> <option_name>
  set_status <issue_number> <status_name>
  set_size <issue_number> <size_name>
  set_type <issue_number> <type_name>
  get_type <issue_number>
  get_iteration <issue_number>
  set_iteration <issue_number> <iteration_title>
  latest_iteration_for_repo <repo>
  get_target_date <issue_number>
  set_target_date <issue_number> <date>

May also be sourced: `source beaver-lib.sh` exposes the public functions.
EOF
      ;;
    *)
      echo "unknown subcommand: $1" >&2
      exit 1
      ;;
  esac
fi
