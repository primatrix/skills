#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-create.sh <subcommand> [args]

Subcommands:
  create-issue <org> <repo> <title> <body-file>
                                Create issue, echo .number
  fetch-ids <org> <repo> <number>
                                Echo "id=N node_id=NID html_url=URL"
  add-labels <org> <repo> <number> <label> [<label> ...]
                                POST labels
  add-to-project <project-number> <org> <issue-url>
                                Add issue to project, echo item id
  set-field <item-id> <project-id> <field-id> <option-id>
                                Set single-select field
  resolve-iteration <org> <project-number> <yyyymm>
                                Echo "project_id=ID field_id=ID iteration_id=ID"
                                Iteration ID is empty if not found.
  set-iteration <project-id> <item-id> <field-id> <iteration-id>
                                Set Iteration field via GraphQL mutation
  link-parent <org> <repo> <parent-number> <child-id>
                                Attach child as sub-issue
EOF
}

case "${1:-}" in
  create-issue)
    org=$2; repo=$3; title=$4; body_file=$5
    gh api "repos/${org}/${repo}/issues" --method POST \
      -f title="$title" \
      -F body=@"$body_file" \
      --jq '.number'
    ;;
  fetch-ids)
    org=$2; repo=$3; num=$4
    id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.id')
    node_id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.node_id')
    url=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.html_url')
    echo "id=${id} node_id=${node_id} html_url=${url}"
    ;;
  add-labels)
    org=$2; repo=$3; num=$4; shift 4
    args=()
    for label in "$@"; do
      args+=(-f "labels[]=${label}")
    done
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  add-to-project)
    project=$2; org=$3; url=$4
    gh project item-add "$project" --owner "$org" --url "$url" \
      --format json --jq '.id'
    ;;
  set-field)
    item=$2; project=$3; field=$4; option=$5
    gh project item-edit --id "$item" --project-id "$project" \
      --field-id "$field" --single-select-option-id "$option"
    ;;
  resolve-iteration)
    org=$2; project=$3; yyyymm=$4
    info=$(gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) {
          projectV2(number: $number) {
            id
            field(name: "Iteration") {
              ... on ProjectV2IterationField {
                id
                configuration { iterations { id title } }
              }
            }
          }
        }
      }' -f owner="$org" -F number="$project")
    project_id=$(echo "$info" | jq -r '.data.organization.projectV2.id')
    field_id=$(echo "$info" | jq -r '.data.organization.projectV2.field.id')
    iteration_id=$(echo "$info" | jq -r --arg yyyymm "$yyyymm" \
      '.data.organization.projectV2.field.configuration.iterations
       | map(select(.title | startswith($yyyymm))) | .[0].id // ""')
    echo "project_id=${project_id} field_id=${field_id} iteration_id=${iteration_id}"
    ;;
  set-iteration)
    project_id=$2; item_id=$3; field_id=$4; iteration_id=$5
    read -r -d '' MUT <<'GRAPHQL' || true
mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
  updateProjectV2ItemFieldValue(input: {
    projectId: $projectId
    itemId: $itemId
    fieldId: $fieldId
    value: { iterationId: $iterationId }
  }) { projectV2Item { id } }
}
GRAPHQL
    gh api graphql \
      -f query="$MUT" \
      -f projectId="$project_id" \
      -f itemId="$item_id" \
      -f fieldId="$field_id" \
      -f iterationId="$iteration_id"
    ;;
  link-parent)
    org=$2; repo=$3; parent=$4; child_id=$5
    gh api "repos/${org}/${repo}/issues/${parent}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
