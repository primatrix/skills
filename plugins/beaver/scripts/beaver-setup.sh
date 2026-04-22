#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-setup.sh <subcommand> [args]

Subcommands:
  auth-status                              gh auth status
  today                                    Print YYYY-MM-DD
  project-id <org> <project-number>        Echo node id of projectV2
  field-list <org> <project-number>        List custom fields (JSON)
  field-create <org> <project-number> <name> <type> [<options-csv>]
                                           Create single-select or number field
  status-replace <field-id>                GraphQL: replace 7 status options
  edit-readme <org> <project-number> <readme-file>
                                           gh project edit --readme
  list-issue-types <org>                   List org issue types
  create-issue-type <org> <name> <color> <desc>
                                           POST issue type, skip on 422
  ensure-label <repo> <name> <color> <desc>
                                           gh label create, skip on dup
  iteration-field-create <project-id> <start-date> <duration> <iterations-json>
                                           Create Iteration field with initial entries
  iteration-field-list <org> <project-number>
                                           Read existing Iteration entries
  iteration-field-append <field-id> <additions-json>
                                           Append missing iteration entries
EOF
}

case "${1:-}" in
  auth-status)
    gh auth status
    ;;
  today)
    date +%Y-%m-%d
    ;;
  project-id)
    org=$2; num=$3
    gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) { projectV2(number: $number) { id } }
      }' -f owner="$org" -F number="$num" \
      --jq '.data.organization.projectV2.id'
    ;;
  field-list)
    org=$2; num=$3
    gh project field-list "$num" --owner "$org" --format json
    ;;
  field-create)
    org=$2; num=$3; name=$4; type=$5
    if [ "$type" = "SINGLE_SELECT" ]; then
      opts=$6
      gh project field-create "$num" --owner "$org" --name "$name" \
        --data-type SINGLE_SELECT --single-select-options "$opts"
    else
      gh project field-create "$num" --owner "$org" --name "$name" --data-type "$type"
    fi
    ;;
  status-replace)
    field_id=$2
    gh api graphql -f query='
      mutation($fieldId: ID!) {
        updateProjectV2Field(input: {
          fieldId: $fieldId
          singleSelectOptions: [
            {name: "Triage", color: GRAY, description: "Awaiting triage"},
            {name: "Ready to Claim", color: BLUE, description: "Added to Iteration, awaiting claim"},
            {name: "Design Pending", color: PURPLE, description: "Design review in progress (size/L)"},
            {name: "Ready to Develop", color: ORANGE, description: "Ready to code (size/L, design approved)"},
            {name: "In Progress", color: YELLOW, description: "Active development"},
            {name: "Blocked", color: RED, description: "Blocked"},
            {name: "Done", color: GREEN, description: "Completed and merged"}
          ]
        }) {
          projectV2Field {
            ... on ProjectV2SingleSelectField { name options { name } }
          }
        }
      }' -f fieldId="$field_id"
    ;;
  edit-readme)
    org=$2; num=$3; file=$4
    gh project edit "$num" --owner "$org" --readme "$(cat "$file")"
    ;;
  list-issue-types)
    org=$2
    gh api "orgs/${org}/issue-types" -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '.[].name'
    ;;
  create-issue-type)
    org=$2; name=$3; color=$4; desc=$5
    gh api "orgs/${org}/issue-types" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -f name="$name" -f color="$color" -f description="$desc" \
      -F is_enabled=true 2>/dev/null || true
    ;;
  ensure-label)
    repo=$2; name=$3; color=$4; desc=$5
    gh label create "$name" --repo "$repo" --color "$color" --description "$desc" 2>/dev/null || true
    ;;
  iteration-field-create)
    project_id=$2; start=$3; duration=$4; iterations=$5
    read -r -d '' M <<'GRAPHQL' || true
mutation($projectId: ID!, $startDate: Date!, $duration: Int!, $iterations: [ProjectV2IterationFieldIterationInput!]!) {
  createProjectV2Field(input: {
    projectId: $projectId
    dataType: ITERATION
    name: "Iteration"
    iterationConfiguration: {
      startDate: $startDate
      duration: $duration
      iterations: $iterations
    }
  }) { projectV2Field { ... on ProjectV2IterationField { id name } } }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f projectId="$project_id" \
      -f startDate="$start" \
      -F duration="$duration" \
      -f iterations="$iterations"
    ;;
  iteration-field-list)
    org=$2; num=$3
    gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) { projectV2(number: $number) {
          field(name: "Iteration") {
            ... on ProjectV2IterationField {
              id
              configuration { iterations { title startDate duration } }
            }
          }
        } }
      }' -f owner="$org" -F number="$num"
    ;;
  iteration-field-append)
    field_id=$2; additions=$3
    read -r -d '' M <<'GRAPHQL' || true
mutation($fieldId: ID!, $additions: [ProjectV2IterationFieldIterationInput!]!) {
  updateProjectV2IterationField(input: {
    iterationFieldId: $fieldId
    additions: $additions
  }) { iterationField { id } }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f fieldId="$field_id" \
      -f additions="$additions"
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
