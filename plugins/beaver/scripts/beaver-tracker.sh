#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-tracker.sh <subcommand> [args]

Subcommands:
  ensure-label <repo> <name> <color> <description>
                                Idempotent label-create (skip on dup)
  find-tracker <repo> <yyyymm>  Search for tracker issue, echo JSON {count, items:[{number,title}]}
  list-carried <prev-number>    Open sub-issues of prior tracker, echo [{number, title}]
  create <repo> <title> <body-file>
                                Create tracker issue, echo .number
  add-labels <repo> <number> <label> [<label> ...]
  resolve-issue-id <repo> <number>
                                Echo numeric DB .id of issue
  attach-sub <tracker-number> <child-id>
                                Attach via sub_issues API
  fetch-triage-backlog          GraphQL fetch primatrix/projects status/triage no-iteration items
  list-tracker-subs <tracker-number>
                                Echo .[].number of sub-issues
  resolve-iteration <yyyymm>    Echo "project_id=ID field_id=ID iteration_id=ID"
  resolve-item-id <repo> <issue-number>
                                Echo ProjectV2Item id for project #14 (empty if not on project)
  add-to-project <project-id> <issue-number>
                                addProjectV2ItemById, echo new item id
  set-iteration <project-id> <item-id> <field-id> <iteration-id>
                                updateProjectV2ItemFieldValue
EOF
}

ORG=primatrix
PROJECT_REPO=projects
PROJECT_NUM=14

case "${1:-}" in
  ensure-label)
    repo=$2; name=$3; color=$4; desc=$5
    gh label create "$name" --repo "${ORG}/${repo}" --color "$color" --description "$desc" 2>/dev/null || true
    ;;
  find-tracker)
    repo=$2; yyyymm=$3
    gh api -X GET search/issues \
      -f q="repo:${ORG}/${PROJECT_REPO} is:issue label:\"tracker/${repo}\" label:\"tracker/${yyyymm}\"" \
      --jq '{count: (.items | length), items: [.items[] | {number, state, title}]}'
    ;;
  list-carried)
    prev=$2
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${prev}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '[.[] | select(.state=="open") | {number, title}]'
    ;;
  create)
    repo=$2; title=$3; body_file=$4
    gh api "repos/${ORG}/${PROJECT_REPO}/issues" --method POST \
      -f title="$title" \
      -F body=@"$body_file" \
      --jq '.number'
    ;;
  add-labels)
    repo=$2; num=$3; shift 3
    args=()
    for label in "$@"; do
      args+=(-f "labels[]=${label}")
    done
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  resolve-issue-id)
    repo=$2; num=$3
    gh api "repos/${ORG}/${repo}/issues/${num}" --jq '.id'
    ;;
  attach-sub)
    tracker=$2; child_id=$3
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  fetch-triage-backlog)
    gh api graphql -f query='
      query {
        organization(login: "primatrix") {
          projectV2(number: 14) {
            items(first: 100) {
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    repository { nameWithOwner }
                    labels(first: 30) { nodes { name } }
                  }
                }
                fieldValueByName(name: "Iteration") {
                  ... on ProjectV2ItemFieldIterationValue { title }
                }
              }
            }
          }
        }
      }' --jq '.data.organization.projectV2.items.nodes
                | map(select(.content != null and .content.repository.nameWithOwner == "primatrix/projects"))
                | map(select(.content.labels.nodes | map(.name) | index("status/triage")))
                | map(select(.fieldValueByName == null))
                | map({number: .content.number, title: .content.title})'
    ;;
  list-tracker-subs)
    tracker=$2
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '.[].number'
    ;;
  resolve-iteration)
    yyyymm=$2
    info=$(gh api graphql -f query='
      query {
        organization(login: "primatrix") {
          projectV2(number: 14) {
            id
            field(name: "Iteration") {
              ... on ProjectV2IterationField {
                id
                configuration { iterations { id title } }
              }
            }
          }
        }
      }')
    project_id=$(echo "$info" | jq -r '.data.organization.projectV2.id')
    field_id=$(echo "$info" | jq -r '.data.organization.projectV2.field.id')
    iteration_id=$(echo "$info" | jq -r --arg yyyymm "$yyyymm" \
      '.data.organization.projectV2.field.configuration.iterations
       | map(select(.title | startswith($yyyymm))) | .[0].id // ""')
    echo "project_id=${project_id} field_id=${field_id} iteration_id=${iteration_id}"
    ;;
  resolve-item-id)
    repo=$2; num=$3
    read -r -d '' Q <<'GRAPHQL' || true
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      projectItems(first: 10) { nodes { id project { number } } }
    }
  }
}
GRAPHQL
    gh api graphql \
      -f query="$Q" \
      -f owner="$ORG" \
      -f repo="$repo" \
      -F number="$num" \
      --jq '.data.repository.issue.projectItems.nodes
            | map(select(.project.number == 14)) | .[0].id // ""'
    ;;
  add-to-project)
    project_id=$2; num=$3
    content_id=$(gh api "repos/${ORG}/${PROJECT_REPO}/issues/${num}" --jq '.node_id')
    read -r -d '' M <<'GRAPHQL' || true
mutation($projectId: ID!, $contentId: ID!) {
  addProjectV2ItemById(input: { projectId: $projectId, contentId: $contentId }) {
    item { id }
  }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f projectId="$project_id" \
      -f contentId="$content_id" \
      --jq '.data.addProjectV2ItemById.item.id'
    ;;
  set-iteration)
    project_id=$2; item_id=$3; field_id=$4; iteration_id=$5
    read -r -d '' M <<'GRAPHQL' || true
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
      -f query="$M" \
      -f projectId="$project_id" \
      -f itemId="$item_id" \
      -f fieldId="$field_id" \
      -f iterationId="$iteration_id"
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
