#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-focus.sh <subcommand> [args]

Subcommands:
  whoami                              Print current gh user
  fetch-my-issues <user>              Print my open Beaver issues from project #14 (JSON array)
  fetch-review-prs <user>             Print PRs awaiting my review (JSON lines)
EOF
}

case "${1:-}" in
  whoami)
    gh api user --jq '.login'
    ;;
  fetch-my-issues)
    user=$2
    gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) {
          projectV2(number: $number) {
            items(first: 100) {
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    state
                    labels(first: 30) { nodes { name } }
                    assignees(first: 10) { nodes { login } }
                  }
                }
                fieldValueByName(name: "Iteration") {
                  ... on ProjectV2ItemFieldIterationValue {
                    title
                    startDate
                    duration
                  }
                }
              }
            }
          }
        }
      }' -f owner=primatrix -F number=14 \
      | jq --arg user "$user" '.data.organization.projectV2.items.nodes
            | map(select(.content != null and .content.state == "OPEN"))
            | map(select(.content.assignees.nodes | map(.login) | index($user)))
            | map(select(.content.labels.nodes | map(.name) | index("Control-By-Beaver")))
            | map({number: .content.number, title: .content.title,
                   labels: [.content.labels.nodes[].name],
                   iteration: (if .fieldValueByName then
                     {title: .fieldValueByName.title,
                      startDate: .fieldValueByName.startDate,
                      duration: .fieldValueByName.duration} else null end)})'
    ;;
  fetch-review-prs)
    user=$2
    gh api "search/issues?q=is:pr+is:open+review-requested:${user}" \
      --jq '.items[] | {number, title, repository_url, created_at, user: .user.login}'
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
