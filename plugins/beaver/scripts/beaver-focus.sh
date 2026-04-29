#!/usr/bin/env bash
#
# beaver-focus.sh — read-only personal dashboard aggregator.
#
# All subcommands are STRICTLY READ-ONLY against remote GitHub state:
# only `gh api graphql -f query=...` (queries) and default-GET `gh api ...`
# REST calls are used. The forbidden write-side tokens enumerated by Issue
# #122 AC5 are absent from this file by design; see the test in
# scripts/tests/test_beaver_focus_fields.sh for the verbatim grep regex.
#
# Subcommands:
#   whoami                              Print current gh user
#   fetch-my-issues <user>              Open issues assigned to <user>,
#                                       with Status/Priority/Type/Iteration fields
#                                       and last-activity (max(updatedAt, latest
#                                       comment createdAt)) for recency sort.
#   fetch-review-prs <user>             PRs awaiting <user>'s review (REST GET).
#   fetch-ready-to-claim                Open issues with Status="Ready to
#                                       Claim" and no assignees.

set -euo pipefail

ORG=primatrix
PROJECT_REPO=projects
PROJECT_NUMBER=14

usage() {
  cat <<'EOF'
Usage: beaver-focus.sh <subcommand> [args]

Subcommands:
  whoami                      Print current gh user
  fetch-my-issues <user>      Print my open issues from project #14 (JSON array)
                              Fields per item: number, title, repo, url, labels,
                              status (Project V2 Status field), priority (Priority
                              field), type (native Issue Type), iteration
                              (Iteration field title or null), createdAt,
                              updatedAt, lastCommentAt, lastActivityAt
                              (= max(updatedAt, lastCommentAt)).
  fetch-review-prs <user>     Print PRs awaiting my review (JSON lines, REST GET)
  fetch-ready-to-claim        Print open issues with Status="Ready to
                              Claim" and no assignees (JSON array)

All subcommands are read-only (GraphQL query / REST GET only).
EOF
}

# Shared GraphQL fragment: pull every Project V2 #14 item with its content
# (Issue), Status / Priority / Iteration single-select+iteration field values,
# native Issue Type, plus the latest comment timestamp for recency sorting.
#
# `comments(last: 1)` gives us the most recent comment's createdAt — combined
# with `updatedAt`, `lastActivityAt` is computed in jq downstream (max of both).
#
# Uses cursor-based pagination (first:100 per page, max enforced by GitHub API)
# and merges all pages into a single JSON array on stdout.
_query_project_items() {
  local cursor_arg=""
  local all_nodes="[]"

  while true; do
    local page
    page=$(gh api -H "GraphQL-Features: issue_types" graphql -f query='
      query($owner: String!, $number: Int!, $cursor: String) {
        organization(login: $owner) {
          projectV2(number: $number) {
            items(first: 100, after: $cursor) {
              pageInfo { hasNextPage endCursor }
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    url
                    state
                    createdAt
                    updatedAt
                    issueType { name }
                    repository { name nameWithOwner }
                    labels(first: 30) { nodes { name } }
                    assignees(first: 10) { nodes { login } }
                    comments(last: 1) { nodes { createdAt } }
                  }
                }
                status: fieldValueByName(name: "Status") {
                  ... on ProjectV2ItemFieldSingleSelectValue { name }
                }
                priority: fieldValueByName(name: "Priority") {
                  ... on ProjectV2ItemFieldSingleSelectValue { name }
                }
                iter: fieldValueByName(name: "Iteration") {
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
      }' -f owner="$ORG" -F number="$PROJECT_NUMBER" ${cursor_arg})

    # Append this page's nodes to the accumulator
    all_nodes=$(printf '%s\n%s' "$all_nodes" "$page" \
      | jq -s '.[0] + (.[1].data.organization.projectV2.items.nodes // [])')

    # Check if there is a next page
    local has_next end_cursor
    has_next=$(echo "$page" | jq -r '.data.organization.projectV2.items.pageInfo.hasNextPage')
    end_cursor=$(echo "$page" | jq -r '.data.organization.projectV2.items.pageInfo.endCursor')

    if [[ "$has_next" != "true" || "$end_cursor" == "null" ]]; then
      break
    fi
    cursor_arg="-f cursor=$end_cursor"
  done

  # Emit the merged result in the same shape as the original single-page query
  jq -n --argjson nodes "$all_nodes" \
    '{data:{organization:{projectV2:{items:{nodes:$nodes}}}}}'
}

# Project items projected into a uniform record shape (used by multiple
# subcommands). Adds lastCommentAt and lastActivityAt fields. Preserves only
# items whose content is an open Issue.
#
# Reads JSON from stdin (output of _query_project_items).
_project_items_to_records() {
  jq '.data.organization.projectV2.items.nodes
      | map(select(.content != null
                   and .content.state == "OPEN"))
      | map(
          ((.content.comments.nodes // []) | .[-1].createdAt // null) as $lastComment
          | {
              number: .content.number,
              title: .content.title,
              url: .content.url,
              repo: .content.repository.name,
              repoFull: .content.repository.nameWithOwner,
              labels: [.content.labels.nodes[].name],
              assignees: [.content.assignees.nodes[].login],
              status: (.status.name // ""),
              priority: (.priority.name // ""),
              type: (.content.issueType.name // ""),
              iteration: (if .iter then
                {title: .iter.title, startDate: .iter.startDate, duration: .iter.duration}
              else null end),
              createdAt: .content.createdAt,
              updatedAt: .content.updatedAt,
              lastCommentAt: $lastComment,
              lastActivityAt: ([.content.updatedAt, $lastComment] | map(select(. != null)) | max)
            }
        )'
}

case "${1:-}" in
  whoami)
    gh api user --jq '.login'
    ;;

  fetch-my-issues)
    user=$2
    # Open issues assigned to $user, with Status/Priority/Type/Iteration
    # and lastActivityAt (max of updatedAt and latest comment createdAt) so the
    # caller can sort each Status group by recency (last commit/comment).
    _query_project_items \
      | _project_items_to_records \
      | jq --arg user "$user" \
          'map(select(.assignees | index($user)))'
    ;;

  fetch-review-prs)
    user=$2
    gh api "search/issues?q=is:pr+is:open+review-requested:${user}" \
      --jq '.items[] | {number, title, repository_url, created_at, user: .user.login, html_url}'
    ;;

  fetch-ready-to-claim)
    # Open issues with Status == "Ready to Claim" AND no assignees,
    # so the caller can surface unclaimed work in the dashboard.
    _query_project_items \
      | _project_items_to_records \
      | jq 'map(select(.status == "Ready to Claim" and (.assignees | length) == 0))'
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
