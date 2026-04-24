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
#   fetch-my-issues <user>              Open Beaver issues assigned to <user>,
#                                       with Status/Priority/Type/Iteration fields
#                                       and last-activity (max(updatedAt, latest
#                                       comment createdAt)) for recency sort.
#   fetch-review-prs <user>             PRs awaiting <user>'s review (REST GET).
#   fetch-ready-to-claim                Open Beaver issues with Status="Ready to
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
  fetch-my-issues <user>      Print my open Beaver issues from project #14 (JSON array)
                              Fields per item: number, title, repo, url, labels,
                              status (Project V2 Status field), priority (Priority
                              field), type (native Issue Type), iteration
                              (Iteration field title or null), createdAt,
                              updatedAt, lastCommentAt, lastActivityAt
                              (= max(updatedAt, lastCommentAt)).
  fetch-review-prs <user>     Print PRs awaiting my review (JSON lines, REST GET)
  fetch-ready-to-claim        Print open Beaver issues with Status="Ready to
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
_query_project_items() {
  gh api -H "GraphQL-Features: issue_types" graphql -f query='
    query($owner: String!, $number: Int!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          items(first: 200) {
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
    }' -f owner="$ORG" -F number="$PROJECT_NUMBER"
}

# Project items projected into a uniform record shape (used by multiple
# subcommands). Adds lastCommentAt and lastActivityAt fields. Preserves only
# items whose content is an open Issue with the Control-By-Beaver label.
#
# Reads JSON from stdin (output of _query_project_items).
_project_items_to_records() {
  jq '.data.organization.projectV2.items.nodes
      | map(select(.content != null
                   and .content.state == "OPEN"
                   and (.content.labels.nodes | map(.name) | index("Control-By-Beaver"))))
      | map({
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
          lastCommentAt: ((.content.comments.nodes // []) | (.[-1].createdAt // null)),
          lastActivityAt: ([
            .content.updatedAt,
            ((.content.comments.nodes // []) | (.[-1].createdAt // null))
          ] | map(select(. != null)) | max)
        })'
}

case "${1:-}" in
  whoami)
    gh api user --jq '.login'
    ;;

  fetch-my-issues)
    user=$2
    # Open Beaver issues assigned to $user, with Status/Priority/Type/Iteration
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
    # Open Beaver issues with Status == "Ready to Claim" AND no assignees,
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
