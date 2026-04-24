#!/usr/bin/env bash
#
# beaver-create.sh — Helper for /beaver-create.
#
# Owns only the operations that are NOT lifecycle-metadata writes:
#   create-issue   POST a new Issue with body from a file (file path is
#                  unique per invocation; callers should use mktemp).
#   fetch-ids      GET .id / .node_id / .html_url for a freshly created
#                  Issue (the POST response only carries .number).
#   add-to-project Add the Issue to Project V2 #<n> and echo the project
#                  item id.
#   link-parent    Attach the Issue as a Sub-Issue of <parent> via the
#                  Sub-Issues REST endpoint (still raw `gh api`, since
#                  this is the native Issues API, not Project V2).
#
# All Project V2 single-select / iteration writes (Status, Size, Type,
# Iteration) and the native Issue Type write live in beaver-lib.sh and
# MUST go through it — this script does not duplicate them.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-create.sh <subcommand> [args]

Subcommands:
  create-issue <org> <repo> <title> <body-file>
                                Create issue, echo .number
                                <body-file> should be a unique path
                                (see `mktemp`); the caller is
                                responsible for cleanup.
  fetch-ids <org> <repo> <number>
                                Echo "id=N node_id=NID html_url=URL"
  add-to-project <project-number> <org> <issue-url>
                                Add issue to project, echo item id
  link-parent <org> <repo> <parent-number> <child-id>
                                Attach child as sub-issue (Sub-Issues API)
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
  add-to-project)
    project=$2; org=$3; url=$4
    gh project item-add "$project" --owner "$org" --url "$url" \
      --format json --jq '.id'
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
