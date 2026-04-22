#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-decompose.sh <subcommand> [args]

Subcommands:
  fetch-parent <org> <repo> <number>
        Print parent issue summary as JSON (number, title, body, labels)
  list-sub-titles <org> <repo> <number>
        Print existing sub-issue titles, one per line
  create-child <org> <repo> <title> <body-file>
        Create child issue from body file; print "number=N id=ID"
  add-labels <org> <repo> <number> <label> [<label> ...]
        Add one or more labels to an issue in a single POST
  link-parent <org> <repo> <parent-number> <child-id>
        Link child issue (by integer id) as sub-issue of parent
  add-to-project <project-number> <org> <issue-url>
        Add issue to Project V2; print created item id
EOF
}

case "${1:-}" in
  fetch-parent)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, body, labels: [.labels[].name]}'
    ;;
  list-sub-titles)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/sub_issues" --jq '.[].title'
    ;;
  create-child)
    org=$2; repo=$3; title=$4; body_file=$5
    num=$(gh api "repos/${org}/${repo}/issues" --method POST \
      -f title="$title" -F body=@"$body_file" --jq '.number')
    id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.id')
    echo "number=${num} id=${id}"
    ;;
  add-labels)
    org=$2; repo=$3; num=$4; shift 4
    args=()
    for label in "$@"; do args+=(-f "labels[]=${label}"); done
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  link-parent)
    org=$2; repo=$3; parent=$4; child_id=$5
    gh api "repos/${org}/${repo}/issues/${parent}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  add-to-project)
    project=$2; org=$3; url=$4
    gh project item-add "$project" --owner "$org" --url "$url" --format json --jq '.id'
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
