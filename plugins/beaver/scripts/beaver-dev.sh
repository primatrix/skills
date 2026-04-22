#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-dev.sh <subcommand> [args]

Subcommands:
  fetch-issue <org> <repo> <number>       Print issue as JSON
  fetch-sub-issues <org> <repo> <number>  Print sub-issues as JSON array
  add-worktree <branch>                   git worktree add .claude/worktrees/<branch> -b <branch>
  swap-to-in-progress <org> <repo> <number>
                                          DELETE status/ready-to-develop, POST status/in-progress
EOF
}

case "${1:-}" in
  fetch-issue)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, body, labels: [.labels[].name]}'
    ;;
  fetch-sub-issues)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/sub_issues" \
      --jq '[.[] | {number, title, body, labels: [.labels[].name]}]'
    ;;
  add-worktree)
    branch=$2
    git worktree add ".claude/worktrees/${branch}" -b "$branch"
    ;;
  swap-to-in-progress)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/labels/status%2Fready-to-develop" --method DELETE
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST \
      -f "labels[]=status/in-progress"
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
