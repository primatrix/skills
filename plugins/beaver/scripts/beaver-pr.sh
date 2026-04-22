#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-pr.sh <subcommand> [args]

Subcommands:
  ctx                                       Print git status / diff stat / branch / recent log
  create-branch <branch>                    Create branch (or switch to it if it already exists)
  commit-push <branch> <msg-file> <file>... git add files, commit -F msg-file, push -u origin <branch>
  check-tests                               List test files changed vs origin/main
  check-labels <org> <repo> <number>        Print issue label names
  add-label <org> <repo> <number> <label>   Add a single label to an issue
  create-pr <title> <body-file>             Create a Draft PR (--body-file)
  mark-ready <pr-number>                    Mark a Draft PR as Ready for Review
EOF
}

case "${1:-}" in
  ctx)
    echo "=== git status ==="
    git status
    echo "=== git diff --stat HEAD ==="
    git diff --stat HEAD
    echo "=== git branch --show-current ==="
    git branch --show-current
    echo "=== git log --oneline -10 ==="
    git log --oneline -10
    ;;
  create-branch)
    branch=$2
    git checkout -b "$branch" 2>/dev/null || git checkout "$branch"
    ;;
  commit-push)
    branch=$2
    msg_file=$3
    shift 3
    git add "$@"
    git commit -F "$msg_file"
    git push -u origin "$branch"
    ;;
  check-tests)
    git diff --name-only origin/main...HEAD | grep -E '(test_|_test\.|/tests/)' || true
    ;;
  check-labels)
    org=$2
    repo=$3
    num=$4
    gh api "repos/${org}/${repo}/issues/${num}/labels" --jq '.[].name'
    ;;
  add-label)
    org=$2
    repo=$3
    num=$4
    label=$5
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST -f "labels[]=${label}"
    ;;
  create-pr)
    title=$2
    body_file=$3
    gh pr create --draft --title "$title" --body-file "$body_file"
    ;;
  mark-ready)
    pr=$2
    gh pr ready "$pr"
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
