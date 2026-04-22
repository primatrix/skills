#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-claim.sh <subcommand> [args]

Subcommands:
  fetch <org> <repo> <number>           Print issue summary as JSON
  whoami                                Print current gh user
  assign <org> <repo> <number> <user>   Assign user to issue
  swap-status <org> <repo> <number> <from-label> <to-label>
                                        Atomic label swap (DELETE old, POST new)
EOF
}

case "${1:-}" in
  fetch)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, state, labels: [.labels[].name], assignees: [.assignees[].login]}'
    ;;
  whoami)
    gh api user --jq '.login'
    ;;
  assign)
    org=$2; repo=$3; num=$4; user=$5
    gh api "repos/${org}/${repo}/issues/${num}/assignees" --method POST \
      -f "assignees[]=${user}"
    ;;
  swap-status)
    org=$2; repo=$3; num=$4; from=$5; to=$6
    from_enc=${from//\//%2F}
    gh api "repos/${org}/${repo}/issues/${num}/labels/${from_enc}" --method DELETE
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST \
      -f "labels[]=${to}"
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
