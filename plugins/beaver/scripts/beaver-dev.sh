#!/usr/bin/env bash
#
# beaver-dev.sh — companion script for /beaver-dev.
#
# Per RFC-0013 §6 and Issue #120 acceptance criteria:
#   * Field-semantic preflight via beaver-lib.sh: read Size, Status, and
#     assignees through Project V2 / GitHub APIs (NO label literals).
#   * /beaver-dev only handles Size=S; non-S Issues are rejected with
#     the message "本命令仅处理 Size=S".
#   * This script does NOT mutate the Status field. The user transitions
#     In Progress ↔ Blocked via the GitHub UI.
#
# Subcommands:
#   preflight <number>          Run all preflight checks; print "OK <type>"
#                               or "FAIL <reason>" and exit 0/1 accordingly.
#   fetch-issue <org> <repo> <number>
#                               Print issue summary as JSON.
#   add-worktree <branch>       git worktree add .claude/worktrees/<branch>
#                               -b <branch> (run from repo root).

set -euo pipefail

# Resolve our own directory so we can source the shared lib regardless of
# the caller's cwd.
_BEAVER_DEV_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage: beaver-dev.sh <subcommand> [args]

Subcommands:
  preflight <number>
        Read Size / Status / assignee for the given Issue via beaver-lib.sh,
        plus the current gh user. Prints "OK <type>" on pass (where <type>
        is the lower-cased Issue Type for branch naming, e.g. "task"); on
        fail prints "FAIL <reason>" to stderr and exits 1.

        Rejection conditions:
          * Size != S          → "本命令仅处理 Size=S"
          * Status != In Progress
          * current user is not in the Issue's assignees

  fetch-issue <org> <repo> <number>
        Print issue summary as JSON {number, title, body, assignees}.

  add-worktree <branch>
        git worktree add .claude/worktrees/<branch> -b <branch>.
        Run from the repo root; <branch> typically follows
        <type>/<n>-<short_desc>.
EOF
}

case "${1:-}" in
  preflight)
    number=$2
    # shellcheck source=/dev/null
    source "${_BEAVER_DEV_SCRIPT_DIR}/beaver-lib.sh"

    size=$(_get_single_select_value "$number" "Size")
    status=$(_get_single_select_value "$number" "Status")
    type_name=$(get_type "$number")
    me=$(gh api user --jq '.login')
    assignees=$(gh api "repos/${ORG}/${PROJECT_REPO}/issues/${number}" \
      --jq '(.assignees // []) | [.[].login] | join(" ")')

    if [ "$size" != "S" ]; then
      echo "FAIL 本命令仅处理 Size=S (当前 Size='${size:-<empty>}')" >&2
      exit 1
    fi
    if [ "$status" != "In Progress" ]; then
      echo "FAIL Status 必须为 'In Progress' (当前='${status:-<empty>}'); /beaver-claim 已删除（见 RFC-0013 §3），请在 GitHub UI assign 自己后手动将 Status 切到 'In Progress'" >&2
      exit 1
    fi
    # Surround both sides with single spaces so substring matches like
    # me="foo" vs assignees="foobar" cannot accidentally pass — GitHub
    # usernames cannot contain whitespace, so this is unambiguous.
    if ! printf ' %s ' "$assignees" | grep -qF " $me "; then
      echo "FAIL 当前用户 (${me}) 不在 Issue 的 assignees 列表中: [${assignees}]" >&2
      exit 1
    fi

    # Lower-case Type for branch prefix; default to "task" if Type is empty
    # (shouldn't happen given Size=S + In Progress, but defensive).
    type_lc=$(printf '%s' "${type_name:-Task}" | tr '[:upper:]' '[:lower:]')
    echo "OK ${type_lc}"
    ;;

  fetch-issue)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, body, assignees: [.assignees[].login]}'
    ;;

  add-worktree)
    branch=$2
    git worktree add ".claude/worktrees/${branch}" -b "$branch"
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
