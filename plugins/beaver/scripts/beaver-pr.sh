#!/usr/bin/env bash
set -euo pipefail

# beaver-pr.sh — RFC-0013 §7 implementation.
#
# G004 / G006 audits emit PR-body warning lines (caller appends them); they do
# NOT post any beaver/* labels on the Issue. G006 also delegates Type/Size
# auto-fill to beaver-lib.sh (set_type, set_size); failure causes the caller to
# append a warning to the PR body.
#
# Every callsite that hands a body to `gh` via --body-file routes through a
# uniquely-named tempfile (mktemp), per RFC §命令规约「临时文件命名约定」.

# Resolve repo root for sourcing beaver-lib.sh, mirroring beaver-claim.sh /
# beaver-create.sh patterns. CLAUDE_PLUGIN_ROOT may be set when invoked from
# the slash-command harness; otherwise fall back to the script's own dirname.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_path="${CLAUDE_PLUGIN_ROOT:-$script_dir/..}/scripts/beaver-lib.sh"
if [ ! -f "$lib_path" ]; then
  lib_path="$script_dir/beaver-lib.sh"
fi

usage() {
  cat <<'EOF'
Usage: beaver-pr.sh <subcommand> [args]

Subcommands:
  ctx                                       Print git status / diff stat / branch / recent log
  infer-issue                               Echo the inferred Issue # (branch prefix → recent commits → empty)
  create-branch <branch>                    Create branch (or switch to it if it already exists)
  commit-push <branch> <msg-file> <file>... git add files, commit -F msg-file, push -u origin <branch>
  check-tests                               List test files changed vs origin/main; exit 1 if none
  check-fields <org> <repo> <number>        Print Type and Size of the issue (one per line: Type=... / Size=...)
  autofill-fields <number> [size]           Set Type=Task (if empty) and Size=<size|S> (if empty); exits 1 on failure
  create-pr <title> <body-string>           Create a Draft PR (writes body to mktemp file, --body-file)
  comment-issue <org> <repo> <number> <body-string>
                                            Post a comment via mktemp + --body-file
  mark-ready <pr-number>                    Mark a Draft PR as Ready for Review
  delete-branch <branch>                    Delete a remote+local branch (used by discard option)
EOF
}

mktemp_body() {
  # Unique tempfile per call so multiple --body-file invocations within a
  # single command do not collide. Pattern matches beaver-design.sh.
  mktemp "/tmp/beaver-pr-body-$$-$RANDOM-XXXXXX.md"
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
  infer-issue)
    # 1) branch prefix: <type>/<number>-<desc>
    branch=$(git branch --show-current)
    num=$(echo "$branch" | sed -nE 's#^[a-z]+/([0-9]+)-.*#\1#p')
    if [ -z "$num" ]; then
      # 2) recent commit messages: look for "#<n>" or "Closes #<n>"
      num=$(git log -20 --pretty=%B | grep -oE '#[0-9]+' | head -1 | tr -d '#')
    fi
    echo "$num"
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
    hits=$(git diff --name-only origin/main...HEAD | grep -E '(test_|_test\.|/tests/)' || true)
    if [ -n "$hits" ]; then
      echo "$hits"
      exit 0
    fi
    exit 1
    ;;
  check-fields)
    org=$2
    repo=$3
    num=$4
    # shellcheck disable=SC1090
    source "$lib_path"
    type_val=$(get_type "$num" || true)
    size_val=$(_get_single_select_value "$num" "Size" || true)
    echo "Type=${type_val}"
    echo "Size=${size_val}"
    ;;
  autofill-fields)
    num=$2
    size_default=${3:-S}
    # shellcheck disable=SC1090
    source "$lib_path"
    rc=0
    type_val=$(get_type "$num" || true)
    size_val=$(_get_single_select_value "$num" "Size" || true)
    if [ -z "$type_val" ]; then
      set_type "$num" "Task" || rc=1
    fi
    if [ -z "$size_val" ]; then
      set_size "$num" "$size_default" || rc=1
    fi
    exit "$rc"
    ;;
  create-pr)
    title=$2
    body_string=$3
    body_file=$(mktemp_body)
    trap 'rm -f "$body_file"' EXIT
    printf '%s' "$body_string" > "$body_file"
    gh pr create --draft --title "$title" --body-file "$body_file"
    ;;
  comment-issue)
    org=$2
    repo=$3
    num=$4
    body_string=$5
    body_file=$(mktemp_body)
    trap 'rm -f "$body_file"' EXIT
    printf '%s' "$body_string" > "$body_file"
    gh issue comment "$num" --repo "${org}/${repo}" --body-file "$body_file"
    ;;
  mark-ready)
    pr=$2
    gh pr ready "$pr"
    ;;
  delete-branch)
    branch=$2
    git push origin --delete "$branch" || true
    git branch -D "$branch"
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
