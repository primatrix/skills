#!/usr/bin/env bash
#
# beaver-design.sh — companion script for /beaver-design.
#
# Per RFC-0013 §4 #4 and Issue #118 acceptance criteria:
#   * No Project V2 field mutations. This script is purely a wiki-side helper
#     (clone / branch / commit / push / PR) plus an Issue-comment helper.
#   * All `gh` calls that pass body content go through `--body-file` fed by a
#     unique tempfile (`mktemp` with a process- and time-disambiguated suffix),
#     per the RFC §命令规约 "临时文件命名约定" preamble.
#
# This script does NOT source beaver-lib.sh — it has no field-ops needs.

set -euo pipefail

WIKI_REPO="primatrix/wiki"
WIKI_DEFAULT="${HOME}/Code/wiki"

usage() {
  cat <<'EOF'
Usage: beaver-design.sh <subcommand> [args]

Subcommands:
  prepare-wiki [<wiki-dir>]
        Default ~/Code/wiki. If exists: cd && git checkout main && git pull.
        Else: gh repo clone primatrix/wiki <wiki-dir>.
  create-branch <wiki-dir> <branch>
        cd <wiki-dir> && git checkout -b <branch>
  commit-push <wiki-dir> <file> <message> <branch>
        cd <wiki-dir>; git add <file> docs/projects/{project}/rfc/index.md; git commit -m <message>;
        git push -u origin <branch>
  create-pr <repo> <title> <body>
        gh pr create --repo <repo> --draft --title <title> --body-file <unique-tempfile>
  comment-issue <org> <repo> <number> <body>
        gh issue comment <number> --repo <org>/<repo> --body-file <unique-tempfile>

Notes:
  * `comment-issue` and `create-pr` always route their body through a unique
    mktemp file (suffix "-$$-$RANDOM-<epoch_ns>"), satisfying the AC6 uniqueness
    requirement even when the same subcommand is called multiple times within
    a single shell invocation (e.g. batch comment loops).
  * The script never writes to Project V2 fields; /beaver-design owns that
    invariant and asserts it post-run.
EOF
}

# Echo path to a freshly-allocated unique tempfile for the given purpose.
# Uses mktemp + process and time disambiguation so back-to-back calls within
# a single shell never collide.
_make_unique_body_file() {
  local purpose=${1:-body}
  local epoch
  epoch=$(date +%s%N 2>/dev/null || date +%s)
  # NOTE: BSD `mktemp` (macOS) requires the `XXXXXX` placeholder to be the
  # final component of the template — any suffix after it (e.g. `.md`) is
  # treated as part of the literal name and the placeholder is NOT randomized,
  # which silently breaks within-loop uniqueness. Keep `XXXXXX` at the end.
  mktemp "/tmp/beaver-design-${purpose}-$$-${RANDOM}-${epoch}.XXXXXX"
}

case "${1:-}" in
  prepare-wiki)
    dir="${2:-$WIKI_DEFAULT}"
    if [ -d "$dir" ]; then
      cd "$dir"
      git checkout main
      git pull
    else
      gh repo clone "$WIKI_REPO" "$dir"
    fi
    ;;
  create-branch)
    dir=$2; branch=$3
    cd "$dir"
    git checkout -b "$branch"
    ;;
  commit-push)
    dir=$2; file=$3; message=$4; branch=$5
    cd "$dir"
    git add "$file"
    if [ -f "docs/projects/{project}/rfc/index.md" ]; then
      git add docs/projects/{project}/rfc/index.md
    fi
    git commit -m "$message"
    git push -u origin "$branch"
    ;;
  create-pr)
    repo=$2; title=$3; body=$4
    # Subshell + EXIT trap ensures the tempfile is removed even if `gh pr
    # create` fails (the script's `set -e` would otherwise abort before our
    # explicit `rm`).
    (
      body_file=$(_make_unique_body_file pr)
      trap 'rm -f "$body_file"' EXIT
      printf '%s\n' "$body" > "$body_file"
      gh pr create --repo "$repo" --draft --title "$title" --body-file "$body_file"
    )
    ;;
  comment-issue)
    org=$2; repo=$3; num=$4; body=$5
    (
      body_file=$(_make_unique_body_file comment)
      trap 'rm -f "$body_file"' EXIT
      printf '%s\n' "$body" > "$body_file"
      gh issue comment "$num" --repo "${org}/${repo}" --body-file "$body_file"
    )
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
