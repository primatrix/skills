#!/usr/bin/env bash
#
# beaver-decompose.sh — Helper for /beaver-decompose.
#
# Per RFC-0013 §5 and Issue #119:
#   * No Project V2 / Issue Type / Status / Size / Iteration writes here —
#     all of those go through beaver-lib.sh.
#   * No `gh api .../labels` writes for the lifecycle metadata families
#     (status/* / type/* / size/*) or for the audit-warning replacements
#     (beaver/missing-test / beaver/needs-split / beaver/missing-context):
#     audit failures are appended as `<!-- audit-warnings -->` blocks in
#     the child Issue body instead.
#   * Each `--body-file` invocation receives a unique mktemp path so that
#     a single shell invocation that creates N children does not collide.
#   * Owns the dependency-landing surface (`add-blocked-by`) which calls
#     the GitHub Issue Dependencies REST endpoint
#     `POST /repos/<owner>/<repo>/issues/<n>/dependencies/blocked_by`.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-decompose.sh <subcommand> [args]

Subcommands:
  fetch-parent <org> <repo> <number>
        Print parent issue summary as JSON
        (number, title, body, labels, assignees)
  parent-fields <org> <repo> <number>
        Print parent's field-semantics state as JSON
        (issueType, status, iteration, targetDate, assignees) — reads via
        beaver-lib.sh for Project V2 fields and via REST for assignees.
  list-sub-titles <org> <repo> <number>
        Print existing sub-issue titles, one per line
  create-child <org> <repo> <title> <body>
        Create child issue. <body> is the literal body string;
        the script writes it to a unique mktemp file and feeds the
        tempfile path to gh via the body-file flag. Echo "number=N id=ID".
  link-parent <org> <repo> <parent-number> <child-id>
        Link child issue (by integer id) as sub-issue of parent
  add-to-project <project-number> <org> <issue-url>
        Add issue to Project V2; echo created item id
  set-assignees <org> <repo> <number> [<login> ...]
        Replace assignees on an issue. With zero logins, clears.
  add-blocked-by <org> <repo> <child-number> <blocker-id>
        Add a "child blocked by blocker" Issue Dependency via REST.
        <blocker-id> is the numeric repo-issue id (gh api .../issues/<n>
        --jq '.id'), NOT the issue number.

Notes:
  * `create-child` always routes its body through a unique mktemp file
    suffixed with $$/$RANDOM/<epoch_ns>, satisfying the AC5 uniqueness
    requirement even when called N times in a single shell.
  * The script never writes Project V2 fields or the native Issue Type;
    /beaver-decompose owns that invariant via beaver-lib.sh.
EOF
}

# Echo path to a freshly-allocated unique tempfile for the given purpose.
_make_unique_body_file() {
  local purpose=${1:-body}
  local epoch
  epoch=$(date +%s%N 2>/dev/null || date +%s)
  # NOTE: BSD `mktemp` (macOS) requires the `XXXXXX` placeholder to be the
  # final component of the template — any suffix after it (e.g. `.md`) is
  # treated as part of the literal name and the placeholder is NOT
  # randomized, silently breaking within-loop uniqueness. Keep `XXXXXX` last.
  mktemp "/tmp/beaver-decompose-${purpose}-$$-${RANDOM}-${epoch}.XXXXXX"
}

case "${1:-}" in
  fetch-parent)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, body, labels: [.labels[].name], assignees: [.assignees[].login]}'
    ;;
  parent-fields)
    org=$2; repo=$3; num=$4
    # Source beaver-lib.sh from the same scripts/ directory as this script.
    lib_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")
    # shellcheck disable=SC1091
    source "${lib_dir}/beaver-lib.sh"
    # Project V2 field reads (Type / Status / Iteration / Target date) are
    # scoped to Project #14 in primatrix/projects regardless of where the
    # issue itself lives — that's where Beaver tracks all lifecycle metadata.
    issue_type=$(get_type "$num")
    status=$(_get_single_select_value "$num" "Status")
    iteration=$(get_iteration "$num")
    target_date=$(get_target_date "$num")
    # Assignees come from the issue's own repo via REST.
    assignees_json=$(gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '[.assignees[].login]')
    jq -n \
      --arg t "$issue_type" \
      --arg s "$status" \
      --arg i "$iteration" \
      --arg d "$target_date" \
      --argjson a "$assignees_json" \
      '{issueType: $t, status: $s, iteration: $i, targetDate: $d, assignees: $a}'
    ;;
  list-sub-titles)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/sub_issues" --jq '.[].title'
    ;;
  create-child)
    org=$2; repo=$3; title=$4; body=$5
    # Allocate the unique tempfile in the parent shell so that a write
    # failure (or any failure inside the gh calls) propagates via set -e
    # instead of silently producing an empty `output` capture that the
    # caller would then `eval` — which would re-link the previous child
    # in a batch loop. Cleanup runs unconditionally via trap.
    body_file=$(_make_unique_body_file child)
    trap 'rm -f "$body_file"' EXIT
    # `%s\n` matches beaver-design.sh and ensures the body ends with a
    # newline (some Markdown blocks render inconsistently without one).
    printf '%s\n' "$body" > "$body_file"
    num=$(gh api "repos/${org}/${repo}/issues" --method POST \
      -f title="$title" -F body=@"$body_file" --jq '.number')
    id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.id')
    echo "number=${num} id=${id}"
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
  set-assignees)
    org=$2; repo=$3; num=$4; shift 4
    # Build a fresh JSON array from the remaining args. PATCH replaces
    # the assignees set — passing [] clears all assignees.
    args=()
    for login in "$@"; do args+=(-f "assignees[]=${login}"); done
    if [ "${#args[@]}" -eq 0 ]; then
      echo '{"assignees":[]}' | gh api "repos/${org}/${repo}/issues/${num}" --method PATCH \
        --input - >/dev/null
    else
      gh api "repos/${org}/${repo}/issues/${num}" --method PATCH \
        "${args[@]}" >/dev/null
    fi
    ;;
  add-blocked-by)
    org=$2; repo=$3; child_num=$4; blocker_id=$5
    # Issue Dependencies REST endpoint expects {"issue_id": <numeric repo
    # issue id>}, NOT the issue number. The caller resolves the id via
    # `gh api repos/.../issues/<n> --jq '.id'`.
    gh api "repos/${org}/${repo}/issues/${child_num}/dependencies/blocked_by" \
      --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F issue_id="$blocker_id"
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
