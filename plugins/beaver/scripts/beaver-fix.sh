#!/usr/bin/env bash
set -euo pipefail

# beaver-fix.sh — helper for /beaver-fix
# 不修改 Project V2 字段.

tmp_files=()

# Default location for the file-list snapshot used by scoped rollback.
# beaver-fix.sh snapshot-files-before <path> writes git diff --name-only HEAD here;
# rollback() reads it and restores ONLY those files (never the whole tree).
: "${BEAVER_FIX_FILES_SNAPSHOT:=/tmp/beaver-fix-files-$$.list}"

_rollback_fired=0
rollback() {
  # Guard: fire only once even if both ERR and EXIT/INT trigger.
  if [ "$_rollback_fired" -eq 1 ]; then
    return 0
  fi
  _rollback_fired=1
  for f in "${tmp_files[@]:-}"; do
    [ -n "${f:-}" ] && [ -f "$f" ] && rm -f "$f"
  done
  if git rev-parse --git-dir >/dev/null 2>&1; then
    # Scoped rollback: ONLY restore files that this command touched.
    # We rely on a pre-edit snapshot at $BEAVER_FIX_FILES_SNAPSHOT (created by
    # the `snapshot-files-before` subcommand). If absent, we do nothing —
    # we MUST NOT fall back to `git checkout -- .` because that would clobber
    # unrelated user work in the worktree.
    if [ -f "$BEAVER_FIX_FILES_SNAPSHOT" ]; then
      while IFS= read -r f; do
        [ -z "$f" ] && continue
        git checkout HEAD -- "$f" 2>/dev/null || true
      done < "$BEAVER_FIX_FILES_SNAPSHOT"
    fi
  fi
  echo "rollback: 回滚已修改文件 (scoped to ${BEAVER_FIX_FILES_SNAPSHOT})" >&2
}
trap 'rollback' INT ERR

write_query() {
  # $1 = output path; reads heredoc-style content from stdin
  cat > "$1"
}

usage() {
  cat <<'EOF'
Usage: beaver-fix.sh <subcommand> [args]

Subcommands:
  verify-author <pr-number>                    Abort unless current user authored the PR
  list-open-comments <pr-number>               List unresolved review threads + PR-level issue comments (graphql)
  resolve-thread <thread-id>                   Call resolveReviewThread mutation
  snapshot-files-before [snapshot-path]        Snapshot `git diff --name-only HEAD` for scoped rollback
  snapshot-projectv2-fields <pr-number>        Snapshot Project V2 field values for the PR
  verify-projectv2-fields <pr-number> <snapshot-file>
                                               Re-snapshot, diff vs <snapshot-file>, exit 1 on mismatch
  commit-and-push <scope>                      Commit staged changes with conventional template
EOF
}

case "${1:-}" in
  verify-author)
    pr=$2
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    author=$(gh pr view "$pr" --repo "$repo" --json author --jq .author.login)
    me=$(gh api user --jq .login)
    if [ "$author" != "$me" ]; then
      echo "只能对自己发起的 PR 运行 /beaver-fix" >&2
      exit 1
    fi
    echo "author-ok: $me"
    ;;
  list-open-comments)
    pr=$2
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    owner=${repo%/*}
    name=${repo#*/}
    qf=$(mktemp -t beaver-fix-q.XXXXXX); tmp_files+=("$qf")
    # Fetch BOTH:
    #   (a) reviewThreads — line-level review-thread comments (filter isResolved=false)
    #   (b) issueComments — PR-level top-level comments (no isResolved field; show all)
    write_query "$qf" <<'GQL'
query($owner:String!, $name:String!, $pr:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$pr) {
      reviewThreads(first:100) {
        nodes { id isResolved comments(first:5){ nodes{ id body path author{login} createdAt } } }
      }
      comments(first:100) {
        nodes { id body author{login} createdAt }
      }
    }
  }
}
GQL
    gh api graphql --body-file "$qf" \
      -F owner="$owner" -F name="$name" -F pr="$pr" \
      --jq '{
        reviewThreads: [.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved==false)],
        issueComments: .data.repository.pullRequest.comments.nodes
      }'
    rm -f "$qf"
    ;;
  resolve-thread)
    tid=$2
    mf=$(mktemp -t beaver-fix-m.XXXXXX); tmp_files+=("$mf")
    write_query "$mf" <<'GQL'
mutation($tid:ID!) {
  resolveReviewThread(input:{threadId:$tid}) { thread { id isResolved } }
}
GQL
    gh api graphql --body-file "$mf" -F tid="$tid"
    rm -f "$mf"
    ;;
  snapshot-files-before)
    # Write the list of files currently dirty vs HEAD; rollback() will scope to these.
    out=${2:-$BEAVER_FIX_FILES_SNAPSHOT}
    git diff --name-only HEAD > "$out"
    echo "$out"
    ;;
  snapshot-projectv2-fields)
    pr=$2
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    owner=${repo%/*}
    name=${repo#*/}
    sf=$(mktemp -t beaver-fix-pv2.XXXXXX); tmp_files+=("$sf")
    qf=$(mktemp -t beaver-fix-pv2q.XXXXXX); tmp_files+=("$qf")
    # Read the PR's projectItems and their fieldValues — the runtime invariant
    # we enforce is that these fields are byte-identical before & after fixups.
    write_query "$qf" <<'GQL'
query($owner:String!, $name:String!, $pr:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$pr) {
      projectItems(first:10) {
        nodes {
          id
          fieldValues(first:50) {
            nodes {
              __typename
              ... on ProjectV2ItemFieldTextValue       { text       field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldNumberValue     { number     field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldDateValue       { date       field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldSingleSelectValue { name     field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldIterationValue  { title      field { ... on ProjectV2FieldCommon { name } } }
            }
          }
        }
      }
    }
  }
}
GQL
    gh api graphql --body-file "$qf" \
      -F owner="$owner" -F name="$name" -F pr="$pr" \
      --jq '.data.repository.pullRequest.projectItems' > "$sf"
    rm -f "$qf"
    # Print the snapshot path so the caller can pass it to verify-projectv2-fields.
    echo "$sf"
    ;;
  verify-projectv2-fields)
    pr=$2
    before=$3
    if [ ! -f "$before" ]; then
      echo "verify-projectv2-fields: snapshot file not found: $before" >&2
      exit 1
    fi
    after=$(mktemp -t beaver-fix-pv2a.XXXXXX); tmp_files+=("$after")
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    owner=${repo%/*}
    name=${repo#*/}
    qf=$(mktemp -t beaver-fix-pv2vq.XXXXXX); tmp_files+=("$qf")
    write_query "$qf" <<'GQL'
query($owner:String!, $name:String!, $pr:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$pr) {
      projectItems(first:10) {
        nodes {
          id
          fieldValues(first:50) {
            nodes {
              __typename
              ... on ProjectV2ItemFieldTextValue       { text       field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldNumberValue     { number     field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldDateValue       { date       field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldSingleSelectValue { name     field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldIterationValue  { title      field { ... on ProjectV2FieldCommon { name } } }
            }
          }
        }
      }
    }
  }
}
GQL
    gh api graphql --body-file "$qf" \
      -F owner="$owner" -F name="$name" -F pr="$pr" \
      --jq '.data.repository.pullRequest.projectItems' > "$after"
    rm -f "$qf"
    if diff -u "$before" "$after" >/dev/null; then
      echo "Project V2 fields unchanged"
      rm -f "$after"
    else
      echo "ERROR: Project V2 fields diverged between snapshot and post-run!" >&2
      diff -u "$before" "$after" >&2 || true
      exit 1
    fi
    ;;
  commit-and-push)
    scope=$2
    # Empty-diff guard: if nothing is staged (e.g. all comments resolved/skipped
    # without code change), there is nothing to commit/push — exit cleanly.
    if git diff --cached --quiet; then
      echo "no staged changes; skipping commit"
      exit 0
    fi
    msg=$(mktemp -t beaver-fix-msg.XXXXXX); tmp_files+=("$msg")
    printf 'fix(%s): address review comments\n' "$scope" > "$msg"
    git commit -F "$msg"
    # -u so freshly-created PR branches set their upstream (matches beaver-pr.sh:41).
    git push -u origin HEAD
    rm -f "$msg"
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
