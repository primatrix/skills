#!/usr/bin/env bash
set -euo pipefail

# beaver-fix.sh — helper for /beaver-fix
# 不修改 Project V2 字段.

tmp_files=()

_rollback_fired=0
rollback() {
  # Guard: fire only once even if both ERR and INT trigger.
  if [ "$_rollback_fired" -eq 1 ]; then
    return 0
  fi
  _rollback_fired=1
  for f in ${tmp_files[@]+"${tmp_files[@]}"}; do
    [ -n "${f:-}" ] && [ -f "$f" ] && rm -f "$f"
  done
  # 不自动回滚已写入的文件，以免覆盖用户 WIP。
  # 如需丢弃改动，请手动: git restore --staged --worktree <file>
  echo "中止: 已清理临时文件；如需撤销已写改动请手动 git restore" >&2
}
trap 'rollback' INT ERR

write_query() {
  # $1 = output path; reads heredoc-style content from stdin
  cat > "$1"
}

usage() {
  cat <<'EOF'
用法: beaver-fix.sh <子命令> [参数]

子命令:
  verify-author <pr-number>          非作者直接终止
  list-open-comments <pr-number>     列出未 resolved 的 review threads + PR 顶层评论
  resolve-thread <thread-id>         调用 resolveReviewThread mutation
  commit-and-push <scope>            按模板提交已 stage 的改动并 push
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
    echo "作者校验通过: $me"
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
    gh api graphql -F query=@"$qf" \
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
    gh api graphql -F query=@"$mf" -F tid="$tid"
    rm -f "$mf"
    ;;
  commit-and-push)
    scope=$2
    # Empty-diff guard: if nothing is staged (e.g. all comments resolved/skipped
    # without code change), there is nothing to commit/push — exit cleanly.
    if git diff --cached --quiet; then
      echo "无 staged 改动，跳过 commit"
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
    echo "未知子命令: $1" >&2
    usage >&2
    exit 1
    ;;
esac
