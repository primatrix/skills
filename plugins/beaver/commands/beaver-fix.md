---
name: beaver-fix
allowed-tools: Bash(gh api:*), Bash(gh pr:*), Bash(git:*)
description: Batch-respond to PR review comments on your own PR. Trigger when the user wants to fix, address, or respond to review comments via /beaver-fix.
argument-hint: "<pr-number>"
---

# /beaver-fix — 批量回应 Review Comments

Batch-process unresolved review threads on a PR you authored. 不修改 Project V2 字段.

## Workflow

### Phase 1: Author Check (AC1)

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh verify-author <pr-number>
```

Compares `gh pr view --json author` against `gh api user --jq .login`.
If they differ, abort with: `只能对自己发起的 PR 运行 /beaver-fix`.

### Phase 2: List Open Review Threads + PR-level Issue Comments (AC2)

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh list-open-comments <pr-number>
```

Collects from BOTH sources:

- **Review threads** (`reviewThreads`) — line-level review comments,
  filtered to `isResolved=false`.
- **Issue comments** (`issueComments` / `pullRequest.comments`) — PR-level
  top-level comments (no resolution state).

If both lists are empty, exit with: `无待处理评论`.

### Phase 3: Per-Comment Triage (AC3)

对每条 open thread / issue comment 逐条 QA, 呈现内容必须包含以下三段(中文):

1. **评论意图解析** — 用一两句话复述 reviewer 在指出什么问题 / 提出什么建议 / 表达什么疑问.
2. **合理性评估** — 结合 diff hunk 与周边代码上下文, 给出 "赞成 / 部分赞成 / 反对" 的判断及简要理由.
3. **推荐选项** — 在下面四个选项中明确推荐其中一个, 并说明为什么.

随后让用户在四个选项中做出选择(中文呈现):

```text
[接受修复]   应用建议的改动，stage 后在最终阶段 resolve 该 thread。
[修改建议]   提出替代 diff，最终阶段 resolve。
[跳过]       保留 thread 为 open，继续下一条。
[仅 resolve] 不改代码，最终阶段仅 resolve（例如已知悉）。
```

**Serialization rule:** 对每条 open comment 逐条处理：呈现三段分析 → 等待用户选择 → 若选择 [接受修复] / [修改建议] **立即写入** 文件变更并 stage(避免事后丢失上下文) → 进入下一条。**不要批量收集决策后再一次性写入文件。** 但 commit / push / resolve 必须留到 Phase 4-5 统一执行, Phase 3 期间不得 commit、不得调用 resolveReviewThread。

### Phase 4: HARD-GATE Confirmation (AC4)

After all comments are triaged, print a HARD-GATE summary listing:

- 每条 comment 的最终选择(接受修复 / 修改建议 / 跳过 / 仅 resolve)
- 所有已 stage 的文件改动列表
- 所有将在 Phase 5 一次性 resolve 的 thread id

用中文要求用户输入 `yes` 才能继续。

If the user answers anything other than `yes`, or sends Ctrl-C, abort. The script's `trap 'rollback' INT ERR` also fires on mid-script crashes (not only Ctrl-C). Rollback **不会** 自动 `git checkout -- .`，以免覆盖用户其他 WIP；如需丢弃已 stage 的改动，请用户手动 `git restore --staged --worktree <file>`。

### Phase 5: Commit, Push, Resolve (AC5)

HARD-GATE 通过后, **一次性** 完成以下三件事(注意顺序):

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh commit-and-push <scope>
# 然后对每个被接受的 thread-id 依次调用:
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh resolve-thread <thread-id>
```

`commit-and-push` early-skips when nothing is staged (e.g. every comment was
仅 resolve / 跳过) via `git diff --cached --quiet`. Push uses `git push -u
origin HEAD` so freshly-created PR branches get an upstream.

Commit message template: `fix(<scope>): address review comments`(单条 commit, 涵盖所有接受的修复).
然后批量调用 GraphQL `resolveReviewThread` mutation 解决所有在 Phase 3 标记为 [接受修复] / [修改建议] / [仅 resolve] 的 thread。

## Constraints

- **所有面向用户的输出（提示、询问、错误、HARD-GATE 文案）一律使用中文。** 仅命令、文件路径、字段名等技术标识可保留英文。
- Only resolves threads the user explicitly accepted; 跳过 leaves them open.
- 不修改 Project V2 字段 — Status/Iteration/Size are owned by /beaver-pr and /beaver-dev. 本命令不再做运行时校验，依赖调用方自律。
- Every `--body-file` in the helper script uses a unique tmp filename (mktemp).
- Ctrl-C **或** 脚本崩溃会触发 trap，但不会自动回滚已写文件（避免误伤用户 WIP）。
