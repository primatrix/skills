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

### Phase 1.5: Snapshot Project V2 fields (runtime invariant)

```bash
PV2_SNAPSHOT=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh snapshot-projectv2-fields <pr-number>)
export BEAVER_FIX_FILES_SNAPSHOT=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh snapshot-files-before)
```

`snapshot-projectv2-fields` writes a JSON snapshot of every Project V2
`fieldValues` node attached to this PR's project items. We re-snapshot at
Phase 6 and assert byte-equality to prove **Project V2 字段未被修改**.

`snapshot-files-before` records the **baseline** of files already dirty vs
HEAD before this command runs. Rollback later computes `current-dirty MINUS
baseline` and restores only that delta — pre-existing user work is never
clobbered. **Must `export`** so subsequent `bash beaver-fix.sh` invocations
(separate processes, different `$$`) see the same baseline path.

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

For each open thread / issue comment, present the comment + diff hunk and prompt with exactly four options:

```text
[接受修复]   Apply the suggested change, stage it, and resolve the thread.
[修改建议]   Open an editor / propose an alternative diff, then resolve.
[跳过]       Leave the thread open, move on.
[仅 resolve] Resolve the thread without code change (e.g. acknowledged).
```

**Serialization rule:** 对每条 open comment 逐条处理：呈现 → 询问 → 等待选择 → 若选择 [接受修复] **立即写入** 文件变更并 stage → 进入下一条。**不要批量收集决策后再一次性写入。** Each comment must be fully processed (immediate write on accept) before moving to the next; the corresponding English wording is "process one comment at a time, immediate write on accept, no batching".

### Phase 4: HARD-GATE Confirmation (AC4)

After all comments are triaged, print a HARD-GATE summary listing every staged change and every thread to be resolved. Require the user to type `yes` to proceed.

If the user answers anything other than `yes`, or sends Ctrl-C, run the rollback path. The script's `trap 'rollback' INT ERR` also fires on mid-script crashes (not only Ctrl-C). Rollback is **scoped** — only files recorded by `snapshot-files-before` are restored via `git checkout HEAD -- "$file"`; unrelated user work is preserved (the script never runs `git checkout -- .`).

### Phase 5: Commit, Push, Resolve (AC5)

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh commit-and-push <scope>
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh resolve-thread <thread-id>
```

`commit-and-push` early-skips when nothing is staged (e.g. every comment was
仅 resolve / 跳过) via `git diff --cached --quiet`. Push uses `git push -u
origin HEAD` so freshly-created PR branches get an upstream.

Commit message template: `fix(<scope>): address review comments`.
Then call the GraphQL `resolveReviewThread` mutation for each accepted thread.

### Phase 6: Verify Project V2 fields unchanged (AC5 runtime assertion)

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-fix.sh verify-projectv2-fields <pr-number> "$PV2_SNAPSHOT"
```

Re-snapshots the PR's Project V2 `fieldValues` and `diff`s against the Phase
1.5 snapshot. Non-zero exit on any divergence. This is the runtime proof
that **Project V2 字段未被修改** — Status / Iteration / Size / Priority /
Progress remain owned by `/beaver-pr` and `/beaver-dev`.

## Constraints

- Only resolves threads the user explicitly accepted; 跳过 leaves them open.
- 不修改 Project V2 字段 — Status/Iteration/Size are owned by /beaver-pr and /beaver-dev. Enforced at runtime via Phase 1.5 + Phase 6.
- Every `--body-file` in the helper script uses a unique tmp filename (mktemp).
- Ctrl-C **or** mid-script crash at any point triggers scoped rollback of unpushed edits (trap on `INT ERR`).
