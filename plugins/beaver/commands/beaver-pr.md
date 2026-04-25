---
allowed-tools: Bash(gh api:*), Bash(gh pr:*), Bash(gh issue:*), Bash(git:*)
description: Commit, push, and open a Draft PR with Beaver compliance checks. Trigger when the user wants to commit, push, or create a pull request.
argument-hint: "[issue-number]"
---

# /beaver-pr — 代码审查

Phase 6 of the Beaver development lifecycle.

## Workflow

### Phase 1: Context Gathering

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh ctx
```

### Phase 2: Issue Association

按以下顺序推断 Issue 编号；任何一步成功即停止：

1. 命令参数（`/beaver-pr <number>`）。
2. 分支前缀：`<type>/<number>-<desc>` 形式的 branch name。
3. 最近 20 条 commit message 中第一个 `#<number>` 引用。
4. 以上都缺失：提示用户输入 Issue 编号（不可省略）。

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh infer-issue
```

PR body 必须包含 `Closes {org}/{issueRepo}#<issue_number>` 一行（**完整 owner/repo 形式**，因为 Beaver Issue 通常位于 `primatrix/projects`，与代码 PR 跨仓库；GitHub 的 `Closes #N` 简写仅在同仓库内生效），确保 PR merge 时 Issue 自动关闭。

### Phase 3: Branch + Commit + Push

1. Create branch if not already on a feature branch:

   ```bash
   BRANCH_NAME="{type}/{issue_number}-{short_desc}"
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh create-branch "$BRANCH_NAME"
   ```

1. Stage, conventional-commit, push.

   把 conventional commit 信息写入唯一命名的临时文件（`mktemp` 或 `/tmp/beaver-pr-msg-$$-$RANDOM.txt`），格式：

   ```text
   {type}({scope}): {description}

   Closes {org}/{issueRepo}#{issue_number}
   ```

   然后：

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh commit-push "$BRANCH_NAME" /tmp/beaver-pr-msg-$$-$RANDOM.txt {relevant_files}
   ```

### Phase 4: Compliance Checks (PR-body warnings, never Issue labels)

G004 / G006 仅产生 PR body 警告附加段；不在原 Issue 上贴任何 `beaver/*` 审计标签。

#### G004 — Test evidence

```bash
if ! bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh check-tests; then
  # 触发 G004：在 PR body 末尾追加一行
  printf '\n> ⚠️ Beaver audit: 本次 PR 未包含 test 文件改动\n' >> "$pr_body_file"
fi
```

#### G006 — Type / Size 自动补齐

读取 Issue 当前 Type / Size：

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh check-fields {org} {issueRepo} {issue_number}
```

若任一为空，调用 `beaver-lib.sh::set_type` / `set_size` 自动补齐（`Type` 默认补 `Task`，`Size` 默认补 `S`）：

```bash
if ! bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh autofill-fields {issue_number} S; then
  # auto-fill failure → 在 PR body 末尾追加警告
  printf '\n> ⚠️ Beaver audit: Issue #%s missing Type/Size fields and auto-fill failed\n' \
    {issue_number} >> "$pr_body_file"
fi
```

> 注意：除 G006 触发的 `set_type` / `set_size` 外，本命令不修改 Issue 的任何 Project V2 字段（不写 `Status` / `Iteration`，不发起任何字段写入的 `gh api graphql` 调用）。

### Phase 5: Create Draft PR

把 PR body 内容（含上面 Phase 4 追加的警告段）以字符串形式传给 `create-pr`；脚本内部走 `mktemp` 生成唯一临时文件后 `--body-file`：

```markdown
## Summary
{2-3 bullet points of changes}

## Test Plan

- [ ] {verification steps}

Closes {org}/{issueRepo}#{issue_number}
```

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh create-pr \
  "{type}({scope}): {description}" \
  "$pr_body_string"
```

### Phase 6: Completion Options (4 互斥选项)

打印恰好 4 个互斥选项（absorbed from superpowers:finishing-a-development-branch）：

```text
Draft PR created. What would you like to do?

1. Keep as Draft PR (self-review first, then mark Open)
1. Mark PR as Ready for Review immediately
1. Keep the branch as-is (我稍后处理)
1. Discard this work
```

- Option 1（默认）: Keep Draft. 输出 `Self-review the Draft PR at {pr_url}. When ready, mark it Open for team review.`
- Option 2: `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh mark-ready {pr_number}`. 输出 `PR marked as Ready for Review.`
- Option 3: 保留分支与 Draft PR；输出 `Branch and Draft PR preserved.`
- Option 4: Discard this work — 必须二次确认。提示用户键入字面字符串 `discard` 才执行；任何其它输入视为取消，回到 Option 1。

  ```bash
  # 用户输入 "discard" 后：
  gh pr close {pr_number}
  bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-pr.sh delete-branch "$BRANCH_NAME"
  ```

### Phase 7: Worktree Cleanup

如果在 worktree 中执行：

- Options 1, 2, 3：保留 worktree（PR 仍存在或分支保留）
- Option 4：移除 worktree

## Code Review Reception (absorbed from superpowers:receiving-code-review)

收到 review 反馈后：

- **Read** complete feedback without reacting
- **Verify** against codebase reality before implementing
- **Push back** with technical reasoning if feedback is wrong
- **Never** use performative agreement ("You're absolutely right!", "Great point!")
- **Just fix** and show in the code — actions speak louder than words
- 来自外部 reviewer 的反馈：保持技术怀疑、先验证再实施
- 反馈与已确定决策冲突：停下来与用户讨论

## Constraints

- PR 默认创建为 **Draft**（用户 self-review 后 mark Open）
- `Closes {org}/{issueRepo}#{issue_number}` 写入 PR body（**完整 owner/repo 形式**，跨仓库自动关闭必需），merge 时 Issue 自动关闭
- G004 / G006 都是 warning-only（不阻断 PR 创建，不在 Issue 上贴 `beaver/*` 标签）
- 除 G006 触发的 Type/Size 自动补齐外，命令不修改任何 Project V2 字段
- 所有 `--body-file` 传给 `gh` CLI 的临时文件必须使用唯一文件名（`mktemp` 或 `/tmp/beaver-pr-body-$$-$RANDOM.md` 等）；该约束由 `beaver-pr.sh` 在内部统一处理
- §7 QA loop 不适用（PR 内容由 git diff 生成，不走用户 Q&A）
