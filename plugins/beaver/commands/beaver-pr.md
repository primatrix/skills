---
allowed-tools: Bash(gh api:*), Bash(gh pr:*), Bash(git:*)
description: Commit, push, and open a Draft PR with Beaver compliance checks. Trigger when the user wants to commit, push, or create a pull request.
argument-hint: "[issue-number]"
---

# /beaver-pr — 代码审查

Phase 6 of the Beaver development lifecycle.

## Workflow

### Phase 1: Context Gathering

Run in parallel:

```bash
git status
git diff --stat HEAD
git branch --show-current
git log --oneline -10
```

### Phase 2: Issue Association

1. If argument provided: use that issue number.
1. Otherwise: detect from branch name (pattern: `{type}/{issue_number}-{desc}`) or commit messages (`#{number}`).
1. If not found: ask user for issue number.

### Phase 3: Branch + Commit + Push

1. Create branch if not already on a feature branch:

   ```bash
   BRANCH_NAME="{type}/{issue_number}-{short_desc}"
   git checkout -b ${BRANCH_NAME} 2>/dev/null || true
   ```

1. Stage, commit (use conventional commit format), push:

   ```bash
   git add {relevant_files}
   git commit -m "{type}({scope}): {description}

   Closes #{issue_number}"
   git push -u origin ${BRANCH_NAME}
   ```

### Phase 4: Compliance Checks

Run guardrail checks and present as table:

| Check | Result | Details |
|---|---|---|
| G004: Test evidence | ✅ PASS / ⚠️ WARN | {test files found / CI status} |
| G006: Label completeness | ✅ PASS / ⚠️ WARN | {type/ + size/ present on issue} |

```bash
# G004: Check for test files in diff
git diff --name-only origin/main...HEAD | grep -E '(test_|_test\.|/tests/)'

# G006: Check issue labels
gh api repos/{org}/{issueRepo}/issues/{issue_number}/labels --jq '.[].name'
```

If G004 warns: add `beaver/missing-test` label to Issue.
If G006 warns: add `beaver/missing-context` label, list missing labels.

### Phase 5: Create Draft PR

```bash
gh pr create --draft \
  --title "{type}({scope}): {description}" \
  --body "$(cat <<'EOF'
## Summary
{2-3 bullet points of changes}

## Test Plan

- [ ] {verification steps}

Closes #{issue_number}
EOF
)"
```

### Phase 6: Completion Options

Present exactly 4 options (absorbed from superpowers:finishing-a-development-branch):

```text
Draft PR created. What would you like to do?

1. Keep as Draft PR (self-review first, then mark Open)
1. Mark PR as Ready for Review immediately
1. Keep the branch as-is (I'll handle it later)
1. Discard this work
```

- Option 1 (default): Keep Draft. Print: "Self-review the Draft PR at {pr_url}. When ready, mark it Open for team review."
- Option 2: `gh pr ready {pr_number}`. Print: "PR marked as Ready for Review."
- Option 3: Print: "Branch and Draft PR preserved."
- Option 4: Require typed "discard" confirmation. Delete branch + close PR.

### Phase 7: Worktree Cleanup

If working in a worktree:

- Options 1, 2: keep worktree (PR still open)
- Option 3: keep worktree
- Option 4: remove worktree

## Code Review Reception (absorbed from superpowers:receiving-code-review)

When the user receives review feedback on the PR, follow these rules:

- **Read** complete feedback without reacting
- **Verify** against codebase reality before implementing
- **Push back** with technical reasoning if feedback is wrong
- **Never** use performative agreement ("You're absolutely right!", "Great point!")
- **Just fix** and show in the code — actions speak louder than words
- When feedback is from external reviewers: verify technically, be skeptical
- When feedback conflicts with prior decisions: stop and discuss with user first

## Constraints

- PR is created as **Draft** by default (user self-reviews before marking Open)
- `Closes #{issue_number}` in PR body ensures Issue auto-closes on merge → `status/done`
- G004 and G006 are warning-only (do not block PR creation)
- §7 QA loop does NOT apply (PR content is generated from git diff, not user Q&A)
