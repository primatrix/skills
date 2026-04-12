---
name: beaver-pr
description: "Commit, push, and open a PR with Beaver compliance checks (LOC guard, label completeness, test evidence). Automatically transitions the linked Issue to review-needed. Trigger when the user wants to commit, push, or create a pull request."
---

# Beaver PR

Commit changes, push, and open a GitHub PR with integrated Beaver compliance checks. Validates LOC limits, label completeness, and test evidence before PR creation. Automatically transitions the linked Issue to `status/review-needed`.

**References beaver-engine for:** guardrails G004-G006 (Section 3), label ops (Section 4), config reading (Section 5), transition execution (Section 6).

## Prerequisites

- `gh auth status` must succeed
- Working directory inside a git repository
- Changes to commit (staged or unstaged)

## Workflow

### Phase 1: Context Gathering (auto)

Run in parallel:
```bash
git status
git diff HEAD
git branch --show-current
git log --oneline -10
```

### Phase 2: Branch + Commit + Push (auto)

1. If on main/master, create branch: `<type>/<issue-number>-<short-desc>`
   - Extract type and issue number from context if available
2. Stage all relevant changes (exclude secrets: `.env`, `credentials.*`)
3. Commit with descriptive message
4. Push with `-u origin <branch>` if new branch

### Phase 3: Issue Association (prompt user)

Detect issue number from:
1. Branch name pattern (e.g., `feat/42-add-login` → #42)
2. Recent commit messages containing `#N`

If detected, confirm with user. If not detected, ask:

> "Associate this PR with a Beaver issue?"
> - Enter issue number (e.g., `42` or `#42`) or full URL
> - `new` → create via beaver-issue first
> - `skip` → no association

Parse response. If `new`: tell user to run `/beaver-issue` first, then resume.

### Phase 4: Compliance Checks (auto, report to user)

Run all checks and present results as a table:

#### G005: LOC Guard

```bash
git diff --numstat origin/main...HEAD
```

Filter to core directories (from `beaver-config`, default: repo root). Exclude:
- `**/*_test.*`, `**/test_*.*`, `**/tests/**`
- `**/*.md`, `**/docs/**`
- `*.pb.go`, `*_generated.*`, `*.lock`

Sum added lines. If > 200:
- Mark check as WARN
- Will add `beaver/needs-split` label after PR creation

#### G006: Label Completeness

If Issue is associated:
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --jq '.[].name'
```
Check for at least one `type/` and one `size/` label. If missing:
- Mark check as FAIL
- List missing label categories

#### G004: Test Evidence

Search for test evidence in order:
1. **Session context**: scan conversation history for test runner output patterns (`PASSED`, `FAILED`, `ok`, `FAIL`, test count summaries)
2. **Diff**: check if PR includes new/modified test files
3. **Note**: CI checks will run after PR creation

If evidence found, extract summary for PR body. If not found:
- Mark check as WARN
- Will add `beaver/missing-test` label after PR creation

#### Present Results

```
## Beaver Compliance Check

| Rule | Status | Details |
|------|--------|---------|
| G005 LOC Guard | PASS/WARN | {N} lines in core dirs (limit: 200) |
| G006 Labels | PASS/FAIL | type/{x}, size/{y} present |
| G004 Test Evidence | PASS/WARN | {source}: {summary} |

{If any FAIL}: "Some checks failed. Fix issues before creating PR."
{If only WARN}: "Warnings found. Continue with PR creation? (y/n)"
{If all PASS}: "All checks passed. Creating PR."
```

Wait for user confirmation if there are warnings.

### Phase 5: Create PR (auto after confirmation)

```bash
gh pr create --title "{title}" --body "$(cat <<'EOF'
## Summary
{bullet points describing changes}

## Test Plan
{test evidence summary from G004, or "TODO: add test evidence"}

{If issue associated: "Relates to {owner}/{repo}#{number}"}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### Phase 6: Post-PR Actions (auto)

If an Issue is associated (user did NOT choose `skip`):

1. Apply beaver labels for any WARN checks:
   ```bash
   # If G005 warned:
   gh api repos/{owner}/{repo}/issues/{issue_number}/labels --method POST -f "labels[]=beaver/needs-split"
   # If G004 warned:
   gh api repos/{owner}/{repo}/issues/{issue_number}/labels --method POST -f "labels[]=beaver/missing-test"
   ```

2. Transition Issue status to `status/review-needed`:
   Execute per engine Section 6 — validate target is legal next state from current status per Section 2.

If no Issue associated (`skip`): skip label and transition steps.

3. Report PR URL to user.

## Constraints

- Never commit `.env`, `credentials.*`, or files likely containing secrets
- Always preview compliance check results before PR creation
- If G006 fails (missing labels), do not create PR — ask user to fix labels first
- Issue body/comments in Chinese where applicable
