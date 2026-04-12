---
name: beaver-audit
description: "Audit the decomposition of a size/L Beaver issue into sub-tasks. Checks coverage, atomicity (200 LOC limit), and test definitions. Trigger when the user wants to review task decomposition quality."
argument-hint: "<issue-number>"
---

# Beaver Audit

Audit the decomposition quality of a `size/L` parent Issue's sub-tasks. Checks three dimensions: coverage, atomicity, and test definitions.

**References beaver-engine for:** guardrails (Section 3), label ops (Section 4), transition execution (Section 6).

## Prerequisites

- `gh auth status` must succeed
- Target Issue must be `size/L` with sub-issues

## Workflow

### Step 1: Load parent Issue

```bash
gh api repos/{owner}/{repo}/issues/{number} \
  --jq '{title, body, labels: [.labels[].name], milestone: (.milestone.title // null)}'
```

Verify it has `size/L` label. If not, inform the user this skill is for size/L issues only.

### Step 2: Load all sub-issues

```bash
gh api repos/{owner}/{repo}/issues/{number}/sub_issues \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  --jq '.[] | {number, title, body, labels: [.labels[].name]}'
```

If no sub-issues found, inform user and exit.

### Step 3: LLM Audit — three checks

For each sub-issue, evaluate:

#### A. Coverage

Compare the parent Issue's 目标 and 验收标准 against the combined scope of all sub-issues. Identify:
- Covered modules/requirements
- **Gaps**: requirements in the parent that no sub-issue addresses

#### B. Atomicity (200 LOC)

For each sub-issue, estimate whether the implementation can fit within 200 lines of core code (excluding tests, docs, generated files). Flag sub-issues that appear too large.

Criteria for "too large":
- Touches multiple independent modules
- Requires both API + UI changes
- Description implies significant new infrastructure

#### C. Test Definition

Check each sub-issue's body for a testing section. Look for:
- Explicit "测试方法" or "How to Test" or "Test Plan" section
- Specific test scenarios or commands
- Mark as missing if no testing guidance found

### Step 4: Generate audit report

Present as a table:

```markdown
## Beaver Audit Report: #{parent_number} {parent_title}

### Coverage Analysis
- Covered: {list of covered requirements}
- Gaps: {list of uncovered requirements, or "None"}

### Sub-task Details

| # | Title | Atomicity | Test Def | Issues |
|---|-------|-----------|----------|--------|
| {n} | {title} | pass/warn | pass/fail | {details} |

### Summary
- Total sub-tasks: {count}
- Passing all checks: {count}
- Needing attention: {count}
```

### Step 5: Apply labels for failures

For each sub-issue with missing test definition:
```bash
gh api repos/{owner}/{repo}/issues/{sub_number}/labels --method POST -f "labels[]=beaver/missing-test"
```

For each sub-issue flagged as too large:
```bash
gh api repos/{owner}/{repo}/issues/{sub_number}/labels --method POST -f "labels[]=beaver/needs-split"
```

If coverage gaps found, add to parent:
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]=beaver/missing-context"
```

### Step 6: Post audit summary as Issue comment

Write the generated report to a temporary file first, then post it:
```bash
cat > /tmp/beaver-audit-report.md << 'BEAVEREOF'
{rendered_audit_report}
BEAVEREOF

gh api repos/{owner}/{repo}/issues/{number}/comments --method POST \
  --raw-field body="$(cat /tmp/beaver-audit-report.md)"
```

### Step 7: Conditional transition

If ALL checks pass (no gaps, all atomic, all have test defs):
- Ask user: "All checks passed. Transition parent to `status/ready-to-develop`?"
- If confirmed: execute transition per engine Section 6

If ANY check fails:
- Keep current status
- Inform user what needs fixing

## Constraints

- Only works on `size/L` issues with sub-issues
- Atomicity is an LLM estimate, not exact — flag as warning not failure
- Never auto-transition without user confirmation
- Audit comments in Chinese where applicable
