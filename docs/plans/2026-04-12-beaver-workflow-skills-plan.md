# Beaver Workflow Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the existing beaver plugin's 2 skills with 6 new skills (1 internal engine + 5 command-layer) that integrate Beaver's label-driven project management into the Claude Code developer workflow.

**Architecture:** Core engine skill (`beaver-engine`) provides shared state machine rules, guardrail checks, label operations, and config reading. Five thin command-layer skills (`beaver-issue`, `beaver-pr`, `beaver-audit`, `beaver-report`, `beaver-focus`) call the engine's rules inline. All skills are pure SKILL.md markdown — no scripts.

**Tech Stack:** SKILL.md (YAML frontmatter + markdown), `gh` CLI for GitHub API, GitHub Projects V2 GraphQL API.

**Design doc:** `docs/plans/2026-04-12-beaver-workflow-skills-design.md`

---

### Task 1: Create beaver-engine (internal skill)

**Files:**
- Create: `plugins/beaver/skills/beaver-engine/SKILL.md`

**Step 1: Write the beaver-engine SKILL.md**

```markdown
---
name: beaver-engine
description: "Internal engine for Beaver workflow skills. DO NOT trigger directly. Provides state machine rules, guardrail checks, label operations, and project config reading used by beaver-issue, beaver-pr, beaver-audit, beaver-report, and beaver-focus."
---

# Beaver Engine

Internal skill. Do not invoke directly. Other beaver skills reference these rules and command templates.

## 1. Label Taxonomy

All labels use a `prefix/name` format:

### Type labels (`type/`)
- `type/feat` — New feature
- `type/bug` — Bug fix
- `type/refactor` — Refactoring
- `type/docs` — Documentation
- `type/chore` — Infrastructure, build, misc

### Priority labels (`p/`)
- `p0/blocker` — Blocking, top of daily report
- `p1/urgent` — Urgent, top of daily report
- `p2/high` — High priority
- `p3/normal` — Normal

### Size labels (`size/`)
- `size/S` — Small, fast-track SOP
- `size/L` — Large, full lifecycle SOP

### Status labels (`status/`)
- `status/triage` — Initial state, awaiting triage
- `status/requirements-gathering` — (size/L only) Requirements refinement
- `status/design-pending` — (size/L only) Design review in progress
- `status/ready-to-develop` — (size/L only) Design approved, ready to code
- `status/in-progress` — Active development
- `status/blocked` — Blocked (must note reason)
- `status/review-needed` — Awaiting code/design review
- `status/done` — Completed and merged

### Beaver agent labels (`beaver/`)
- `beaver/needs-split` — PR LOC exceeds 200 in core dirs
- `beaver/missing-test` — No test evidence found before done
- `beaver/missing-context` — Incomplete labels or description
- `beaver/stale` — Stuck in same status > 3 days
- `beaver/overdue` — Past DDL and not done
- `beaver/upstream-blocked` — Upstream dependency is blocked
- `beaver/wontfix` — Will not fix, skip stale/overdue detection

## 2. State Machine

### size/S Fast Track
```
triage → in-progress → review-needed → done
```

### size/L Standard SOP
```
triage → requirements-gathering → design-pending → ready-to-develop → in-progress → review-needed → done
```

### Universal transitions
- Any status → `blocked` (must note reason in Issue comment)
- `blocked` → restore to previous status

### Legal next-states lookup

| Current Status | size/S next | size/L next |
|---|---|---|
| triage | in-progress | requirements-gathering |
| requirements-gathering | N/A | design-pending |
| design-pending | N/A | ready-to-develop |
| ready-to-develop | N/A | in-progress |
| in-progress | review-needed | review-needed |
| review-needed | done | done |
| blocked | (previous) | (previous) |

## 3. Guardrail Rules

### G001: Size required before leaving triage
- **Check:** Issue has a `size/S` or `size/L` label
- **When:** Any transition FROM `status/triage`
- **Fail action:** Block transition, comment on Issue requesting size classification

### G002: size/L must not skip stages
- **Check:** Target status is the legal next state per size/L SOP
- **When:** Any transition of a `size/L` Issue
- **Fail action:** Block transition, comment listing required intermediate stages

### G003: Cannot skip review
- **Check:** `in-progress` cannot go directly to `done`
- **When:** Transition to `status/done`
- **Fail action:** Block transition, require `status/review-needed` first

### G004: Test evidence required for done
- **Check:** Find test evidence from (in priority order):
  1. Current session context — scan conversation for test runner output (pytest, go test, npm test, cargo test, etc.)
  2. PR diff — new/modified test files (`*_test.*`, `test_*.*`, `tests/**`)
  3. CI status — GitHub Actions / Check Runs on associated PR
- **When:** Transition to `status/done` or PR creation
- **Fail action:** Add `beaver/missing-test` label, comment requesting evidence
- **On success:** Write test summary to PR body's Test Plan section or Issue comment

### G005: LOC limit on PR
- **Check:** Count added lines in core directories, excluding:
  - `**/*_test.*`, `**/test_*.*`, `**/tests/**`
  - `**/*.md`, `**/docs/**`
  - `*.pb.go`, `*_generated.*`, `*.lock`
- **Threshold:** 200 lines
- **When:** PR creation
- **Fail action:** Add `beaver/needs-split` label, warn developer (do not block — let developer confirm)

### G006: PR must have complete labels
- **Check:** Associated Issue has at least one `type/` label AND one `size/` label
- **When:** PR creation
- **Fail action:** Add `beaver/missing-context` label, list missing labels

## 4. Label Operations (gh command templates)

### Read all labels on an Issue
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --jq '.[].name'
```
Parse into structured data:
- `type`: first label matching `type/*`
- `size`: first label matching `size/*`
- `status`: first label matching `status/*`
- `priority`: first label matching `p*/*`
- `beaver_flags`: all labels matching `beaver/*`

### Set status label (atomic swap)
Remove all existing `status/*` labels, then add the new one:
```bash
# Remove current status label
gh api repos/{owner}/{repo}/issues/{number}/labels/{current_status_label} --method DELETE

# Add new status label
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={new_status_label}"
```

### Add beaver flag label
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={beaver_label}"
```

## 5. Project Config Reading

Read from Project V2 README's `beaver-config` YAML block:

```bash
gh project view {number} --owner {org} --format json --jq '.readme'
```

Parse the ```` ```yaml beaver-config ```` fenced block for:
- `repositories`: list of observed repos
- `issueRepo`: the repo hosting Beaver issues
- `customFields`: field name overrides (default: Level, Status, Progress)

## 6. Transition Execution Template

When a command-layer skill needs to transition an Issue's status:

1. Read current labels (Section 4)
2. Determine `size` from labels
3. Look up legal next states (Section 2)
4. Validate target state against guardrails (Section 3)
5. If all checks pass: execute atomic label swap (Section 4)
6. If any check fails: report failure, do NOT swap labels
```

**Step 2: Verify frontmatter is valid YAML**

Run: `head -5 plugins/beaver/skills/beaver-engine/SKILL.md`
Expected: Valid `---` fenced YAML with `name` and `description` fields.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-engine/SKILL.md
git commit -m "feat(beaver): add beaver-engine internal skill

Core engine providing state machine, guardrail rules (G001-G006),
label operations, and project config reading for all beaver skills."
```

---

### Task 2: Update create-beaver-project to create full label taxonomy

**Files:**
- Modify: `plugins/beaver/commands/create-beaver-project.md`

**Step 1: Update the Labels section**

In `create-beaver-project.md`, replace the existing "Create Labels" section (the table with 4 labels and the `gh label create` command) with the full label taxonomy. Keep the existing structure but expand the label table:

Replace the current labels table and command with:

```markdown
### Create Labels

Create on the issue repository. Skip any that already exist (`--force` updates color/description if label exists).

**Type labels:**

| Label | Color | Description |
|-------|-------|-------------|
| type/feat | 0E8A16 | New feature |
| type/bug | D73A4A | Bug fix |
| type/refactor | E4E669 | Code refactoring |
| type/docs | 0075CA | Documentation |
| type/chore | BFD4F2 | Infrastructure, build, misc |

**Priority labels:**

| Label | Color | Description |
|-------|-------|-------------|
| p0/blocker | B60205 | Blocking — top of daily report |
| p1/urgent | D93F0B | Urgent — top of daily report |
| p2/high | FBCA04 | High priority |
| p3/normal | C2E0C6 | Normal priority |

**Size labels:**

| Label | Color | Description |
|-------|-------|-------------|
| size/S | C5DEF5 | Small task — fast-track SOP |
| size/L | 1D76DB | Large task — full lifecycle SOP |

**Status labels:**

| Label | Color | Description |
|-------|-------|-------------|
| status/triage | E4E669 | Awaiting triage |
| status/requirements-gathering | D4C5F9 | Requirements refinement (size/L) |
| status/design-pending | D4C5F9 | Design review in progress (size/L) |
| status/ready-to-develop | 0E8A16 | Ready to code (size/L) |
| status/in-progress | FBCA04 | Active development |
| status/blocked | B60205 | Blocked |
| status/review-needed | 1D76DB | Awaiting review |
| status/done | 0E8A16 | Completed and merged |

**Beaver agent labels:**

| Label | Color | Description |
|-------|-------|-------------|
| beaver/needs-split | D93F0B | PR LOC exceeds 200 in core dirs |
| beaver/missing-test | D93F0B | No test evidence before done |
| beaver/missing-context | D93F0B | Incomplete labels or description |
| beaver/stale | E4E669 | Stuck in same status > 3 days |
| beaver/overdue | B60205 | Past DDL and not done |
| beaver/upstream-blocked | D93F0B | Upstream dependency blocked |
| beaver/wontfix | BFDADC | Will not fix |

**Control label:**

| Label | Color | Description |
|-------|-------|-------------|
| Control-By-Beaver | 7B61FF | Issue managed by Beaver automation |

```bash
gh label create "{label}" --repo {org}/{issueRepo} --color "{color}" --description "{desc}" --force
```

Create all labels in sequence. `--force` ensures idempotency (updates existing labels).
```

Also remove the old standalone labels (Approve-Design, Reviewing, Waiting-For-Merge) from the table — they are replaced by the `status/` labels.

**Step 2: Verify the file**

Run: `head -3 plugins/beaver/commands/create-beaver-project.md`
Expected: Valid `---` YAML frontmatter.

**Step 3: Commit**

```bash
git add plugins/beaver/commands/create-beaver-project.md
git commit -m "feat(beaver): expand create-beaver-project with full label taxonomy

Replace 4 simple labels with complete prefix-based label system:
type/, p/, size/, status/, beaver/ categories."
```

---

### Task 3: Create beaver-issue skill

**Files:**
- Create: `plugins/beaver/skills/beaver-issue/SKILL.md`

**Step 1: Write the beaver-issue SKILL.md**

```markdown
---
name: beaver-issue
description: "Create or claim a Beaver-tracked GitHub Issue with automatic status transitions and guardrail checks. Trigger when the user wants to create a GitHub issue, claim/start a task, or pick up work."
argument-hint: "[issue-number to claim, or omit to create new]"
---

# Beaver Issue

Create a new Beaver-tracked Issue or claim an existing one. Handles Project V2 field setup, label assignment, parent linking, and automatic status transitions with guardrail enforcement.

**References beaver-engine for:** state machine (Section 2), guardrails G001 (Section 3), label ops (Section 4), config reading (Section 5), transition execution (Section 6).

## Prerequisites

- `gh auth status` must succeed
- Project scope: `gh auth refresh -s project` if missing
- Project README must contain `beaver-config` YAML block

## Detect Mode

- If an argument is provided (issue number): **Claim mode**
- If no argument: **Create mode**

---

## Create Mode

### Step 1: Load project config

Check auto memory for `beaver-issue-defaults.md`. If found, present defaults for confirmation. Parse project config per engine Section 5.

### Step 2: Collect issue details

Collect one at a time:
1. **Level**: Goal / Task / SubTask
2. **Parent issue** (Task/SubTask only): list project items, filter by parent level, let user pick
3. **Title**: concise issue title
4. **Description**: structured as 目标 (objective) and 验收标准 (acceptance criteria), in Chinese
5. **Type label** (`type/`): feat / bug / refactor / docs / chore
6. **Priority label** (`p/`): p0/blocker / p1/urgent / p2/high / p3/normal

### Step 3: Auto-classify size

Analyze the description and suggest `size/S` or `size/L`:
- If description mentions multiple components, API changes + frontend + tests → suggest `size/L`
- If description is focused on a single change → suggest `size/S`
Present suggestion with reasoning. Wait for user confirmation.

### Step 4: Preview and confirm

Show complete issue details in a structured preview. Wait for explicit approval.

### Step 5: Create the Issue

```bash
gh api repos/{org}/{issueRepo}/issues --method POST \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  -f title="{title}" --raw-field body="$(cat "$BODY_FILE")" \
  -f type="{level}" \
  -f "labels[]=Control-By-Beaver" \
  -f "labels[]=type/{type}" \
  -f "labels[]=size/{size}" \
  -f "labels[]=p{n}/{priority}" \
  -f "labels[]=status/triage"
```
Add `-f milestone={number}` if selected. If issue type API fails, retry without `-f type`.

### Step 6: Add to Project V2 and set fields

```bash
gh project item-add {projectNumber} --owner {org} --url {issue_url} --format json
```
Set Level, Status (Not Started), Progress (0) fields per engine Section 4.

### Step 7: Link to parent (Task/SubTask only)

```bash
CHILD_ID=$(gh api repos/{org}/{issueRepo}/issues/{number} --jq '.id')
gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues \
  --method POST -H "X-GitHub-Api-Version: 2026-03-10" \
  -f sub_issue_id="$CHILD_ID"
```

### Step 8: Auto-transition from triage

Per engine state machine:
- `size/S`: transition `status/triage` → `status/in-progress`
- `size/L`: transition `status/triage` → `status/requirements-gathering`

Execute transition per engine Section 6 (validates G001).

### Step 9: Report and save defaults

Print summary: issue URL, level, labels, milestone, status, parent. Silently save defaults to `beaver-issue-defaults.md`.

---

## Claim Mode

### Step 1: Load issue

```bash
gh api repos/{owner}/{repo}/issues/{number} --jq '{title, state, labels: [.labels[].name], assignees: [.assignees[].login]}'
```

### Step 2: Validate claimable

Parse labels per engine Section 4. Check status is `status/triage` or `status/ready-to-develop`. If not, inform the developer the issue cannot be claimed in its current state.

### Step 3: Assign

```bash
CURRENT_USER=$(gh api user --jq '.login')
gh api repos/{owner}/{repo}/issues/{number}/assignees --method POST -f "assignees[]=$CURRENT_USER"
```

### Step 4: Auto-transition to in-progress

Execute transition per engine Section 6:
- From `status/triage`: validate G001 (size label required)
- From `status/ready-to-develop`: direct transition
- Target: `status/in-progress`

### Step 5: Report

Print: issue URL, title, assigned user, new status.

---

## Issue Body Template

### Goal
```
## 目标

{objective}

## 验收标准

{acceptance_criteria}
```

### Task / SubTask
```
## 目标

{objective}

## 验收标准

{acceptance_criteria}

<!-- beaver-tracking
repos:
  - {repo1}
paths:
  - {path1}
keywords:
  - {keyword1}
-->
```

## Constraints

- One issue at a time
- Always preview before creating
- Issue body in Chinese (中文)
- Read-only project config
- Never modify existing issues (except label transitions during claim)
```

**Step 2: Verify frontmatter**

Run: `head -5 plugins/beaver/skills/beaver-issue/SKILL.md`
Expected: Valid YAML frontmatter.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "feat(beaver): add beaver-issue skill with create and claim modes

Replaces create-beaver-issue. Adds auto-classification of size,
automatic status transitions, and guardrail enforcement."
```

---

### Task 4: Create beaver-pr skill

**Files:**
- Create: `plugins/beaver/skills/beaver-pr/SKILL.md` (overwrite existing)

**Step 1: Write the new beaver-pr SKILL.md**

```markdown
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

Relates to {owner}/{repo}#{number}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### Phase 6: Post-PR Actions (auto)

1. Apply beaver labels for any WARN checks:
   ```bash
   # If G005 warned:
   gh api repos/{owner}/{repo}/issues/{issue_number}/labels --method POST -f "labels[]=beaver/needs-split"
   # If G004 warned:
   gh api repos/{owner}/{repo}/issues/{issue_number}/labels --method POST -f "labels[]=beaver/missing-test"
   ```

2. Transition Issue status to `status/review-needed`:
   Execute per engine Section 6 — validates G003 (must come from `in-progress`).

3. Report PR URL to user.

## Constraints

- Never commit `.env`, `credentials.*`, or files likely containing secrets
- Always preview compliance check results before PR creation
- If G006 fails (missing labels), do not create PR — ask user to fix labels first
- Issue body/comments in Chinese where applicable
```

**Step 2: Verify frontmatter**

Run: `head -4 plugins/beaver/skills/beaver-pr/SKILL.md`
Expected: Valid YAML frontmatter.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-pr/SKILL.md
git commit -m "feat(beaver): rewrite beaver-pr with compliance checks and auto-transition

Adds LOC guard (G005), label completeness (G006), test evidence
extraction from session context (G004), and auto-transition to
review-needed."
```

---

### Task 5: Create beaver-audit skill

**Files:**
- Create: `plugins/beaver/skills/beaver-audit/SKILL.md`

**Step 1: Write the beaver-audit SKILL.md**

```markdown
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
  --jq '{title, body, labels: [.labels[].name], milestone: .milestone.title}'
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

```
## Beaver Audit Report: #{parent_number} {parent_title}

### Coverage Analysis
- ✅ Covered: {list of covered requirements}
- ❌ Gaps: {list of uncovered requirements, or "None"}

### Sub-task Details

| # | Title | Atomicity | Test Def | Issues |
|---|-------|-----------|----------|--------|
| {n} | {title} | ✅/⚠️ | ✅/❌ | {details} |

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

```bash
gh api repos/{owner}/{repo}/issues/{number}/comments --method POST \
  --raw-field body="$(cat "$AUDIT_REPORT_FILE")"
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
- Atomicity is an LLM estimate, not exact — flag as ⚠️ not ❌
- Never auto-transition without user confirmation
- Audit comments in Chinese where applicable
```

**Step 2: Verify frontmatter**

Run: `head -5 plugins/beaver/skills/beaver-audit/SKILL.md`
Expected: Valid YAML frontmatter.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-audit/SKILL.md
git commit -m "feat(beaver): add beaver-audit skill for task decomposition review

LLM-powered audit checking coverage, atomicity (200 LOC), and test
definitions for size/L issue decompositions."
```

---

### Task 6: Create beaver-report skill

**Files:**
- Create: `plugins/beaver/skills/beaver-report/SKILL.md`

**Step 1: Write the beaver-report SKILL.md**

```markdown
---
name: beaver-report
description: "Generate a project health report with milestone progress, stale/overdue detection, blocking chains, and risk analysis. Trigger when the user asks about project status, progress, health, or risks."
---

# Beaver Report

Generate a comprehensive project health report covering milestone progress, health indicators, risk analysis, and sub-task rollup.

**References beaver-engine for:** label taxonomy (Section 1), label ops (Section 4), config reading (Section 5).

## Prerequisites

- `gh auth status` must succeed
- A Beaver-configured Project V2 exists

## Workflow

### Step 1: Load project config

Read `beaver-config` per engine Section 5. Identify issue repo and observed repos.

### Step 2: Fetch active milestone

```bash
gh api repos/{org}/{issueRepo}/milestones --jq '[.[] | select(.state=="open")] | sort_by(.due_on) | .[0] | {number, title, due_on, open_issues, closed_issues}'
```

### Step 3: Fetch all open Beaver issues

```bash
gh api repos/{org}/{issueRepo}/issues?labels=Control-By-Beaver\&state=open\&per_page=100 \
  --jq '.[] | {number, title, labels: [.labels[].name], assignees: [.assignees[].login], updated_at, created_at, milestone: .milestone.title}'
```

### Step 4: Compute health indicators

Parse each issue's labels per engine Section 4 and check:

#### Stale Detection
Issues where `status/in-progress` or `status/review-needed` and `updated_at` > 3 days ago.
For each: add `beaver/stale` label if not already present.

#### Overdue Detection
Issues with milestone whose `due_on` has passed and status is not `status/done`.
For each: add `beaver/overdue` label if not already present.
Skip issues with `beaver/wontfix` label.

#### Upstream Blocked
Scan issue bodies for `Depends on #{N}` patterns. Check if referenced issues have `status/blocked`. Mark downstream issues with `beaver/upstream-blocked`.

#### Missing Context
Issues without `type/` or `size/` label (excluding those in `status/triage`).

### Step 5: Compute sub-task rollup

For issues with `size/L`, fetch sub-issues:
```bash
gh api repos/{org}/{issueRepo}/issues/{number}/sub_issues \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  --jq '.[] | {number, title, labels: [.labels[].name]}'
```
Calculate: total sub-tasks, completed (has `status/done`), percentage.

### Step 6: Generate report

```
# Beaver Project Report

**Generated:** {date} | **Milestone:** {title} (due {due_on})

## Milestone Progress

| Metric | Value |
|--------|-------|
| Total Issues | {open + closed} |
| Closed | {closed} ({pct}%) |
| size/L | {count} ({completed}/{total}) |
| size/S | {count} ({completed}/{total}) |

## Health Indicators

### 🔴 Overdue ({count})
| # | Title | Assignee | Days Overdue |
|---|-------|----------|-------------|

### 🟡 Stale ({count})
| # | Title | Status | Days Since Update |
|---|-------|--------|------------------|

### 🔴 Blocked ({count})
| # | Title | Blocked By |
|---|-------|-----------|

### 🟡 Missing Context ({count})
| # | Title | Missing |
|---|-------|---------|

## size/L Progress Rollup

| # | Title | Sub-tasks | Completed | Progress |
|---|-------|-----------|-----------|----------|

## Risk Summary

{LLM analysis of the above data: top 3 risks with recommended actions}
```

### Step 7: Optional publish

Ask user: "Publish this report as an Issue comment? (provide issue number, or skip)"

If yes:
```bash
gh api repos/{org}/{issueRepo}/issues/{target_number}/comments --method POST \
  --raw-field body="$(cat "$REPORT_FILE")"
```

## Constraints

- Read-only by default — only adds beaver/ labels for health flags
- Report in terminal markdown, not written to file
- Skip `beaver/wontfix` issues for stale/overdue detection
```

**Step 2: Verify frontmatter**

Run: `head -4 plugins/beaver/skills/beaver-report/SKILL.md`
Expected: Valid YAML frontmatter.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-report/SKILL.md
git commit -m "feat(beaver): add beaver-report skill for project health reporting

Covers milestone progress, stale/overdue detection, blocking chains,
sub-task rollup, and LLM risk analysis."
```

---

### Task 7: Create beaver-focus skill

**Files:**
- Create: `plugins/beaver/skills/beaver-focus/SKILL.md`

**Step 1: Write the beaver-focus SKILL.md**

```markdown
---
name: beaver-focus
description: "Show your personal Beaver work status: today's tasks, pending reviews, blockers, and DDL warnings with priority recommendations. Trigger when the user asks about their tasks, what to work on, or personal status."
---

# Beaver Focus

Show the current developer's personal work dashboard: active tasks, pending reviews, blockers, DDL warnings, and LLM-powered priority recommendations.

**References beaver-engine for:** label taxonomy (Section 1), label ops (Section 4), config reading (Section 5).

## Prerequisites

- `gh auth status` must succeed
- A Beaver-configured Project V2 exists

## Workflow

### Step 1: Identify current user

```bash
CURRENT_USER=$(gh api user --jq '.login')
```

### Step 2: Load project config

Read `beaver-config` per engine Section 5.

### Step 3: Fetch my active issues

```bash
gh api "repos/{org}/{issueRepo}/issues?labels=Control-By-Beaver&assignee=$CURRENT_USER&state=open&per_page=100" \
  --jq '.[] | {number, title, labels: [.labels[].name], milestone: {title: .milestone.title, due_on: .milestone.due_on}, updated_at}'
```

Parse labels per engine Section 4. Group by status.

### Step 4: Fetch PRs needing my review

```bash
gh api "search/issues?q=is:pr+is:open+review-requested:$CURRENT_USER" \
  --jq '.items[] | {number, title, repository_url, created_at, user: .user.login}'
```

### Step 5: Compute DDL warnings

For issues with milestones, check if `due_on` is within 48 hours. Flag as ⏳.

### Step 6: Generate dashboard

```
# Beaver Focus: @{username}

**Date:** {today}

## 🔨 In Progress ({count})
| # | Title | Priority | Updated |
|---|-------|----------|---------|

## 📋 Ready to Develop ({count})
| # | Title | Priority |
|---|-------|----------|

## 👀 Awaiting My Review ({count})
| # | Title | Repo | Waiting Since |
|---|-------|------|--------------|

## 🚫 My Blockers ({count})
| # | Title | Blocked Since |
|---|-------|--------------|

## ⏳ DDL Warnings ({count})
| # | Title | Due | Days Left |
|---|-------|-----|-----------|

## 🎯 Today's Top 3 Priorities

{LLM recommendation based on:
 1. p0/blocker and p1/urgent issues first
 2. DDL < 48h issues next
 3. Longest-waiting review requests
 Explain WHY each is prioritized.}
```

## Constraints

- Read-only — no label changes, no status transitions
- Only shows issues assigned to the current `gh` authenticated user
- Dashboard in terminal markdown, not written to file
```

**Step 2: Verify frontmatter**

Run: `head -4 plugins/beaver/skills/beaver-focus/SKILL.md`
Expected: Valid YAML frontmatter.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-focus/SKILL.md
git commit -m "feat(beaver): add beaver-focus skill for personal work dashboard

Shows active tasks, pending reviews, blockers, DDL warnings, and
LLM-powered priority recommendations for the current developer."
```

---

### Task 8: Update plugin.json and clean up old skills

**Files:**
- Modify: `plugins/beaver/.claude-plugin/plugin.json`
- Delete: `plugins/beaver/skills/create-beaver-issue/SKILL.md`

**Step 1: Update plugin.json**

Replace the content of `plugins/beaver/.claude-plugin/plugin.json` with:

```json
{
  "name": "beaver",
  "description": "Beaver project management: GitHub-native issue lifecycle, compliance checks, task audit, and developer dashboards for Projects V2",
  "version": "2.0.0"
}
```

**Step 2: Delete old create-beaver-issue skill**

```bash
rm -r plugins/beaver/skills/create-beaver-issue
```

The old `beaver-pr/SKILL.md` was already overwritten in Task 4.

**Step 3: Verify directory structure**

```bash
find plugins/beaver -type f | sort
```

Expected:
```
plugins/beaver/.claude-plugin/plugin.json
plugins/beaver/commands/create-beaver-project.md
plugins/beaver/skills/beaver-audit/SKILL.md
plugins/beaver/skills/beaver-engine/SKILL.md
plugins/beaver/skills/beaver-focus/SKILL.md
plugins/beaver/skills/beaver-issue/SKILL.md
plugins/beaver/skills/beaver-pr/SKILL.md
plugins/beaver/skills/beaver-report/SKILL.md
```

**Step 4: Commit**

```bash
git add -A plugins/beaver/
git commit -m "feat(beaver): update plugin to v2.0.0, remove old create-beaver-issue

Plugin now has 6 skills (1 internal engine + 5 command-layer) replacing
the previous 2 skills."
```

---

### Task 9: Update CLAUDE.md and verify everything

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the beaver plugin count in CLAUDE.md**

In the "Repository Layout" section, update `**beaver** (3 skills)` to `**beaver** (6 skills)`.

**Step 2: Verify all SKILL.md frontmatter**

```bash
for f in plugins/beaver/skills/*/SKILL.md; do
  echo "=== $f ==="
  head -4 "$f"
  echo ""
done
```

Expected: Each file starts with `---`, has `name:` and `description:`, ends with `---`.

**Step 3: Verify plugin.json is valid JSON**

```bash
python3 -c "import json; json.load(open('plugins/beaver/.claude-plugin/plugin.json'))"
```

Expected: No output (success).

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect beaver plugin v2.0.0 (6 skills)"
```
