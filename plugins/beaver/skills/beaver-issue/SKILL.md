---
name: beaver-issue
description: "Create or claim a Beaver-tracked GitHub Issue with automatic status transitions and guardrail checks. Trigger when the user wants to create a GitHub issue, claim/start a task, or pick up work."
argument-hint: "[issue-number to claim, or omit to create new]"
---

# Beaver Issue

Create a new Beaver-tracked Issue or claim an existing one. Handles Project V2 field setup, label assignment, parent linking, and automatic status transitions with guardrail enforcement.

**References beaver-engine for:** state machine (Section 2), guardrails G001 (Section 3), label ops (Section 4), config reading (Section 5), transition execution (Section 6), QA loop & HARD-GATE (Section 7), Discovery Triad (Section 8), doc quality constraints (Section 9).

## Prerequisites

- `gh auth status` must succeed
- Project scope: `gh auth refresh -s project` if missing
- Project README must contain `beaver-config` YAML block

## Detect Mode

- If an argument is provided (issue number): **Claim mode** (skip §7/§8/§9)
- If no argument: **Create mode**, then run **Step 0** below to choose Feature submode or Bug submode.

### Step 0: Type detection (Create mode only)

Ask the user once: "What kind of issue is this? (feat / bug / refactor / docs / chore)"

- If the answer is `bug`: enter **Bug submode** (see "Bug Submode" section near the bottom of this file).
- Otherwise: enter **Feature submode** (continues with Step 1 below).

---

## Create Mode

### Step 1: Load project config

Check auto memory for `beaver-issue-defaults.md`. If found, present defaults for confirmation. Parse project config per engine Section 5.

### Step 1.5: Discovery Triad

Execute engine Section 8 (Discovery Triad) using the user's draft title + objective as the keyword source. Print the Discovery Brief in the §8.3 format. The user does not need to "approve" the Brief — it is informational input for Step 2.

HARD-GATE: Do NOT proceed to Step 2 until the Brief has been printed.

### Step 2: Collect issue details (engine §7 Q&A loop)

Enter engine Section 7 Q&A loop. The first question MUST be size, because it routes the rest of the loop:

1. **Size**: ask "size/S (small, single change) or size/L (multi-component / needs design)?"

Then route:

- **size/S route — minimal Q&A (3 questions):**
  1. Title (concise)
  2. Objective (一句话, Chinese)
  3. Acceptance criteria (≥ 2 verifiable items per §9.4)
  Defaults are used for: Level (= Task unless user mentions parent), Parent (skip if Level=Task at top of project), Type (= feat unless user mentioned bug in Step 0 — in which case the caller is already in Bug submode), Priority (= p3/normal unless user names urgency).

- **size/L route — full Q&A (4 sections, each with §7.5 approval and §9.3 checklist):**
  1. Level + Parent: Goal / Task / SubTask; for Task/SubTask list project items and let user pick parent.
  2. Title.
  3. Objective + Scope (which subsystems / boundaries).
  4. Acceptance criteria + Stakeholders (who reviews / who is impacted).

In both routes, Type and Priority labels are collected at the END (after the section-by-section loop), as separate single questions. Type defaults to `feat`; Priority is required and asked explicitly.

HARD-GATE per §7.2: until the user approves the section per §7.5, do NOT call any `gh api` POST/PATCH or `gh project` write command.

### Step 3: Preview and §9.4 checklist

Show complete issue details in a structured preview, then present the §9.4 issue-body checklist (Objective is one user-facing sentence / ≥ 2 verifiable acceptance items / no invented file paths). All three rows must be ☑ before continuing. Then ask `Approved? (y/revise)` per §7.5.

### Step 4: Create the Issue

Write the issue body (from template in "Issue Body Template" section) to a temp file, then pass it via `-F body=@`:

```bash
BODY_FILE=$(mktemp)
cat > "$BODY_FILE" << 'BEAVEREOF'
{rendered_body}
BEAVEREOF

gh api repos/{org}/{issueRepo}/issues --method POST \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  -f title="{title}" -F body=@"$BODY_FILE" \
  -f type="{level}" \
  -f "labels[]=Control-By-Beaver" \
  -f "labels[]=type/{type}" \
  -f "labels[]=size/{size}" \
  -f "labels[]={priority}" \
  -f "labels[]=status/triage"
```
Add `-f milestone={number}` if selected. If issue type API fails, retry without `-f type`.
After the API call, clean up: `rm "$BODY_FILE"`

> **NOTE:** Only `-F`/`--field` supports `@file` syntax for reading file contents. `-f`/`--raw-field` passes values as literal strings (so `body=@path` would send the literal text `@path`).

### Step 5: Add to Project V2 and set fields

```bash
gh project item-add {projectNumber} --owner {org} --url {issue_url} --format json
```
Set Level, Status (Not Started), Progress (0) fields via `gh project item-edit`.

### Step 6: Link to parent (Task/SubTask only)

```bash
CHILD_ID=$(gh api repos/{org}/{issueRepo}/issues/{number} --jq '.id')
gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues \
  --method POST -H "X-GitHub-Api-Version: 2026-03-10" \
  -F sub_issue_id=$CHILD_ID
```

> **NOTE:** The sub-issues API requires `sub_issue_id` to be an integer. Use `-F` (uppercase, `--field`) which infers numeric types automatically. `-f` (lowercase, `--raw-field`) always sends strings, causing a 422 error.

### Step 7: Auto-transition from triage

Per engine state machine:
- `size/S`: transition `status/triage` → `status/in-progress`
- `size/L`: transition `status/triage` → `status/design-pending`

Execute transition per engine Section 6 (validates G001).

### Step 8: Report and save defaults

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

{objective_in_chinese}

## 验收标准

{acceptance_criteria_in_chinese}
```

### Task / SubTask
```
## 目标

{objective_in_chinese}

## 验收标准

{acceptance_criteria_in_chinese}

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
- Never modify existing issues (except label transitions and assignee updates during claim)
