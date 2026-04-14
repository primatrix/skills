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
4. **Description**: structured as Objective and Acceptance Criteria
5. **Type label** (`type/`): feat / bug / refactor / docs / chore
6. **Priority label** (`p/`): choose one: `p0/blocker` / `p1/urgent` / `p2/high` / `p3/normal`

### Step 3: Auto-classify size

Analyze the description and suggest `size/S` or `size/L`:
- If description mentions multiple components, API changes + frontend + tests → suggest `size/L`
- If description is focused on a single change → suggest `size/S`
Present suggestion with reasoning. Wait for user confirmation.

### Step 4: Preview and confirm

Show complete issue details in a structured preview. Wait for explicit approval.

### Step 5: Create the Issue

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

### Step 6: Add to Project V2 and set fields

```bash
gh project item-add {projectNumber} --owner {org} --url {issue_url} --format json
```
Set Level, Status (Not Started), Progress (0) fields via `gh project item-edit`.

### Step 7: Link to parent (Task/SubTask only)

```bash
CHILD_ID=$(gh api repos/{org}/{issueRepo}/issues/{number} --jq '.id')
gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues \
  --method POST -H "X-GitHub-Api-Version: 2026-03-10" \
  -F sub_issue_id=$CHILD_ID
```

> **NOTE:** The sub-issues API requires `sub_issue_id` to be an integer. Use `-F` (uppercase, `--field`) which infers numeric types automatically. `-f` (lowercase, `--raw-field`) always sends strings, causing a 422 error.

### Step 8: Auto-transition from triage

Per engine state machine:
- `size/S`: transition `status/triage` → `status/in-progress`
- `size/L`: transition `status/triage` → `status/design-pending`

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
## Objective

{objective}

## Acceptance Criteria

{acceptance_criteria}
```

### Task / SubTask
```
## Objective

{objective}

## Acceptance Criteria

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
- Never modify existing issues (except label transitions and assignee updates during claim)
