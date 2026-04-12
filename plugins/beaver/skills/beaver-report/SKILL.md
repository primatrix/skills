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

### Overdue ({count})
| # | Title | Assignee | Days Overdue |
|---|-------|----------|-------------|

### Stale ({count})
| # | Title | Status | Days Since Update |
|---|-------|--------|------------------|

### Blocked ({count})
| # | Title | Blocked By |
|---|-------|-----------|

### Missing Context ({count})
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
