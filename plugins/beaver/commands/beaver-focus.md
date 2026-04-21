---
allowed-tools: Bash(gh api:*), Bash(gh search:*)
description: "Show your personal Beaver work status. Trigger when the user asks about their tasks, what to work on, or personal status."
---

# /beaver-focus — 个人看板

Utility command, usable at any time.

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
  --jq '.[] | {number, title, labels: [.labels[].name], milestone: {title: (.milestone.title // null), due_on: (.milestone.due_on // null)}, updated_at}'
```

Parse labels per engine Section 4. Group by status.

Within each group, sort p/0-blocker bugs to the TOP of results. For any p/0-blocker issue open longer than 24 hours, display a ⚠️ warning alongside the issue row.

### Step 4: Fetch PRs needing my review

```bash
gh api "search/issues?q=is:pr+is:open+review-requested:$CURRENT_USER" \
  --jq '.items[] | {number, title, repository_url, created_at, user: .user.login}'
```

### Step 5: Compute DDL warnings

For issues with milestones, check if `due_on` is within 48 hours. Flag accordingly with a warning indicator.

### Step 6: Generate dashboard

```markdown
# Beaver Focus: @{username}

**Date:** {today}

## P/0 Blockers ({count})
| # | Title | Age | Warning |
|---|-------|-----|---------|
(⚠️ shown if open > 24h)

## In Progress ({count})
| # | Title | Priority | Updated |
|---|-------|----------|---------|

## Bugs ({count})
| # | Title | Priority | Updated |
|---|-------|----------|---------|

## Ready to Develop ({count})
| # | Title | Priority |
|---|-------|----------|

## Ready to Claim ({count})
| # | Title | Priority |
|---|-------|----------|

## Awaiting My Review ({count})
| # | Title | Repo | Waiting Since |
|---|-------|------|--------------|

## My Blockers ({count})
| # | Title | Blocked Since |
|---|-------|--------------|

## DDL Warnings ({count})
| # | Title | Due | Days Left |
|---|-------|-----|-----------|
(⚠️ shown if milestone due within 48h)

## Today's Top 3 Priorities

{LLM recommendation based on:
 1. p/0-blocker issues first (especially if open > 24h)
 2. p1/urgent issues next
 3. DDL < 48h issues next
 4. Longest-waiting review requests
 Explain WHY each is prioritized.}
```

## Constraints

Strictly read-only — no label changes or status transitions.

- Only shows issues assigned to the current `gh` authenticated user
- Dashboard in terminal markdown, not written to file
