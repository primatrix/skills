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
  --jq '.[] | {number, title, labels: [.labels[].name], milestone: {title: (.milestone.title // null), due_on: (.milestone.due_on // null)}, updated_at}'
```

Parse labels per engine Section 4. Group by status.

### Step 4: Fetch PRs needing my review

```bash
gh api "search/issues?q=is:pr+is:open+review-requested:$CURRENT_USER" \
  --jq '.items[] | {number, title, repository_url, created_at, user: .user.login}'
```

### Step 5: Compute DDL warnings

For issues with milestones, check if `due_on` is within 48 hours. Flag accordingly.

### Step 6: Generate dashboard

```markdown
# Beaver Focus: @{username}

**Date:** {today}

## In Progress ({count})
| # | Title | Priority | Updated |
|---|-------|----------|---------|

## Ready to Develop ({count})
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

## Today's Top 3 Priorities

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
