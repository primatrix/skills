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
gh api graphql -f query='
  query($owner: String!, $number: Int!) {
    organization(login: $owner) {
      projectV2(number: $number) {
        items(first: 100) {
          nodes {
            content {
              ... on Issue {
                number
                title
                state
                labels(first: 30) { nodes { name } }
                assignees(first: 10) { nodes { login } }
              }
            }
            fieldValueByName(name: "Iteration") {
              ... on ProjectV2ItemFieldIterationValue {
                title
                startDate
                duration
              }
            }
          }
        }
      }
    }
  }' -f owner=primatrix -F number=14 \
  --jq '.data.organization.projectV2.items.nodes
        | map(select(.content != null and .content.state == "OPEN"))
        | map(select(.content.assignees.nodes | map(.login) | index("'"$CURRENT_USER"'")))
        | map(select(.content.labels.nodes | map(.name) | index("Control-By-Beaver")))
        | map({number: .content.number, title: .content.title,
               labels: [.content.labels.nodes[].name],
               iteration: (if .fieldValueByName then
                 {title: .fieldValueByName.title,
                  startDate: .fieldValueByName.startDate,
                  duration: .fieldValueByName.duration} else null end)})'
```

Parse labels per engine Section 4. Group by status.

Within each group, sort p/0-blocker bugs to the TOP of results. For any p/0-blocker issue open longer than 24 hours, display a ⚠️ warning alongside the issue row.

### Step 4: Fetch PRs needing my review

```bash
gh api "search/issues?q=is:pr+is:open+review-requested:$CURRENT_USER" \
  --jq '.items[] | {number, title, repository_url, created_at, user: .user.login}'
```

### Step 5: Compute DDL warnings

For issues with an Iteration assignment, compute `iteration_end = startDate + duration days`. If `iteration_end - now <= 48h`, flag with a warning indicator.

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
| # | Title | Iteration End | Days Left |
|---|-------|---------------|-----------|
(⚠️ shown if Iteration ends within 48h)

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
