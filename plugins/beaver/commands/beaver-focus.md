---
allowed-tools: Bash(gh api:*), Bash(gh search:*), Bash(gh auth:*), Bash(date:*)
description: "Show your personal Beaver work status. Trigger when the user asks about their tasks, what to work on, or personal status."
---

# /beaver-focus — 个人看板

Utility command, usable at any time.

Show the current developer's personal work dashboard, aggregating six sections from Project V2 #14 fields plus a single LLM-generated priorities block. **This command is strictly read-only against all remote state** — it never mutates Issues, comments, labels, PRs, or Project fields.

**References beaver-engine for:** Status taxonomy (§2 Status field), Priority taxonomy (§2 Priority field), config reading (§5).

## Prerequisites

- `gh auth status` must succeed
- A Beaver-configured Project V2 #14 exists (run `/beaver-setup` if not)

## Read-Only Constraint (HARD)

This command source contains zero remote-write tokens (per Issue #122 AC1/AC5; the verbatim grep regex lives in `scripts/tests/test_beaver_focus_fields.sh`). All data is fetched via `gh api graphql -f query=...` (queries) and default-GET `gh api ...` REST endpoints only. The dashboard is rendered to stdout as terminal markdown — nothing is written back to GitHub.

## Workflow

### Step 1: Identify current user

```bash
CURRENT_USER=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh whoami)
```

### Step 2: Fetch my open Beaver issues (Project V2 fields)

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh fetch-my-issues "$CURRENT_USER"
```

Each record carries: `number`, `title`, `repo`, `url`, `labels`, `assignees`, `status` (Project V2 Status field), `priority` (Project V2 Priority field), `type` (native Issue Type), `iteration` (Iteration field title + startDate + duration, or null), `createdAt`, `updatedAt`, `lastCommentAt`, `lastActivityAt` (= max(`updatedAt`, `lastCommentAt`)).

### Step 3: Fetch PRs needing my review

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh fetch-review-prs "$CURRENT_USER"
```

REST GET — read-only.

### Step 4: Compute DDL warnings

For every record from Step 2 whose `iteration` is non-null, compute `iteration_end = startDate + duration days`. If `iteration_end - now <= 48h`, flag the row with a ⚠️ warning indicator and surface in the DDL Warnings section.

### Step 5: Group My Open Issues by Status

Group the Step 2 records secondarily by their `status` field (Project V2 Status, single-select). The required group order is:

1. **In Progress**
2. **Blocked**
3. **Design Pending**
4. **Ready to Develop**
5. **Ready to Claim**
6. **Triage**

Within each group, sort by **last commit / comment time** (`lastActivityAt`, descending) — most recently active first.

### Step 6: P0 Bugs (special section)

Across all open issues from Step 2 whose `labels` contain `p/0-blocker`, build a separate **P0 Bugs** section. Sort those rows by **issue 持续时间** (open duration = `now - createdAt`), longest-open first. For any p/0-blocker row whose open duration exceeds **24h**, append a ⚠️警示 marker to its row.

### Step 7: Fetch unclaimed candidates

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh fetch-ready-to-claim
```

Returns open Beaver issues whose Status field equals `Ready to Claim` and which have zero assignees. Used to populate the dashboard's claimable-work section.

### Step 8: Render the 6-section dashboard

```markdown
# Beaver Focus: @{username}

**Date:** {today}

## My Open Issues ({count})

### In Progress ({n_in_progress})
| # | Title | Priority | Type | Last Activity |
|---|-------|----------|------|---------------|

### Blocked ({n_blocked})
| # | Title | Priority | Type | Last Activity |
|---|-------|----------|------|---------------|

### Design Pending ({n_design_pending})
| # | Title | Priority | Type | Last Activity |
|---|-------|----------|------|---------------|

### Ready to Develop ({n_ready_to_develop})
| # | Title | Priority | Type | Last Activity |
|---|-------|----------|------|---------------|

### Ready to Claim ({n_ready_to_claim})
| # | Title | Priority | Type | Last Activity |
|---|-------|----------|------|---------------|

### Triage ({n_triage})
| # | Title | Priority | Type | Last Activity |
|---|-------|----------|------|---------------|

## P0 Bugs ({count})
| # | Title | Open Since | Age | Warning |
|---|-------|-----------|-----|---------|
(⚠️ shown if age > 24h, sorted by age desc)

## Awaiting Review ({count})
| # | Title | Repo | Waiting Since |
|---|-------|------|--------------|

## Ready to Claim ({count})
| # | Title | Priority | Repo |
|---|-------|----------|------|
(unassigned Beaver issues with Status=Ready to Claim)

## DDL Warnings ({count})
| # | Title | Iteration End | Hours Left |
|---|-------|---------------|-----------|
(⚠️ shown if Iteration ends within 48h)

## Today's Top 3 Priorities

{LLM-generated block — see Step 9.}
```

### Step 9: Single LLM call for `Today's Top 3 Priorities`

Issue **exactly one** LLM completion (one-shot, once per `/beaver-focus` invocation — not per row). Prompt the model with the aggregated dashboard data from Steps 2/3/4/6/7 and request **3 actionable next steps** rather than an issue list. Each priority must be a concrete, actionable next step the developer can perform within the next work block (e.g. "Push the branch for #122 and request review from X" rather than "Work on #122").

Inputs the model considers, in this order:
 1. p/0-blocker issues (from Step 6), especially any with the ⚠️警示 24h marker.
 2. p1/urgent priority issues from My Open Issues.
 3. DDL < 48h issues from the DDL Warnings section.
 4. Longest-waiting review requests from Awaiting Review.

Output format:

```markdown
1. [Actionable next step] — Why: <one-line reason referencing the source row>
2. [Actionable next step] — Why: <one-line reason>
3. [Actionable next step] — Why: <one-line reason>
```

Each item must be the next concrete physical action (一次性、可执行、下一步), not a restatement of the issue title.

## Constraints

- Strictly read-only: this command never mutates remote state (Issue #122 AC1/AC5; see the verbatim grep regex in `scripts/tests/test_beaver_focus_fields.sh`).
- Only shows issues assigned to the current `gh` authenticated user (My Open Issues / DDL Warnings sections); the unclaimed-Status and Awaiting Review sections are scoped per Step 7 / Step 3 respectively.
- Status grouping uses the Project V2 Status field (single-select), not legacy `status/*` labels.
- Priority highlighting (P0) reads the `p/0-blocker` label per beaver-engine §1.
- Within each Status group, rows are sorted by `lastActivityAt` (max of `updatedAt` and the latest comment `createdAt`) descending — i.e., most recent commit/comment first.
- The P0 Bugs section is sorted by issue 持续时间 (open age) descending; rows older than 24h are flagged ⚠️.
- Step 9 issues exactly **one** LLM call per command invocation; the priorities are framed as actionable next steps, not as a bullet list of issue titles.
- Dashboard renders to terminal markdown; no file is written.
