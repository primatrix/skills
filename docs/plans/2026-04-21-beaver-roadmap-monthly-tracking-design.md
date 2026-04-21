# Design: beaver-roadmap → 月度 Roadmap Tracking Issue

**Date:** 2026-04-21
**Scope:** Rewrite `plugins/beaver/commands/beaver-roadmap.md`
**Status:** Approved

## Background

Current `beaver-roadmap.md` generates a project health report (milestone progress, stale/overdue detection, sub-task rollup, risk analysis) for the configured Beaver project. Health reporting is being retired. The command will be repurposed to manage monthly roadmap tracking issues.

## Goal

Rewrite `/beaver-roadmap` so that, given a repo, it:

1. Creates a monthly tracking issue in `primatrix/projects` titled `[Roadmap] <repo> YYYY-MM`.
2. Carries forward unfinished tasks from the prior month's roadmap as sub-issues of the new roadmap.

## Decisions

| Topic | Decision |
|---|---|
| Default issueRepo | Hardcoded `primatrix/projects`. Skip `beaver-config` lookup. |
| Repo selection | Required CLI argument: `/beaver-roadmap <repo> [YYYY-MM]`. |
| Roadmap ↔ milestone | Coexist. Milestone provides `due_on`; roadmap issue provides sub-issue tree. |
| Roadmap identity | Title `[Roadmap] <repo> YYYY-MM` + labels `roadmap`, `roadmap/<repo>`, `roadmap/YYYY-MM`. |
| "Unfinished" definition | Open (un-closed) task issues attached as sub-issues of the prior roadmap. Their own sub-tasks travel with them. |
| Migration granularity | Re-parent the task only. Sub-tasks stay attached to their task (not flattened). |
| Duplicate run | Detect existing roadmap for current month → ask user whether to re-run migration into the existing one. |
| Health report logic | Removed entirely. No stub. |

## Command Signature

```
/beaver-roadmap <repo> [YYYY-MM]
```

- `<repo>` required. Plain repo name (e.g. `nirvana`); owner is fixed at `primatrix`. Roadmap issues are created in `primatrix/projects`.
- `[YYYY-MM]` optional, defaults to the current local year-month.

## Data Model

**Roadmap issue (created in `primatrix/projects`):**

- Title: `[Roadmap] <repo> YYYY-MM`
- Labels: `roadmap`, `roadmap/<repo>`, `roadmap/YYYY-MM`
- Body:

  ```markdown
  ## 月度 Roadmap — <repo> YYYY-MM

  本 issue 作为 <repo> YYYY-MM 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

  ## 来源
  - 上月 roadmap: #<N>（迁移 <K> 个未完成 task）

  <!-- beaver-roadmap
  repo: <repo>
  month: YYYY-MM
  carried-from: #<N>
  -->
  ```

**Prior-roadmap lookup** (label-based search, exact match):

```bash
gh api -X GET search/issues \
  -f q='repo:primatrix/projects is:issue label:"roadmap/<repo>" label:"roadmap/<prevYYYY-MM>"' \
  --jq '.items[0] | {number, state}'
```

**Open sub-issues of the prior roadmap:**

```bash
gh api repos/primatrix/projects/issues/<N>/sub_issues \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  --jq '.[] | select(.state=="open") | {number, title}'
```

## Workflow

```
Step 1  Parse args
        - repo (required), month (default current), prevMonth = month - 1

Step 2  Locate prior roadmap
        - Search by labels: roadmap/<repo> + roadmap/<prevYYYY-MM>
        - 0 hits → carried = []; jump to Step 4
        - 1 hit  → record prev_number; continue Step 3
        - >1 hit → error, list matches, ask user to clean up

Step 3  Collect open sub-issues from prior roadmap
        - Fetch sub_issues, filter state == "open"
        - Build carried list (number + title) for Step 5 preview

Step 4  Check whether current month's roadmap already exists
        - Search by labels: roadmap/<repo> + roadmap/<YYYY-MM>
        - 0 hits → continue Step 5
        - 1 hit  → ask user:
            "本月 roadmap 已存在 #<M>。是否将 carried 列表重新挂到 #<M> 下？(y/skip)"
            y    → set target = #<M>, jump to Step 7
            skip → exit
        - >1 hit → error

Step 5  Preview + approval gate (HARD-GATE per engine §7.2)
        - Print proposed title, body, labels
        - Print carried list
        - Wait for explicit approval per engine §7.5 (y/yes/ok/lgtm/继续/通过)

Step 6  Create new roadmap issue
        - POST repos/primatrix/projects/issues
        - Add labels (roadmap, roadmap/<repo>, roadmap/<YYYY-MM>)
        - Record new_number

Step 7  Migrate open tasks as sub-issues
        - For each carried task:
            POST repos/primatrix/projects/issues/<target>/sub_issues
              -F sub_issue_id=<task_id>
        - GitHub auto-detaches from prior parent (one-parent invariant)
        - Per-task failures recorded; do not abort the batch

Step 8  Report
        - Print new roadmap URL
        - Print migration summary: success X / failure Y (with reasons)
```

## Error Handling & Boundaries

| Scenario | Behavior |
|---|---|
| `<repo>` missing | Exit with usage hint. |
| `roadmap` / `roadmap/<repo>` / `roadmap/YYYY-MM` label absent | Auto `gh label create` (idempotent). |
| Prior roadmap not found | No error. carried = []. Create empty roadmap. |
| Prior roadmap matched multiple | Error, list candidate issue numbers, request manual cleanup. |
| Current roadmap already exists | Step 4 prompt resolves. |
| Prior roadmap closed but has open sub-issues | Still migrate. Completion is judged at sub-issue level, not parent. |
| sub_issues API failure (auth/missing) | Record reason, continue with next; summarize at Step 8. |
| `gh auth status` fails | Pre-flight check; abort. |

## Document Structure

New `beaver-roadmap.md` sections:

```
frontmatter (allowed-tools, description, argument-hint)
# /beaver-roadmap — 月度 Roadmap 创建与迁移
Phase 2 of the Beaver development lifecycle.
## Prerequisites
## Workflow (Step 1 — Step 8)
## Issue Body Template
## Constraints
```

**Removed in full (no stub):**

- "Fetch active milestone"
- "Compute health indicators" (stale, overdue, bug stats, upstream blocked, missing context)
- "Compute sub-task rollup"
- "Generate report" markdown template
- "Optional publish" step
- All G010 (stale/overdue flag-label) references inside this command

**Engine impact:** None. `beaver-engine/SKILL.md` G010 stays — it remains valid guidance for any future command that applies stale/overdue flags. This rewrite simply stops being a caller.

## Out of Scope

- Modifying `beaver-engine/SKILL.md`.
- Restoring health reporting under another command (e.g. `/beaver-report`).
- Auto-running on a schedule.
- Cross-repo aggregation in a single roadmap issue.

<!-- provenance
- "Default issueRepo hardcoded primatrix/projects" ← QA round 2
- "Roadmap ↔ milestone coexist" ← QA round 1
- "Title + label identity scheme" ← QA round 4
- "Unfinished = open task + its sub-tasks" ← QA round 5
- "Re-parent task only, sub-tasks stay" ← QA round 6
- "Duplicate run prompts user" ← QA round 7
- "Health report removed entirely" ← QA round 8
- "Approach A (rewrite, no stub)" ← approach selection
- "sub_issues API + X-GitHub-Api-Version: 2026-03-10" ← plugins/beaver/commands/beaver-roadmap.md:74-77 (current file)
- "GitHub one-parent invariant for sub_issues" ← plugins/beaver/commands/beaver-create.md:60-63 (existing parent linkage usage)
- "Engine §7.2 HARD-GATE / §7.5 approval grammar" ← plugins/beaver/skills/beaver-engine/SKILL.md:194-242
-->
