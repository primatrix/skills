---
allowed-tools: Bash(gh api:*), Bash(gh label:*), Bash(gh auth:*), Bash(date:*)
description: "Create a monthly Iteration tracker issue in primatrix/projects, carry forward unfinished sub-issues from the prior month, optionally pull from the triage backlog, and sync the Iteration field on all sub-issues. Trigger when the user wants to start a new month's tracker or roll over open tasks."
argument-hint: "<repo> [YYYY-MM]"
---

# /beaver-tracker — 月度 Tracker 创建与迁移

Phase 2 of the Beaver development lifecycle.

Create a monthly tracker issue in `primatrix/projects` for a given repo. Carry forward unfinished tasks from the prior month's tracker as sub-issues of the new one.

**References beaver-engine for:** HARD-GATE rule (§7.2), approval grammar (§7.5).

## Prerequisites

- `gh auth status` must succeed
- `gh` must support the Sub-Issues API (header `X-GitHub-Api-Version: 2026-03-10`)
- The repo `primatrix/projects` must exist and be writable by the current user

## Workflow

### Step 1: Parse arguments

- `<repo>` — required, plain repo name (e.g. `nirvana`). Owner is fixed at `primatrix`.
- `[YYYY-MM]` — optional, defaults to the current local year-month.
- Compute `prevYYYY-MM` = `YYYY-MM` minus 1 month (handle Jan → previous year Dec).

If `<repo>` is missing, exit with:
```
Usage: /beaver-tracker <repo> [YYYY-MM]
```

### Step 2: Ensure required labels exist

Idempotent label bootstrap (no-op if already present):

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh ensure-label projects tracker BFD4F2 "Monthly Iteration tracker issue"
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh ensure-label projects "tracker/<repo>" BFD4F2 "Tracker for <repo>"
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh ensure-label projects "tracker/<YYYY-MM>" BFD4F2 "Tracker for <YYYY-MM>"
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh ensure-label projects "tracker/<prevYYYY-MM>" BFD4F2 "Tracker for <prevYYYY-MM>"
```

### Step 3: Locate prior month's tracker

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh find-tracker <repo> <prevYYYY-MM>
```

- `count == 0` → `prev_number = null`, `carried = []`. Skip to Step 5.
- `count == 1` → record `prev_number = .items[0].number`. Continue to Step 4.
- `count > 1` → ERROR. Print all matches and abort with:
  ```
  Multiple prior trackers matched for <repo> <prevYYYY-MM>: #<n1>, #<n2>, ...
  Please consolidate or remove duplicates, then re-run.
  ```

### Step 4: Collect open sub-issues from the prior tracker

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh list-carried <prev_number>
```

Store as `carried` (array of `{number, title}`).

> **Definition of unfinished:** any sub-issue of the prior tracker whose `state == "open"`. Sub-tasks beneath those tasks are NOT separately re-parented; they remain attached to their parent task and travel with it implicitly.

### Step 5: Check whether current month's tracker already exists

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh find-tracker <repo> <YYYY-MM>
```

- `count == 0` → continue to Step 6 (create).
- `count == 1` → set `existing_number = .items[0].number`, ask the user:
  ```
  本月 tracker 已存在 #<existing_number>。是否将 carried 列表 (<K> 个 task) 重新挂到 #<existing_number> 下？(y/skip)
  ```
  - User answer `y` → set `target = existing_number`, jump to Step 8 (skip create).
  - Anything else → exit without changes.
- `count > 1` → ERROR. Print matches and abort.

### Step 6: Preview + approval gate (HARD-GATE per engine §7.2)

Print the full preview:

```
Will create tracker issue:
  Repo:   primatrix/projects
  Title:  [Iteration] <repo> <YYYY-MM>
  Labels: tracker, tracker/<repo>, tracker/<YYYY-MM>
  Body:
    ---
    ## 月度 Tracker — <repo> <YYYY-MM>

    本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

    ## 来源
    - 上月 tracker: #<prev_number 或 "无">（迁移 <K> 个未完成 task）

    <!-- beaver-tracker
    repo: <repo>
    month: <YYYY-MM>
    carried-from: #<prev_number 或 "none">
    -->
    ---

Will then re-parent <K> open tasks as sub-issues of the new tracker:
  - #<n1> <title1>
  - #<n2> <title2>
  ...

Will then prompt to pull backlog issues from triage queue (Step 8.5).
After tracker is populated, will set Iteration field on all sub-issues
to <YYYY-MM> entry (Step 8.6).

Approved? (y/revise)
```

Wait for explicit approval per engine §7.5 (only `y`/`yes`/`ok`/`approve`/`approved`/`lgtm`/`继续`/`通过` count). Anything else → revise.

### Step 7: Create the tracker issue

Render the body template (see §Issue Body Template) with placeholders substituted, write it to a temp file via the `Write` tool (e.g. `/tmp/beaver-tracker-body.md`), then:

```bash
new_number=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh create <repo> "[Iteration] <repo> <YYYY-MM>" /tmp/beaver-tracker-body.md)

bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh add-labels <repo> $new_number \
  tracker "tracker/<repo>" "tracker/<YYYY-MM>"
```

Set `target = $new_number`.

### Step 8: Re-parent open tasks as sub-issues

For each `task` in `carried`:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh attach-sub <target> <task.id>
```

> **Note:** GitHub's Sub-Issues API enforces one parent per issue. POSTing to the new parent automatically detaches from the prior parent. Use the issue **node ID / numeric DB id** (`.id`, not `.number`) per the API contract.

To resolve `task.id` from `task.number`:
```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh resolve-issue-id projects <task.number>
```

On per-task failure: record reason, do NOT abort the batch. Continue with the next task.

### Step 8.5: Pull backlog from triage queue (interactive)

Query candidates: `status/triage` issues in `primatrix/projects` not yet assigned to any Iteration.

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh fetch-triage-backlog
```

Print as numbered list:

```
以下 primatrix/projects 中 triage 队列尚未分配 Iteration 的 issue：
  1. #<n1> <title1>
  2. #<n2> <title2>
  ...

请输入要纳入本月 tracker 的编号（逗号分隔，如 "1,3,5"），或输入 "skip" 跳过：
```

If the candidate list is empty after filtering, print `No triage candidates without Iteration found.` and skip directly to Step 8.6.

If user inputs `skip` (case-insensitive) or empty → no-op for this step, continue to Step 8.6.

Otherwise, parse the comma-separated indices, ignoring whitespace. If any index is out of range or non-numeric, print the offending input and re-prompt (do not silently drop). Accept literal `skip` (case-insensitive) to skip backlog selection entirely.

For each selected issue, resolve `.number` → numeric DB `.id` via `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh resolve-issue-id projects <number>`, then attach via `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh attach-sub <tracker_number> <id>` (same pattern as Step 8). After attachment, Step 8.6 will set the Iteration field.

Per-issue failures: collect, do NOT abort batch; surface in Step 9.

### Step 8.6: Sync Iteration field for all sub-issues

Resolve current month's Iteration entry id:

```bash
eval $(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh resolve-iteration <YYYY-MM>)
PROJECT_ID=$project_id
ITERATION_FIELD_ID=$field_id
ITERATION_ID=$iteration_id
```

If `ITERATION_ID` is `null` or empty, abort Step 8.6 with the following message and continue to Step 9 (do NOT abort the whole command — Steps 1-8 already succeeded):

```
Iteration entry for <YYYY-MM> not found on Project #14.
Run /beaver-setup to extend iterations into <YYYY-MM>.
Sub-issue iteration sync skipped — fix this and re-run /beaver-tracker if needed.
```

For every sub-issue currently attached to the tracker (fetch fresh via `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh list-tracker-subs <tracker_number>`), resolve its ProjectV2Item id and set the Iteration field:

```bash
# Resolve item id for an issue (assuming the issue is already on the project):
ITEM_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh resolve-item-id projects <issue_number>)

# Set Iteration field
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh set-iteration "$PROJECT_ID" "$ITEM_ID" "$ITERATION_FIELD_ID" "$ITERATION_ID"
```

> **Note (add-if-missing, then update):** A sub-issue parented via the Sub-Issues API is NOT auto-added to Project v2 #14. If `ITEM_ID` resolves to empty/null for a given sub-issue, first add it to the project, then retry the field update:
>
> ```bash
> ITEM_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh add-to-project "$PROJECT_ID" <issue_number>)
> ```
>
> Then re-run the `set-iteration` invocation above with the new `ITEM_ID`. If the add itself fails, log the failure (per the existing "Per-issue failures: collect, do NOT abort" behavior below) and continue.

Per-issue failures: collect, do NOT abort batch; surface in Step 9.

### Step 9: Report

Print:

```
Tracker created: <new tracker URL>   (or "Reused existing #<existing_number>")
Migration: <success_count> succeeded, <failure_count> failed
Backlog pulled: <N> succeeded, <M> failed
Iteration sync: <X> succeeded, <Y> failed
  Failed tasks (if any):
    - #<n> <title> — <error reason>
```

## Issue Body Template

```markdown
## 月度 Tracker — <repo> <YYYY-MM>

本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

## 来源
- 上月 tracker: #<N>（迁移 <K> 个未完成 task）

<!-- beaver-tracker
repo: <repo>
month: <YYYY-MM>
carried-from: #<N>
-->
```

## Constraints

- IssueRepo is hardcoded to `primatrix/projects`. Do NOT read `beaver-config`.
- Step 6 is a HARD-GATE per engine §7.2; Steps 7/8 must not run before explicit §7.5 approval.
- Sub-issue re-parent uses the `sub_issues` API with `X-GitHub-Api-Version: 2026-03-10`. The payload field is `sub_issue_id` and takes the issue's numeric DB id (`.id`), not its repo-local `.number`.
- Per-task migration failures must be collected and reported in Step 9, not aborted.
- "Unfinished" = any sub-issue of the prior tracker with `state == "open"`. Sub-tasks travel with their task and are not separately re-parented.
- Step 8.5 backlog selection is interactive; only runs after HARD-GATE approval; user can skip and run migration only.
- Step 8.6 sets the Iteration field on every sub-issue under the tracker, mapping to the entry whose title starts with <YYYY-MM>.
- Health reporting (stale/overdue/bug-stats/upstream-blocked/missing-context/sub-task rollup) is intentionally NOT in this command anymore.
