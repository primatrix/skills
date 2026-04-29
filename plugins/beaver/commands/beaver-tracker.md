---
allowed-tools: Bash(gh api:*), Bash(gh label:*), Bash(gh auth:*), Bash(date:*), Bash(mktemp:*)
description: "Create a monthly Iteration tracker issue in primatrix/projects, carry forward unfinished sub-issues from the prior month, optionally pull from the triage backlog, and sync the Iteration field on all sub-issues. Trigger when the user wants to start a new month's tracker or roll over open tasks."
argument-hint: "<repo> [YYYY-MM]"
---

# /beaver-tracker — 月度 Tracker 创建与差集同步

Phase 2 of the Beaver development lifecycle.

Create or reuse a monthly tracker issue in `primatrix/projects` for a given repo. Carry forward unfinished tasks from the prior month, pull additional Tasks/Bugs from the triage backlog, and synchronize the tracker's sub-issue set with the Project V2 #14 projection `{Iteration=<YYYY-MM> ∧ repo归属=<repo> ∧ Type ∈ {Task, Bug}}` (per RFC-0013 §2 #2).

**References beaver-engine for:** HARD-GATE rule (§7.2), approval grammar (§7.5), field operations (§4 — all Project V2 field writes go through `beaver-lib.sh`).

## Prerequisites

- `gh auth status` must succeed
- `gh` must support the Sub-Issues API (header `X-GitHub-Api-Version: 2026-03-10`)
- `primatrix/projects` must exist and be writable by the current user
- Project V2 #14 has Iteration / Status / native Issue Type fields populated (run `/beaver-setup` first if any are missing)

## Workflow

### Step 1: Parse arguments

- `<repo>` — required, plain repo name (e.g. `nirvana`). Owner is fixed at `primatrix`.
- `[YYYY-MM]` — optional, defaults to the current local year-month.
- Compute `prevYYYY-MM` = `YYYY-MM` minus 1 month (handle Jan → previous year Dec).

If `<repo>` is missing, exit with:
```
Usage: /beaver-tracker <repo> [YYYY-MM]
```

The command also validates that an Iteration entry whose title starts with `<YYYY-MM>` exists on Project #14. If absent, abort with:
```
Iteration entry for <YYYY-MM> not found on Project #14.
Run /beaver-setup to extend iterations into <YYYY-MM>, then re-run.
```

### Step 2: Ensure required labels exist

The three repository-level tracker labels are Beaver metadata and remain in the labels API (per beaver-engine §1, `tracker/*` is exempt from the legacy-taxonomy retirement):

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

### Step 4: Collect carry-over candidates from the prior tracker (全选 / 全拒 / 逐项)

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh list-carried <prev_number>
```

Display the full carry-over candidate list to the user with three interaction modes:
- 全选 (`all`) — accept every open sub-issue from the prior tracker.
- 全拒 (`none` / `skip`) — accept none.
- 逐项 (comma-separated indices, e.g. `1,3,5`) — per-item selection.

Re-prompt on out-of-range indices. Store the user's selection as `carried`.

> **Definition of unfinished:** any sub-issue of the prior tracker whose `state == "open"`. Sub-tasks beneath those tasks are NOT separately re-parented; they remain attached to their parent task and travel with it implicitly.

### Step 5: Check whether current month's tracker already exists

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh find-tracker <repo> <YYYY-MM>
```

- `count == 0` → continue to Step 6 (create).
- `count == 1` → set `existing_number = .items[0].number`, ask the user:
  ```
  本月 tracker 已存在 #<existing_number>。是否复用并继续同步？(y/skip)
  ```
  - User answer `y` → set `target = existing_number`, jump to Step 8.
  - Anything else → exit without changes.
- `count > 1` → ERROR. Print matches and abort.

### Step 6: Preview + approval gate (HARD-GATE per engine §7.2)

Print the full preview, including the carry-over selection from Step 4 and the planned Step 8 / 8.5 / 8.6 / 8.7 actions:

```
Will create tracker issue:
  Repo:   primatrix/projects
  Title:  [Iteration] <repo> <YYYY-MM>
  Labels: tracker, tracker/<repo>, tracker/<YYYY-MM>
  Body:
    ---
    ## 月度 Tracker — <repo> <YYYY-MM>
    ...
    ---

Will then perform:
  - Step 8:    Re-parent <K> carried tasks as sub-issues of the new tracker
  - Step 8.5:  Pull backlog candidates from Project V2 (Iteration empty ∧ Status=Triage ∧ Type ∈ {Task, Bug})
  - Step 8.6:  Set Iteration=<YYYY-MM> and Status=In Progress on the tracker itself + Iteration on every sub-issue
  - Step 8.7:  Unmount stale sub-issues whose Iteration ≠ <YYYY-MM> or repo mismatch

Approved? (y/revise)
```

Wait for explicit approval per engine §7.5. Anything else → revise.

### Step 7: Create the tracker issue

Render the body template (see §Issue Body Template) with placeholders substituted, write it to a **unique** temporary file (per engine §「临时文件命名约定」 — use `mktemp` to avoid collisions when this command is re-run):

```bash
BODY_FILE=$(mktemp /tmp/beaver-tracker-body-XXXXXX)
# ... write body content into "$BODY_FILE" ...

new_number=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh create <repo> "[Iteration] <repo> <YYYY-MM>" "$BODY_FILE")

bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh add-labels <repo> $new_number \
  tracker "tracker/<repo>" "tracker/<YYYY-MM>"

# Write Iteration field on the tracker itself via beaver-lib (per AC1).
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh set-tracker-iteration $new_number <YYYY-MM>

# Write Status=In Progress so the tracker appears on the Project board.
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh set-tracker-status $new_number "In Progress"

rm -f "$BODY_FILE"
```

Set `target = $new_number`.

### Step 8: Re-parent carried sub-issues

For each `task` in `carried`:

```bash
TASK_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh resolve-issue-id projects <task.number>)
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh attach-sub <target> $TASK_ID
```

> GitHub's Sub-Issues API enforces one parent per issue. POSTing to the new parent automatically detaches from the prior parent. Use the issue **numeric DB id** (`.id`, not `.number`).

On per-task failure: record reason; do NOT abort. Continue with the next task.

### Step 8.5: Pull backlog from Project V2 (interactive, 全选 / 全拒 / 逐项)

Query candidates via the **Project V2 fields** (no `status/triage` label query — per AC3):

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh fetch-backlog <repo>
```

This returns Issues where `Iteration` is empty AND `Status == "Triage"` AND repo matches `<repo>` AND native Issue Type ∈ `{Task, Bug}`.

Print as numbered list:

```
以下 Project V2 #14 中尚未分配 Iteration 且 Status=Triage 的 <repo> Task/Bug：
  1. #<n1> <title1>
  2. #<n2> <title2>
  ...

请输入要纳入本月 tracker 的编号（全选输入 "all"，全拒输入 "skip"，逐项输入逗号分隔编号如 "1,3,5"）：
```

If the candidate list is empty, print `No backlog candidates found.` and skip to Step 8.6.

Parse the input:
- `all` → select every candidate.
- `skip` / empty → select none.
- comma-separated indices → per-item; re-prompt on out-of-range or non-numeric tokens.

For each selected issue, attach to the tracker (Step 8 pattern).

Per-issue failures: collect; do NOT abort batch.

### Step 8.6: Sync Iteration field for all sub-issues

For every sub-issue currently attached to the tracker (fetch fresh via `list-tracker-subs`), write the Iteration via `beaver-lib.sh::set_iteration` (delegated through `set-issue-iteration`):

```bash
for sub in $(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh list-tracker-subs <target>); do
  bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh set-issue-iteration $sub <YYYY-MM>
done
```

`beaver-lib.sh::set_iteration` auto-adds the issue to Project #14 if missing, then writes the iteration. Per-issue failures: collect; do NOT abort batch.

### Step 8.7: Unmount stale sub-issues (差集同步)

Identify sub-issues that should NOT be under the tracker — those whose current Iteration title does not start with `<YYYY-MM>`, or whose home repo ≠ `<repo>`:

```bash
BEAVER_EXPECTED_YYYYMM=<YYYY-MM> bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh \
  list-tracker-subs-meta <target> <repo>
```

Each result row contains `{number, id, iteration_title, repo, repo_match, iteration_match}`. For every row where `repo_match == false` OR `iteration_match == false`:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-tracker.sh detach-sub <target> <id>
```

This calls the Sub-Issues API DELETE endpoint. The Issue itself and all other fields are untouched — only the parent link is removed. Per-issue failures: collect.

### Step 9: Report (4 statistics)

Print:

```
Tracker created: <new tracker URL>   (or "Reused existing #<existing_number>")
Sub-issue 总数:    <total>
Carry-over 数:     <K_carried>
新拉取数:          <N_pulled>
解挂数:            <M_unmounted>
  Failed actions (if any):
    - #<n> <title> — <error reason>
```

These four statistics map to RFC §2 #8 «sub-issue 数量、carry-over 数量、新拉取数量、解挂数量».

## Issue Body Template

```markdown
## 月度 Tracker — <repo> <YYYY-MM>

本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

仓库: primatrix/<repo>

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
- Step 6 is a HARD-GATE per engine §7.2; Steps 7/8/8.5/8.6/8.7 must not run before explicit §7.5 approval.
- All Project V2 field writes (Iteration on tracker + sub-issues) go through `beaver-lib.sh::set_iteration` (per beaver-engine §4 / §1 single-source-of-truth rule).
- Sub-issue re-parent uses the `sub_issues` API with `X-GitHub-Api-Version: 2026-03-10`. The payload field is `sub_issue_id` and takes the issue's numeric DB id (`.id`), not its repo-local `.number`.
- Backlog query reads Project V2 fields (`Iteration`, `Status`, native `issueType`); does NOT query `status/*`, `type/*`, or `size/*` labels (per AC3 / RFC-0013 成功指标 3).
- "Unfinished" = any sub-issue of the prior tracker with `state == "open"`. Sub-tasks travel with their task and are not separately re-parented.
- Steps 4 and 8.5 are interactive with three modes: 全选 / 全拒 / 逐项.
- Step 8.7 is the unmount-half of the diff sync that makes the tracker's sub-issue set equal `{Project V2 #14 ∧ Iteration=<YYYY-MM> ∧ repo归属=<repo> ∧ Type ∈ {Task, Bug}}` (RFC §2 #2 期望终态).
- Per-issue failures throughout Steps 8 / 8.5 / 8.6 / 8.7 are collected and surfaced in Step 9; the batch is never aborted on a single failure.
- Per engine §「临时文件命名约定」, all body files passed to `gh ... --body-file` use unique names (`mktemp /tmp/beaver-tracker-body-XXXXXX` — BSD mktemp requires `XXXXXX` at end, no suffix).
- The three repo-level tracker labels (`tracker / tracker/<repo> / tracker/<YYYY-MM>`) are Beaver metadata and remain managed via the labels API; they are NOT part of the deprecated `status/* / type/* / size/*` taxonomy.
- Health reporting (stale/overdue/bug-stats/upstream-blocked/missing-context/sub-task rollup) is intentionally NOT in this command anymore.
