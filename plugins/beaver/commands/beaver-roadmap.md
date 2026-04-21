---
allowed-tools: Bash(gh api:*), Bash(gh label:*), Bash(gh auth:*), Bash(date:*)
description: "Create a monthly roadmap tracking issue for a given repo in primatrix/projects, and carry forward unfinished tasks from the prior month's roadmap as sub-issues. Trigger when the user asks to start a new month's roadmap or roll over open tasks."
argument-hint: "<repo> [YYYY-MM]"
---

# /beaver-roadmap — 月度 Roadmap 创建与迁移

Phase 2 of the Beaver development lifecycle.

Create a monthly roadmap tracking issue in `primatrix/projects` for a given repo. Carry forward unfinished tasks from the prior month's roadmap as sub-issues of the new one.

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
Usage: /beaver-roadmap <repo> [YYYY-MM]
```

### Step 2: Ensure required labels exist

Idempotent label bootstrap (no-op if already present):

```bash
gh label create roadmap --repo primatrix/projects --color BFD4F2 --description "Monthly roadmap tracking issue" 2>/dev/null || true
gh label create "roadmap/<repo>" --repo primatrix/projects --color BFD4F2 --description "Roadmap for <repo>" 2>/dev/null || true
gh label create "roadmap/<YYYY-MM>" --repo primatrix/projects --color BFD4F2 --description "Roadmap for <YYYY-MM>" 2>/dev/null || true
gh label create "roadmap/<prevYYYY-MM>" --repo primatrix/projects --color BFD4F2 --description "Roadmap for <prevYYYY-MM>" 2>/dev/null || true
```

### Step 3: Locate prior month's roadmap

```bash
gh api -X GET search/issues \
  -f q='repo:primatrix/projects is:issue label:"roadmap/<repo>" label:"roadmap/<prevYYYY-MM>"' \
  --jq '{count: (.items | length), items: [.items[] | {number, state, title}]}'
```

- `count == 0` → `prev_number = null`, `carried = []`. Skip to Step 5.
- `count == 1` → record `prev_number = .items[0].number`. Continue to Step 4.
- `count > 1` → ERROR. Print all matches and abort with:
  ```
  Multiple prior roadmaps matched for <repo> <prevYYYY-MM>: #<n1>, #<n2>, ...
  Please consolidate or remove duplicates, then re-run.
  ```

### Step 4: Collect open sub-issues from the prior roadmap

```bash
gh api repos/primatrix/projects/issues/<prev_number>/sub_issues \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  --jq '[.[] | select(.state=="open") | {number, title}]'
```

Store as `carried` (array of `{number, title}`).

> **Definition of unfinished:** any sub-issue of the prior roadmap whose `state == "open"`. Sub-tasks beneath those tasks are NOT separately re-parented; they remain attached to their parent task and travel with it implicitly.

### Step 5: Check whether current month's roadmap already exists

```bash
gh api -X GET search/issues \
  -f q='repo:primatrix/projects is:issue label:"roadmap/<repo>" label:"roadmap/<YYYY-MM>"' \
  --jq '{count: (.items | length), items: [.items[] | {number, title}]}'
```

- `count == 0` → continue to Step 6 (create).
- `count == 1` → set `existing_number = .items[0].number`, ask the user:
  ```
  本月 roadmap 已存在 #<existing_number>。是否将 carried 列表 (<K> 个 task) 重新挂到 #<existing_number> 下？(y/skip)
  ```
  - User answer `y` → set `target = existing_number`, jump to Step 8 (skip create).
  - Anything else → exit without changes.
- `count > 1` → ERROR. Print matches and abort.

### Step 6: Preview + approval gate (HARD-GATE per engine §7.2)

Print the full preview:

```
Will create roadmap issue:
  Repo:   primatrix/projects
  Title:  [Roadmap] <repo> <YYYY-MM>
  Labels: roadmap, roadmap/<repo>, roadmap/<YYYY-MM>
  Body:
    ---
    ## 月度 Roadmap — <repo> <YYYY-MM>

    本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

    ## 来源
    - 上月 roadmap: #<prev_number 或 "无">（迁移 <K> 个未完成 task）

    <!-- beaver-roadmap
    repo: <repo>
    month: <YYYY-MM>
    carried-from: #<prev_number 或 "none">
    -->
    ---

Will then re-parent <K> open tasks as sub-issues of the new roadmap:
  - #<n1> <title1>
  - #<n2> <title2>
  ...

Approved? (y/revise)
```

Wait for explicit approval per engine §7.5 (only `y`/`yes`/`ok`/`approve`/`approved`/`lgtm`/`继续`/`通过` count). Anything else → revise.

### Step 7: Create the roadmap issue

```bash
BODY_FILE=$(mktemp)
cat > "$BODY_FILE" << 'BEAVEREOF'
## 月度 Roadmap — <repo> <YYYY-MM>

本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

## 来源
- 上月 roadmap: #<prev_number 或 "无">（迁移 <K> 个未完成 task）

<!-- beaver-roadmap
repo: <repo>
month: <YYYY-MM>
carried-from: #<prev_number 或 "none">
-->
BEAVEREOF

new_number=$(gh api repos/primatrix/projects/issues --method POST \
  -f title="[Roadmap] <repo> <YYYY-MM>" \
  -F body=@"$BODY_FILE" \
  --jq '.number')
rm "$BODY_FILE"

gh api repos/primatrix/projects/issues/$new_number/labels --method POST \
  -f "labels[]=roadmap" \
  -f "labels[]=roadmap/<repo>" \
  -f "labels[]=roadmap/<YYYY-MM>"
```

Set `target = $new_number`.

### Step 8: Re-parent open tasks as sub-issues

For each `task` in `carried`:

```bash
gh api repos/primatrix/projects/issues/<target>/sub_issues --method POST \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  -F sub_issue_id=<task.id>
```

> **Note:** GitHub's Sub-Issues API enforces one parent per issue. POSTing to the new parent automatically detaches from the prior parent. Use the issue **node ID / numeric DB id** (`.id`, not `.number`) per the API contract.

To resolve `task.id` from `task.number`:
```bash
gh api repos/primatrix/projects/issues/<task.number> --jq '.id'
```

On per-task failure: record reason, do NOT abort the batch. Continue with the next task.

### Step 9: Report

Print:

```
Roadmap created: <new roadmap URL>   (or "Reused existing #<existing_number>")
Migration: <success_count> succeeded, <failure_count> failed
  Failed tasks (if any):
    - #<n> <title> — <error reason>
```

## Issue Body Template

```markdown
## 月度 Roadmap — <repo> <YYYY-MM>

本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

## 来源
- 上月 roadmap: #<N>（迁移 <K> 个未完成 task）

<!-- beaver-roadmap
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
- "Unfinished" = any sub-issue of the prior roadmap with `state == "open"`. Sub-tasks travel with their task and are not separately re-parented.
- Health reporting (stale/overdue/bug-stats/upstream-blocked/missing-context/sub-task rollup) is intentionally NOT in this command anymore.
