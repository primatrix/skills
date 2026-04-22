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
gh label create tracker --repo primatrix/projects --color BFD4F2 --description "Monthly Iteration tracker issue" 2>/dev/null || true
gh label create "tracker/<repo>" --repo primatrix/projects --color BFD4F2 --description "Tracker for <repo>" 2>/dev/null || true
gh label create "tracker/<YYYY-MM>" --repo primatrix/projects --color BFD4F2 --description "Tracker for <YYYY-MM>" 2>/dev/null || true
gh label create "tracker/<prevYYYY-MM>" --repo primatrix/projects --color BFD4F2 --description "Tracker for <prevYYYY-MM>" 2>/dev/null || true
```

### Step 3: Locate prior month's tracker

```bash
gh api -X GET search/issues \
  -f q='repo:primatrix/projects is:issue label:"tracker/<repo>" label:"tracker/<prevYYYY-MM>"' \
  --jq '{count: (.items | length), items: [.items[] | {number, state, title}]}'
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
gh api repos/primatrix/projects/issues/<prev_number>/sub_issues \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  --jq '[.[] | select(.state=="open") | {number, title}]'
```

Store as `carried` (array of `{number, title}`).

> **Definition of unfinished:** any sub-issue of the prior tracker whose `state == "open"`. Sub-tasks beneath those tasks are NOT separately re-parented; they remain attached to their parent task and travel with it implicitly.

### Step 5: Check whether current month's tracker already exists

```bash
gh api -X GET search/issues \
  -f q='repo:primatrix/projects is:issue label:"tracker/<repo>" label:"tracker/<YYYY-MM>"' \
  --jq '{count: (.items | length), items: [.items[] | {number, title}]}'
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

```bash
BODY_FILE=$(mktemp)
cat > "$BODY_FILE" << 'BEAVEREOF'
## 月度 Tracker — <repo> <YYYY-MM>

本 issue 作为 <repo> <YYYY-MM> 月度 tracking 容器。所有本月 task issue 作为 sub-issue 挂在下方。

## 来源
- 上月 tracker: #<prev_number 或 "无">（迁移 <K> 个未完成 task）

<!-- beaver-tracker
repo: <repo>
month: <YYYY-MM>
carried-from: #<prev_number 或 "none">
-->
BEAVEREOF

new_number=$(gh api repos/primatrix/projects/issues --method POST \
  -f title="[Iteration] <repo> <YYYY-MM>" \
  -F body=@"$BODY_FILE" \
  --jq '.number')
rm "$BODY_FILE"

gh api repos/primatrix/projects/issues/$new_number/labels --method POST \
  -f "labels[]=tracker" \
  -f "labels[]=tracker/<repo>" \
  -f "labels[]=tracker/<YYYY-MM>"
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

### Step 8.5: Pull backlog from triage queue (interactive)

Query candidates: `status/triage` issues in `primatrix/projects` not yet assigned to any Iteration.

```bash
gh api graphql -f query='
  query {
    organization(login: "primatrix") {
      projectV2(number: 14) {
        items(first: 100) {
          nodes {
            content {
              ... on Issue {
                number
                title
                repository { nameWithOwner }
                labels(first: 30) { nodes { name } }
              }
            }
            fieldValueByName(name: "Iteration") {
              ... on ProjectV2ItemFieldIterationValue { title }
            }
          }
        }
      }
    }
  }' --jq '.data.organization.projectV2.items.nodes
            | map(select(.content != null and .content.repository.nameWithOwner == "primatrix/projects"))
            | map(select(.content.labels.nodes | map(.name) | index("status/triage")))
            | map(select(.fieldValueByName == null))
            | map({number: .content.number, title: .content.title})'
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

For each selected issue, resolve `.number` → numeric DB `.id` via `gh api repos/primatrix/projects/issues/<number> --jq '.id'`, then POST to `/repos/primatrix/projects/issues/<tracker_number>/sub_issues` with `sub_issue_id=<id>` (same pattern as Step 8). After attachment, Step 8.6 will set the Iteration field.

Per-issue failures: collect, do NOT abort batch; surface in Step 9.

### Step 8.6: Sync Iteration field for all sub-issues

Resolve current month's Iteration entry id:

```bash
ITERATION_INFO=$(gh api graphql -f query='
  query {
    organization(login: "primatrix") {
      projectV2(number: 14) {
        id
        field(name: "Iteration") {
          ... on ProjectV2IterationField {
            id
            configuration { iterations { id title } }
          }
        }
      }
    }
  }')
PROJECT_ID=$(echo "$ITERATION_INFO" | jq -r '.data.organization.projectV2.id')
ITERATION_FIELD_ID=$(echo "$ITERATION_INFO" | jq -r '.data.organization.projectV2.field.id')
ITERATION_ID=$(echo "$ITERATION_INFO" | jq -r --arg yyyymm "<YYYY-MM>" \
  '.data.organization.projectV2.field.configuration.iterations
   | map(select(.title | startswith($yyyymm))) | .[0].id')
```

If `ITERATION_ID` is `null` or empty, abort Step 8.6 with the following message and continue to Step 9 (do NOT abort the whole command — Steps 1-8 already succeeded):

```
Iteration entry for <YYYY-MM> not found on Project #14.
Run /beaver-setup to extend iterations into <YYYY-MM>.
Sub-issue iteration sync skipped — fix this and re-run /beaver-tracker if needed.
```

For every sub-issue currently attached to the tracker (fetch fresh via `gh api /repos/primatrix/projects/issues/<tracker_number>/sub_issues --jq '.[].number'`), resolve its ProjectV2Item id and set the Iteration field:

```bash
# Resolve item id for an issue (assuming the issue is already on the project):
ITEM_ID=$(gh api graphql -f query="
  query {
    repository(owner: \"primatrix\", name: \"projects\") {
      issue(number: <issue_number>) {
        projectItems(first: 10) {
          nodes { id project { number } }
        }
      }
    }
  }" --jq '.data.repository.issue.projectItems.nodes
            | map(select(.project.number == 14)) | .[0].id')

# Set Iteration field
gh api graphql -f query="
  mutation {
    updateProjectV2ItemFieldValue(input: {
      projectId: \"$PROJECT_ID\"
      itemId: \"$ITEM_ID\"
      fieldId: \"$ITERATION_FIELD_ID\"
      value: { iterationId: \"$ITERATION_ID\" }
    }) { projectV2Item { id } }
  }"
```

> **Note (add-if-missing, then update):** A sub-issue parented via the Sub-Issues API is NOT auto-added to Project v2 #14. If `ITEM_ID` resolves to `null` for a given sub-issue, first add it to the project, then retry the field update:
>
> ```bash
> # Resolve the issue's node id, then add to project
> CONTENT_ID=$(gh api repos/primatrix/projects/issues/<issue_number> --jq '.node_id')
> ITEM_ID=$(gh api graphql -f query="
>   mutation {
>     addProjectV2ItemById(input: {
>       projectId: \"$PROJECT_ID\"
>       contentId: \"$CONTENT_ID\"
>     }) { item { id } }
>   }" --jq '.data.addProjectV2ItemById.item.id')
> ```
>
> Then re-run the `updateProjectV2ItemFieldValue` mutation above with the new `ITEM_ID`. If the add itself fails, log the failure (per the existing "Per-issue failures: collect, do NOT abort" behavior below) and continue.

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
