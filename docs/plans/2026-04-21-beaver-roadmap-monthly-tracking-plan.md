# beaver-roadmap Monthly Tracking Issue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `/beaver-roadmap` to create a monthly roadmap tracking issue in `primatrix/projects` for a given repo, and carry forward unfinished tasks from the prior month's roadmap as sub-issues.

**Architecture:** Single-file rewrite of `plugins/beaver/commands/beaver-roadmap.md`. The command becomes a 4-stage pipeline (locate prior → collect open tasks → preview/approve → create + re-parent). Hardcodes `primatrix/projects` as issueRepo. Removes the entire health-report path with no stub.

**Tech Stack:** Markdown command file (Claude Code slash command), `gh` CLI (`gh api`, `gh label`), GitHub Sub-Issues API (header `X-GitHub-Api-Version: 2026-03-10`).

**Reference design:** `docs/plans/2026-04-21-beaver-roadmap-monthly-tracking-design.md`

---

## Pre-Task Checklist

This codebase has no test runner, no linter, no build. "Verify" means:
- JSON validity for `.claude-plugin/*.json` (none touched here, so skip)
- YAML frontmatter validity (between `---` fences) for the command file
- `gh` command shapes are syntactically reasonable (validated by reading)
- Manual review against the design doc

There is **no automated test** for slash-command markdown files. Verification per task is by visual diff + grep against the design doc's decision list.

---

## Task 1: Replace the command file with the new structure

**Files:**
- Modify: `plugins/beaver/commands/beaver-roadmap.md` (full rewrite, 156 lines → ~140 lines)

**Step 1: Read the current file once for confirmation**

Run: `wc -l plugins/beaver/commands/beaver-roadmap.md`
Expected: `156 plugins/beaver/commands/beaver-roadmap.md`

**Step 2: Overwrite with the new content**

Use the `Write` tool to replace the entire file with:

````markdown
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
````

**Step 3: Verify the file shape**

Run: `head -5 plugins/beaver/commands/beaver-roadmap.md`
Expected: starts with `---`, contains `allowed-tools:`, `description:`, `argument-hint: "<repo> [YYYY-MM]"`, ends with `---` on line 5.

Run: `grep -c '^### Step ' plugins/beaver/commands/beaver-roadmap.md`
Expected: `9`

Run: `grep -c '^## ' plugins/beaver/commands/beaver-roadmap.md`
Expected: at least `5` (Prerequisites, Workflow, Issue Body Template, Constraints, plus the H1 is `#`).

**Step 4: Verify removed sections are gone**

Run: `grep -n 'Health Indicators\|Stale Detection\|Overdue Detection\|Bug Statistics\|Upstream Blocked\|Missing Context\|Risk Summary\|Optional publish\|stale/overdue\|G010' plugins/beaver/commands/beaver-roadmap.md`
Expected: no matches.

**Step 5: Verify new key elements are present**

Run: `grep -n 'primatrix/projects\|sub_issues\|X-GitHub-Api-Version: 2026-03-10\|HARD-GATE\|roadmap/<repo>' plugins/beaver/commands/beaver-roadmap.md`
Expected: each pattern matches at least once.

**Step 6: Commit**

```bash
git add plugins/beaver/commands/beaver-roadmap.md
git commit -m "$(cat <<'EOF'
refactor(beaver): rewrite beaver-roadmap as monthly tracking issue creator

Repurpose /beaver-roadmap from health reporting to monthly roadmap
tracking issue creation in primatrix/projects, with carry-forward of
unfinished tasks (open sub-issues of the prior month's roadmap) as
sub-issues of the new roadmap.

- Hardcode issueRepo to primatrix/projects (skip beaver-config)
- Add required CLI argument <repo> + optional [YYYY-MM]
- Identify roadmaps via title prefix [Roadmap] + labels
  roadmap, roadmap/<repo>, roadmap/<YYYY-MM>
- Re-parent only the task; sub-tasks stay attached to their task
- Prompt user when current month's roadmap already exists
- Remove all health-report logic (stale/overdue/bug-stats/rollup)
- Per-task migration failures are collected, not fatal

Refs: docs/plans/2026-04-21-beaver-roadmap-monthly-tracking-design.md
EOF
)"
```

---

## Task 2: Bump plugin version

The plugin manifest tracks a version that should bump on a behavior-changing rewrite.

**Files:**
- Modify: `plugins/beaver/.claude-plugin/plugin.json`

**Step 1: Read current version**

Use the `Read` tool on `plugins/beaver/.claude-plugin/plugin.json`. Note the current `version` value (likely `3.0.0` based on recent commits like `8a60fe5 chore(beaver): bump to v3.0.0`).

**Step 2: Decide bump**

This is a breaking behavioral change to one command (no longer produces a health report; signature changed; default repo changed). Bump **minor** if v3.x → `3.1.0`, or **major** if you treat slash-command behavior as public API. Choose `3.1.0` unless the project owner says otherwise.

**Step 3: Edit version**

Use the `Edit` tool. Replace `"version": "3.0.0"` with `"version": "3.1.0"`.

**Step 4: Verify JSON validity**

Run: `python3 -c 'import json; json.load(open("plugins/beaver/.claude-plugin/plugin.json"))' && echo OK`
Expected: `OK`

**Step 5: Commit**

```bash
git add plugins/beaver/.claude-plugin/plugin.json
git commit -m "chore(beaver): bump to v3.1.0 for beaver-roadmap rewrite"
```

---

## Task 3: Bump marketplace registry version (if it pins beaver)

**Files:**
- Modify: `.claude-plugin/marketplace.json` (only if it pins a version for `beaver`)

**Step 1: Inspect**

Use the `Read` tool on `.claude-plugin/marketplace.json`. Find the `beaver` entry.

**Step 2: Decide**

- If the entry has no `version` field, **skip this task entirely** — proceed to Task 4.
- If it has `"version": "3.0.0"`, update to `"3.1.0"`.

**Step 3: Edit (if applicable)**

Use the `Edit` tool to update the version string.

**Step 4: Verify JSON validity (if edited)**

Run: `python3 -c 'import json; json.load(open(".claude-plugin/marketplace.json"))' && echo OK`
Expected: `OK`

**Step 5: Commit (if edited)**

```bash
git add .claude-plugin/marketplace.json
git commit -m "chore(marketplace): pin beaver to v3.1.0"
```

---

## Task 4: Final cross-check against design doc

**Step 1: Re-read the design**

Use the `Read` tool on `docs/plans/2026-04-21-beaver-roadmap-monthly-tracking-design.md`.

**Step 2: Walk the Decisions table**

For each row in the design's "Decisions" table, run a `grep` against `plugins/beaver/commands/beaver-roadmap.md` confirming the decision is honored:

| Decision | Verification command | Expected |
|---|---|---|
| Hardcoded `primatrix/projects` | `grep -c primatrix/projects plugins/beaver/commands/beaver-roadmap.md` | ≥ 5 |
| `beaver-config` not read | `grep -c beaver-config plugins/beaver/commands/beaver-roadmap.md` | 0 |
| Required `<repo>` arg | `grep -c 'argument-hint: "<repo>' plugins/beaver/commands/beaver-roadmap.md` | 1 |
| Title `[Roadmap] <repo> YYYY-MM` | `grep -c '\[Roadmap\] <repo>' plugins/beaver/commands/beaver-roadmap.md` | ≥ 2 |
| Three roadmap labels | `grep -c 'roadmap/<repo>\|roadmap/<YYYY-MM>' plugins/beaver/commands/beaver-roadmap.md` | ≥ 4 |
| Duplicate-current-month prompt | `grep -c '本月 roadmap 已存在' plugins/beaver/commands/beaver-roadmap.md` | ≥ 1 |
| Sub-task NOT separately re-parented | `grep -c 'sub-tasks travel\|sub-task' plugins/beaver/commands/beaver-roadmap.md` | ≥ 1 |
| Per-task failure non-fatal | `grep -c 'do NOT abort\|not abort the batch' plugins/beaver/commands/beaver-roadmap.md` | ≥ 1 |
| HARD-GATE preserved | `grep -c 'HARD-GATE\|engine §7' plugins/beaver/commands/beaver-roadmap.md` | ≥ 2 |

If any row fails, return to Task 1 and patch.

**Step 3: Final visual diff vs main**

Run: `git log --oneline main..HEAD`
Expected: 1–3 commits (Task 1 always, Task 2 always, Task 3 maybe).

Run: `git diff main -- plugins/beaver/commands/beaver-roadmap.md | head -50`
Expected: large `-` block for the old health-report content, large `+` block for the new structure.

**Step 4: No commit needed for cross-check.**

---

## Out of Scope (do NOT do)

- Touching `plugins/beaver/skills/beaver-engine/SKILL.md` (G010 stays for other potential callers).
- Restoring health reporting under `/beaver-report` or any other command.
- Adding scheduling / cron / automation around the new command.
- Cross-repo aggregation in a single roadmap.
- Modifying any other beaver command file.
