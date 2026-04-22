---
allowed-tools: Bash(gh api:*), Bash(gh project:*), Bash(gh label:*), Bash(cat > /tmp/*), Bash(git log:*), Bash(git diff:*)
description: Create a Beaver-tracked GitHub Issue with brainstorming QA, automatic status transitions and guardrail checks. Trigger when the user wants to create a GitHub issue or report a bug.
argument-hint: "[issue-type]"
---

# /beaver-create — Task/Bug 创建

Phase 1 of the Beaver development lifecycle.

## Workflow

1. **Type detection**: Determine issue type (feat/bug/refactor/docs/chore). If argument provided, use it. Otherwise ask.
   - `type/bug` → enter Bug Submode (see §Bug below)

1. **Load project config**: Read beaver-config from Project V2 README per engine §5.

1. **Discovery Triad**: Execute engine §8 (D1 recent activity, D2 keyword search, D3 project conventions). Print Discovery Brief before first question.

1. **Iterative QA**: Follow engine §7 strictly.
   - **size/S path** (3 questions minimum):
     1. Title (one-line, imperative)
     2. Objective (one user-facing outcome sentence)
     3. Acceptance criteria (≥ 2 verifiable items)
   - **size/L path** (4 sectional approvals):
     1. Level + parent Issue (Goal/Task/SubTask hierarchy)
     2. Title
     3. Objective + scope
     4. Acceptance criteria + stakeholders
   - System auto-suggests size (S/L) with reasoning after collecting objective. User confirms or overrides.

1. **Preview + approval gate**: Engine §7.2 HARD-GATE. Present full Issue preview with §9.4 checklist. Wait for explicit approval per §7.5.

1. **Create Issue**:

   The POST response body contains the issue body Markdown with embedded newlines. Capturing the whole JSON into a shell variable and then re-parsing with `jq` corrupts those control chars and yields empty fields. Always extract each field via `--jq` directly on the `gh api` call — never round-trip through a captured variable. To avoid losing the issue id when only `--jq '.number'` is requested, perform the POST once for `.number`, then a single follow-up GET for `.id` / `.node_id` / `.html_url`. Render the issue body to a temp file (e.g. `/tmp/beaver-issue-body.md`) using the templates in §Issue Body Templates, then:

   ```bash
   # Create issue (extracts .number)
   NEW_NUM=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh create-issue {org} {issueRepo} "{title}" /tmp/beaver-issue-body.md)

   # Re-fetch to obtain remaining ids (cheap GET, idempotent)
   eval $(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh fetch-ids {org} {issueRepo} $NEW_NUM)
   NEW_ID=$id; NEW_NODE_ID=$node_id; NEW_URL=$html_url

   # Add labels
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh add-labels {org} {issueRepo} $NEW_NUM \
     Control-By-Beaver {type_label} {size_label} status/triage

   # Add to Project V2 (capture the project item id for the field-edit step)
   ITEM_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh add-to-project {project_number} {org} "$NEW_URL")

   # Set custom fields (Level, Status, Progress)
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh set-field "$ITEM_ID" {project_id} {level_field_id} {level_option_id}
   ```

1. **Iteration assignment (interactive)** — skipped in Bug Submode (G007 exempts bugs):

   Ask the user:

   ```
   将本 Issue 加入哪个 Iteration？
     - skip      不分配，留给 /beaver-tracker 后续同步
     - current   当前月份 (YYYY-MM)
     - YYYY-MM   指定月份（如 2026-05）
   ```

   - `skip` (case-insensitive) or empty → no-op, continue to next step.
   - `current` → set `target_yyyymm` to local current year-month.
   - `YYYY-MM` literal → use as `target_yyyymm`.

   Resolve the Iteration field id and target iteration id (mirrors beaver-tracker §8.6):

   ```bash
   eval $(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh resolve-iteration primatrix 14 "$target_yyyymm")
   PROJECT_ID=$project_id
   ITERATION_FIELD_ID=$field_id
   ITERATION_ID=$iteration_id
   ```

   If `ITERATION_ID` is empty, warn (do NOT abort the whole command — the issue already exists):

   ```
   Iteration entry for <target_yyyymm> not found on Project #14.
   Run /beaver-setup to extend iterations into <target_yyyymm>.
   Iteration assignment skipped — assign manually or re-run after setup.
   ```

   Otherwise apply the mutation against the project item created above:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh set-iteration "$PROJECT_ID" "$ITEM_ID" "$ITERATION_FIELD_ID" "$ITERATION_ID"
   ```

   On success, the issue gains the Iteration assignment and (per engine §G007) becomes eligible for `status/ready-to-claim`. This step does NOT auto-transition the status label — that remains `/beaver-claim`'s job.

1. **Link to parent** (if Task or SubTask):

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh link-parent {org} {issueRepo} {parent_number} $NEW_ID
   ```

1. **Initial status**: `status/triage` for all. Exception: p0/blocker Bug → `status/in-progress` + @CODEOWNERS (see Bug Submode).

1. **Report**: Print created Issue URL, labels, Iteration (if assigned, else `unassigned`), and next-step hint.

## Bug Submode

Activated when `type/bug` is detected. Overrides:

- **Forced size/S**: G008 enforced. System sets `size/S` automatically, user cannot change to `size/L`.
- **Mandatory priority**: Must ask priority (`p0/blocker`, `p1/urgent`, `p2/high`, `p3/normal`).
- **Bug QA template** (4 sections, each requires §7.5 approval):
  1. 复现步骤 (must be runnable/clickable per §9.5)
  2. 期望行为
  3. 实际行为
  4. 影响范围 + 环境信息
- **p0/blocker fast path**:
  - Skip `status/triage`, set directly to `status/in-progress`
  - Resolve CODEOWNERS for relevant files, @mention in Issue body
  - No Iteration required (G007 exempt)

## Issue Body Templates

### Feature (Task/SubTask)

```markdown
## 目标
{objective}

## 验收标准
{acceptance_criteria}

<!-- beaver-tracking
type: {type}
size: {size}
created-by: beaver-create
-->
```

### Bug

```markdown
## 复现步骤
{reproduction_steps}

## 期望行为
{expected}

## 实际行为
{actual}

## 影响范围
{impact}

## 环境信息
{environment}

<!-- beaver-tracking
type: type/bug
size: size/S
priority: {priority}
created-by: beaver-create
-->
```

## Constraints

- Engine §7.2 HARD-GATE applies to all write operations
- Engine §9.4 checklist must pass before approval
- Engine §9.5 applies for bug-mode
- All labels must use exact names from engine §1
