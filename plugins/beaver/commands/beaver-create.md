---
allowed-tools: Bash(gh api:*), Bash(gh project:*), Bash(gh issue:*), Bash(mktemp:*), Bash(git log:*), Bash(git diff:*), Bash(bash:*)
description: Create a Beaver-tracked GitHub Issue with brainstorming QA, automatic status transitions and guardrail checks. Trigger when the user wants to create a GitHub issue or report a bug.
argument-hint: "[--type task|subtask|bug]"
---

# /beaver-create — Task / SubTask / Bug 创建

Phase 1 of the Beaver development lifecycle. All lifecycle metadata is written to **Project V2 #14** custom fields and the **native GitHub Issue Type** through `beaver-lib.sh` — never via repository labels. See `plugins/beaver/skills/beaver-engine/SKILL.md` §1 (Field Taxonomy) and §4 (Field Operations).

## Workflow

1. **Type inference + override**

   Determine the Issue Type as one of `task / subtask / bug` (these are the three native Issue Types created by `/beaver-setup`). The user may either:

   - pass `--type task`, `--type subtask`, or `--type bug` explicitly on the slash command, in which case that value is used verbatim, or
   - omit `--type`, in which case Claude infers the Type from the user's request:
     - mention of defect / 复现 / regression / "broken" → `bug`
     - mention of an existing parent Goal/Task being decomposed → `subtask`
     - otherwise → `task`

   When `--type bug`, enter **Bug Submode** (see §Bug Submode).

1. **Load project config**: Read `beaver-config` from the Project V2 README per engine §5.

1. **Discovery Triad**: Execute engine §8 (D1 recent activity, D2 keyword search, D3 project conventions). Print the Discovery Brief before the first question.

1. **Iterative QA**: Follow engine §7 strictly.
   - **Size=S path** (3 questions minimum):
     1. Title (one-line, imperative)
     2. Objective (one user-facing outcome sentence)
     3. Acceptance criteria (≥ 2 verifiable items)
   - **Size=L path** (4 sectional approvals):
     1. Type + parent Issue (Task/SubTask hierarchy via native Issue Type)
     2. Title
     3. Objective + scope
     4. Acceptance criteria + stakeholders
   - The system auto-suggests Size (S/L) with reasoning after collecting the objective. The user confirms or overrides.

1. **Preview + approval gate**: Engine §7.2 HARD-GATE. Present the full Issue preview with the §9.4 checklist. Wait for explicit approval per §7.5.

1. **落库 (Issue creation + field writes)**

   Per RFC-0013 §1 step 9, perform exactly this ordered sequence. Each step has a single owner — never duplicate field writes inside this command, all metadata writes flow through `beaver-lib.sh`.

   1. **9a — Create the Issue.**

      Render the Issue body to a *unique* temp file (use `mktemp` — never reuse a fixed path; concurrent invocations would clobber each other) using the templates in §Issue Body Templates. The body includes `@CODEOWNERS` only for the P0 Bug case (see Bug Submode).

      ```bash
      BODY_FILE=$(mktemp -t beaver-create-body.XXXXXX)
      # ... render the chosen template into "$BODY_FILE" ...

      # Create the Issue (POST returns only .number — re-fetch ids next).
      NEW_NUM=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh \
        create-issue {org} {issueRepo} "{title}" "$BODY_FILE")

      # Re-fetch to obtain remaining ids (cheap GET, idempotent).
      eval $(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh \
        fetch-ids {org} {issueRepo} $NEW_NUM)
      NEW_ID=$id; NEW_NODE_ID=$node_id; NEW_URL=$html_url

      rm -f "$BODY_FILE"
      ```

      Do not capture the POST response into a shell variable and re-parse with `jq` — embedded newlines in the body Markdown corrupt control characters. Always use `--jq` directly on each `gh api` call.

   2. **9b — Link to parent (SubTask only).**

      For SubTasks, link via the Sub-Issues API **before** add-to-project. If add-to-project runs first, the parent's project card shows "1 sub-issue not in this project" because Projects V2 emits the parent→child rollup at the moment the project item is created. Top-level Tasks and Bugs have no parent and skip this step.

      ```bash
      bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh link-parent \
        {org} {issueRepo} {parent_number} $NEW_ID
      ```

   3. **9c — Add to Project V2 #14.**

      ```bash
      ITEM_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-create.sh \
        add-to-project {project_number} {org} "$NEW_URL")
      ```

   4. **9d — Write Project V2 fields via `beaver-lib.sh`.**

      Two branches based on the Type chosen in step 1.

      **Task / SubTask** — write Type, Size, Status=Triage, and (optionally) Iteration:

      ```bash
      source ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-lib.sh

      set_type      "$NEW_NUM" "{Task|SubTask}"   # native Issue Type
      set_size      "$NEW_NUM" "{S|L}"             # Project V2 Size
      set_status    "$NEW_NUM" "Triage"            # Project V2 Status
      # Iteration is optional for Task/SubTask — see §Iteration assignment.
      ```

      **Bug** — write Type=Bug, Priority, Status, and Iteration. The Bug path **does NOT write Size** (Size has no Bug semantics in RFC-0013):

      ```bash
      source ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-lib.sh

      set_type   "$NEW_NUM" "Bug"
      # Priority is mandatory for Bug — value ∈ { P0, P1, P2 } collected in QA.
      # set_priority is currently called via the generic single-select path:
      PRIORITY_FIELD_ID=$(get_field_id "Priority")
      PRIORITY_OPT_ID=$(get_option_id "Priority" "{P0|P1|P2}")
      ITEM_ID_FOR_PRIO=$(resolve_item_id "$NEW_URL")
      _set_single_select "$ITEM_ID_FOR_PRIO" "$PRIORITY_FIELD_ID" "$PRIORITY_OPT_ID"

      # Status mapping (RFC-0013 §3 Bug path):
      #   P0       → "In Progress"  (fast-path, work begins immediately)
      #   P1 / P2  → "Ready to Claim"
      if [ "$priority" = "P0" ]; then
        set_status "$NEW_NUM" "In Progress"
      else
        set_status "$NEW_NUM" "Ready to Claim"
      fi

      # Iteration is MANDATORY for Bug — resolved by G011.
      ITER_TITLE=$(latest_iteration_for_repo {issueRepo})
      if [ -z "$ITER_TITLE" ]; then
        echo "G011 fail: no current or future Iteration on Project #14 for {issueRepo}." >&2
        echo "Run /beaver-tracker {issueRepo} to create this month's Iteration entry, then retry." >&2
        exit 1
      fi
      set_iteration "$NEW_NUM" "$ITER_TITLE"
      ```

   5. **9e — Parent-tracker linkage (out of scope for this step; handled by `/beaver-tracker`).**

1. **Iteration assignment (Task / SubTask interactive path)**

   For Task / SubTask only — Bug already wrote Iteration via G011 in step 9d. Ask the user:

   ```
   将本 Issue 加入哪个 Iteration？
     - skip      不分配，留给 /beaver-tracker 后续同步
     - current   当前月份 (YYYY-MM)
     - YYYY-MM   指定月份（如 2026-05）
   ```

   - `skip` (case-insensitive) or empty → no-op, continue.
   - `current` → `target=$(date -u +%Y-%m)`.
   - `YYYY-MM` literal → use as `target`.

   Then:

   ```bash
   source ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-lib.sh
   set_iteration "$NEW_NUM" "$target" || \
     echo "Iteration entry for $target not found on Project #14. Run /beaver-setup to extend iterations, or assign manually." >&2
   ```

   `set_iteration` accepts a `YYYY-MM` prefix and matches the first iteration whose title starts with that prefix. A failure here is a warning — the Issue already exists.

1. **Initial Status summary**: All Tasks / SubTasks land at `Status = Triage`. Bug routes per priority (P0 → In Progress, P1/P2 → Ready to Claim). The native Issue Type (`Task` / `SubTask` / `Bug`) plus Status / Size (Task/SubTask only) / Priority (Bug only) / Iteration (Bug always; Task/SubTask optional) are now written. No `status/* / type/* / size/*` repository labels are touched.

1. **Report**: Print the created Issue URL, native Issue Type, Status, Size (if applicable), Priority (if Bug), Iteration (if assigned, else `unassigned`), and the next-step hint: "`/beaver-claim` 已删除（见 RFC-0013 §3）：对于 Ready to Claim 的 Issue，请在 GitHub UI assign 自己后手动将 Status 切到对应值；Triage 状态则等待 triage。"

## Bug Submode

Activated when `--type bug` (explicit) or Type inference selects `bug`. Overrides:

- **Mandatory Priority**: Must ask priority. Allowed values are exactly `P0 / P1 / P2`. Asked before the §7.2 HARD-GATE preview so the gate can render the full Bug record.
- **No Size write**: The Bug path explicitly skips Size — RFC-0013 has no Size semantics for Bug. Do not call `set_size` in the Bug branch.
- **Bug QA template** (4 sections, each requires §7.5 approval):
  1. 复现步骤 (must be runnable / clickable per §9.5)
  2. 期望行为
  3. 实际行为
  4. 影响范围 + 环境信息
- **P0 fast path**:
  - Status routes directly to `In Progress` (skip `Triage` / `Ready to Claim`).
  - Resolve `CODEOWNERS` for the relevant files and `@`-mention them in the Issue body — see the Bug template below for the placement of `@CODEOWNERS`.
- **P1 / P2 path**:
  - Status routes to `Ready to Claim`.
  - No `@CODEOWNERS` mention — the Bug body omits the `@CODEOWNERS` line.
- **Iteration mandatory (G011)**:
  - The Bug path calls `beaver-lib.sh::latest_iteration_for_repo <issueRepo>` to resolve the target Iteration title (G011 algorithm: current iteration if any, else the next future iteration).
  - On `null` / error, **G011** fails: the command MUST abort and print `Run /beaver-tracker <issueRepo> to create this month's Iteration entry, then retry.` Do not partially write the Bug — resolve the Iteration first, then re-run.

## Issue Body Templates

### Task / SubTask

```markdown
## 目标
{objective}

## 验收标准
{acceptance_criteria}

<!-- beaver-tracking
issue-type: {Task|SubTask}
size: {S|L}
created-by: beaver-create
-->
```

### Bug — P0 (includes `@CODEOWNERS`)

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

cc @CODEOWNERS

<!-- beaver-tracking
issue-type: Bug
priority: P0
created-by: beaver-create
-->
```

### Bug — P1 / P2 (no `@CODEOWNERS`)

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
issue-type: Bug
priority: {P1|P2}
created-by: beaver-create
-->
```

## Constraints

- Engine §7.2 HARD-GATE applies to all write operations.
- Engine §9.4 checklist must pass before approval.
- Engine §9.5 applies in Bug Submode.
- All lifecycle metadata writes go through `beaver-lib.sh` — this command does not write repository labels for `Status` / `Type` / `Size`, and it does not POST to the issue labels endpoint.
- Issue body files are unique per invocation — always created via `mktemp`.
