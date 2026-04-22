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

   The POST response body contains the issue body Markdown with embedded newlines. Capturing the whole JSON into a shell variable and then re-parsing with `jq` corrupts those control chars and yields empty fields. Always extract each field via `--jq` directly on the `gh api` call — never round-trip through a captured variable. To avoid losing the issue id when only `--jq '.number'` is requested, perform the POST once for `.number`, then a single follow-up GET for `.id` / `.node_id` / `.html_url`:

   ```bash
   # Create issue (extract .number directly via --jq; do NOT capture the full JSON body)
   NEW_NUM=$(gh api repos/{org}/{issueRepo}/issues --method POST \
     -f title="{title}" \
     -F body=@/tmp/beaver-issue-body.md \
     --jq '.number')

   # Re-fetch to obtain remaining ids (cheap GET, idempotent)
   NEW_ID=$(gh api repos/{org}/{issueRepo}/issues/$NEW_NUM --jq '.id')
   NEW_NODE_ID=$(gh api repos/{org}/{issueRepo}/issues/$NEW_NUM --jq '.node_id')
   NEW_URL=$(gh api repos/{org}/{issueRepo}/issues/$NEW_NUM --jq '.html_url')

   # Add labels
   gh api repos/{org}/{issueRepo}/issues/$NEW_NUM/labels --method POST \
     -f "labels[]=Control-By-Beaver" \
     -f "labels[]={type_label}" \
     -f "labels[]={size_label}" \
     -f "labels[]=status/triage"

   # Add to Project V2 (capture the project item id for the field-edit step)
   ITEM_ID=$(gh project item-add {project_number} --owner {org} --url "$NEW_URL" \
     --format json --jq '.id')

   # Set custom fields (Level, Status, Progress)
   gh project item-edit --id "$ITEM_ID" --project-id {project_id} \
     --field-id {level_field_id} --single-select-option-id {level_option_id}
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
   ITERATION_ID=$(echo "$ITERATION_INFO" | jq -r --arg yyyymm "$target_yyyymm" \
     '.data.organization.projectV2.field.configuration.iterations
      | map(select(.title | startswith($yyyymm))) | .[0].id')
   ```

   If `ITERATION_ID` is `null` or empty, warn (do NOT abort the whole command — the issue already exists):

   ```
   Iteration entry for <target_yyyymm> not found on Project #14.
   Run /beaver-setup to extend iterations into <target_yyyymm>.
   Iteration assignment skipped — assign manually or re-run after setup.
   ```

   Otherwise apply the mutation against the project item created above:

   ```bash
   read -r -d '' SET_ITERATION_MUTATION <<'GRAPHQL'
   mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
     updateProjectV2ItemFieldValue(input: {
       projectId: $projectId
       itemId: $itemId
       fieldId: $fieldId
       value: { iterationId: $iterationId }
     }) { projectV2Item { id } }
   }
   GRAPHQL

   gh api graphql \
     -f query="$SET_ITERATION_MUTATION" \
     -f projectId="$PROJECT_ID" \
     -f itemId="$ITEM_ID" \
     -f fieldId="$ITERATION_FIELD_ID" \
     -f iterationId="$ITERATION_ID"
   ```

   On success, the issue gains the Iteration assignment and (per engine §G007) becomes eligible for `status/ready-to-claim`. This step does NOT auto-transition the status label — that remains `/beaver-claim`'s job.

1. **Link to parent** (if Task or SubTask):

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues --method POST \
     -H "X-GitHub-Api-Version: 2026-03-10" \
     -F sub_issue_id=$NEW_ID
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
