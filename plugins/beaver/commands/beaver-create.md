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

   ```bash
   # Create issue
   gh api repos/{org}/{issueRepo}/issues --method POST \
     -f title="{title}" \
     --raw-field body=@/tmp/beaver-issue-body.md \
     --jq '.number'

   # Add labels
   gh api repos/{org}/{issueRepo}/issues/{number}/labels --method POST \
     -f "labels[]=Control-By-Beaver" \
     -f "labels[]={type_label}" \
     -f "labels[]={size_label}" \
     -f "labels[]=status/triage"

   # Add to Project V2
   gh project item-add {project_number} --owner {org} --url {issue_url}

   # Set custom fields (Level, Status, Progress)
   gh project item-edit {item_id} --project-id {project_id} \
     --field-id {level_field_id} --single-select-option-id {level_option_id}
   ```

1. **Link to parent** (if Task or SubTask):

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues --method POST \
     -F sub_issue_id={child_issue_id}
   ```

1. **Initial status**: `status/triage` for all. Exception: p0/blocker Bug → `status/in-progress` + @CODEOWNERS (see Bug Submode).

1. **Report**: Print created Issue URL, labels, and next-step hint.

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
  - No Milestone required (G007 exempt)

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
