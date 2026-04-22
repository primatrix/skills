---
allowed-tools: Bash(gh api:*), Bash(gh project:*)
description: Decompose a Beaver Goal into Tasks (size/L) or a Task into SubTasks (size/S), guided by a design doc. Trigger when the user wants to split, breakdown, or decompose an issue into sub-issues.
argument-hint: "<issue-number> --design-doc <url-or-path>"
---

# /beaver-decompose — 任务拆解

Phase 5 of the Beaver development lifecycle (size/L only).

## Workflow

Arguments required: parent issue number AND design doc reference (`--design-doc <url-or-path>`).

### Phase 1: Load & Validate

1. Parse arguments: extract issue number and design doc reference.

1. Fetch parent Issue:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh fetch-parent {org} {issueRepo} {number}
   ```

1. Validate:
   - Issue type is Goal or Task (check issue type or Level field)
   - Issue has `status/ready-to-develop` (confirms Design Doc PR has been merged)
   - Fail with clear message if checks fail

1. Fetch existing sub-issues to avoid duplication:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh list-sub-titles {org} {issueRepo} {number}
   ```

### Phase 2: Read Design Doc

Support three reference formats:

- **GitHub PR URL** (`https://github.com/.../pull/N`): fetch files from PR
- **GitHub blob URL** (`https://github.com/.../blob/...`): fetch file content via API
- **Local path**: read directly

Extract the design doc content for decomposition analysis.

### Phase 3: Draft Decomposition

1. Analyze design doc to identify logical work units:
   - Goal → Task children (each gets `size/L`, will need own design doc)
   - Task → SubTask children (each gets `size/S`, should be independently deliverable)

1. For each child, draft:
   - Title (imperative, specific)
   - Objective (one sentence)
   - Acceptance criteria (≥ 2 verifiable items)
   - Estimated scope

1. Skip any scope already covered by existing sub-issues.

### Phase 4: Per-Child QA Confirm

Engine §7 applies. For each proposed child issue:

1. Present the child with title, objective, acceptance criteria
1. User can: accept / edit / delete / insert (add new child before this one)
1. One child per turn, wait for approval per §7.5

### Phase 5: Create Sub-Issues

For each approved child, sequentially:

1. Render the body template (shown below) via Write tool to a temp file (e.g. `/tmp/beaver-sub-issue.md`), then create the Issue:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh create-child {org} {issueRepo} "{title}" /tmp/beaver-sub-issue.md
   ```

   Body template:

   ```markdown
   ## 目标
   {objective}

   ## 验收标准
   {acceptance_criteria}

   <!-- beaver-tracking
   type: {parent_type}
   size: {child_size}
   parent: #{parent_number}
   created-by: beaver-decompose
   -->
   ```

1. Add labels:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh add-labels {org} {issueRepo} {child_number} Control-By-Beaver {type_label} {size_label} status/triage
   ```

1. Link to parent:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh link-parent {org} {issueRepo} {parent_number} {child_issue_id}
   ```

1. Add to Project V2:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh add-to-project {project_number} {org} {child_url}
   ```

1. For Goal → Task children only: set initial status to `status/triage`. Note: these children are size/L and will need to be added to an Iteration (-> ready-to-claim) and claimed (-> design-pending) before design work begins. Do NOT auto-transition to design-pending directly — that would skip the ready-to-claim state.

### Phase 6: Auto-Audit

After all children created, automatically run audit checks:

1. **Coverage**: Compare parent objective + acceptance criteria against combined child scope. Flag gaps.
1. **Atomicity**: Estimate whether each child fits within reasonable scope (SubTasks should be independently deliverable).
1. **Test Definition**: Check each child body for testable acceptance criteria.

Generate audit report table:

| Child | Coverage | Atomicity | Tests | Status |
|---|---|---|---|---|
| #N title | ✅/⚠️ | ✅/⚠️ | ✅/⚠️ | PASS/WARN |

Apply labels for failures:

- `beaver/missing-test` if no testable criteria
- `beaver/needs-split` if scope too large
- `beaver/missing-context` if incomplete

Post audit summary as comment on parent Issue.

### Phase 7: Report

Print summary:

- Number of children created
- Audit results
- Next-step hint: "Children are in `status/triage`. Add them to an Iteration, then team members can claim with `/beaver-claim <number>`."

## Constraints

- Engine §7 QA applies to Phase 4 (per-child confirmation)
- Engine §7.2 HARD-GATE applies to Phase 5 (issue creation)
- Use `-F sub_issue_id=` (not `-f`) for integer fields in sub_issues API
- Use `--template '{{.html_url}}'` instead of `--jq` when quoting issues arise
