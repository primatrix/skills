---
allowed-tools: Bash(gh api:*), Bash(gh project:*), Bash(cat > /tmp/*)
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
   gh api repos/{org}/{issueRepo}/issues/{number} --jq '{number, title, body, labels: [.labels[].name]}'
   ```

1. Validate:
   - Issue type is Goal or Task (check issue type or Level field)
   - Issue has `status/ready-to-develop` (confirms Design Doc PR has been merged)
   - Fail with clear message if checks fail

1. Fetch existing sub-issues to avoid duplication:

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{number}/sub_issues --jq '.[].title'
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

1. Write Issue body to temp file:

   ```bash
   cat > /tmp/beaver-sub-issue.md << 'BODY'
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
   BODY
   ```

1. Create Issue:

   ```bash
   gh api repos/{org}/{issueRepo}/issues --method POST \
     -f title="{title}" \
     -F body=@/tmp/beaver-sub-issue.md \
     --jq '.number'
   ```

1. Add labels:

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{child_number}/labels --method POST \
     -f "labels[]=Control-By-Beaver" \
     -f "labels[]={type_label}" \
     -f "labels[]={size_label}" \
     -f "labels[]=status/triage"
   ```

1. Link to parent:

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues --method POST \
     -F sub_issue_id={child_issue_id}
   ```

1. Add to Project V2:

   ```bash
   gh project item-add {project_number} --owner {org} --url {child_url}
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
