---
allowed-tools: Bash(gh api:*), Bash(git:*)
description: TDD development with subagent dispatch for a Beaver-tracked Issue. Trigger when the user wants to start coding, implement, or develop a claimed task.
argument-hint: "<issue-number>"
---

# /beaver-dev — TDD 开发

Phase 3+ of the Beaver development lifecycle (development stage).

## Workflow

Argument is required: the issue number to develop.

### Phase 1: Load Context

1. Fetch Issue:

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{number} --jq '{number, title, body, labels: [.labels[].name]}'
   ```

1. Extract from Issue body:
   - Objective
   - Acceptance criteria
   - Design doc link (if referenced)

1. If size/L: fetch sub-issues list:

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{number}/sub_issues --jq '[.[] | {number, title, body, labels: [.labels[].name]}]'
   ```

### Phase 2: Guardrail Check

- G009: If size/L, verify:
  - Issue has `status/ready-to-develop` (or `status/in-progress` if already started)
  - Issue has at least one sub-issue
  - If checks fail: stop with message directing user to `/beaver-design` or `/beaver-decompose`

- If size/S: verify Issue has `status/in-progress`
  - If not: stop with message directing user to `/beaver-claim`

### Phase 3: Workspace Setup

1. Create git worktree for isolation:

   ```bash
   BRANCH_NAME="{type}/{issue_number}-{short_desc}"
   git worktree add .claude/worktrees/${BRANCH_NAME} -b ${BRANCH_NAME}
   ```

1. Transition size/L issues from `status/ready-to-develop` to `status/in-progress` (first-time only):

   ```bash
   # Only if current status is ready-to-develop
   gh api repos/{org}/{issueRepo}/issues/{number}/labels/status%2Fready-to-develop --method DELETE
   gh api repos/{org}/{issueRepo}/issues/{number}/labels --method POST \
     -f "labels[]=status/in-progress"
   ```

### Phase 4: Subagent-Driven Development

For each work unit (SubTask if size/L, or the task itself if size/S):

#### 4.1 Dispatch TDD Subagent

Dispatch a subagent with the following context:

- Work unit: title, objective, acceptance criteria
- Codebase location (worktree path)
- TDD discipline (absorbed from superpowers:test-driven-development):

**TDD Iron Law: NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST.**

Red-Green-Refactor cycle:

1. **RED**: Write one minimal failing test showing desired behavior
1. **Verify RED**: Run test, confirm it fails for the right reason (missing feature, not typo)
1. **GREEN**: Write simplest code to pass the test
1. **Verify GREEN**: Run test, confirm pass. Run all tests, confirm no regressions.
1. **REFACTOR**: Clean up (remove duplication, improve names). Keep tests green.
1. **Commit**: `git add` changed files, `git commit -m "{message}"`

Red flags that mean STOP and restart:

- Code written before test → delete code, start over
- Test passes immediately → testing wrong thing, fix test
- "Too simple to test" → simple code breaks, test takes 30 seconds
- "I'll test after" → tests-after prove nothing

#### 4.2 On Failure: Dispatch Debugging Subagent

If tests fail unexpectedly during implementation, dispatch a debugging subagent with systematic debugging discipline (absorbed from superpowers:systematic-debugging):

**Debugging Iron Law: NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST.**

Four phases:

1. **Root Cause Investigation**: Read errors carefully, reproduce consistently, check recent changes, trace data flow
1. **Pattern Analysis**: Find working examples, compare against references, identify differences
1. **Hypothesis Testing**: Form single hypothesis, test minimally, verify before continuing
1. **Implementation**: Create failing test, single fix, verify

If 3+ fixes fail: STOP. Question the architecture. Escalate to user.

#### 4.3 After Implementation: Dispatch Code Review Subagent

Two-stage review per work unit:

**Stage 1 — Spec Compliance Review**:

- Does the implementation match the acceptance criteria?
- Is anything missing? Is anything extra (not requested)?
- If issues found: implementer subagent fixes, re-review

**Stage 2 — Code Quality Review**:

- Issue severity: Critical (must fix) / Important (fix before proceeding) / Minor (note for later)
- Only after spec compliance is ✅

#### 4.4 Parallel Agents for Independent Failures

If multiple independent test failures occur in different subsystems, dispatch debugging subagents in parallel (one per failure domain). Do NOT dispatch multiple implementation subagents in parallel (conflict risk).

### Phase 5: Verification

Before claiming completion (absorbed from superpowers:verification-before-completion):

**Verification Iron Law: NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE.**

1. Run the full test suite fresh
1. Read complete output, check exit code, count failures
1. Only claim "all tests pass" if output shows 0 failures
1. Forbidden words: "should", "probably", "seems to"

### Phase 6: Report

Print completion status:

- Tests passing (with evidence)
- Files changed
- Commits made
- Next-step hint: "Use `/beaver-pr {number}` to create a Draft PR."

## Constraints

- TDD is mandatory, not optional. No exceptions without user's explicit permission.
- Subagents get fresh context (no session history leakage)
- Subagents are dispatched sequentially (no parallel implementation)
- Debugging subagents may be dispatched in parallel for independent failures
- §7 QA loop does NOT apply (this is local development, not Issue content creation)
