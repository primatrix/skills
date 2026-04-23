---
allowed-tools: Bash(gh api:*), Bash(git:*)
description: TDD development for a Beaver-tracked Issue. Trigger when the user wants to start coding, implement, or develop a claimed Size=S task.
argument-hint: "<issue-number>"
---

# /beaver-dev — TDD 开发

Phase 3+ of the Beaver development lifecycle.

Only `Size=S` issues are supported by this command.

## Workflow

Argument is required: the issue number to develop.

### Phase 1: Load Context

1. Fetch the Issue together with Beaver Project V2 context. `beaver-dev.sh` must load the Project item fields through `beaver-lib.sh`, not by reading Issue labels directly.

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-dev.sh fetch-issue {org} {issueRepo} {number}
   ```

1. Extract and print:
   - Issue title and objective
   - Acceptance criteria
   - Issue type (used for branch naming)
   - Project V2 fields: `Size`, `Status`
   - Current assignee
   - Design doc link, if present

### Phase 2: Guardrail Check

Reject immediately unless all of the following are true:

1. `Size = S`
1. `Status = In Progress`
1. Assignee is the current GitHub user

Failure handling:

- If `Size != S`: stop and print `本命令仅处理 Size=S`
- If `Status != In Progress`: stop and tell the user to return the Issue to `In Progress` before using `/beaver-dev`
- If assignee is not the current user: stop and tell the user to claim the Issue first

Do not change the Project `Status` field in this command. If the work becomes blocked, the user must update it manually in the GitHub UI.

### Phase 3: Workspace Setup

1. Create an isolated worktree with branch name `<type>/<n>-<short_desc>`:

   ```bash
   BRANCH_NAME="<type>/{issue_number}-{short_desc}"
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-dev.sh add-worktree "$BRANCH_NAME"
   ```

1. Enter the worktree and keep all coding there. The main worktree must remain clean.

1. Generate an implementation plan before editing code. The plan should include:
   - Which acceptance criteria will be addressed
   - Expected test coverage
   - Files likely to change
   - Verification commands to run
   - Any known risks or open questions

1. Show the plan to the user and require confirmation before starting implementation. If the user requests adjustments, revise the plan first.

### Phase 4: Subagent-Driven Development

Use the Beaver issue context, acceptance criteria, and worktree path as shared inputs for all subagents. Run implementation sequentially inside the same worktree.

#### 4.1 TDD Implementer

Dispatch the `test-driven-development` superpower first.

Required discipline:

- No production code before a failing test
- Follow Red → Green → Refactor strictly
- After each Green step, rerun the relevant tests
- Keep the implementation minimal and acceptance-criteria driven

#### 4.2 Debugging Fallback

If implementation hits an unexpected failure, dispatch the `systematic-debugging` superpower before attempting more fixes.

Required discipline:

- Investigate root cause first
- Compare against known-good behavior or references
- Test one hypothesis at a time
- Return to TDD once the root cause is confirmed

#### 4.3 Code Review Gate

After implementation is complete, dispatch the `requesting-code-review` superpower.

Review must happen in two stages:

1. Spec compliance review
1. Code quality review

Fix all Critical and Important findings before proceeding to final verification.

### Phase 5: Verification

Before claiming completion, enforce the Verification Iron Law:

**NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE.**

1. Run the project's full test suite from the worktree
1. Read the full output and exit code
1. Only enter the completion branch if the result is `0 failures`
1. If any test fails, return to TDD or systematic debugging

Do not use tentative wording such as `should`, `probably`, or `seems to` when reporting verification status.

### Phase 6: Completion Branch

1. Re-read the Project V2 item through `beaver-lib.sh` and assert that `Status` is still `In Progress`
1. Print:
   - Test results with concrete evidence
   - Files changed
   - Worktree path
1. Ask the user exactly:

   ```text
   是否直接 /beaver-pr {number}？(y/N)
   ```

1. If the user answers `y`, invoke `/beaver-pr {number}` directly
1. Otherwise, print a manual next step hint:

   ```text
   Use /beaver-pr {number} when you are ready to open the Draft PR.
   ```

## Constraints

- This command only handles `Size=S`
- Project V2 field reads must come from `beaver-lib.sh`
- `Status` must remain `In Progress` throughout the command
- Blocked transitions are manual in the GitHub UI
- TDD is mandatory
- The full test suite must pass with `0 failures` before completion
