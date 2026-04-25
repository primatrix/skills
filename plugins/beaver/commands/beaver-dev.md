---
allowed-tools: Bash(gh api:*), Bash(gh issue:*), Bash(git:*), Bash(bash:*)
description: TDD development with subagent dispatch for a Beaver-tracked Size=S Issue. Trigger when the user wants to start coding, implement, or develop a claimed task.
argument-hint: "<issue-number>"
---

# /beaver-dev — TDD 开发

Phase 6 of the Beaver development lifecycle (development stage). **Only handles Size=S Issues.**

> Per RFC-0013 §6: Size=L Tasks are decomposed into Size=S SubTasks (via
> `/beaver-decompose`); each SubTask runs through `/beaver-dev` independently.

## Workflow

Argument is required: the issue number to develop (e.g. `120`).

### Phase 1: Preflight (field semantics)

Run the preflight check. The script reads **Project V2 fields** via
`beaver-lib.sh` (Size, Status) plus assignees and the current user — no
label literals are consulted.

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-dev.sh preflight {number}
```

Rejection conditions (script exits 1 with message):

- `Size != S` → "本命令仅处理 Size=S" (use `/beaver-decompose` first)
- `Status != In Progress` → `/beaver-claim` 已删除（见 RFC-0013 §3），请在 GitHub UI assign 自己后手动将 Status 切到 `In Progress`，再重新执行 `/beaver-dev`。
- current `gh` user is not in the Issue's assignees

On success the script prints `OK <type>` (e.g. `OK task`); capture the
type token for use in the worktree branch name.

Then fetch Issue body for objective + acceptance criteria + (optional) design
doc reference:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-dev.sh fetch-issue {org} {issueRepo} {number}
```

### Phase 2: Workspace Setup

Create an isolated worktree. Branch follows `<type>/<n>-<short_desc>`,
where `<type>` is the lower-cased Issue Type from preflight, `<n>` is the
issue number, and `<short_desc>` is a kebab-case slug derived from the
Issue title (max ~5 words).

```bash
BRANCH_NAME="<type>/<n>-<short_desc>"
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-dev.sh add-worktree "$BRANCH_NAME"
```

All subsequent editing/commits happen inside the worktree.

> The script does NOT touch the Status field. The Issue stays in
> `In Progress` for the entire `/beaver-dev` lifetime. If the user gets
> blocked on an external dependency, they manually flip Status to
> `Blocked` in the GitHub UI per RFC-0013 §6 step 5.

### Phase 3: Implementation Plan + User Confirmation

Generate a local implementation plan in markdown and print it to the
terminal. **方案 review 时用中文输出**（计划正文、说明、与用户确认的
对话均使用中文；文件路径、命令、代码片段保持原样）。For each work item
include:

- exact file paths to be created/modified
- the failing test snippet to write first
- minimal implementation idea
- verification command and expected output
- proposed commit message

Wait for the user to confirm or amend each item. Only proceed to Phase 4
after the user explicitly approves the full plan.

### Phase 4: TDD Subagent

Dispatch the `superpowers:test-driven-development` skill in a subagent
for each work item, with this context:

- worktree path + branch name
- work item: objective + acceptance criteria + planned files
- the implementation plan item agreed in Phase 3

**TDD Iron Law: NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST.**

Red-Green-Refactor cycle (the subagent enforces this):

1. **RED**: write one minimal failing test
1. **Verify RED**: run it, confirm it fails for the right reason
1. **GREEN**: write the simplest code that passes
1. **Verify GREEN**: rerun the test + adjacent suite, confirm no regressions
1. **REFACTOR**: clean up while keeping tests green
1. **Commit**: `git add` only the changed files, conventional `git commit -m`

Restart triggers:

- Code written before test → delete the code, start over
- Test passes immediately → it's testing the wrong thing, fix the test
- "Too simple to test" → simple code still breaks; the test takes 30 seconds
- "I'll test after" → tests-after prove nothing

### Phase 5: Debugging Fallback

If a test fails unexpectedly, dispatch the
`superpowers:systematic-debugging` skill in a subagent.

**Debugging Iron Law: NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST.**

Four phases the subagent follows:

1. **Root Cause Investigation**: read errors carefully, reproduce
   consistently, check recent changes, trace data flow
1. **Pattern Analysis**: find working examples, compare against
   references, identify differences
1. **Hypothesis Testing**: form one hypothesis, test minimally, verify
   before continuing
1. **Implementation**: create the failing-test reproducer, single fix,
   verify

If 3+ fixes fail: STOP, question the architecture, escalate to user.

For multiple **independent** failures in different subsystems, dispatch
debugging subagents in parallel (one per failure domain). Do NOT run
multiple implementation subagents in parallel (conflict risk).

### Phase 6: Two-Stage Code Review

When the developer claims "feature complete", dispatch the
`superpowers:requesting-code-review` skill in two passes per work item:

**Stage 1 — Spec compliance review**

- Does the implementation match each acceptance criterion?
- Anything missing? Anything extra (not requested)?
- If issues found: implementer subagent fixes them, re-review.

**Stage 2 — Code quality review** (only after Stage 1 ✅)

- Severity: Critical (must fix) / Important (fix before completion) /
  Minor (note for later)

Present each finding to the user; the user accepts or rejects with
reason. Accepted findings re-enter Phase 4.

### Phase 7: Verification Iron Law

**Verification Iron Law: NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE.**

Before declaring done:

1. Run the project's full test suite fresh (`make test` / `pnpm test` /
   the per-plugin test runner — whatever the repo uses).
1. Read the complete output, check the exit code, count failures.
1. Only claim "all tests pass" if the output shows **0 failures**.
1. Forbidden hedge words in the completion claim: "should", "probably",
   "seems to". Any of those means re-run and re-read.

Any single failure forces a return to Phase 5 (debugging).

### Phase 8: Completion + ask `/beaver-pr`

Print completion summary:

- worktree path + branch name
- commits added (with subjects)
- files changed
- full test-suite output with exit code 0

Then assert that the Issue's Status field is **unchanged** (still
`In Progress`); the script never wrote to Status, so this is a sanity
check, not a mutation.

Finally, ask the user:

> 是否直接 `/beaver-pr <n>`？(y/N)

(`<n>` here is the Issue number known from the command argument.)

- On `y`: invoke `/beaver-pr <n>` immediately, passing the Issue number.
- On `N` (or anything else): print the manual hint
  `Use \`/beaver-pr <n>\` to create a Draft PR.` and exit.

## Constraints

- TDD is mandatory; no production code without a failing test first.
- This command **only** handles Size=S; Size != S is rejected at preflight.
- Subagents are dispatched sequentially for implementation (parallel only
  for independent debugging).
- The command does **not** mutate the Project V2 Status field; Blocked
  transitions are user-driven in the GitHub UI.
- §7 QA loop does NOT apply (this is local development, not Issue
  content creation).
