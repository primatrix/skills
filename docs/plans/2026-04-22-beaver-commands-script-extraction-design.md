# Beaver Commands — Bash Extraction Design

**Date:** 2026-04-22
**Status:** Approved
**Scope:** `plugins/beaver/commands/*.md` and new `plugins/beaver/scripts/`

## Problem

The 9 files in `plugins/beaver/commands/` mix two concerns:

1. **Workflow logic** — what Claude must reason about (validation rules, QA flow, status transitions, HARD-GATE, templates, constraints).
2. **Mechanical bash plumbing** — `gh api` calls, GraphQL heredocs, jq filter chains, label CRUD.

Embedded bash blocks (some 50+ lines of GraphQL) inflate the command files (1714 total lines) and pollute Claude's context on every command invocation. Claude does not need to re-parse `gh api graphql -f query='...'` invocations to make decisions — the calls only need to *run*.

## Goal

Move all `gh api` / `git` / `gh project` / GraphQL bash blocks out of `commands/*.md` into `plugins/beaver/scripts/*.sh`. Commands keep only workflow logic and short invocation lines.

## Layout

```
plugins/beaver/
  commands/                  # workflow only, with bash invocation lines
    beaver-claim.md
    beaver-create.md
    beaver-decompose.md
    beaver-design.md
    beaver-dev.md
    beaver-focus.md
    beaver-pr.md
    beaver-setup.md
    beaver-tracker.md
  scripts/                   # one script per command, subcommands inside
    beaver-claim.sh
    beaver-create.sh
    beaver-decompose.sh
    beaver-design.sh
    beaver-dev.sh
    beaver-focus.sh
    beaver-pr.sh
    beaver-setup.sh
    beaver-tracker.sh
  skills/beaver-engine/SKILL.md   # unchanged
```

No shared `lib/` for now (YAGNI — extract only after 3+ duplications surface).

## Script conventions

- Shebang: `#!/usr/bin/env bash`
- Strict mode: `set -euo pipefail`
- Dispatch: `case "$1" in ...; esac` on the first arg (subcommand)
- Output: machine-readable on stdout (raw value, JSON, or `key=value`); human messages to stderr
- Errors: non-zero exit + stderr message
- No prompts: scripts never call `read`. Args are positional or env vars
- Help: bare `--help` lists subcommands
- Body templates: scripts accept body content as a file path (`-F body=@$BODY_FILE`); Claude renders templates and writes the temp file

## Subcommand inventory

### `beaver-claim.sh`
- `fetch <number>` — get issue summary (number, title, state, labels, assignees)
- `assign <number>` — assign current `gh` user
- `swap-status <number> <from-label> <to-label>` — atomic label swap

### `beaver-create.sh`
- `create-issue <title> <body-file>` — POST issue, echo `.number`
- `fetch-ids <number>` — echo `id`, `node_id`, `html_url` as `key=value`
- `add-labels <number> <label> [<label> ...]` — POST labels
- `add-to-project <issue-url>` — `gh project item-add`, echo item id
- `set-field <item-id> <field-id> <option-id>` — single-select field edit
- `resolve-iteration <yyyymm>` — echo `project_id`, `field_id`, `iteration_id`
- `set-iteration <project-id> <item-id> <field-id> <iteration-id>` — GraphQL mutation
- `link-parent <parent-number> <child-id>` — sub_issues POST

### `beaver-decompose.sh`
- `fetch-parent <number>` — number, title, body, labels
- `list-sub-issues <number>` — array of titles
- `create-child <title> <body-file>` — POST, echo `.number` and `.id`
- `attach <parent-number> <child-id>` — sub_issues POST
- `add-to-project <child-url>` — same as beaver-create

### `beaver-design.sh`
- `prepare-wiki` — clone or pull `~/Code/wiki`, checkout main
- `next-rfc-num` — read `docs/rfc/index.md`, echo next NNNN
- `create-branch <branch>` — `git checkout -b`
- `commit-push <file> <message> <branch>` — add+commit+push
- `create-pr <title> <body>` — `gh pr create --draft` against wiki
- `comment-issue <number> <body>` — POST comment to original repo

### `beaver-dev.sh`
- `fetch-issue <number>` — number, title, body, labels
- `fetch-sub-issues <number>` — list with labels
- `swap-to-in-progress <number>` — DELETE design-pending + POST in-progress
- `add-worktree <branch>` — `git worktree add .claude/worktrees/<branch> -b <branch>`

### `beaver-focus.sh`
- `whoami` — echo current `gh` user
- `fetch-my-issues` — GraphQL on Project #14, filtered to current user OPEN issues with labels + iteration
- `fetch-review-prs` — search/issues query for review-requested

### `beaver-pr.sh`
- `ctx` — git status / diff stat / current branch / log -10 (parallel-safe single call)
- `create-branch <branch>` — checkout -b (idempotent)
- `commit-push <files> <message> <branch>` — stage + commit + push -u
- `check-tests` — `git diff --name-only origin/main...HEAD | grep -E '(test_|_test\.|/tests/)'`
- `check-labels <number>` — list labels on issue
- `add-warn-label <number> <label>` — add `beaver/missing-test` etc
- `create-pr <title> <body-file>` — `gh pr create --draft`
- `mark-ready <pr-number>` — `gh pr ready`

### `beaver-setup.sh`
- `auth-status` — `gh auth status`
- `today` — `date +%Y-%m-%d`
- `ensure-field <name> <type> [options]` — idempotent field-create, skip if exists
- `update-status-options <field-id>` — GraphQL replace 7 status options
- `write-readme <body-file>` — `gh project edit --readme`
- `list-issue-types` — `gh api orgs/.../issue-types`
- `create-issue-type <name> <color> <desc>` — POST issue-type, skip on 422
- `ensure-label <name> <color> <desc>` — `gh label create`, skip on dup
- `ensure-iteration-field <iterations-json>` — create or append-missing

### `beaver-tracker.sh`
- `ensure-tracker-labels <repo> <yyyymm> <prev-yyyymm>` — idempotent label-create
- `find-tracker <repo> <yyyymm>` — search/issues, echo count + numbers
- `list-carried <prev-number>` — sub_issues filtered to state=open
- `create <repo> <yyyymm> <body-file>` — POST issue, label, echo number
- `attach-sub <tracker-number> <issue-number>` — resolve issue id, POST sub_issues
- `fetch-triage-backlog` — GraphQL filtered to status/triage + no iteration
- `resolve-iteration <yyyymm>` — same as beaver-create
- `resolve-item-id <issue-number>` — find ProjectV2Item id for project #14
- `add-to-project <issue-number>` — addProjectV2ItemById, echo new item id
- `set-iteration <project-id> <item-id> <field-id> <iteration-id>` — same mutation
- `list-tracker-subs <tracker-number>` — `.[].number`

## Command file pattern

Each command keeps:
- Frontmatter (`allowed-tools`, `description`, `argument-hint`)
- Workflow narrative (phases, validation rules, decision tables)
- Templates (issue body markdown — Claude fills placeholders, writes temp file, passes path)
- Constraints
- Invocation lines:
  ```bash
  bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-claim.sh fetch "$NUMBER"
  ```

Removed:
- All raw `gh api` flag soup
- All GraphQL heredocs
- All jq filter chains
- All `cat > /tmp/...` patterns (Claude still uses `Write` to create body files)

## What does NOT change

- `skills/beaver-engine/SKILL.md` — unchanged
- `.claude-plugin/plugin.json` — unchanged
- `.claude-plugin/marketplace.json` — unchanged
- Behavior of any command — pure mechanical extraction

## Path resolution

Scripts referenced as `${CLAUDE_PLUGIN_ROOT}/scripts/<name>.sh`. Claude Code injects `CLAUDE_PLUGIN_ROOT` for plugin commands.

## Allowed-tools

Existing `allowed-tools` lines preserved. The underlying `gh api` calls still happen — just inside `bash` invocation. The current declarations (`Bash(gh api:*)` etc) remain valid since the bash subprocess invokes those tools.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Argument escaping for titles with quotes/newlines | Bodies always passed as file paths, never inline strings |
| BSD vs GNU `date` differences | Existing dual-form logic preserved verbatim inside `beaver-tracker.sh` |
| Silent behavior regression | Every flag and header preserved during move; manual diff review |
| Subcommand name churn | Inventory above is the contract; locked before writing |

## Verification

No build/test infra exists. Verify by:

1. `bash -n scripts/*.sh` — syntax check on every script
2. `bash scripts/<name>.sh --help` — confirm subcommand list
3. `python -c 'import json; json.load(open("plugins/beaver/.claude-plugin/plugin.json"))'` — JSON still valid (no changes expected)
4. `python -c 'import json; json.load(open(".claude-plugin/marketplace.json"))'` — JSON still valid
5. Manual diff review of each command file: workflow narrative preserved verbatim; only bash blocks replaced

## Out of scope

- Adding tests for scripts (no test infra, would be net-new complexity)
- Extracting `lib/gh-helpers.sh` (YAGNI; revisit if 3+ duplications appear)
- Behavior changes (this is a pure refactor)
- Touching the `beaver-engine` skill
