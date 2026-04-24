---
allowed-tools: Bash(gh auth status:*), Bash(gh auth refresh:*), Bash(gh project edit:*), Bash(gh project field-create:*), Bash(gh project field-list:*), Bash(gh api:*), Bash(gh label create:*), Bash(date:*)
description: "Initialize Beaver on primatrix/projects#14: ensure native Issue Types (Bug/Task/SubTask), Status (7), Size (5), Priority (3), Progress, Iteration fields, and labels."
---

# Initialize Beaver Project

Idempotent initializer for the Beaver project at `primatrix` org, project #14. Updates custom fields, README, issue types, labels, and Iteration field on `primatrix/projects`.

## Constants

| Parameter | Value |
|-----------|-------|
| Organization | `primatrix` |
| Project number | `14` |
| Project URL | `https://github.com/orgs/primatrix/projects/14` |
| Issue repo | `primatrix/projects` |
| Observed repos | all |

## Context

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh auth-status
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh today
```

Verify token scopes include `project` and `admin:org`. If missing, prompt the user to run `gh auth refresh -h github.com -s project` and/or `gh auth refresh -h github.com -s admin:org`.

## Preview and Confirm

Show a full preview before executing. Include:

- All constants above
- Custom fields: Status (Single Select: 7 options — see below), Size (Single Select: XS, S, M, L, XL), Priority (Single Select: P0, P1, P2), Progress (Number: 0-100), Iteration (Iteration: monthly entries from current month to year-end)
- Native Issue Types: Bug, Task, SubTask
- README beaver-config block
- Labels to create
- Iteration field with N monthly entries (YYYY-MM ... YYYY-12)

Wait for explicit user confirmation. If changes requested, adjust and re-preview.

**Note on the legacy Project V2 field**: Earlier Beaver iterations created a single-select project field named `Level`. Per RFC-0013, that field is superseded by native GitHub Issue Types. This command does **not** create or read it; if it already exists on Project #14 it is left untouched (the field carries no semantic meaning for current Beaver commands).

## Execution

### Update Custom Fields

List existing fields once and reuse the result to decide which create/replace calls to issue:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh field-list primatrix 14
```

**Status:** find the Status field id from the JSON output and replace its options with the 7 RFC values:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh status-replace <existing_status_field_id>
```

The `status-replace` subcommand applies these 7 options exactly:
- Triage (GRAY) — Awaiting triage
- Ready to Claim (BLUE) — Added to Iteration, awaiting claim
- Design Pending (PURPLE) — Design review in progress (size/L)
- Ready to Develop (ORANGE) — Ready to code (size/L, design approved)
- In Progress (YELLOW) — Active development
- Blocked (RED) — Blocked
- Done (GREEN) — Completed and merged

If no Status field exists, create it first:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh field-create primatrix 14 "Status" SINGLE_SELECT "Triage,Ready to Claim,Design Pending,Ready to Develop,In Progress,Blocked,Done"
```

**Size:** if absent in the field-list output, create it. Skip if already present:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh field-create primatrix 14 "Size" SINGLE_SELECT "XS,S,M,L,XL"
```

The five options match the spec; the current Beaver command surface uses only `S` and `L`, but the field carries the full set so the spec stays addressable from Project #14 UI.

**Priority:** if absent, create it. Skip if already present:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh field-create primatrix 14 "Priority" SINGLE_SELECT "P0,P1,P2"
```

**Progress:** if absent, create it. Skip if already present:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh field-create primatrix 14 "Progress" NUMBER
```

For every field, skip creation when it already exists and inform the user.

### Write README with beaver-config

````
# Primatrix Projects

```yaml beaver-config
repositories: all
issueRepo: projects
customFields:
  status: Status
  size: Size
  priority: Priority
  progress: Progress
  iteration: Iteration
```
````

Claude renders the README content above to a temp file via `Write` (path `/tmp/beaver-project-readme.md`), then runs:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh edit-readme primatrix 14 /tmp/beaver-project-readme.md
```

### Create Native Issue Types

Issue types are org-scoped. Requires `admin:org` scope and `X-GitHub-Api-Version: 2026-03-10` header. List existing first to avoid duplicates:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh list-issue-types primatrix
```

Create each if not already present:

| Name | Color | Description |
|------|-------|-------------|
| Bug | red | Defect or regression |
| Task | green | Top-level deliverable (S or L) |
| SubTask | gray | Finest granularity work item |

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh create-issue-type primatrix <name> <color> <desc>
```

The `create-issue-type` subcommand swallows duplicate (422) errors so the loop is idempotent. Warn and continue on 404 (org plan may not support issue types).

**Definition vs assignment**: this `create-issue-type` path defines the org-level Type taxonomy. Per-issue Type assignment (e.g. setting `Type=Bug` on a specific issue) goes through `beaver-lib.sh::set_type`, never through this command.

### Create Labels

Create on the issue repository. For each row in the label tables below, run:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh ensure-label primatrix/projects <name> <color> <desc>
```

The `ensure-label` subcommand swallows duplicate-label errors so the loop is idempotent.

**Type labels (legacy parity, retained for repo-level coloring; Beaver commands no longer read these):**

| Label | Color | Description |
|-------|-------|-------------|
| type/feat | 0E8A16 | New feature |
| type/bug | D73A4A | Bug fix |
| type/refactor | E4E669 | Code refactoring |
| type/docs | 0075CA | Documentation |
| type/chore | BFD4F2 | Infrastructure, build, misc |

**Priority labels (legacy parity, retained for repo-level coloring; Beaver commands now read the Project V2 Priority field):**

| Label | Color | Description |
|-------|-------|-------------|
| p0/blocker | B60205 | Blocking — top of daily report |
| p1/urgent | D93F0B | Urgent — top of daily report |
| p2/high | FBCA04 | High priority |
| p3/normal | C2E0C6 | Normal priority |

**Size labels (legacy parity, retained for repo-level coloring; Beaver commands now read the Project V2 Size field):**

| Label | Color | Description |
|-------|-------|-------------|
| size/S | C5DEF5 | Small task — fast-track SOP |
| size/L | 1D76DB | Large task — full lifecycle SOP |

**Status labels (legacy parity, retained for repo-level coloring; Beaver commands now read the Project V2 Status field):**

| Label | Color | Description |
|-------|-------|-------------|
| status/triage | E4E669 | Awaiting triage |
| status/ready-to-claim | C2E0C6 | Added to Iteration, awaiting claim |
| status/design-pending | D4C5F9 | Design review in progress (size/L) |
| status/ready-to-develop | 0E8A16 | Ready to code (size/L, design approved) |
| status/in-progress | FBCA04 | Active development |
| status/blocked | B60205 | Blocked |
| status/done | 0E8A16 | Completed and merged |

**Beaver agent labels:**

| Label | Color | Description |
|-------|-------|-------------|
| beaver/needs-split | D93F0B | PR LOC exceeds 200 in core dirs |
| beaver/missing-test | D93F0B | No test evidence before done |
| beaver/missing-context | D93F0B | Incomplete labels or description |
| beaver/stale | E4E669 | Stuck in same status > 3 days |
| beaver/overdue | B60205 | Past DDL and not done |
| beaver/upstream-blocked | D93F0B | Upstream dependency blocked |
| beaver/wontfix | BFDADC | Will not fix |

**Control label:**

| Label | Color | Description |
|-------|-------|-------------|
| Control-By-Beaver | 7B61FF | Issue managed by Beaver automation |

### Create Iteration Field

Create the Iteration custom field on Project #14 and populate one entry per natural month from the current month through December of the current year.

**Field creation (skip if exists):**

First fetch the project node id:

```bash
PROJECT_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh project-id primatrix 14)
```

GraphQL ITERATION input requires the full iterationConfiguration object inline,
so build the iterations array (one entry per month from current month through
December of the current year) and embed it as a JSON literal inside the mutation.
Each entry: `{startDate: "YYYY-MM-01", duration: <days_in_month>, title: "YYYY-MM (MonthShort)"}` —
e.g. `{startDate: "2026-04-01", duration: 30, title: "2026-04 (Apr)"}`.

`ITERATIONS_JSON` is a JSON array of iteration entries, e.g.
`[{"startDate":"2026-04-01","duration":30,"title":"2026-04 (Apr)"}, ...]`.
Build it via `jq` from the computed month list rather than string concatenation.

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh iteration-field-create "$PROJECT_ID" <first_day_of_current_month> <days_in_current_month> "$ITERATIONS_JSON"
```

The `iteration-field-create` subcommand passes scalars as GraphQL variables (`-f` for strings, `-F` for the integer `duration`); the `$iterations` variable is declared as a GraphQL list type so `gh api` parses the JSON string correctly.

**Note:** The top-level `startDate` / `duration` in `iterationConfiguration` set the **default cadence** GitHub uses to auto-generate future iterations. The explicit `iterations` array fully defines the initial set — set top-level `startDate` to the first day of the current month and `duration` to that month's length so future auto-generated iterations align to month starts.

Build the `iterations` array by iterating from the current month to December of the current year. For each month: title = `YYYY-MM (MonthShort)` (e.g. `2026-04 (Apr)`), startDate = first of that month (`YYYY-MM-01`), duration = days in that month (28/29/30/31). Compute month lengths via `cal` or `date -v1d -v+1m -v-1d +%d` (BSD/macOS) / `date -d "$(date +%Y-%m-01) +1 month -1 day" +%d` (GNU/Linux) to handle February correctly.

If the Iteration field already exists, skip creation and instead read existing entries via:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh iteration-field-list primatrix 14
```

Compute which monthly entries are missing (compare titles to the expected `YYYY-MM (MonthShort)` set) and append only the missing ones via `iteration-field-append` (preserve existing entries).

Use `additions` to append iterations without disturbing existing ones. Pass only the entries whose titles are absent from the current configuration.

`ADDITIONS_JSON` contains only the entries whose titles are absent from the
current configuration, e.g. `[{"startDate":"2026-05-01","duration":31,"title":"2026-05 (May)"}]`.

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh iteration-field-append <iteration_field_id> "$ADDITIONS_JSON"
```

## Success Report

Print summary with: project URL (`https://github.com/orgs/primatrix/projects/14`), custom fields (Status with 7 options, Size with 5 options, Priority with 3 options, Progress, Iteration), native Issue Types (Bug / Task / SubTask), label count, Iteration entry count and range. Inform the user they can now use `beaver-create` against `primatrix/14`.

## Constraints

- Idempotent — safe to run multiple times; skips existing fields, labels, Iteration entries
- Organization: `primatrix` only
- Project: `#14` only (does not create new projects)
- Fixed field names: Status (7 options), Size (5 options: XS/S/M/L/XL), Priority (3 options: P0/P1/P2), Progress, Iteration
- Status field is a full replacement — only the 7 specified options will exist after update
- Iteration field — only missing months are appended; existing entries preserved
- Native Issue Types — Bug / Task / SubTask defined here; per-issue assignment goes through `beaver-lib.sh::set_type`
- Always confirm before executing — never execute without preview and approval
- On failure, report the error and let the user decide how to proceed (no auto-retry)
