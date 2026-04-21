---
allowed-tools: Bash(gh auth status:*), Bash(gh auth refresh:*), Bash(gh project edit:*), Bash(gh project field-create:*), Bash(gh project field-list:*), Bash(gh api:*), Bash(gh label create:*), Bash(cat > /tmp/*), Bash(date:*)
description: "Initialize Beaver on primatrix/projects#14: update Status field to 7 options, create labels, and create remaining-year weekly milestones."
---

# Initialize Beaver Project

Idempotent initializer for the Beaver project at `primatrix` org, project #14. Updates custom fields, README, issue types, labels, and milestones on `primatrix/projects`.

## Constants

| Parameter | Value |
|-----------|-------|
| Organization | `primatrix` |
| Project number | `14` |
| Project URL | `https://github.com/orgs/primatrix/projects/14` |
| Issue repo | `primatrix/projects` |
| Observed repos | all |

## Context

! gh auth status
! date +%Y-%m-%d

Verify token scopes include `project` and `admin:org`. If missing, prompt the user to run `gh auth refresh -h github.com -s project` and/or `gh auth refresh -h github.com -s admin:org`.

## Preview and Confirm

Show a full preview before executing. Include:

- All constants above
- Custom fields: Level (Single Select: Goal, Task, SubTask), Status (Single Select: 7 options — see below), Progress (Number: 0-100)
- README beaver-config block
- Issue types, labels, milestones to create
- Number of remaining-year weekly milestones to create

Wait for explicit user confirmation. If changes requested, adjust and re-preview.

## Execution

### Update Custom Fields

**Level:**
```bash
gh project field-create 14 --owner primatrix --name "Level" --data-type SINGLE_SELECT --single-select-options "Goal,Task,SubTask"
```

Skip if field already exists.

**Status:** List fields with `gh project field-list 14 --owner primatrix --format json`, find the Status field ID, then update via GraphQL to exactly 7 options. This is a full replacement — any pre-existing options not in this list are removed.

```bash
gh api graphql -f query='
mutation {
  updateProjectV2Field(input: {
    fieldId: "{existing_status_field_id}"
    singleSelectOptions: [
      {name: "Triage", color: GRAY, description: "Awaiting triage"},
      {name: "Ready to Claim", color: BLUE, description: "Added to Milestone, awaiting claim"},
      {name: "Design Pending", color: PURPLE, description: "Design review in progress (size/L)"},
      {name: "Ready to Develop", color: ORANGE, description: "Ready to code (size/L, design approved)"},
      {name: "In Progress", color: YELLOW, description: "Active development"},
      {name: "Blocked", color: RED, description: "Blocked"},
      {name: "Done", color: GREEN, description: "Completed and merged"}
    ]
  }) {
    projectV2Field {
      ... on ProjectV2SingleSelectField {
        name
        options {
          name
        }
      }
    }
  }
}'
```

If no Status field exists, create it:
```bash
gh project field-create 14 --owner primatrix --name "Status" --data-type SINGLE_SELECT --single-select-options "Triage,Ready to Claim,Design Pending,Ready to Develop,In Progress,Blocked,Done"
```

**Progress:**
```bash
gh project field-create 14 --owner primatrix --name "Progress" --data-type NUMBER
```

Skip any field that already exists and inform the user.

### Write README with beaver-config

````
# Primatrix Projects

```yaml beaver-config
repositories: all
issueRepo: projects
customFields:
  level: Level
  progress: Progress
  status: Status
```
````

```bash
cat > /tmp/beaver-project-readme.md << 'BEAVEREOF'
# Primatrix Projects

```yaml beaver-config
repositories: all
issueRepo: projects
customFields:
  level: Level
  progress: Progress
  status: Status
```
BEAVEREOF
```

```bash
gh project edit 14 --owner primatrix --readme "$(cat /tmp/beaver-project-readme.md)"
```

### Create Issue Types

Issue types are org-scoped. Requires `admin:org` scope and `X-GitHub-Api-Version: 2026-03-10` header. List existing first to avoid duplicates:

```bash
gh api orgs/primatrix/issue-types -H "X-GitHub-Api-Version: 2026-03-10" --jq '.[].name'
```

Create each if not already present:

| Name | Color | Description |
|------|-------|-------------|
| Goal | blue | High-level objective |
| Task | green | Breakdown of a Goal |
| SubTask | gray | Finest granularity work item |

```bash
gh api orgs/primatrix/issue-types --method POST -H "X-GitHub-Api-Version: 2026-03-10" -f name="{name}" -f color="{color}" -f description="{desc}" -F is_enabled=true
```

Skip on 422 (already exists). Warn and continue on 404 (org plan may not support issue types).

### Create Labels

Create on the issue repository. Skip any that already exist (on 422 error, continue to next label).

**Type labels:**

| Label | Color | Description |
|-------|-------|-------------|
| type/feat | 0E8A16 | New feature |
| type/bug | D73A4A | Bug fix |
| type/refactor | E4E669 | Code refactoring |
| type/docs | 0075CA | Documentation |
| type/chore | BFD4F2 | Infrastructure, build, misc |

**Priority labels:**

| Label | Color | Description |
|-------|-------|-------------|
| p0/blocker | B60205 | Blocking — top of daily report |
| p1/urgent | D93F0B | Urgent — top of daily report |
| p2/high | FBCA04 | High priority |
| p3/normal | C2E0C6 | Normal priority |

**Size labels:**

| Label | Color | Description |
|-------|-------|-------------|
| size/S | C5DEF5 | Small task — fast-track SOP |
| size/L | 1D76DB | Large task — full lifecycle SOP |

**Status labels:**

| Label | Color | Description |
|-------|-------|-------------|
| status/triage | E4E669 | Awaiting triage |
| status/ready-to-claim | C2E0C6 | Added to Milestone, awaiting claim |
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

```bash
gh label create "{label}" --repo primatrix/projects --color "{color}" --description "{desc}"
```

Create all labels in sequence. Skip on error (label already exists).

### Create Milestones

Calculate weekly milestones for the remaining weeks of the current year. Each milestone represents one ISO week (Monday to Sunday).

- **Start:** Monday of the current ISO week
- **End:** Sunday of the ISO week containing December 31
- **Title format:** `Week {iso_week_number} (Mon DD - Mon DD)` where dates use short month names (e.g. `Apr 21`)
- **due_on:** Sunday of that week, `T23:59:59Z`

```bash
gh api repos/primatrix/projects/milestones --method POST -f title="Week {n} ({start} - {end})" -f due_on="{end_iso}T23:59:59Z" -f state="open"
```

Skip on 422 (title already exists).

## Success Report

Print summary with: project URL (`https://github.com/orgs/primatrix/projects/14`), custom fields (Level, Status with 7 options, Progress), issue types, label count, milestone count and range. Inform the user they can now use `beaver-issue` with project identifier `primatrix/14`.

## Constraints

- Idempotent — safe to run multiple times; skips existing fields, labels, milestones
- Organization: `primatrix` only
- Project: `#14` only (does not create new projects)
- Fixed field names: Level, Status (7 options), Progress
- Status field is a full replacement — only the 7 specified options will exist after update
- Always confirm before executing — never execute without preview and approval
- On failure, report the error and let the user decide how to proceed (no auto-retry)
