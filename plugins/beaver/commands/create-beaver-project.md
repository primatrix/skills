---
allowed-tools: Bash(gh auth status:*), Bash(gh auth refresh:*), Bash(gh repo list:*), Bash(gh project create:*), Bash(gh project edit:*), Bash(gh project field-create:*), Bash(gh project field-list:*), Bash(gh api:*), Bash(gh label create:*), Bash(cat > /tmp/*), Bash(date:*)
description: Create a GitHub Project V2 with Beaver-required custom fields, README config, issue types, labels, and milestones
---

# Create Beaver Project

Create a GitHub Project V2 with Beaver-required custom fields (Level, Status, Progress), a README containing `beaver-config`, and initialize the issue repository with Issue Types, Labels, and Milestones.

## Context

! gh auth status
! date +%Y-%m-%d

Verify token scopes include `project` and `admin:org`. If missing, prompt the user to run `gh auth refresh -h github.com -s project` and/or `gh auth refresh -h github.com -s admin:org`.

## Workflow

Collect from the user, one at a time:
1. **Organization** — verify access with `gh repo list {org} --json name --limit 1`
2. **Project title**
3. **Repositories to observe** — list with `gh repo list {org} --json name --limit 100 --jq '.[].name'`, accept comma-separated names or "all"
4. **Issue repository** — which observed repo hosts Beaver-tracked issues

### Preview and Confirm

Always show a full preview before executing. Include:

- Organization, project title, observed repos, issue repo
- Custom fields: Level (Single Select: Goal, Task, SubTask), Status (Single Select: Not Started, In Progress, Blocked, Done), Progress (Number: 0-100)
- README beaver-config block
- Issue types, labels, milestones to create

Wait for explicit user confirmation. If changes requested, adjust and re-preview.

## Execution

### Create Project

```bash
gh project create --owner {org} --title "{title}" --format json
```

Extract `number` and `url` from the output.

### Create Custom Fields

**Level:**
```bash
gh project field-create {number} --owner {org} --name "Level" --data-type SINGLE_SELECT --single-select-options "Goal,Task,SubTask"
```

**Status:** The built-in Status field cannot be deleted. List fields with `gh project field-list {number} --owner {org} --format json`, find the Status field ID, then update via GraphQL:

```bash
gh api graphql -f query='
mutation {
  updateProjectV2Field(input: {
    fieldId: "{existing_status_field_id}"
    singleSelectOptions: [
      {name: "Not Started", color: GRAY, description: ""},
      {name: "In Progress", color: YELLOW, description: ""},
      {name: "Blocked", color: RED, description: ""},
      {name: "Done", color: GREEN, description: ""}
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
gh project field-create {number} --owner {org} --name "Status" --data-type SINGLE_SELECT --single-select-options "Not Started,In Progress,Blocked,Done"
```

**Progress:**
```bash
gh project field-create {number} --owner {org} --name "Progress" --data-type NUMBER
```

Skip any field that already exists and inform the user.

### Write README with beaver-config

````
# {title}

```yaml beaver-config
repositories:
  - {repo1}
  - {repo2}
issueRepo: {issueRepo}
customFields:
  level: Level
  progress: Progress
  status: Status
```
````

```bash
cat > /tmp/beaver-project-readme.md << 'BEAVEREOF'
{readme_content}
BEAVEREOF
```

```bash
gh project edit {number} --owner {org} --readme "$(cat /tmp/beaver-project-readme.md)"
```

### Create Issue Types

Issue types are org-scoped. Requires `admin:org` scope and `X-GitHub-Api-Version: 2026-03-10` header. List existing first to avoid duplicates:

```bash
gh api orgs/{org}/issue-types -H "X-GitHub-Api-Version: 2026-03-10" --jq '.[].name'
```

Create each if not already present:

| Name | Color | Description |
|------|-------|-------------|
| Goal | blue | High-level objective |
| Task | green | Breakdown of a Goal |
| SubTask | gray | Finest granularity work item |

```bash
gh api orgs/{org}/issue-types --method POST -H "X-GitHub-Api-Version: 2026-03-10" -f name="{name}" -f color="{color}" -f description="{desc}" -F is_enabled=true
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
| status/design-pending | D4C5F9 | Design review in progress (size/L) |
| status/ready-to-develop | 0E8A16 | Ready to code (size/L) |
| status/in-progress | FBCA04 | Active development |
| status/blocked | B60205 | Blocked |
| status/review-needed | 1D76DB | Awaiting review |
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
gh label create "{label}" --repo {org}/{issueRepo} --color "{color}" --description "{desc}"
```

Create all labels in sequence. Skip on error (label already exists).

### Create Milestones

Calculate 4 weekly milestones starting from today. Each week spans 7 days. Title format: `Week N (Mon DD - Mon DD)`.

```bash
gh api repos/{org}/{issueRepo}/milestones --method POST -f title="Week {n} ({start} - {end})" -f due_on="{end_iso}T23:59:59Z" -f state="open"
```

Skip on 422 (title already exists).

## Success Report

Print summary with: project URL, custom fields, observed repos, issue repo, issue types, labels, milestones. Inform the user they can now use `beaver-issue` with project identifier `{org}/{number}`.

## Constraints

- Organization-level projects only (not user-level)
- Fixed field names: Level, Status, Progress
- Fixed labels, issue types, and milestone count (4 weeks)
- Always confirm before executing — never create without preview and approval
- Does not modify existing projects — new projects only
- On failure, report the error and let the user decide how to proceed (no auto-retry)
