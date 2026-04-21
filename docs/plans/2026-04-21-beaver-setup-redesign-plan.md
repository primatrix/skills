# Beaver Setup Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite beaver-setup.md to be a zero-interaction idempotent initializer for primatrix project #14 with unified 7-option Status and full-year milestones.

**Architecture:** Single file modification — rewrite `plugins/beaver/commands/beaver-setup.md`. All parameters hardcoded, only preview confirmation remains as user interaction.

**Tech Stack:** Markdown skill file, GitHub CLI (`gh`), GraphQL API

---

### Task 1: Rewrite frontmatter and header

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md:1-8`

**Step 1: Replace frontmatter and header**

Replace lines 1-8 with:

```markdown
---
allowed-tools: Bash(gh auth status:*), Bash(gh auth refresh:*), Bash(gh project edit:*), Bash(gh project field-create:*), Bash(gh project field-list:*), Bash(gh api:*), Bash(gh label create:*), Bash(cat > /tmp/*), Bash(date:*)
description: "Initialize Beaver on primatrix/projects#14: update Status field to 7 options, create labels, and create remaining-year weekly milestones."
---

# Initialize Beaver Project

Idempotent initializer for the Beaver project at `primatrix` org, project #14. Updates custom fields, README, issue types, labels, and milestones on `primatrix/projects`.
```

Note: Removed `Bash(gh repo list:*)` and `Bash(gh project create:*)` from allowed-tools since we no longer list repos or create projects.

**Step 2: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): update frontmatter for idempotent initializer"
```

---

### Task 2: Replace Context and Workflow sections

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md:10-34`

**Step 1: Replace Context and Workflow**

Replace lines 10-34 (from `## Context` through the end of `### Preview and Confirm`) with:

```markdown
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
```

**Step 2: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): hardcode constants, remove interactive workflow"
```

---

### Task 3: Replace Create Project and Custom Fields sections

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md` — the `### Create Project` and `### Create Custom Fields` sections

**Step 1: Replace Create Project section**

Remove the entire `### Create Project` section (lines 38-44 in original). Replace with:

```markdown
### Update Custom Fields
```

(No project creation — project #14 already exists.)

**Step 2: Replace Custom Fields section**

Replace the entire `### Create Custom Fields` section with:

````markdown
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
````

**Step 3: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): update Status to 7 options, hardcode project 14"
```

---

### Task 4: Replace README section

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md` — the `### Write README with beaver-config` section

**Step 1: Replace README section**

Replace the entire section with:

`````markdown
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
`````

**Step 2: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): hardcode README beaver-config for primatrix"
```

---

### Task 5: Hardcode Issue Types and Labels sections

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md` — `### Create Issue Types` and `### Create Labels` sections

**Step 1: Replace Issue Types section**

Replace `{org}` with `primatrix` in all commands:

```bash
gh api orgs/primatrix/issue-types -H "X-GitHub-Api-Version: 2026-03-10" --jq '.[].name'
```

```bash
gh api orgs/primatrix/issue-types --method POST -H "X-GitHub-Api-Version: 2026-03-10" -f name="{name}" -f color="{color}" -f description="{desc}" -F is_enabled=true
```

**Step 2: Replace Labels section**

Replace `{org}/{issueRepo}` with `primatrix/projects` in the label create command:

```bash
gh label create "{label}" --repo primatrix/projects --color "{color}" --description "{desc}"
```

The label tables remain unchanged.

**Step 3: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): hardcode primatrix/projects in issue types and labels"
```

---

### Task 6: Replace Milestones section

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md` — `### Create Milestones` section

**Step 1: Replace Milestones section**

Replace the entire section with:

```markdown
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
```

**Step 2: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): remaining-year weekly milestones instead of 4 weeks"
```

---

### Task 7: Replace Success Report and Constraints sections

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md` — `## Success Report` and `## Constraints` sections

**Step 1: Replace Success Report**

```markdown
## Success Report

Print summary with: project URL (`https://github.com/orgs/primatrix/projects/14`), custom fields (Level, Status with 7 options, Progress), issue types, label count, milestone count and range. Inform the user they can now use `beaver-issue` with project identifier `primatrix/14`.
```

**Step 2: Replace Constraints**

```markdown
## Constraints

- Idempotent — safe to run multiple times; skips existing fields, labels, milestones
- Organization: `primatrix` only
- Project: `#14` only (does not create new projects)
- Fixed field names: Level, Status (7 options), Progress
- Status field is a full replacement — only the 7 specified options will exist after update
- Always confirm before executing — never execute without preview and approval
- On failure, report the error and let the user decide how to proceed (no auto-retry)
```

**Step 3: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): update success report and constraints for idempotent mode"
```

---

### Task 8: Final review — read complete file and verify consistency

**Step 1: Read the complete file**

```bash
cat plugins/beaver/commands/beaver-setup.md
```

**Step 2: Verify checklist**

- [ ] No `{org}`, `{title}`, `{issueRepo}`, `{repo1}`, `{repo2}`, `{number}` template variables remain (except `{name}`, `{color}`, `{desc}`, `{label}` for loop iteration, and `{existing_status_field_id}` for runtime lookup)
- [ ] All `gh` commands reference `primatrix`, `14`, or `primatrix/projects` directly
- [ ] Status field has exactly 7 options matching the 7 status labels
- [ ] Milestones section describes remaining-year weekly milestones
- [ ] No interactive user input steps remain (only preview confirmation)
- [ ] Frontmatter allowed-tools list matches the commands used in the file

**Step 3: Fix any issues found, then commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver-setup): final cleanup and consistency check"
```
