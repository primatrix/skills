# Beaver Setup Redesign

## Goal

Rewrite `beaver-setup.md` from an interactive project bootstrapper to a zero-interaction idempotent initializer for the existing `primatrix` project #14.

## Hardcoded Parameters

| Parameter | Value |
|-----------|-------|
| Organization | `primatrix` |
| Project | `#14` (`https://github.com/orgs/primatrix/projects/14`) |
| Issue repo | `primatrix/projects` |
| Observed repos | all |

## Execution Flow

```
1. Context (automatic)
   - gh auth status (verify project + admin:org scopes)
   - date (for milestone calculation)

2. Preview
   - Show all fields, labels, milestones to create/update
   - Wait for user confirmation

3. Execution
   3a. Update Project V2 Status field → exactly 7 options (replaces all existing)
   3b. Create/skip Level, Progress fields
   3c. Update README beaver-config
   3d. Create Issue Types (Goal/Task/SubTask)
   3e. Create 26 labels (skip existing)
   3f. Create remaining-year weekly milestones (skip existing)

4. Success Report
```

## Project V2 Status Field (7 options, unified with labels)

The GraphQL `updateProjectV2Field` mutation with `singleSelectOptions` is a full replacement — only the 7 specified options will exist after the update. Any pre-existing options not in this list (e.g. "Not Started") are removed.

| Option | Color | Corresponding Label |
|--------|-------|-------------------|
| Triage | GRAY | status/triage |
| Ready to Claim | BLUE | status/ready-to-claim |
| Design Pending | PURPLE | status/design-pending |
| Ready to Develop | ORANGE | status/ready-to-develop |
| In Progress | YELLOW | status/in-progress |
| Blocked | RED | status/blocked |
| Done | GREEN | status/done |

## Milestones (remaining weeks of current year)

- Start: Monday of the current ISO week
- End: Sunday of the week containing December 31
- Title format: `Week {iso_week_number} (Mon DD - Mon DD)`
- `due_on`: Sunday 23:59:59Z of each week
- Skip milestones whose title already exists (422 error → continue)

## README beaver-config

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

## Labels, Issue Types, Other Fields

No changes to the existing label set (26 labels), issue types (Goal/Task/SubTask), or Level/Progress fields. These remain as currently defined in beaver-setup.md.

## Removed Interaction

The original 4-step user input flow (organization, project title, observed repos, issue repo) is entirely removed. The only user interaction remaining is the preview confirmation before execution.
