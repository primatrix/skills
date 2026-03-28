---
name: create-beaver-issue
description: Create a Beaver-tracked GitHub Issue (Goal/Task/SubTask) with Project V2 field setup and tracking rules. Trigger this skill whenever the user wants to create a GitHub issue.
---

# Create Beaver Issue

Create a single Beaver-tracked GitHub Issue (Goal / Task / SubTask) with automatic Project V2 field setup and tracking rule configuration. The wizard collects issue details, builds a structured body with a `beaver-tracking` comment block, creates the issue via `gh api`, adds it to the Project V2, sets Level/Status/Progress fields, and links to the parent issue via GitHub's native sub-issues if applicable.

## Prerequisites

- `gh auth status` must succeed (run `gh auth login` if not)
- If associating with a Project: project scope required (`gh auth refresh -s project` if missing)
- If associating with a Project: Project README must contain a fenced ```` ```yaml beaver-config ```` block with `repositories` and `issueRepo`

## Defaults

Check auto memory for `beaver-issue-defaults.md`. If found, present batch confirmation and allow the user to confirm, edit, or reset. The `project` field is only relevant when associating with a Project. Expected format:

```yaml
project: primatrix/4
trackingRepos:
  - ant-pretrain
  - pallas-kernel
milestone: 1
milestoneTitle: "Week 1 (Mar 17 - Mar 23)"
```

## Workflow

1. **Confirm Project association** -- Ask user: "是否要关联到 GitHub Project?" If yes, continue to step 2. If no, ask for target repo (`org/repo`) and skip to step 3 (all project-related steps are skipped).
2. **Load project config** (project only) -- Parse project identifier (`org/number` or URL), run `gh project view {number} --owner {org} --format json`, extract `readme` and `id` (project node ID). Parse `beaver-config` YAML block from the README for `repositories`, `issueRepo`, and optional `customFields`.
3. **Choose level** -- Goal (high-level objective), Task (breakdown of a Goal), or SubTask (finest granularity).
4. **Associate parent** (Task/SubTask only) -- If project associated: list project items via `gh project item-list`, filter by parent level, let user pick. If no project: list issues in target repo via `gh api repos/{org}/{repo}/issues`, filter by parent level, let user pick.
5. **Collect title and description** -- Structure into 目标 (objective) and 验收标准 (acceptance criteria) sections.
6. **Tracking rules** (Task/SubTask only) -- If project associated: select repos from config's `repositories` list (pre-check remembered defaults). If no project: let user specify repos manually. Optionally add paths and keywords. Goal issues use child rollup.
7. **Select milestone** -- Query `gh api repos/{org}/{repo}/milestones`, let user pick or skip.
8. **Preview and confirm** -- Show full issue details; wait for explicit user approval before creating.

## Issue Body Templates

### Goal (no tracking rule)

```
## 目标

{structured description}

## 验收标准

{acceptance criteria}
```

### Task / SubTask (with tracking rule)

```
## 目标

{structured description}

## 验收标准

{acceptance criteria}

<!-- beaver-tracking
repos:
  - {repo1}
  - {repo2}
paths:
  - {path1}
keywords:
  - {keyword1}
  - {keyword2}
-->
```

## Creation Commands

**Create issue:**
```bash
gh api repos/{org}/{issueRepo}/issues --method POST \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  -f title="{title}" --raw-field body="$(cat "$BODY_FILE")" \
  -f type="{level}" -f "labels[]=Control-By-Beaver"
```
Add `-f milestone={number}` if a milestone was selected. The `X-GitHub-Api-Version: 2026-03-10` header is required for the `type` field to be recognized. If issue type API fails (e.g., org plan does not support issue types), retry without `-f type` and warn the user that the issue type was not set.

**Add to project and set fields** (project only — skip entirely if no project association):
```bash
gh project item-add {projectNumber} --owner {org} --url {issue_url} --format json
gh project field-list {projectNumber} --owner {org} --format json

gh project item-edit --id {item_id} --project-id {project_id} \
  --field-id {level_field_id} --single-select-option-id {level_option_id}
gh project item-edit --id {item_id} --project-id {project_id} \
  --field-id {status_field_id} --single-select-option-id {not_started_option_id}
gh project item-edit --id {item_id} --project-id {project_id} \
  --field-id {progress_field_id} --number 0
```

**Link to parent** (Task/SubTask only): Use GitHub's native sub-issues API to establish the parent-child relationship.
```bash
# Get the internal issue ID of the newly created child issue
CHILD_ID=$(gh api repos/{org}/{issueRepo}/issues/{new_issue_number} --jq '.id')

# Add as sub-issue of the parent
gh api repos/{org}/{issueRepo}/issues/{parent_number}/sub_issues \
  --method POST -H "X-GitHub-Api-Version: 2026-03-10" \
  -f sub_issue_id="$CHILD_ID"
```

## After Creation

1. Print summary (issue URL, level, label, milestone, status, progress, parent link, project — omit project fields if no project association).
2. Silently save defaults to `beaver-issue-defaults.md` (project, trackingRepos, milestone, milestoneTitle — skip project if not associated).

## Constraints

- **One issue at a time** -- single issue per invocation
- **Always confirm before creating** -- never create without user preview and approval
- **Use Chinese (中文) for issue body content** -- matching existing project convention
- **Do not modify the Project README** -- config is read-only
- **Do not close or modify existing issues** -- parent linking uses GitHub's native sub-issues API, not body edits
