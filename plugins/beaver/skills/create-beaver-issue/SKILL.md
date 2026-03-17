---
name: create-beaver-issue
description: Create a Beaver-tracked GitHub Issue (Goal/Task/SubTask) with Project V2 field setup and tracking rules. Trigger this skill whenever the user wants to create a GitHub issue.
---

# Create Beaver Issue

Create a single Beaver-tracked GitHub Issue (Goal / Task / SubTask) with automatic Project V2 field setup and tracking rule configuration. The wizard collects issue details, builds a structured body with a `beaver-tracking` comment block, creates the issue via `gh api`, adds it to the Project V2, sets Level/Status/Progress fields, and updates the parent issue if applicable.

## Prerequisites

- `gh auth status` must succeed (run `gh auth login` if not)
- Project scope required (`gh auth refresh -s project` if missing)
- Project README must contain a fenced ```` ```yaml beaver-config ```` block with `repositories` and `issueRepo`

## Defaults

Check auto memory for `beaver-issue-defaults.md`. If found, present batch confirmation and allow the user to confirm, edit, or reset. Expected format:

```yaml
project: primatrix/4
trackingRepos:
  - ant-pretrain
  - pallas-kernel
milestone: 1
milestoneTitle: "Week 1 (Mar 17 - Mar 23)"
```

## Workflow

1. **Load project config** -- Parse project identifier (`org/number` or URL), run `gh project view {number} --owner {org} --format json`, extract `readme` and `id` (project node ID). Parse `beaver-config` YAML block from the README for `repositories`, `issueRepo`, and optional `customFields`.
2. **Choose level** -- Goal (high-level objective), Task (breakdown of a Goal), or SubTask (finest granularity).
3. **Associate parent** (Task/SubTask only) -- List project items via `gh project item-list`, filter by parent level, let user pick.
4. **Collect title and description** -- Structure into 目标 (objective) and 验收标准 (acceptance criteria) sections.
5. **Tracking rules** (Task/SubTask only) -- Select repos from config's `repositories` list (pre-check remembered defaults), optionally add paths and keywords. Goal issues use child rollup.
6. **Select milestone** -- Query `gh api repos/{org}/{issueRepo}/milestones`, let user pick or skip.
7. **Preview and confirm** -- Show full issue details; wait for explicit user approval before creating.

## Issue Body Templates

### Goal (no tracking rule)

```
## 目标

{structured description}

## 验收标准

{acceptance criteria}

## 子任务

(子任务将在创建后自动添加)
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

**Add to project and set fields:**
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

**Update parent** (Task/SubTask only): Append `- [ ] #{new_issue_number}` to the parent's 子任务 section via `gh issue edit --body-file`.

## After Creation

1. Print summary (issue URL, level, label, milestone, status, progress, parent link, project).
2. Silently save defaults to `beaver-issue-defaults.md` (project, trackingRepos, milestone, milestoneTitle).

## Constraints

- **One issue at a time** -- single issue per invocation
- **Always confirm before creating** -- never create without user preview and approval
- **Use Chinese (中文) for issue body content** -- matching existing project convention
- **Do not modify the Project README** -- config is read-only
- **Do not close or modify existing issues** (except appending to parent's task list)
