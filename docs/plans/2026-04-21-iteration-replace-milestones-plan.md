# Replace Weekly Milestones with Project Iteration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `beaver-setup` off weekly repo Milestones onto a monthly Project Iteration field, update G007 + beaver-focus to use Iteration, rename `beaver-roadmap` → `beaver-tracker` with new backlog-selection step, and bump beaver plugin to v3.2.0.

**Architecture:** Five command/skill markdown files plus two manifests. No code or tests — verification is read-back + JSON validity + manual smoke against the live `primatrix/projects` Project #14. Each task is one focused edit followed by an inline verification step, then a commit.

**Tech Stack:** GitHub Projects v2 GraphQL (`createProjectV2Field`, `ProjectV2ItemFieldIterationValue`), `gh api`, markdown.

**Reference design:** `docs/plans/2026-04-21-iteration-replace-milestones-design.md`

---

## Pre-flight

Confirm worktree state and tooling.

**Step 1:** Verify branch and clean tree.

Run: `git status && git branch --show-current`
Expected: branch `worktree-greedy-exploring-liskov`, working tree clean (only the design doc commit `7268d69` ahead of `main`).

**Step 2:** Verify `gh` auth has needed scopes.

Run: `gh auth status`
Expected: scopes include `project` and `repo`.

**Step 3:** Verify upstream state — milestones empty, no roadmap labels.

Run:
```bash
gh api repos/primatrix/projects/milestones --jq 'length'
gh api repos/primatrix/projects/labels --paginate --jq '.[].name' | grep -ci roadmap || echo 0
```
Expected: `0` and `0`.

---

## Task 1: Update `beaver-setup.md` — remove milestones, add Iteration field

**Files:**
- Modify: `plugins/beaver/commands/beaver-setup.md`

**Step 1: Update frontmatter description**

Replace the `description:` value (line 3) with:
```
description: "Initialize Beaver on primatrix/projects#14: update Status field to 7 options, create labels, and create Iteration field with monthly entries."
```

**Step 2: Update Preview block (lines 31–35)**

Replace the bullet list under "Show a full preview before executing. Include:" so it reads:
```
- All constants above
- Custom fields: Level (Single Select: Milestone, Task, SubTask), Status (Single Select: 7 options — see below), Progress (Number: 0-100), Iteration (Iteration: monthly entries from current month to year-end)
- README beaver-config block
- Issue types and labels to create
- Iteration field with N monthly entries (YYYY-MM ... YYYY-12)
```

**Step 3: Update Status option description (line 59)**

Change:
```
{name: "Ready to Claim", color: BLUE, description: "Added to Milestone, awaiting claim"},
```
to:
```
{name: "Ready to Claim", color: BLUE, description: "Added to Iteration, awaiting claim"},
```

**Step 4: Update README beaver-config block (lines 96–103 and 110–117)**

In both the displayed example and the heredoc body, change:
```yaml
customFields:
  level: Level
  progress: Progress
  status: Status
```
to:
```yaml
customFields:
  level: Level
  progress: Progress
  status: Status
  iteration: Iteration
```

**Step 5: Update status label description (line 182)**

Change `status/ready-to-claim` row description from `Added to Milestone, awaiting claim` to `Added to Iteration, awaiting claim`.

**Step 6: Replace "Create Milestones" section with "Create Iteration Field"**

Delete lines 213–226 (entire `### Create Milestones` section). Replace with:

````markdown
### Create Iteration Field

Create the Iteration custom field on Project #14 and populate one entry per natural month from the current month through December of the current year.

**Field creation (skip if exists):**

```bash
PROJECT_ID=$(gh api graphql -f query='
  query { organization(login: "primatrix") { projectV2(number: 14) { id } } }' \
  --jq '.data.organization.projectV2.id')

gh api graphql -f query='
mutation($projectId: ID!) {
  createProjectV2Field(input: {
    projectId: $projectId
    dataType: ITERATION
    name: "Iteration"
    iterationConfiguration: {
      startDate: "{first_day_of_current_month}"
      duration: {days_in_current_month}
      iterations: [
        {startDate: "{YYYY-MM-01}", duration: {days_in_month}, title: "<YYYY-MM> (<MonthShort>)"},
        ...
      ]
    }
  }) { projectV2Field { ... on ProjectV2IterationField { id name } } }
}' -f projectId="$PROJECT_ID"
```

Build the `iterations` array by iterating from the current month to December of the current year. For each month: title = `YYYY-MM (MonthShort)` (e.g. `2026-04 (Apr)`), startDate = first of that month (`YYYY-MM-01`), duration = days in that month (28/29/30/31).

If the Iteration field already exists, skip creation and instead read existing entries via:

```bash
gh api graphql -f query='
  query { organization(login: "primatrix") { projectV2(number: 14) {
    field(name: "Iteration") {
      ... on ProjectV2IterationField {
        id
        configuration { iterations { title startDate duration } }
      }
    }
  } } }'
```

Compute which monthly entries are missing (compare titles to the expected `YYYY-MM (MonthShort)` set) and append only the missing ones via `updateProjectV2IterationField` (preserve existing entries).
````

**Step 7: Update Constraints section (line 232+)**

Append a new bullet after "Status field is a full replacement":
```
- Iteration field — only missing months are appended; existing entries preserved
```

Remove the original "milestones" reference from the "Idempotent" line (line 234):
```
- Idempotent — safe to run multiple times; skips existing fields, labels, Iteration entries
```

**Step 8: Update Success Report (line 230)**

Replace `milestone count and range` with `Iteration entry count and range`. Final line should read:
```
Print summary with: project URL (`https://github.com/orgs/primatrix/projects/14`), custom fields (Level, Status with 7 options, Progress, Iteration), issue types, label count, Iteration entry count and range. Inform the user they can now use `beaver-issue` with project identifier `primatrix/14`.
```

**Step 9: Verify no `milestone` references remain (case-insensitive, excluding hierarchy `Level: Milestone`)**

Run: `grep -in 'milestone' plugins/beaver/commands/beaver-setup.md`
Expected: only references to `Milestone` as a Level option (line ~32 and ~45 and ~137–138 — the issue type / Level value), zero references to repo milestones or `gh api .../milestones`.

**Step 10: Commit**

```bash
git add plugins/beaver/commands/beaver-setup.md
git commit -m "feat(beaver-setup): replace weekly milestones with monthly Project Iteration field"
```

---

## Task 2: Update `beaver-engine/SKILL.md` — G007 and string updates

**Files:**
- Modify: `plugins/beaver/skills/beaver-engine/SKILL.md`

**Step 1: Replace G007 block (lines 107–111)**

Replace exactly:
```
### G007: ready-to-claim requires Milestone
- **Check:** Issue is associated with a Milestone
- **When:** Transition to `status/ready-to-claim`
- **Exempt:** `type/bug` issues (bugs skip Roadmap)
- **Fail action:** Block transition, comment requesting Milestone assignment
```
with:
```
### G007: ready-to-claim requires Iteration
- **Check:** Issue is assigned to an Iteration entry on Project #14 (custom field "Iteration" non-null). Read via GraphQL `projectV2Item.fieldValueByName(name: "Iteration")`.
- **When:** Transition to `status/ready-to-claim`
- **Exempt:** `type/bug` issues (bugs skip tracker)
- **Fail action:** Block transition, comment requesting Iteration assignment
```

**Step 2: Update line 33**

Change `- status/ready-to-claim — Added to Milestone, awaiting claim` to `- status/ready-to-claim — Added to Iteration, awaiting claim`.

**Step 3: Update line 64**

Change `Other bugs: triage → in-progress (skip ready-to-claim, no Milestone required)` to `Other bugs: triage → in-progress (skip ready-to-claim, no Iteration required)`.

**Step 4: Verify Level: Milestone hierarchy references untouched**

Run: `grep -n 'Milestone' plugins/beaver/skills/beaver-engine/SKILL.md`
Expected: any remaining matches refer to the Level hierarchy value (e.g. `Level: Milestone`, `Milestone → Task → SubTask`), NOT to repo milestones or G007.

**Step 5: Commit**

```bash
git add plugins/beaver/skills/beaver-engine/SKILL.md
git commit -m "refactor(beaver-engine): G007 now requires Iteration assignment instead of Milestone"
```

---

## Task 3: Update `beaver-focus.md` — read Iteration via GraphQL

**Files:**
- Modify: `plugins/beaver/commands/beaver-focus.md`

**Step 1: Replace REST issue listing (around line 33–35) with GraphQL Project query**

Replace the existing `gh api "repos/{org}/{issueRepo}/issues?...` block with:

```bash
gh api graphql -f query='
  query($owner: String!, $number: Int!, $login: String!) {
    organization(login: $owner) {
      projectV2(number: $number) {
        items(first: 100) {
          nodes {
            content {
              ... on Issue {
                number
                title
                state
                labels(first: 30) { nodes { name } }
                assignees(first: 10) { nodes { login } }
              }
            }
            fieldValueByName(name: "Iteration") {
              ... on ProjectV2ItemFieldIterationValue {
                title
                startDate
                duration
              }
            }
          }
        }
      }
    }
  }' -f owner=primatrix -F number=14 -f login="$CURRENT_USER" \
  --jq '.data.organization.projectV2.items.nodes
        | map(select(.content.assignees.nodes | map(.login) | index($login)))
        | map(select(.content.labels.nodes | map(.name) | index("Control-By-Beaver")))
        | map({number: .content.number, title: .content.title,
               labels: [.content.labels.nodes[].name],
               iteration: (if .fieldValueByName then
                 {title: .fieldValueByName.title,
                  start: .fieldValueByName.startDate,
                  duration: .fieldValueByName.duration} else null end)})'
```

**Step 2: Update Step 5 description (lines 49–51)**

Replace:
```
For issues with milestones, check if `due_on` is within 48 hours. Flag accordingly with a warning indicator.
```
with:
```
For issues with an Iteration assignment, compute `iteration_end = startDate + duration days`. If `iteration_end - now <= 48h`, flag with a warning indicator.
```

**Step 3: Update table header / footnote (lines 90–92)**

Change column header `Due` to `Iteration End`. Change footnote `(⚠️ shown if milestone due within 48h)` to `(⚠️ shown if Iteration ends within 48h)`.

**Step 4: Verify all `milestone` strings replaced**

Run: `grep -in 'milestone' plugins/beaver/commands/beaver-focus.md`
Expected: no matches (focus.md has no Level hierarchy references; all mentions were repo milestones).

**Step 5: Commit**

```bash
git add plugins/beaver/commands/beaver-focus.md
git commit -m "refactor(beaver-focus): read Iteration field via GraphQL for DDL warnings"
```

---

## Task 4: Rename `beaver-roadmap.md` → `beaver-tracker.md` and rewrite

**Files:**
- Rename: `plugins/beaver/commands/beaver-roadmap.md` → `plugins/beaver/commands/beaver-tracker.md`
- Modify: the renamed file

**Step 1: Rename via git**

```bash
git mv plugins/beaver/commands/beaver-roadmap.md plugins/beaver/commands/beaver-tracker.md
```

**Step 2: Replace all "roadmap" → "tracker" in the file**

Substitutions to apply (apply in this order to avoid double-replacement):

| Find | Replace |
|---|---|
| `[Roadmap] <repo>` | `[Iteration] <repo>` |
| `roadmap/<repo>` | `tracker/<repo>` |
| `roadmap/<YYYY-MM>` | `tracker/<YYYY-MM>` |
| `roadmap/<prevYYYY-MM>` | `tracker/<prevYYYY-MM>` |
| `roadmap` (the bare label) | `tracker` |
| `<!-- beaver-roadmap` | `<!-- beaver-tracker` |
| `## 月度 Roadmap` | `## 月度 Tracker` |
| `月度 roadmap` | `月度 tracker` |
| `月度 tracking 容器` | unchanged (Chinese term — keep "tracking") |
| `本月 roadmap` | `本月 tracker` |
| `上月 roadmap` | `上月 tracker` |
| `/beaver-roadmap` | `/beaver-tracker` |
| Description frontmatter `Create a monthly roadmap tracking issue ...` | `Create a monthly Iteration tracker issue ...` |
| Argument-hint unchanged | — |
| `Phase 2 of the Beaver development lifecycle.` | unchanged |
| Label color `BFD4F2` and color of `tracker/*` labels | unchanged |

**Step 3: Insert new Step 8.5 — Backlog Selection**

After the existing "Step 8: Re-parent open tasks as sub-issues" section, before "Step 9: Report", add:

````markdown
### Step 8.5: Pull backlog from triage queue (interactive)

Query candidates: `status/triage` issues in `primatrix/projects` not yet assigned to any Iteration.

```bash
gh api graphql -f query='
  query {
    organization(login: "primatrix") {
      projectV2(number: 14) {
        items(first: 100) {
          nodes {
            content {
              ... on Issue {
                number
                title
                repository { nameWithOwner }
                labels(first: 30) { nodes { name } }
              }
            }
            fieldValueByName(name: "Iteration") {
              ... on ProjectV2ItemFieldIterationValue { title }
            }
          }
        }
      }
    }
  }' --jq '.data.organization.projectV2.items.nodes
            | map(select(.content.repository.nameWithOwner == "primatrix/projects"))
            | map(select(.content.labels.nodes | map(.name) | index("status/triage")))
            | map(select(.fieldValueByName == null))
            | map({number: .content.number, title: .content.title})'
```

Print as numbered list:

```
以下 primatrix/projects 中 triage 队列尚未分配 Iteration 的 issue：
  1. #<n1> <title1>
  2. #<n2> <title2>
  ...

请输入要纳入本月 tracker 的编号（逗号分隔，如 "1,3,5"），或输入 "skip" 跳过：
```

If user inputs `skip` or empty → no-op for this step. Otherwise, parse comma-separated indices, resolve to issue numbers, and for each:

1. Attach as sub-issue of the tracker (same `sub_issues` API as Step 8).
2. Continue to Step 8.6 to set Iteration field.

Per-issue failures: collect, do NOT abort batch; surface in Step 9.
````

**Step 4: Insert new Step 8.6 — Iteration Field Sync**

After Step 8.5, before Step 9, add:

````markdown
### Step 8.6: Sync Iteration field for all sub-issues

Resolve current month's Iteration entry id:

```bash
ITERATION_ID=$(gh api graphql -f query='
  query {
    organization(login: "primatrix") {
      projectV2(number: 14) {
        field(name: "Iteration") {
          ... on ProjectV2IterationField {
            configuration { iterations { id title } }
          }
        }
      }
    }
  }' --jq --arg yyyymm "<YYYY-MM>" \
    '.data.organization.projectV2.field.configuration.iterations
     | map(select(.title | startswith($yyyymm))) | .[0].id')
```

For each sub-issue under the tracker (carried + backlog-added), resolve its ProjectV2Item id and set the Iteration field:

```bash
gh api graphql -f query='
  mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
    updateProjectV2ItemFieldValue(input: {
      projectId: $projectId
      itemId: $itemId
      fieldId: "<iteration_field_id>"
      value: { iterationId: $iterationId }
    }) { projectV2Item { id } }
  }' -f projectId="$PROJECT_ID" -f itemId="$ITEM_ID" \
    -f fieldId="$ITERATION_FIELD_ID" -f iterationId="$ITERATION_ID"
```

Per-issue failures: collect, do NOT abort batch; surface in Step 9.
````

**Step 5: Update Step 6 preview text**

In the preview block, after the "Will then re-parent" line, add:

```
Will then prompt to pull backlog issues from triage queue (Step 8.5).
After tracker is populated, will set Iteration field on all sub-issues
to <YYYY-MM> entry (Step 8.6).
```

**Step 6: Update Step 9 report**

Add lines for backlog and Iteration sync:
```
Backlog pulled: <N> succeeded, <M> failed
Iteration sync: <X> succeeded, <Y> failed
```

**Step 7: Update Constraints section**

Add bullets:
```
- Step 8.5 backlog selection is interactive; only runs after HARD-GATE approval; user can skip and run migration only.
- Step 8.6 sets the Iteration field on every sub-issue under the tracker, mapping to the entry whose title starts with <YYYY-MM>.
```

Update existing constraint about IssueRepo: still hardcoded to `primatrix/projects`. Update `roadmap` references inside Constraints section to `tracker`.

**Step 8: Verify file integrity**

Run: `head -5 plugins/beaver/commands/beaver-tracker.md`
Expected: frontmatter intact with updated description, no stray `roadmap` words.

Run: `grep -ic 'roadmap' plugins/beaver/commands/beaver-tracker.md`
Expected: `0`.

**Step 9: Commit**

```bash
git add plugins/beaver/commands/beaver-roadmap.md plugins/beaver/commands/beaver-tracker.md
git commit -m "feat(beaver): rename beaver-roadmap to beaver-tracker, add backlog selection and Iteration sync"
```

---

## Task 5: Update plugin manifests

**Files:**
- Modify: `plugins/beaver/.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`

**Step 1: Edit `plugins/beaver/.claude-plugin/plugin.json`**

Replace contents with:
```json
{
  "name": "beaver",
  "description": "Beaver project management: GitHub-native issue lifecycle with explicit /beaver-xxx commands covering the full development cycle (create, claim, design, decompose, dev, PR, tracker, focus, setup)",
  "version": "3.2.0"
}
```

**Step 2: Edit `.claude-plugin/marketplace.json` (beaver entry, lines 28–36)**

Update the `beaver` plugin entry's `description` (replace `roadmap` with `tracker`) and `version` (`3.1.0` → `3.2.0`).

**Step 3: Verify both JSON files parse**

Run:
```bash
python -c 'import json; json.load(open("plugins/beaver/.claude-plugin/plugin.json"))' && echo OK
python -c 'import json; json.load(open(".claude-plugin/marketplace.json"))' && echo OK
```
Expected: `OK` printed twice.

**Step 4: Commit**

```bash
git add plugins/beaver/.claude-plugin/plugin.json .claude-plugin/marketplace.json
git commit -m "chore(beaver): bump to v3.2.0 — Iteration migration + tracker rename"
```

---

## Task 6: Smoke verification against live project

Run a dry-read against `primatrix/projects` Project #14 to confirm GraphQL queries used in the docs are syntactically correct.

**Step 1: Verify Project ID query works**

Run:
```bash
gh api graphql -f query='
  query { organization(login: "primatrix") { projectV2(number: 14) { id title } } }'
```
Expected: returns the project id and title without errors.

**Step 2: Verify Iteration field query works (field may not exist yet)**

Run:
```bash
gh api graphql -f query='
  query { organization(login: "primatrix") { projectV2(number: 14) {
    field(name: "Iteration") { ... on ProjectV2IterationField { id name } }
  } } }'
```
Expected: either returns `field: null` (field not yet created — expected before users run new beaver-setup) or returns `{id, name: "Iteration"}` if already created. Either is acceptable; an HTTP error or schema error is a failure.

**Step 3: Final lint — frontmatter validity**

For each touched markdown file, verify YAML frontmatter is intact:
```bash
for f in plugins/beaver/commands/beaver-setup.md \
         plugins/beaver/commands/beaver-focus.md \
         plugins/beaver/commands/beaver-tracker.md \
         plugins/beaver/skills/beaver-engine/SKILL.md; do
  python -c "
import sys, re
content = open('$f').read()
m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
if not m:
    print('$f: NO FRONTMATTER'); sys.exit(1)
import yaml
try: yaml.safe_load(m.group(1))
except Exception as e: print('$f: BAD YAML —', e); sys.exit(1)
print('$f: OK')
"
done
```
Expected: all four files print `OK`.

**Step 4: Verify task list has no orphaned references**

Run: `grep -rn 'beaver-roadmap' plugins/ .claude-plugin/`
Expected: no matches (file renamed, manifests updated).

Run: `grep -rin 'milestone' plugins/beaver/ | grep -v 'Level' | grep -v 'Milestone, Task, SubTask' | grep -v 'High-level objective' | grep -v 'Breakdown of a Milestone'`
Expected: only references that legitimately refer to the Level hierarchy value remain. Surface for human review.

**Step 5: No commit (verification only).**

---

## Task 7: Open PR

**Step 1: Push branch**

```bash
git push -u origin worktree-greedy-exploring-liskov
```

**Step 2: Open PR via gh**

```bash
gh pr create --title "feat(beaver): replace weekly milestones with monthly Project Iteration" --body "$(cat <<'EOF'
## Summary
- `beaver-setup` now creates a Project Iteration field with monthly entries (current month → year-end) instead of weekly repo Milestones.
- G007 `ready-to-claim requires Milestone` → `requires Iteration`; reads `projectV2Item.fieldValueByName(name: "Iteration")` via GraphQL.
- `beaver-focus` reads Iteration entry end date for DDL warnings instead of `milestone.due_on`.
- `beaver-roadmap` renamed to `beaver-tracker`: `[Iteration] <repo> <YYYY-MM>` titles, `tracker/*` labels, plus new interactive backlog selection (Step 8.5) and Iteration field sync (Step 8.6).
- Plugin bumped to v3.2.0.

40 stale weekly milestones in `primatrix/projects` were manually deleted prior to this PR; no `roadmap*` labels existed, so migration starts from a clean state.

Design doc: `docs/plans/2026-04-21-iteration-replace-milestones-design.md`

## Test plan
- [ ] `beaver-setup` dry-run preview shows Iteration field creation with N monthly entries; no milestone block.
- [ ] After running `beaver-setup` on Project #14, `gh api graphql` confirms `Iteration` field exists with correct entries.
- [ ] `beaver-tracker primatrix/nirvana` preview lists migration + backlog candidates; HARD-GATE approval gate works.
- [ ] G007 blocks `ready-to-claim` transition for an issue with no Iteration assignment.
- [ ] `beaver-focus` DDL column reads from Iteration entry end date.
EOF
)"
```

Expected: PR URL printed.

---

## Notes

- All edits are markdown-only. No runtime tests because skills are interpreted by the agent at runtime; verification is read-back + JSON validity + smoke against live GraphQL.
- The new `beaver-setup` Iteration creation logic is described declaratively in the SKILL.md body — the agent will execute it by following the documented `gh api graphql` calls. There is no Python script to test.
- After PR merge, the user must run `/beaver-setup` once on `primatrix/14` to create the Iteration field. This plan does NOT execute that — it only updates the skill definitions.
