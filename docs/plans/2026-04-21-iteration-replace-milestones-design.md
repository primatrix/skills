# Replace Weekly repo Milestones with Project Iteration (monthly)

**Date:** 2026-04-21
**Scope:** `plugins/beaver/` — `beaver-setup`, `beaver-engine`, `beaver-focus`, `beaver-roadmap` → `beaver-tracker`, plus marketplace registration.
**Status:** Design approved 2026-04-21.

## 1. Goal & Scope

Replace `beaver-setup`'s weekly repo Milestones with a monthly **Project Iteration** field on `primatrix` Project #14. Rename `beaver-roadmap` to `beaver-tracker`, keeping monthly tracker issues but aligned to the new `[Iteration]` naming and Iteration field.

**Touched files:**
- `plugins/beaver/commands/beaver-setup.md`
- `plugins/beaver/skills/beaver-engine/SKILL.md`
- `plugins/beaver/commands/beaver-focus.md`
- `plugins/beaver/commands/beaver-roadmap.md` → renamed to `beaver-tracker.md`, body rewritten
- `plugins/beaver/.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`

**Disambiguation note:** "Milestone" appears in two contexts: (a) GitHub repo Milestone (the time-box being replaced), (b) `Level: Milestone` (issue hierarchy value, recently renamed from "Goal" — unrelated to time, kept as-is).

## 2. Project Iteration Field Design

| Property | Value |
|---|---|
| Field name | `Iteration` |
| Project | `primatrix/14` |
| Data type | `ITERATION` (GraphQL) |
| Entry title format | `<YYYY-MM> (<MonthShortName>)` (e.g. `2026-04 (Apr)`) |
| Entry start date | First day of month |
| Entry duration | Actual days in that month (28/29/30/31) |
| Initial fill | Current month → end of year |

**Idempotency:**
- If `Iteration` field exists → skip creation.
- Existing entries preserved; only missing months appended.

**API:** `createProjectV2Field(dataType: ITERATION)` to create the field; `updateProjectV2IterationField` (or equivalent) to configure entries — durations vary per month, submitted in one mutation as an array.

## 3. beaver-setup Changes

### Removed
- Entire **"Create Milestones"** section (current lines 213–226).
- Preview line "Number of remaining-year weekly milestones to create".

### Added
- **"Create Iteration Field"** subsection inside "Update Custom Fields", peer to Level / Status / Progress.
- Preview line: "Iteration field with N monthly entries (YYYY-MM ... YYYY-12)".
- Preview "Custom fields" listing now includes Iteration.

### Modified
- **README beaver-config block:** add `iteration: Iteration` under `customFields`.
- **Status option description:** `Ready to Claim` description: `"Added to Milestone, awaiting claim"` → `"Added to Iteration, awaiting claim"`.
- **Status label description:** `status/ready-to-claim` description same change.
- **Frontmatter description:** `"Initialize Beaver on primatrix/projects#14: update Status field to 7 options, create labels, and create Iteration field with monthly entries."`
- **Constraints section:** add "Iteration field — only missing months are appended; existing entries preserved."

## 4. beaver-engine G007 Change

**Before (lines 107–111):**
```
### G007: ready-to-claim requires Milestone
- Check: Issue is associated with a Milestone
- When: Transition to status/ready-to-claim
- Exempt: type/bug issues (bugs skip Roadmap)
- Fail action: Block transition, comment requesting Milestone assignment
```

**After:**
```
### G007: ready-to-claim requires Iteration
- Check: Issue is assigned to an Iteration entry on Project #14 (custom field "Iteration" non-null)
- When: Transition to status/ready-to-claim
- Exempt: type/bug issues (bugs skip tracker)
- Fail action: Block transition, comment requesting Iteration assignment
```

**Implementation:** Use GraphQL `projectV2Item.fieldValueByName(name: "Iteration")` to check non-null.

**Other engine string updates:**
- Line 33 `status/ready-to-claim` description: `Added to Milestone, awaiting claim` → `Added to Iteration, awaiting claim`.
- Line 64 bug path: `no Milestone required` → `no Iteration required`.
- **Preserve:** any reference to `Level: Milestone` (the hierarchy value) — do not change.

## 5. beaver-focus Changes

**Replace REST milestone reads with GraphQL Project Iteration reads.**

```bash
gh api graphql -f query='
  query($owner: String!, $number: Int!) {
    organization(login: $owner) {
      projectV2(number: $number) {
        items(first: 100) {
          nodes {
            content { ... on Issue { number title assignees { nodes { login } } } }
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
  }' -f owner=primatrix -F number=14
```

**DDL computation:** `iteration_end = startDate + duration days`; warn ⚠️ if `iteration_end - now <= 48h`.

**Other updates:**
- All `milestone:` jq paths → `iteration:`.
- Table column header: `Due` → `Iteration End`.
- Step 5 description rephrased to reference Iteration entry end instead of `due_on`.

## 6. beaver-roadmap → beaver-tracker

### Rename
- File: `beaver-roadmap.md` → `beaver-tracker.md`.
- Command: `/beaver-roadmap` → `/beaver-tracker`.
- Argument signature unchanged: `/beaver-tracker <repo> [YYYY-MM]`.

### Diff vs original

| Aspect | beaver-roadmap | beaver-tracker |
|---|---|---|
| Tracker title | `[Roadmap] <repo> <YYYY-MM>` | `[Iteration] <repo> <YYYY-MM>` |
| Labels | `roadmap`, `roadmap/<repo>`, `roadmap/<YYYY-MM>` | `tracker`, `tracker/<repo>`, `tracker/<YYYY-MM>` |
| Body comment | `<!-- beaver-roadmap ... -->` | `<!-- beaver-tracker ... -->` |
| Prior-month migration | ✅ Keep | ✅ Keep |
| Backlog selection | ❌ — | ✅ **New: Step 8.5** |
| Iteration field sync | ❌ — | ✅ **New: Step 8.6** |

### New Step 8.5: Backlog Selection (interactive)

Query candidates: `status/triage` issues in `primatrix/projects` not yet assigned an Iteration.

```bash
gh api graphql -f query='
  query {
    organization(login: "primatrix") {
      projectV2(number: 14) {
        items(first: 100, query: "repo:primatrix/projects label:status/triage no:iteration") {
          nodes { content { ... on Issue { number title } } }
        }
      }
    }
  }'
```

Print numbered list and prompt:
```
以下 primatrix/projects 中 triage 队列尚未分配 Iteration 的 issue：
  1. #123 feat: add foo
  2. #145 bug: fix bar
  ...

请输入要纳入本月 tracker 的编号（逗号分隔，如 "1,3,5"），或输入 "skip" 跳过：
```

Selected issues attached to tracker via sub-issues API.

### New Step 8.6: Iteration Field Sync

For every sub-issue under the tracker (carried + backlog-added), set its Project Iteration field to the entry matching `<YYYY-MM>`:

```bash
gh api graphql -f query='
  mutation {
    updateProjectV2ItemFieldValue(input: {
      projectId: "<project_id>"
      itemId: "<item_id>"
      fieldId: "<iteration_field_id>"
      value: { iterationId: "<entry_id>" }
    }) { projectV2Item { id } }
  }'
```

### Other updates
- Step 6 preview text: replace all "roadmap" with "tracker"; add backlog candidate preview block.
- Constraints: add "Step 8.5 backlog selection is interactive; only runs after HARD-GATE approval; user can skip and run migration only."

## 7. Marketplace / Plugin Registration

**`plugins/beaver/.claude-plugin/plugin.json`**
- `version`: `3.1.0` → `3.2.0`
- `description`: command list `roadmap` → `tracker`

```json
{
  "name": "beaver",
  "description": "Beaver project management: GitHub-native issue lifecycle with explicit /beaver-xxx commands covering the full development cycle (create, claim, design, decompose, dev, PR, tracker, focus, setup)",
  "version": "3.2.0"
}
```

**`.claude-plugin/marketplace.json`** (beaver entry, lines 28–36)
- Same `version` bump and `description` update.

## 8. Migration & Rollback

| Asset | State |
|---|---|
| Weekly repo Milestones | **Cleared** — 40 milestones manually deleted on 2026-04-21 prior to design freeze. |
| Issues attached to weekly milestones | None — every milestone had `open_issues=0, closed_issues=0` at deletion. |
| `[Roadmap] ...` tracker issues | None in `primatrix/projects`. |
| `roadmap*` labels | None in `primatrix/projects`. |

Clean state — no migration burden. New `beaver-setup` / `beaver-tracker` initialize a fresh structure.

**Rollback:** Single PR; revert if needed. Manually-deleted milestones are not restored on revert (and don't need to be).
