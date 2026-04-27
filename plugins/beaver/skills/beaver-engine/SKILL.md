---
name: beaver-engine
description: "Internal engine for Beaver commands. DO NOT trigger directly. Provides state machine rules, guardrail checks, field operations, and project config reading used by beaver-create, beaver-design, beaver-decompose, beaver-dev, beaver-pr, beaver-tracker, beaver-focus, and beaver-setup."
---

# Beaver Engine

Internal skill. Do not invoke directly. Other beaver skills reference these rules and command templates.

Single source of truth for an Issue's lifecycle metadata is **GitHub Projects V2 #14** custom fields and the **native GitHub Issue Type** — not labels. All read/write operations on lifecycle metadata go through `plugins/beaver/scripts/beaver-lib.sh` (see §4 Field Operations). Native GitHub Issues APIs that fall outside the lifecycle-metadata surface (e.g., the sub-issues API used by G009, the labels API used for `beaver/*` flag labels in §4) are called directly with `gh api` since they do not correspond to Project V2 fields or Issue Type.

<!-- BEGIN 废弃说明 -->
## 废弃说明 (Deprecation Notice)

Earlier revisions of this engine modeled lifecycle metadata as `status/*`, `type/*`, and `size/*` repository labels. Those label families are deprecated and replaced by Project V2 fields and native Issue Types per [RFC-0013](https://github.com/primatrix/wiki/blob/main/docs/rfc/0013-beaver-commands-realignment.md). The old labels are kept on `primatrix/projects` for historical Issues but are no longer read or written by any Beaver command. New code MUST go through `beaver-lib.sh`.

Mapping for reference (legacy label → new field semantics):

- `status/triage` → Project V2 `Status = Triage`
- `status/ready-to-claim` → Project V2 `Status = Ready to Claim`
- `status/design-pending` → Project V2 `Status = Design Pending`
- `status/ready-to-develop` → Project V2 `Status = Ready to Develop`
- `status/in-progress` → Project V2 `Status = In Progress`
- `status/blocked` → Project V2 `Status = Blocked`
- `status/done` → Project V2 `Status = Done`
- `type/feat|bug|refactor|docs|chore` → native Issue Type ∈ {`Bug`, `Task`, `SubTask`}
- `size/S|L` → Project V2 `Size = S | L`

The lookup tables below also use the old label syntax for traceability when reading legacy Issues; the canonical names live in Project V2 single-select options.
<!-- END 废弃说明 -->

## 1. Field Taxonomy

All lifecycle metadata is stored either on **Project V2 #14 single-select fields** or on the **native GitHub Issue Type**. There are no `prefix/name` labels in this taxonomy.

### Issue Type (native, organization-level)

Three values managed via `gh api /orgs/primatrix/issue-types`:

- **Bug** — Defects requiring fast-path handling and Iteration assignment.
- **Task** — Top-level deliverable. May be Size=S (fast track) or Size=L (full SOP with Design Doc + SubTasks).
- **SubTask** — Child of a Size=L Task; always Size=S.

Read via `beaver-lib.sh::get_type`. Write via `beaver-lib.sh::set_type`.

### Status (Project V2 single-select, 7 values)

Lifecycle stages, in canonical order: **Triage → Ready to Claim → Design Pending → Ready to Develop → In Progress → Blocked → Done**.

Read via `_get_single_select_value <issue> "Status"`. Write via `beaver-lib.sh::set_status`.

### Size (Project V2 single-select)

`beaver-setup` ensures the spec's full set (`XS / S / M / L / XL`) on the field. Beaver commands consume only `S` and `L`. Read/write via `beaver-lib.sh::set_size`.

### Priority (Project V2 single-select, Bug only)

`P0 / P1 / P2`. `P0` Bugs route to `Status = In Progress`; `P1`/`P2` route to `Status = Ready to Claim` (per RFC-0013 §3 Bug path).

### Iteration (Project V2 iteration field)

Sprint container. Read via `beaver-lib.sh::get_iteration`. Write via `beaver-lib.sh::set_iteration`. The "current or next future" lookup used by `/beaver-create` Bug path is `beaver-lib.sh::latest_iteration_for_repo` (see §3 G011).

### Target date (Project V2 date field)

Deadline date (YYYY-MM-DD). Read via `beaver-lib.sh::get_target_date`. Write via `beaver-lib.sh::set_target_date`. `/beaver-decompose` silently inherits the parent Task's Target date onto each child SubTask (skipped when the parent has no Target date).

### Beaver agent labels (still labels)

These are operational flags written by Beaver tooling, not lifecycle metadata, and remain implemented as repository labels:

- `beaver/needs-split` — PR LOC exceeds 200 in core dirs
- `beaver/missing-test` — No test evidence found before done
- `beaver/missing-context` — Incomplete fields or description
- `beaver/stale` — Stuck in same Status > 3 days
- `beaver/overdue` — Past DDL and not done
- `beaver/upstream-blocked` — Upstream dependency is blocked
- `beaver/wontfix` — Will not fix; skip stale/overdue detection

The repo-level `Control-By-Beaver` and tracker-issue `tracker-*` labels are also retained as Beaver metadata and are out of scope of the Status / Type / Size taxonomy migration.

## 2. State Machine

Status flow on Project V2 #14, by Size:

| Current Status | Size=S next (Task / Bug) | Size=L next (Task only) | Trigger |
|---|---|---|---|
| Triage | Ready to Claim (Task) / In Progress (P0 Bug) / Ready to Claim (P1\|P2 Bug) | Ready to Claim | `/beaver-create` writes initial Status; transition out by `/beaver-tracker` (Iteration assign) or user UI |
| Ready to Claim | In Progress | Design Pending | User assigns self in GitHub UI, then user manually sets Status (the system-side auto-migration is out-of-scope per RFC-0013 §「范围」) |
| Design Pending | N/A | Ready to Develop | Design Doc PR merged (system migration, out-of-scope; user sets manually in UI during transition period) |
| Ready to Develop | N/A | In Progress | `/beaver-dev` writes via `beaver-lib.sh::set_status` |
| In Progress | Done | Done | PR merged (system migration, out-of-scope; user sets manually in UI during transition period) |
| Blocked | (previous) | (previous) | User manually toggles in GitHub UI; `/beaver-engine` does not encapsulate this transition |
| Done | (terminal) | (terminal) | — |

### Size=S Fast Track (Task)

`Triage → Ready to Claim → In Progress → Done`. Skips Design Pending and Ready to Develop.

### Size=L Standard SOP (Task)

`Triage → Ready to Claim → Design Pending → Ready to Develop → In Progress → Done`.

### Bug Track

- Per RFC-0013, Bugs are no longer forced to Size=S (G008 deleted). Bugs do not write the `Size` field at all.
- `Priority = P0` Bugs are created directly at `Status = In Progress` (skips Triage and Ready to Claim).
- `Priority = P1 | P2` Bugs are created at `Status = Ready to Claim` (skips Triage; no Design Pending stage).
- All Bugs MUST have an Iteration assigned at creation (G011).

### Universal Transitions

- Any Status → `Blocked` (user manually toggles in GitHub UI; commands do not encapsulate this transition; reason should be noted in an Issue comment).
- `Blocked` → previous Status (user manually toggles).

### System Migrations (out-of-scope, transition period only)

Per RFC-0013, three transitions are designated as system behavior but are not yet implemented:

1. Iteration assignment → auto-set `Status = Design Pending` for Size=L Task.
2. Design Doc PR merged → auto-set `Status = Ready to Develop` for parent Size=L Task.
3. All SubTasks of a Size=L Task closed → auto-set parent `Status = Done`.

Until automation lands, users perform these transitions manually in the Project V2 UI. Beaver commands MUST NOT print "waiting for system migration" prompts.

## 3. Guardrail Rules

All checks below read **Project V2 fields** and **native Issue Type** via `beaver-lib.sh`. None of them read or write the legacy lifecycle labels (see §「废弃说明」 for the deprecated label families).

### G001: Size required before leaving Triage

- **Check:** Issue's Project V2 `Size` field is set to `S` or `L`. Read via `_get_single_select_value <number> "Size"`.
- **When:** Any transition out of `Status = Triage` for a Task.
- **Fail action:** Block transition; comment on Issue requesting Size classification (`/beaver-create` should set this at creation; missing values indicate a legacy Issue).
- **Bug exemption:** Bugs do not have a `Size` field per RFC-0013; G001 only applies to Task / SubTask Issue Types (read via `beaver-lib.sh::get_type`).

### G002: Size=L Task must not skip stages

- **Check:** Target `Status` value matches the legal next state in the Size=L SOP table (§2). Read current Status via `_get_single_select_value <number> "Status"` and Size via `_get_single_select_value <number> "Size"`.
- **When:** Any Status transition of a Task whose Project V2 `Size = L`.
- **Fail action:** Block transition; comment listing the required intermediate stages.

### G004: Test evidence required for Done

- **Check:** Find test evidence from (in priority order):
  1. Current session context — scan conversation for test runner output (pytest, go test, npm test, cargo test, etc.).
  2. PR diff — new/modified test files (`*_test.*`, `test_*.*`, `tests/**`).
  3. CI status — GitHub Actions / Check Runs on associated PR.
- **When:** PR creation (`beaver-pr`).
- **Fail action:** Add `beaver/missing-test` label, comment requesting evidence.
- **On success:** Write test summary to PR body's Test Plan section or Issue comment.

### G006: PR must have complete fields

- **Check:** Associated Issue has a non-empty native Issue Type AND a Project V2 `Size` value (the latter only when Issue Type ∈ {Task, SubTask}; Bug exempt). Read via `beaver-lib.sh::get_type` and `_get_single_select_value <number> "Size"`.
- **When:** PR creation.
- **Fail action:** Attempt auto-fill via `beaver-lib.sh::set_type` (inferred from branch prefix) and `beaver-lib.sh::set_size` (default `S`). If auto-fill fails, append a warning to the PR body (e.g., `> ⚠️ Beaver audit: Issue missing Type/Size fields and auto-fill failed`). Do not block PR creation; do not write any of the deprecated lifecycle labels.

### G007: Ready to Claim requires Iteration

- **Check:** Issue is assigned to an Iteration entry on Project #14 (custom field `Iteration` non-null). Read via `beaver-lib.sh::get_iteration`.
- **When:** Transition to `Status = Ready to Claim`.
- **Exempt:** Bugs (Bug path uses G011 instead).
- **Fail action:** Block transition; comment requesting Iteration assignment via `/beaver-tracker <repo>`.

**Read via lib:**

```bash
# get_iteration <issue_number> echoes Iteration title or empty string
title=$(bash plugins/beaver/scripts/beaver-lib.sh get_iteration "$NUMBER")
[ -z "$title" ] && echo "G007 fail: Iteration unassigned" >&2
```

### G009: Size=L Task must be Ready to Develop before In Progress

- **Check:** Project V2 `Status = Ready to Develop` AND the Issue has at least one sub-issue. Read via `_get_single_select_value <number> "Status"` and `gh api repos/{owner}/{repo}/issues/{number}/sub_issues`.
- **When:** Transition to `Status = In Progress` for a Task whose Project V2 `Size = L`.
- **Fail action:** Block transition; comment listing what is missing (Status not yet `Ready to Develop`, or zero sub-issues).

### G010: stale / overdue are flag labels

- **Check:** `beaver/stale` and `beaver/overdue` are Beaver flag labels, not Status values.
- **When:** Applied by health-reporting tooling (not `beaver-tracker`, which only handles monthly tracker creation + Iteration sync).
- **Note:** These labels coexist with any Status field value and do not participate in state machine transitions.

### G011: Bug must have Iteration assigned

- **Check:** When `/beaver-create` is processing a Bug, resolve the target Iteration via `beaver-lib.sh::latest_iteration_for_repo <repo>` and write it via `beaver-lib.sh::set_iteration`. The resolution algorithm (RFC-0013 §「`latest_iteration_for_repo` 解析算法」):
  1. Pull all Iteration entries on Project #14 (`title / startDate / duration`).
  2. Compute `endDate = startDate + duration days` for each entry. Today is evaluated in **UTC** to avoid boundary-day ambiguity.
  3. **Step A:** Select entries where `startDate <= today < endDate` (the "current" Iteration). If exactly one matches, return its title.
  4. **Step B:** If Step A returns zero, select the entry with `startDate > today` and minimum `startDate` (the "next future" Iteration). Return its title.
  5. **Step C:** If both A and B are empty, fail with the message `"No current or future Iteration found on Project #14 for <repo>. Run /beaver-tracker <repo> to create this month's Iteration entry."`
  6. Step A returning > 1 entry is an ambiguity error: `"Ambiguous current Iteration for <repo>: <list>. Resolve overlap before retrying."` (defensive — should not occur on a monthly cadence).
- **When:** Issue creation (`/beaver-create`) for native Issue Type = `Bug`.
- **Fail action:** Abort Issue creation; surface the algorithm's error message verbatim and direct the user to `/beaver-tracker <repo>`.
- **Replaces:** former G008 (Bug forced to Size=S), which is deleted per RFC-0013 §「Phase A.2 中的 guardrail 改写」.

## 4. Field Operations

Canonical wrappers around Project V2 #14 fields and the native Issue Type. All call sites MUST go through `plugins/beaver/scripts/beaver-lib.sh` rather than issuing raw `gh api graphql` mutations or label-API calls.

Source the library once per script:

```bash
source "$(git rev-parse --show-toplevel)/plugins/beaver/scripts/beaver-lib.sh"
```

Or invoke the CLI form for one-shot use:

```bash
bash plugins/beaver/scripts/beaver-lib.sh <function-name> <args...>
```

### Read operations

| Op | Function | Returns |
|---|---|---|
| Resolve Project V2 item id from Issue URL | `resolve_item_id <issue_url>` | item node id or empty string |
| Look up Project V2 field id | `get_field_id <field_name>` | field node id or empty string |
| Look up single-select option id | `get_option_id <field_name> <option_name>` | option id or empty string |
| Read native Issue Type | `get_type <issue_number>` | Type name (`Bug`/`Task`/`SubTask`) or empty |
| Read Iteration title | `get_iteration <issue_number>` | Iteration title or empty |
| Read Target date | `get_target_date <issue_number>` | `YYYY-MM-DD` or empty |

For ad-hoc Status/Size reads in scripts that have already sourced the library, use the internal helper `_get_single_select_value <number> <field_name>`.

### Write operations

| Op | Function | Backing mutation |
|---|---|---|
| Set Project V2 `Status` | `set_status <issue_number> <status_name>` | `updateProjectV2ItemFieldValue` (singleSelectOptionId) |
| Set Project V2 `Size` | `set_size <issue_number> <size_name>` | `updateProjectV2ItemFieldValue` (singleSelectOptionId) |
| Set native Issue Type | `set_type <issue_number> <type_name>` | `updateIssueIssueType` (requires `admin:org` scope and `GraphQL-Features: issue_types` header) |
| Set Iteration | `set_iteration <issue_number> <iteration_title>` | `updateProjectV2ItemFieldValue` (iterationId) |
| Set Target date | `set_target_date <issue_number> <date>` | `updateProjectV2ItemFieldValue` (date) |

`set_type` is an instance assignment — it selects an existing organization-level Issue Type for the given Issue. Creating new organization-level Type definitions stays in `beaver-setup.sh` and goes through the REST endpoint `POST /orgs/{org}/issue-types`.

### Composite helpers

| Op | Function | Algorithm |
|---|---|---|
| Resolve target Iteration for a Bug | `latest_iteration_for_repo <repo>` | RFC-0013 G011 algorithm: current Iteration if uniquely defined, else next future Iteration, else fail |

### Self-test

```bash
bash plugins/beaver/scripts/beaver-lib.sh self-test
```

Creates a sandbox Issue in `primatrix/projects`, exercises `set_status` / `set_size` / `set_type` round-trips, then deletes the sandbox Issue. Run this whenever changing the library.

### Beaver flag labels

`beaver/*` labels are still managed via the standard labels API (they are operational flags, not lifecycle metadata):

```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={beaver_flag}"
```

## 5. Project Config Reading

Read from Project V2 README's `beaver-config` YAML block:

```bash
gh project view {number} --owner {org} --format json --jq '.readme'
```

Parse the `yaml beaver-config` fenced block for:

- `repositories`: list of observed repos
- `issueRepo`: the repo hosting Beaver issues
- `customFields`: field name overrides (default: Status, Size, Priority, Iteration; Level retained read-only for legacy items)

## 6. Transition Execution Template

When a command-layer skill needs to transition an Issue's Status:

1. Read current Status via `_get_single_select_value <number> "Status"`.
2. Read native Issue Type via `beaver-lib.sh::get_type` and Project V2 `Size` via `_get_single_select_value <number> "Size"`.
3. Look up legal next states (§2).
4. Validate target state against guardrails (§3).
5. If all checks pass: call `beaver-lib.sh::set_status <number> <target>`.
6. If any check fails: report failure; do NOT mutate any field.

## 7. QA Loop & HARD-GATE

Reusable Q&A discipline. Other beaver skills reference this section before any state-changing action (`gh api ... POST`, `git commit`, `gh pr create`, `gh project item-add`, Project V2 field writes via `beaver-lib.sh`).

### 7.1 When callers must invoke

A caller MUST invoke this section before any state-changing action when:

- Creating a new GitHub Issue (`beaver-create`)
- Drafting a design doc section (`beaver-design` Phase 2/3)
- Decomposing into sub-issues (`beaver-decompose`)

A caller MAY skip this section only when the action is a pure Status field write on an Issue that already has approved content.

### 7.2 HARD-GATE rule

Until the user has explicitly approved the current section per §7.5, the caller MUST NOT:

- Write files (other than ephemeral scratch buffers like `mktemp` used purely to assemble content for review)
- Run `gh api ... --method POST` / `--method PATCH` / `--method PUT`
- Run `gh project item-add` / `gh project item-edit`
- Call any `beaver-lib.sh` `set_*` function
- Run `git commit` / `git push`
- Run `gh pr create`
- Add or remove GitHub labels (including `beaver/*` flag labels)
- Assign or unassign users

The single permitted action is read-only codebase discovery (used by §8).

### 7.3 Sectional Q&A loop

For each section the caller is collecting:

1. Ask exactly one question per turn. Prefer multiple-choice options when ≥ 2 distinct alternatives exist; otherwise open-ended is acceptable.
2. Never batch multiple questions in one turn (no "and also" / "additionally" questions).
3. After each user answer, echo the answer verbatim into the running "已确认要点" (Confirmed Points) list at the top of the next turn. This guards against drift across long Q&A.
4. When the caller believes the section is complete, present:
   - The full "已确认要点" for this section
   - The §9.3 checklist (5 rows, all marked ☐ or ☑)
   - The literal prompt: `Approved? (y/revise)`
5. Do NOT advance to the next section until §7.5 approval is received.

### 7.4 Skip-detection — STOP if the caller (you, the agent) catches itself thinking

| Thought | Reality |
|---|---|
| "信息够了，先开始写吧" | 每节都有隐藏约束。继续提问。 |
| "我帮用户合并几个问题省时间" | 一次一问。批量提问会得到肤浅回答。 |
| "issue body 已经写得很全" | issue body 是起点，不是输入。继续 Q&A。 |
| "我可以推断技术细节" | 推断 = 幻觉。问。 |
| "用户看起来很忙" | 烂 doc 比 Q&A 更费时。继续问。 |
| "这个 section 很简单，可以跳过 checklist" | §9.3 是 HARD-GATE。不可跳。 |
| "用户说 '差不多' / '看起来还行'" | 模糊回答 = revise。重新呈现并请求显式 approve。 |

### 7.5 Approval grammar

Approval is granted ONLY when the user's response, stripped of leading/trailing whitespace and case-folded, exactly matches one of:

- `y`
- `yes`
- `ok`
- `approve`
- `approved`
- `lgtm`
- `继续`
- `通过`

Anything else (including `差不多`, `看起来还行`, `应该可以`, silence, or substring matches like `yes but ...`) MUST be treated as `revise`. The caller must re-present the section, address the implied feedback, and ask again.

## 8. Discovery Triad

Mandatory codebase discovery executed by the caller BEFORE the first §7 question. Output goes into a fixed-format "Discovery Brief" presented to the user. **HARD-GATE:** §7 Q&A may not begin until the Brief has been printed. Order is: §8 discovery → Brief printed → §7 Q&A → §7.5 approval → §7.2 state change.

### 8.1 The three required actions

| ID | Action | Tool | Purpose |
|---|---|---|---|
| D1 | Recent activity | `Bash`: `git log --oneline -20` AND `git log --all --since="14 days ago" --oneline` | Anchor the issue against recent project direction |
| D2 | Keyword search | `Glob` over file names + `Grep` over file contents, for ≤ 5 keywords | Locate related code |
| D3 | Project conventions | `Read` repo-root `README.md` and `CLAUDE.md` (when present), plus `*/README.md` under any directory hit by D2 | Surface conventions, tech stack, special instructions |

### 8.2 Keyword extraction rules

- Pull keywords ONLY from the literal text of the issue title + objective. Do NOT invent synonyms or related terms.
- Keep slash-bearing identifiers intact (e.g. `beaver-engine`, `latest_iteration_for_repo`, `add-to-project`).
- Keep ≤ 5 keywords; trim down by removing stop words and single-character tokens.
- For a Chinese-only title where simple whitespace splitting yields fewer than 2 useful tokens, ask the user "Which 2-5 keywords should I search?" before running D2.

### 8.3 Discovery Brief output format

The caller must print exactly this structure to the user before the first §7 question:

```text
## Discovery Brief

### D1 Recent activity
- {hash} {subject}        ← up to 10 lines
- ...

### D2 Keyword hits
- keyword1 → 3 files: a.py, b.py, c.py
- keyword2 → 0 files (NEW AREA)

### D3 Conventions / docs
- README.md: <one-line summary>
- CLAUDE.md: <one-line summary or "absent">
- relevant docs: <list or "none">

### Open questions surfaced
- <每个发现衍生出的待确认问题，逐条列出>
```

### 8.4 Anti-hallucination rules

- Every line in the Brief must correspond 1:1 to actual tool output. Do NOT paraphrase or summarize beyond the source.
- 0 hits MUST be written as `0 files` / `absent` / `none`. The words `似乎`, `可能`, `应该`, `seems`, `probably`, `should be` are FORBIDDEN in the Brief.
- If a `Read` fails (file not found), record the file as `absent` rather than guessing its contents.

### 8.5 Bug exception

When the caller has already detected native Issue Type = `Bug` and the Issue has `Priority = P0`:

- D1 and D3 are still mandatory (skipping is forbidden).
- D2 keywords are restricted to error messages / stack-trace tokens / API names from the user's reproduction steps.
- The Brief must still be printed before any §7.2 state-changing action; `Priority = P0` shortens latency but does NOT skip discovery.

## 9. Doc Quality Constraints

Constraints on issue bodies and design-doc sections produced by callers. Used by §7's per-section approval gate.

### 9.1 Bilingual rule

| Element | Language |
|---|---|
| Section headings (`## 目标` / `## 验收标准` / `## 1. Context & Scope` etc.) | Follow caller's existing template (issue uses Chinese, design doc uses English numbered headings) |
| Body prose | Chinese as primary language |
| Technical nouns: API names, file paths, field names (`Status`, `Iteration`), commands (`gh api`, `beaver-lib.sh`), commit hashes | English / original form, untranslated |
| Quoted code blocks, error messages | Original, untranslated |

### 9.2 Anti-hallucination rule

The following content is FORBIDDEN unless verified:

| Type | Verification source |
|---|---|
| Library / framework name | Discovery Brief D2 or D3 hit, OR explicit user mention in §7 Q&A |
| File path | D2 hit, OR `Read` confirmed |
| API endpoint / function signature | User-provided, OR grep hit |
| Quantitative metric ("延迟降低 30%", "覆盖 80% 场景") | User-provided with named source |

Every claimed fact MUST be traceable. The caller appends a Provenance block:

- For design docs (markdown): an HTML comment at the end of the document:

  ```markdown
  <!-- provenance
  - "<fact 1>" ← <source: Discovery D1/D2/D3 line, or QA round N>
  - "<fact 2>" ← <source>
  -->
  ```

- For issue bodies: provenance is implicit — every fact must come from §7 Q&A answers or the Discovery Brief; no separate block is required, but no fact may exceed those sources.

### 9.3 Section completeness checklist

Before requesting approval per §7.3 step 4, the caller MUST present this 5-row table for the section being approved:

| Check | Condition | Pass? |
|---|---|---|
| Why | Does this section answer "为什么这么做", not just "做什么"? | ☐ |
| Verifiable | Can a reader verify each statement via `gh` / `git` / `Read`? | ☐ |
| No invented facts | Are all facts traceable to Discovery Brief or §7 Q&A answers per §9.2? | ☐ |
| Bilingual rule | Chinese prose + English technical terms per §9.1? | ☐ |
| Length scaled | Simple topic ≤ a few sentences; complex topic ≤ 300 words? | ☐ |

If ANY row is ☐, the caller MUST revise the section first, then re-present the table with all rows ☑, THEN ask `Approved? (y/revise)`.

### 9.4 Issue body simplified checklist

Issue bodies (created by `beaver-create`) do not require a Provenance block but MUST satisfy:

- **Objective**: one sentence stating the user-facing outcome (not the implementation).
- **Acceptance criteria**: ≥ 2 items, each starting with a verb that yields a verifiable check (`运行 X 返回 Y` / `打开 URL Z 看到 W` / `pytest tests/foo.py 全部通过`). Avoid the verbs "improve" / "refactor" / "optimize" as acceptance-criterion phrasing without a measurable target. (The underlying intent of a "refactor" task is unaffected — the prohibition is only on the verb appearing in acceptance criteria; a Task whose work is structural cleanup remains a valid native Issue Type = `Task`.)
- **No invented file paths**: every path mentioned must appear in the Discovery Brief D2 hits or D3 file list.

### 9.5 Bug-mode adjustment

For Issues whose native Issue Type = `Bug`, the body uses the Bug template (复现步骤 / 期望 / 实际 / 影响 / 环境). §9.4 applies with these substitutions:

- "Objective" → 复现步骤 (must be runnable / clickable, not abstract description).
- "Acceptance criteria" → 期望行为 + 实际行为, both concrete.
- Traceability for Bug issues: the reproduction steps must cite their origin inline (e.g., `(来源: §7 Q&A round 2)` or `(log: path/to/file)`). No separate Provenance block is required, consistent with §9.2's issue-body rule.
