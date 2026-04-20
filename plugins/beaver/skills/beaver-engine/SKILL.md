---
name: beaver-engine
description: "Internal engine for Beaver workflow skills. DO NOT trigger directly. Provides state machine rules, guardrail checks, label operations, and project config reading used by beaver-issue, beaver-pr, beaver-audit, beaver-report, beaver-focus, and beaver-design-doc."
---

# Beaver Engine

Internal skill. Do not invoke directly. Other beaver skills reference these rules and command templates.

## 1. Label Taxonomy

All labels use a `prefix/name` format:

### Type labels (`type/`)
- `type/feat` — New feature
- `type/bug` — Bug fix
- `type/refactor` — Refactoring
- `type/docs` — Documentation
- `type/chore` — Infrastructure, build, misc

### Priority labels (`p/`)
- `p0/blocker` — Blocking, top of daily report
- `p1/urgent` — Urgent, top of daily report
- `p2/high` — High priority
- `p3/normal` — Normal

### Size labels (`size/`)
- `size/S` — Small, fast-track SOP
- `size/L` — Large, full lifecycle SOP

### Status labels (`status/`)
- `status/triage` — Initial state, awaiting triage
- `status/design-pending` — (size/L only) Design review in progress
- `status/ready-to-develop` — (size/L only) Design approved, ready to code
- `status/in-progress` — Active development
- `status/blocked` — Blocked (must note reason)
- `status/review-needed` — Awaiting code/design review
- `status/done` — Completed and merged

### Beaver agent labels (`beaver/`)
- `beaver/needs-split` — PR LOC exceeds 200 in core dirs
- `beaver/missing-test` — No test evidence found before done
- `beaver/missing-context` — Incomplete labels or description
- `beaver/stale` — Stuck in same status > 3 days
- `beaver/overdue` — Past DDL and not done
- `beaver/upstream-blocked` — Upstream dependency is blocked
- `beaver/wontfix` — Will not fix, skip stale/overdue detection

## 2. State Machine

### size/S Fast Track
```text
triage → in-progress → review-needed → done
```

### size/L Standard SOP
```text
triage → design-pending → ready-to-develop → in-progress → review-needed → done
```

### Universal transitions
- Any status → `blocked` (must note reason in Issue comment)
- `blocked` → restore to previous status

### Legal next-states lookup

| Current Status | size/S next | size/L next |
|---|---|---|
| triage | in-progress | design-pending |
| design-pending | N/A | ready-to-develop |
| ready-to-develop | N/A | in-progress |
| in-progress | review-needed | review-needed |
| review-needed | done | done |
| blocked | (previous) | (previous) |

## 3. Guardrail Rules

### G001: Size required before leaving triage
- **Check:** Issue has a `size/S` or `size/L` label
- **When:** Any transition FROM `status/triage`
- **Fail action:** Block transition, comment on Issue requesting size classification

### G002: size/L must not skip stages
- **Check:** Target status is the legal next state per size/L SOP
- **When:** Any transition of a `size/L` Issue
- **Fail action:** Block transition, comment listing required intermediate stages

### G003: Cannot skip review
- **Check:** `in-progress` cannot go directly to `done`
- **When:** Transition to `status/done`
- **Fail action:** Block transition, require `status/review-needed` first

### G004: Test evidence required for done
- **Check:** Find test evidence from (in priority order):
  1. Current session context — scan conversation for test runner output (pytest, go test, npm test, cargo test, etc.)
  2. PR diff — new/modified test files (`*_test.*`, `test_*.*`, `tests/**`)
  3. CI status — GitHub Actions / Check Runs on associated PR
- **When:** Transition to `status/done` or PR creation
- **Fail action:** Add `beaver/missing-test` label, comment requesting evidence
- **On success:** Write test summary to PR body's Test Plan section or Issue comment

### G005: LOC limit on PR
- **Check:** Count added lines in core directories, excluding:
  - `**/*_test.*`, `**/test_*.*`, `**/tests/**`
  - `**/*.md`, `**/docs/**`
  - `*.pb.go`, `*_generated.*`, `*.lock`
- **Threshold:** 200 lines
- **When:** PR creation
- **Fail action:** Add `beaver/needs-split` label, warn developer (do not block — let developer confirm)

### G006: PR must have complete labels
- **Check:** Associated Issue has at least one `type/` label AND one `size/` label
- **When:** PR creation
- **Fail action:** Add `beaver/missing-context` label, list missing labels

## 4. Label Operations (gh command templates)

### Read all labels on an Issue
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --jq '.[].name'
```
Parse into structured data:
- `type`: first label matching `type/*`
- `size`: first label matching `size/*`
- `status`: first label matching `status/*`
- `priority`: first label matching `p*/*`
- `beaver_flags`: all labels matching `beaver/*`

### Set status label (atomic swap)
Remove all existing `status/*` labels, then add the new one. **Note:** label names containing `/` must be URL-encoded in the DELETE path (e.g., `status/triage` → `status%2Ftriage`).
```bash
# Remove current status label (URL-encode the label name)
gh api repos/{owner}/{repo}/issues/{number}/labels/{url_encoded_current_status_label} --method DELETE

# Add new status label
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={new_status_label}"
```

### Add beaver flag label
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={beaver_label}"
```

## 5. Project Config Reading

Read from Project V2 README's `beaver-config` YAML block:

```bash
gh project view {number} --owner {org} --format json --jq '.readme'
```

Parse the `yaml beaver-config` fenced block for:
- `repositories`: list of observed repos
- `issueRepo`: the repo hosting Beaver issues
- `customFields`: field name overrides (default: Level, Status, Progress)

## 6. Transition Execution Template

When a command-layer skill needs to transition an Issue's status:

1. Read current labels (Section 4)
2. Determine `size` from labels
3. Look up legal next states (Section 2)
4. Validate target state against guardrails (Section 3)
5. If all checks pass: execute atomic label swap (Section 4)
6. If any check fails: report failure, do NOT swap labels

## 7. QA Loop & HARD-GATE

Reusable Q&A discipline. Other beaver skills reference this section before any state-changing action (`gh api ... POST`, `git commit`, `gh pr create`, `gh project item-add`, label transitions).

### 7.1 When callers must invoke

A caller MUST invoke this section before any state-changing action when:
- Creating a new GitHub Issue (beaver-issue Create mode)
- Drafting a design doc section (beaver-design-doc Phase 2/3)
- Decomposing into sub-issues (beaver-decompose)

A caller MAY skip this section only when the action is purely a label transition / assignee update on an Issue that already has approved content (e.g., beaver-issue Claim mode).

### 7.2 HARD-GATE rule

Until the user has explicitly approved the current section per §7.5, the caller MUST NOT:
- Write files (other than ephemeral scratch buffers like `mktemp` used purely to assemble content for review)
- Run `gh api ... --method POST` / `--method PATCH` / `--method PUT`
- Run `gh project item-add` / `gh project item-edit`
- Run `git commit` / `git push`
- Run `gh pr create`
- Add or remove GitHub labels
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

Mandatory codebase discovery executed by the caller BEFORE the first §7 question. Output goes into a fixed-format "Discovery Brief" presented to the user. **HARD-GATE:** §7 Q&A may not begin until the Brief has been printed.

### 8.1 The three required actions

| ID | Action | Tool | Purpose |
|---|---|---|---|
| D1 | Recent activity | `Bash`: `git log --oneline -20` AND `git log --all --since="14 days ago" --oneline` | Anchor the issue against recent project direction |
| D2 | Keyword search | `Glob` over file names + `Grep` over file contents, for ≤ 5 keywords | Locate related code |
| D3 | Project conventions | `Read` repo-root `README.md` and `CLAUDE.md` (when present), plus `*/README.md` under any directory hit by D2 | Surface conventions, tech stack, special instructions |

### 8.2 Keyword extraction rules

- Pull keywords ONLY from the literal text of the issue title + objective. Do NOT invent synonyms or related terms.
- Keep slash-bearing identifiers intact (`status/triage`, `beaver-engine`, `type/bug`).
- Keep ≤ 5 keywords; trim down by removing stop words and single-character tokens.
- For a Chinese-only title, split on whitespace and `/`. If fewer than 2 tokens emerge, ask the user "Which 2-5 keywords should I search?" before running D2.

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

When the caller has already detected `type/bug` and the issue is `p/0-blocker`:
- D1 and D3 are still mandatory (skipping is forbidden).
- D2 keywords are restricted to error messages / stack-trace tokens / API names from the user's reproduction steps.
- The Brief must still be printed before any state-changing action; `p/0-blocker` shortens latency but does NOT skip discovery.
