# Beaver QA / Discovery / Doc Quality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Embed superpowers' QA loop, codebase discovery, and doc-writing discipline into `beaver-issue` and `beaver-design-doc` by adding three reusable Sections (7/8/9) to `beaver-engine` and refactoring the two skills to reference them.

**Architecture:** Approach A — `beaver-engine` becomes the shared host of QA & HARD-GATE rules, the Discovery Triad recipe, and Doc Quality constraints. `beaver-issue` Create mode and `beaver-design-doc` Phase 2/3/4 reference these new sections. `beaver-issue` additionally gains a Bug submode (type/bug → forced size/S, p/0-blocker → direct in-progress + @CODEOWNERS). Existing engine sections 1–6 (and G001–G006 numbering) are untouched.

**Tech Stack:** Markdown (SKILL.md), YAML frontmatter, `gh` CLI, `git`. No runtime/build/test framework — verification is grep/Read assertions against the rendered SKILL.md text.

**Source design:** `docs/plans/2026-04-20-beaver-qa-discovery-doc-quality-design.md`

**Verification model (TDD-for-markdown):**
For each SKILL edit:
1. Define an exact `grep -F` / `grep -E` pattern that MUST appear in the final SKILL.md.
2. Run grep BEFORE the edit → expect 0 matches (proves the edit is needed).
3. Make the Edit/Write.
4. Run grep AFTER the edit → expect ≥1 matches.
5. Re-read the file via `Read` to spot-check formatting (no broken fences, no orphan headings).
6. Commit.

---

## Task 1: Append Section 7 (QA Loop & HARD-GATE) to beaver-engine

**Files:**
- Modify: `plugins/beaver/skills/beaver-engine/SKILL.md` (append after current EOF, line 167)

**Step 1: Verify section is missing**

Run:
```bash
grep -cF "## 7. QA Loop & HARD-GATE" plugins/beaver/skills/beaver-engine/SKILL.md
```
Expected: `0`

**Step 2: Append Section 7**

Use Edit tool. `old_string`: the last existing line of the file (line 167 is `6. If any check fails: report failure, do NOT swap labels`). `new_string`: the same line followed by `\n\n` and the section content below.

Section content to append:

````markdown

## 7. QA Loop & HARD-GATE

Reusable Q&A discipline. Other beaver skills reference this section before any state-changing action (`gh api ... POST`, `git commit`, `gh pr create`, `gh project item-add`, label transitions).

### 7.1 When callers must invoke

A caller MUST invoke this section before any state-changing action when:
- Creating a new GitHub Issue (beaver-issue Create mode)
- Drafting a design doc section (beaver-design-doc Phase 2/3)
- Decomposing into sub-issues (future: beaver-decompose)

A caller MAY skip this section only when the action is purely a label transition / assignee update on an Issue that already has approved content (e.g., beaver-issue Claim mode).

### 7.2 HARD-GATE rule

Until the user has explicitly approved the current section per §7.5, the caller MUST NOT:
- Write files (other than `mktemp` scratch buffers used inside Step 5 of the caller's own workflow)
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

````

**Step 3: Verify section was added**

Run:
```bash
grep -cF "## 7. QA Loop & HARD-GATE" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "Approved? (y/revise)" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "### 7.5 Approval grammar" plugins/beaver/skills/beaver-engine/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Spot-check formatting**

Read the last 100 lines of `plugins/beaver/skills/beaver-engine/SKILL.md` and confirm:
- Section 6's last bullet is preserved untouched
- Section 7 has 5 sub-headings (7.1 / 7.2 / 7.3 / 7.4 / 7.5)
- No code fences are unclosed (every ```` opens and closes)

**Step 5: Commit**

```bash
git add plugins/beaver/skills/beaver-engine/SKILL.md
git commit -m "feat(beaver-engine): add Section 7 QA Loop & HARD-GATE

Reusable Q&A discipline for issue/design-doc/decompose. Defines HARD-GATE
on state-changing actions, sectional Q&A loop, skip-detection table, and
strict approval grammar.
"
```

---

## Task 2: Append Section 8 (Discovery Triad) to beaver-engine

**Files:**
- Modify: `plugins/beaver/skills/beaver-engine/SKILL.md` (append after Section 7)

**Step 1: Verify section is missing**

Run:
```bash
grep -cF "## 8. Discovery Triad" plugins/beaver/skills/beaver-engine/SKILL.md
```
Expected: `0`

**Step 2: Append Section 8**

Use Edit tool. `old_string`: the last line of Section 7 (the `MUST be treated as ...` paragraph end). To make `old_string` unique, include the closing line `Anything else ... ask again.` plus the surrounding context.

Section content to append:

````markdown

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

````

**Step 3: Verify**

Run:
```bash
grep -cF "## 8. Discovery Triad" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "### D1 Recent activity" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "0 files (NEW AREA)" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "### 8.5 Bug exception" plugins/beaver/skills/beaver-engine/SKILL.md
```
Expected: `1`, `1`, `1`, `1`

**Step 4: Spot-check** — Read the file tail; confirm Section 7 still ends correctly and Section 8 has 5 sub-sections.

**Step 5: Commit**

```bash
git add plugins/beaver/skills/beaver-engine/SKILL.md
git commit -m "feat(beaver-engine): add Section 8 Discovery Triad

Three required pre-QA actions (git log / grep / docs read), keyword
extraction rules, fixed-format Discovery Brief, anti-hallucination rules,
and Bug exception clause.
"
```

---

## Task 3: Append Section 9 (Doc Quality Constraints) to beaver-engine

**Files:**
- Modify: `plugins/beaver/skills/beaver-engine/SKILL.md` (append after Section 8)

**Step 1: Verify section is missing**

Run:
```bash
grep -cF "## 9. Doc Quality Constraints" plugins/beaver/skills/beaver-engine/SKILL.md
```
Expected: `0`

**Step 2: Append Section 9**

Use Edit tool. `old_string`: the closing line of Section 8 (`shortens latency but does NOT skip discovery.`) including enough context to be unique.

Section content to append:

````markdown

## 9. Doc Quality Constraints

Constraints on issue bodies and design-doc sections produced by callers. Used by §7's per-section approval gate.

### 9.1 Bilingual rule

| Element | Language |
|---|---|
| Section headings (`## 目标` / `## 验收标准` / `## 1. Context & Scope` etc.) | Follow caller's existing template (issue uses Chinese, design doc uses English numbered headings) |
| Body prose | Chinese as primary language |
| Technical nouns: API names, file paths, label names (`status/triage`), commands (`gh api`), commit hashes | English / original form, untranslated |
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

Issue bodies (created by beaver-issue) do not require a Provenance block but MUST satisfy:

- **Objective**: one sentence stating the user-facing outcome (not the implementation).
- **Acceptance criteria**: ≥ 2 items, each starting with a verb that yields a verifiable check (`运行 X 返回 Y` / `打开 URL Z 看到 W` / `pytest tests/foo.py 全部通过`). Avoid `improve` / `refactor` / `optimize` without a measurable target.
- **No invented file paths**: every path mentioned must appear in the Discovery Brief D2 hits or D3 file list.

### 9.5 Bug-mode adjustment

For `type/bug` issues, the body uses the Bug template (复现步骤 / 期望 / 实际 / 影响 / 环境). §9.4 applies with these substitutions:
- "Objective" → 复现步骤 (must be runnable / clickable, not abstract description).
- "Acceptance criteria" → 期望行为 + 实际行为, both concrete.
- Provenance for Bug issues is the source of the reproduction steps (e.g., "user-reported in §7 Q&A round 2" or "log file path X").

````

**Step 3: Verify**

Run:
```bash
grep -cF "## 9. Doc Quality Constraints" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "### 9.3 Section completeness checklist" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "<!-- provenance" plugins/beaver/skills/beaver-engine/SKILL.md
grep -cF "### 9.5 Bug-mode adjustment" plugins/beaver/skills/beaver-engine/SKILL.md
```
Expected: `1`, `1`, `≥1`, `1`

**Step 4: Spot-check** — Read the full file (`Read` plugins/beaver/skills/beaver-engine/SKILL.md). Confirm:
- Sections 1–6 unchanged
- Sections 7, 8, 9 appended in order
- File ends cleanly (no orphan code fences)

**Step 5: Commit**

```bash
git add plugins/beaver/skills/beaver-engine/SKILL.md
git commit -m "feat(beaver-engine): add Section 9 Doc Quality Constraints

Bilingual rule, anti-hallucination rule with Provenance block format,
5-row section completeness checklist, issue-body simplified checklist,
and Bug-mode adjustment.
"
```

---

## Task 4: Update beaver-issue header to reference §7/§8/§9

**Files:**
- Modify: `plugins/beaver/skills/beaver-issue/SKILL.md:11`

**Step 1: Verify current reference list does NOT include §7/§8/§9**

Run:
```bash
grep -F "References beaver-engine for:" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected output (line 11):
```
**References beaver-engine for:** state machine (Section 2), guardrails G001 (Section 3), label ops (Section 4), config reading (Section 5), transition execution (Section 6).
```

**Step 2: Update the reference line**

Use Edit:
- `old_string`:
  ```
  **References beaver-engine for:** state machine (Section 2), guardrails G001 (Section 3), label ops (Section 4), config reading (Section 5), transition execution (Section 6).
  ```
- `new_string`:
  ```
  **References beaver-engine for:** state machine (Section 2), guardrails G001 (Section 3), label ops (Section 4), config reading (Section 5), transition execution (Section 6), QA loop & HARD-GATE (Section 7), Discovery Triad (Section 8), doc quality constraints (Section 9).
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "QA loop & HARD-GATE (Section 7)" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "Discovery Triad (Section 8)" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "doc quality constraints (Section 9)" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "refactor(beaver-issue): reference engine §7/§8/§9 in header"
```

---

## Task 5: Add Mode Detect "Step 0: Type detection" to beaver-issue

**Files:**
- Modify: `plugins/beaver/skills/beaver-issue/SKILL.md` (Detect Mode section, lines 19–23)

**Step 1: Verify current Mode Detect lacks Bug branch**

Run:
```bash
grep -cF "Bug submode" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `0`

**Step 2: Replace the Detect Mode section**

Use Edit:
- `old_string`:
  ```
  ## Detect Mode

  - If an argument is provided (issue number): **Claim mode**
  - If no argument: **Create mode**

  ---
  ```
- `new_string`:
  ```
  ## Detect Mode

  - If an argument is provided (issue number): **Claim mode** (skip §7/§8/§9)
  - If no argument: **Create mode**, then run **Step 0** below to choose Feature submode or Bug submode.

  ### Step 0: Type detection (Create mode only)

  Ask the user once: "What kind of issue is this? (feat / bug / refactor / docs / chore)"

  - If the answer is `bug`: enter **Bug submode** (see "Bug Submode" section near the bottom of this file).
  - Otherwise: enter **Feature submode** (continues with Step 1 below).

  ---
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "### Step 0: Type detection (Create mode only)" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "Bug submode" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `1`, `≥2` (one in Step 0, one in cross-reference; the actual Bug submode body is added in Task 7)

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "feat(beaver-issue): add Step 0 type detection with Feature/Bug submode split"
```

---

## Task 6: Refactor Feature submode (Steps 1.5 & 2 & 3 & 4) to use §7/§8/§9

**Files:**
- Modify: `plugins/beaver/skills/beaver-issue/SKILL.md` (Create Mode body — Steps 1, 2, 3, 4)

**Step 1: Verify "Step 1.5" does not yet exist and Step 3 still auto-classifies size**

Run:
```bash
grep -cF "### Step 1.5: Discovery Triad" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "### Step 3: Auto-classify size" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `0`, `1`

**Step 2: Insert Step 1.5 (Discovery Triad) after Step 1**

Use Edit:
- `old_string`:
  ```
  ### Step 1: Load project config

  Check auto memory for `beaver-issue-defaults.md`. If found, present defaults for confirmation. Parse project config per engine Section 5.

  ### Step 2: Collect issue details
  ```
- `new_string`:
  ```
  ### Step 1: Load project config

  Check auto memory for `beaver-issue-defaults.md`. If found, present defaults for confirmation. Parse project config per engine Section 5.

  ### Step 1.5: Discovery Triad

  Execute engine Section 8 (Discovery Triad) using the user's draft title + objective as the keyword source. Print the Discovery Brief in the §8.3 format. The user does not need to "approve" the Brief — it is informational input for Step 2.

  HARD-GATE: Do NOT proceed to Step 2 until the Brief has been printed.

  ### Step 2: Collect issue details
  ```

**Step 3: Replace Step 2 body (one-shot collect → §7 Q&A loop with size-routed strength)**

Use Edit:
- `old_string`:
  ```
  ### Step 2: Collect issue details

  Collect one at a time:
  1. **Level**: Goal / Task / SubTask
  2. **Parent issue** (Task/SubTask only): list project items, filter by parent level, let user pick
  3. **Title**: concise issue title
  4. **Description**: structured as 目标 (Objective) and 验收标准 (Acceptance Criteria), written in Chinese
  5. **Type label** (`type/`): feat / bug / refactor / docs / chore
  6. **Priority label** (`p/`): choose one: `p0/blocker` / `p1/urgent` / `p2/high` / `p3/normal`
  ```
- `new_string`:
  ```
  ### Step 2: Collect issue details (engine §7 Q&A loop)

  Enter engine Section 7 Q&A loop. The first question MUST be size, because it routes the rest of the loop:

  1. **Size**: ask "size/S (small, single change) or size/L (multi-component / needs design)?"

  Then route:

  - **size/S route — minimal Q&A (3 questions):**
    1. Title (concise)
    2. Objective (一句话, Chinese)
    3. Acceptance criteria (≥ 2 verifiable items per §9.4)
    Defaults are used for: Level (= Task unless user mentions parent), Parent (skip if Level=Task at top of project), Type (= feat unless user mentioned bug in Step 0 — in which case the caller is already in Bug submode), Priority (= p3/normal unless user names urgency).

  - **size/L route — full Q&A (4 sections, each with §7.5 approval and §9.3 checklist):**
    1. Level + Parent: Goal / Task / SubTask; for Task/SubTask list project items and let user pick parent.
    2. Title.
    3. Objective + Scope (which subsystems / boundaries).
    4. Acceptance criteria + Stakeholders (who reviews / who is impacted).

  In both routes, Type and Priority labels are collected at the END (after the section-by-section loop), as separate single questions. Type defaults to `feat`; Priority is required and asked explicitly.

  HARD-GATE per §7.2: until the user approves the section per §7.5, do NOT call any `gh api` POST/PATCH or `gh project` write command.
  ```

**Step 4: Delete Step 3 (auto-classify) — size is now collected in Step 2**

Use Edit:
- `old_string`:
  ```
  ### Step 3: Auto-classify size

  Analyze the description and suggest `size/S` or `size/L`:
  - If description mentions multiple components, API changes + frontend + tests → suggest `size/L`
  - If description is focused on a single change → suggest `size/S`
  Present suggestion with reasoning. Wait for user confirmation.

  ### Step 4: Preview and confirm

  Show complete issue details in a structured preview. Wait for explicit approval.
  ```
- `new_string`:
  ```
  ### Step 3: Preview and §9.4 checklist

  Show complete issue details in a structured preview, then present the §9.4 issue-body checklist (Objective is one user-facing sentence / ≥ 2 verifiable acceptance items / no invented file paths). All three rows must be ☑ before continuing. Then ask `Approved? (y/revise)` per §7.5.
  ```

**Step 5: Renumber the remaining Steps (4 → was 5, 5 → was 6, 6 → was 7, 7 → was 8, 8 → was 9)**

Use 5 separate Edit calls to renumber, in order:
- `### Step 5: Create the Issue` → `### Step 4: Create the Issue`
- `### Step 6: Add to Project V2 and set fields` → `### Step 5: Add to Project V2 and set fields`
- `### Step 7: Link to parent (Task/SubTask only)` → `### Step 6: Link to parent (Task/SubTask only)`
- `### Step 8: Auto-transition from triage` → `### Step 7: Auto-transition from triage`
- `### Step 9: Report and save defaults` → `### Step 8: Report and save defaults`

**Step 6: Verify**

Run:
```bash
grep -cF "### Step 1.5: Discovery Triad" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "### Step 3: Preview and §9.4 checklist" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "### Step 3: Auto-classify size" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "### Step 9: Report and save defaults" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "### Step 8: Report and save defaults" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `1`, `1`, `0`, `0`, `1`

**Step 7: Spot-check** — Read the Create Mode section end-to-end. Confirm: Step 0 / 1 / 1.5 / 2 / 3 / 4 / 5 / 6 / 7 / 8 in order, no orphan headings.

**Step 8: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "refactor(beaver-issue): wire Create mode through engine §7/§8/§9

Insert Step 1.5 Discovery Triad. Replace one-shot 6-field collect with
size-routed §7 Q&A (size/S → 3 questions, size/L → 4 sections with
sectional approval). Drop auto-classify-size step. Add §9.4 checklist
preview gate. Renumber subsequent steps.
"
```

---

## Task 7: Add Bug submode body to beaver-issue

**Files:**
- Modify: `plugins/beaver/skills/beaver-issue/SKILL.md` (insert before `## Claim Mode` section)

**Step 1: Verify the Bug submode body is missing**

Run:
```bash
grep -cF "## Bug Submode" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `0`

**Step 2: Insert Bug Submode section before Claim Mode**

Use Edit:
- `old_string`:
  ```
  ## Claim Mode
  ```
- `new_string`:
  ```
  ## Bug Submode

  Entered when Step 0 detects `type/bug`. Differs from Feature submode: forced `size/S`, mandatory priority, and `p/0-blocker` direct-to-in-progress flow with CODEOWNERS @-mention.

  ### Bug Step 1: Load project config + Discovery Triad

  Same as Feature Step 1 + Step 1.5. For D2 keywords, prefer error messages, stack-trace tokens, API names from the user's reproduction steps. If those are not yet known, ask: "Paste the error message or stack trace so I can search the codebase." THEN run §8.

  ### Bug Step 2: Forced size/S

  Set `size/S` automatically. If the user objects ("this is actually a big bug, mark it size/L"), refuse with: "Bug issues are restricted to size/S per the project-management framework. If the underlying work is multi-component, please file a `type/refactor` or `type/feat` Issue instead." Do NOT proceed.

  ### Bug Step 3: §7 Q&A — Bug template (4 questions)

  Run engine §7 with these four required sections, each individually approved per §7.5:
  1. **复现步骤** (Reproduction steps): concrete, runnable / clickable. No abstract description.
  2. **期望行为** (Expected behavior).
  3. **实际行为** (Actual behavior): include the verbatim error log / screenshot reference.
  4. **影响范围 + 环境** (Impact + Environment): scope, OS, version/commit.

  ### Bug Step 4: Required priority

  Ask: "Priority? (p0/blocker / p1/urgent / p2/high / p3/normal)". This is a required field; do NOT default.

  If the answer is `p0/blocker`:
  - Mark for direct transition to `status/in-progress` (skip `status/triage`) in Bug Step 7.
  - Resolve CODEOWNERS @-mentions (Bug Step 5).

  ### Bug Step 5: Resolve @CODEOWNERS (p0/blocker only)

  Run:
  ```bash
  gh api repos/{owner}/{repo}/contents/.github/CODEOWNERS --jq '.content' | base64 -d
  ```
  If the file does not exist or returns 404, skip CODEOWNERS resolution and continue.

  Otherwise: parse CODEOWNERS, match against file paths surfaced in the Discovery Brief D2 hits (best-effort glob match: `*` matches segments, `**` matches subtrees). Collect the union of matched owner handles.

  Append to the Issue body:
  ```
  cc {@owner1 @owner2 ...}
  ```

  If no D2 hits matched any CODEOWNERS rule, leave a comment instead of failing: `cc (CODEOWNERS lookup found no match — please assign manually)`.

  ### Bug Step 6: Preview + §9.5 checklist

  Show the Bug Issue preview + §9.5-adjusted checklist. All rows must be ☑. Then ask `Approved? (y/revise)`.

  ### Bug Step 7: Create + transition

  Use the Bug body template (in "Issue Body Template" section). Same `gh api` create call as Feature Step 4. Then:
  - Always: `-f "labels[]=type/bug" -f "labels[]=size/S" -f "labels[]={priority}"`.
  - If `p0/blocker`: skip the standard "transition from triage" step. Directly atomic-swap `status/triage` → `status/in-progress` per engine §4. Validate G001 (size label present — guaranteed since size/S was forced).
  - Otherwise: same as Feature Step 7 (size/S → in-progress per §6).

  ### Bug Step 8: Report

  Same as Feature Step 8. Additionally, when `p0/blocker`: print "@-mentioned owners: {@owner1 @owner2}".

  ---

  ## Claim Mode
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "## Bug Submode" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "### Bug Step 5: Resolve @CODEOWNERS (p0/blocker only)" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "Bug issues are restricted to size/S" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "feat(beaver-issue): add Bug Submode

Type/bug → forced size/S, mandatory priority, p/0-blocker direct-to-
in-progress with CODEOWNERS @-mention. Bug Q&A uses the 4-section bug
template (复现/期望/实际/影响+环境) per engine §7.
"
```

---

## Task 8: Add Bug body template to beaver-issue's Issue Body Template section

**Files:**
- Modify: `plugins/beaver/skills/beaver-issue/SKILL.md` (Issue Body Template section)

**Step 1: Verify Bug template is missing**

Run:
```bash
grep -cF "### Bug" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `0`

**Step 2: Append Bug template after the existing Task / SubTask template**

Use Edit:
- `old_string`:
  ```
  ### Task / SubTask
  ```
  ```
  ## 目标

  {objective_in_chinese}

  ## 验收标准

  {acceptance_criteria_in_chinese}

  <!-- beaver-tracking
  repos:
    - {repo1}
  paths:
    - {path1}
  keywords:
    - {keyword1}
  -->
  ```
  ```
  (If this multi-block `old_string` is not unique, fall back to using just the closing fence `keywords:\n    - {keyword1}\n-->\n```` plus 2 surrounding lines.)
- `new_string`: same content, then a blank line, then:
  ```
  ### Bug
  ```
  ```
  ## 复现步骤

  {steps_in_chinese}

  ## 期望行为

  {expected}

  ## 实际行为

  {actual}
  （错误日志 / 截图：原文粘贴）

  ## 影响范围

  {scope}

  ## 环境

  - OS: {os}
  - Version/commit: {version}
  ```
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "### Bug" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "## 复现步骤" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "Version/commit" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "feat(beaver-issue): add Bug body template (复现步骤/期望/实际/影响/环境)"
```

---

## Task 9: Update beaver-issue Constraints

**Files:**
- Modify: `plugins/beaver/skills/beaver-issue/SKILL.md` (Constraints section near EOF)

**Step 1: Read current Constraints**

Read the last 10 lines of `plugins/beaver/skills/beaver-issue/SKILL.md`. Current text:
```
## Constraints

- One issue at a time
- Always preview before creating
- Never modify existing issues (except label transitions and assignee updates during claim)
```

**Step 2: Replace Constraints section**

Use Edit:
- `old_string`:
  ```
  ## Constraints

  - One issue at a time
  - Always preview before creating
  - Never modify existing issues (except label transitions and assignee updates during claim)
  ```
- `new_string`:
  ```
  ## Constraints

  - One issue at a time
  - Always preview before creating
  - Never modify existing issues (except label transitions and assignee updates during claim)
  - Create mode MUST run engine §8 Discovery Triad before the first §7 question
  - Bug submode forbids `size/L` (refuse with the message in Bug Step 2)
  - `p0/blocker` skips `status/triage` but does NOT skip §8 Discovery or §7 Q&A
  - Approval per §7.5 grammar is required before any `gh api` POST / `gh project` write
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "Create mode MUST run engine §8 Discovery Triad" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "Bug submode forbids \`size/L\`" plugins/beaver/skills/beaver-issue/SKILL.md
grep -cF "p0/blocker\` skips \`status/triage\` but does NOT skip" plugins/beaver/skills/beaver-issue/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-issue/SKILL.md
git commit -m "docs(beaver-issue): update Constraints to enforce §7/§8/§9 and Bug rules"
```

---

## Task 10: Update beaver-design-doc header to reference §7/§8/§9

**Files:**
- Modify: `plugins/beaver/skills/beaver-design-doc/SKILL.md:11`

**Step 1: Verify**

Run:
```bash
grep -F "References beaver-engine for:" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected:
```
**References beaver-engine for:** label ops (Section 4), state machine validation (Section 2).
```

**Step 2: Update**

Use Edit:
- `old_string`:
  ```
  **References beaver-engine for:** label ops (Section 4), state machine validation (Section 2).
  ```
- `new_string`:
  ```
  **References beaver-engine for:** label ops (Section 4), state machine validation (Section 2), QA loop & HARD-GATE (Section 7), Discovery Triad (Section 8), doc quality constraints (Section 9).
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "QA loop & HARD-GATE (Section 7)" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Discovery Triad (Section 8)" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "doc quality constraints (Section 9)" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-design-doc/SKILL.md
git commit -m "refactor(beaver-design-doc): reference engine §7/§8/§9 in header"
```

---

## Task 11: Refactor design-doc Phase 2 to invoke §8 first then run §7 per section

**Files:**
- Modify: `plugins/beaver/skills/beaver-design-doc/SKILL.md` (Phase 2 body, currently lines 73–141)

**Step 1: Verify Phase 2 currently has the inline Red Flags table and HARD-GATE block**

Run:
```bash
grep -cF "<HARD-GATE>" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "## Red Flags — STOP If You Catch Yourself Thinking" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1` (the inline Phase 2 HARD-GATE), `1` (the local Red Flags table at file end)

**Step 2: Replace Phase 2 General Rules block to delegate to engine §7/§8**

Use Edit:
- `old_string`:
  ```
  ## Phase 2: Context Collection (Iterative Q&A)

  <HARD-GATE>
  Do NOT skip Q&A. Do NOT "derive reasonable assumptions" from the issue body. Do NOT draft any design content until ALL 4 sections have been explored through Q&A with the user. The issue body is a starting point, NOT sufficient input for a design doc.
  </HARD-GATE>

  **General Rules:**
  - Ask only one question at a time
  - Agent must proactively search the codebase first (existing architecture, related files, test infrastructure, etc.), then ask questions based on search results
  - Continuously ask the user for additional context — for each section, ask whether there are related docs, code, or designs to reference
  - Encourage the user to use @ to reference files or paste relevant content
  - Do not move to the next section until the current one is clear
  - Do not fabricate technical details (library names, frameworks, architecture components) — all technical decisions must come from user input
  - Do not skip questions or make assumptions without sufficient context
  - For every design decision, ask about trade-offs — "Why this approach instead of alternatives?"

  Collect information across the following 4 sections one by one. Each section does not have a fixed list of questions; instead, dynamically determine the next question based on already-collected information and codebase search results:
  ```
- `new_string`:
  ```
  ## Phase 2: Context Collection (Iterative Q&A)

  <HARD-GATE>
  Do NOT skip Q&A. Do NOT "derive reasonable assumptions" from the issue body. Do NOT draft any design content until ALL 4 sections have been explored through Q&A with the user per engine §7.3. The issue body is a starting point, NOT sufficient input for a design doc.
  </HARD-GATE>

  ### Phase 2 Step 0: Discovery Triad (engine §8)

  Before the first question, execute engine §8 Discovery Triad. Use the issue title + objective as the keyword source. Print the Discovery Brief in the §8.3 format. Do NOT proceed to Phase 2 Section 1 until the Brief has been printed (HARD-GATE per §8 introduction).

  ### Phase 2 General Rules

  Q&A discipline is governed by engine §7 (one question at a time, approval grammar per §7.5, skip-detection per §7.4). Doc quality is governed by engine §9 (bilingual rule §9.1, anti-hallucination §9.2, completeness checklist §9.3 to be presented before each section's `Approved? (y/revise)` prompt).

  Design-doc-specific additions:
  - For every design decision, ask about trade-offs — "Why this approach instead of alternatives?"
  - Encourage the user to use @ to reference files or paste relevant content; the caller MUST `Read` any @-referenced file before continuing.
  - When the user mentions an external doc (e.g. wiki page), `WebFetch` or `Read` it before continuing.

  Collect information across the following 4 sections one by one. Each section does not have a fixed list of questions; dynamically determine the next question based on Discovery Brief findings and previously-collected answers:
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "### Phase 2 Step 0: Discovery Triad (engine §8)" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Q&A discipline is governed by engine §7" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Doc quality is governed by engine §9" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Ask only one question at a time" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `1`, `1`, `0` (the bullet was removed; rule now lives in §7)

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-design-doc/SKILL.md
git commit -m "refactor(beaver-design-doc): delegate Phase 2 Q&A to engine §7/§8/§9

Add Phase 2 Step 0 Discovery Triad gate. Replace inline general-rules
list with delegation to engine §7 (Q&A loop) and §9 (doc quality).
"
```

---

## Task 12: Add §9.3 checklist to Phase 3 Sectional Review

**Files:**
- Modify: `plugins/beaver/skills/beaver-design-doc/SKILL.md` (Phase 3, currently lines 143–159)

**Step 1: Verify**

Run:
```bash
grep -cF "Phase 3" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Is this section accurate?" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `1`

**Step 2: Replace Phase 3 Step 2**

Use Edit:
- `old_string`:
  ```
  ### Step 2: Present each section for approval

  Present each of the 4 sections individually. For each section:
  - Show the section content
  - Ask: "Is this section accurate? Any changes needed?"
  - If user requests changes, revise and re-present
  - Only proceed to next section after approval
  ```
- `new_string`:
  ```
  ### Step 2: Present each section for approval

  Present each of the 4 sections individually. For each section:
  - Show the section content
  - Show the engine §9.3 completeness checklist as a 5-row table; all rows MUST be ☑ before requesting approval. If any row is ☐, revise the section first, re-present, then re-show the table.
  - Ask the literal prompt: `Approved? (y/revise)`
  - Apply engine §7.5 approval grammar strictly (only `y/yes/ok/approve/approved/lgtm/继续/通过` count). Anything else means revise.
  - If user requests changes, revise and re-present
  - Only proceed to next section after explicit approval
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "engine §9.3 completeness checklist" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "literal prompt: \`Approved? (y/revise)\`" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Is this section accurate?" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `1`, `0`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-design-doc/SKILL.md
git commit -m "refactor(beaver-design-doc): Phase 3 sectional review uses §9.3 checklist + §7.5 approval grammar"
```

---

## Task 13: Add Provenance block + Open Questions to design doc template + Phase 4 Step 7 next-step hint

**Files:**
- Modify: `plugins/beaver/skills/beaver-design-doc/SKILL.md` (Design Doc Template section + Phase 4 Step 7)

**Step 1: Verify**

Run:
```bash
grep -cF "## 4. Alternatives Considered" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "## 5. Open Questions" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "<!-- provenance" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "### Step 7: Report" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `0`, `0`, `1`

**Step 2: Append §5 Open Questions + Provenance block to the design-doc template**

Use Edit:
- `old_string`: the closing template fence — find the unique block:
  ```
  ## 4. Alternatives Considered

  {Other viable approaches and their trade-offs. Focus on the trade-offs of each alternative and why the current approach is better given the stated goals.}
  ```
- `new_string`:
  ```
  ## 4. Alternatives Considered

  {Other viable approaches and their trade-offs. Focus on the trade-offs of each alternative and why the current approach is better given the stated goals.}

  ## 5. Open Questions

  {Items raised during Q&A that are recorded but not yet decided. For each: question, owner, expected resolution time. Empty list is acceptable but the section header must be present.}

  <!-- provenance
  - "<fact 1>" ← <source: Discovery D1/D2/D3 line, or QA round N>
  - "<fact 2>" ← <source>
  -->
  ```

**Step 3: Update Phase 4 Step 7 (Report) to include the next-step hint**

Use Edit:
- `old_string`:
  ```
  ### Step 7: Report

  Print summary: PR URL, design doc path, issue status (remains `design-pending`).
  ```
- `new_string`:
  ```
  ### Step 7: Report

  Print summary: PR URL, design doc path, issue status (remains `design-pending`).

  Then print the next-step hint (do NOT auto-invoke):
  ```
  Next step: after the design doc PR is reviewed and merged, run:
    beaver-decompose {owner}/{repo}#{number}
  to break the size/L Task into SubTasks.
  ```
  ```

**Step 4: Verify**

Run:
```bash
grep -cF "## 5. Open Questions" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "<!-- provenance" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Next step: after the design doc PR is reviewed and merged" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "beaver-decompose {owner}/{repo}#{number}" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `1`, `1`, `1`

**Step 5: Commit**

```bash
git add plugins/beaver/skills/beaver-design-doc/SKILL.md
git commit -m "feat(beaver-design-doc): add §5 Open Questions + Provenance block + Phase 4 next-step hint to decompose"
```

---

## Task 14: Trim local Red Flags table to design-doc-specific rows; rest delegated to §7.4

**Files:**
- Modify: `plugins/beaver/skills/beaver-design-doc/SKILL.md` (Red Flags table near EOF, currently lines 301–311)

**Step 1: Verify**

Run:
```bash
grep -cF "## Red Flags — STOP If You Catch Yourself Thinking" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "see engine §7.4" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `0`

**Step 2: Replace the Red Flags section**

Use Edit:
- `old_string`:
  ```
  ## Red Flags — STOP If You Catch Yourself Thinking

  | Thought | Reality |
  |---------|---------|
  | "Issue body has enough info to start drafting" | Issue body is a starting point. Q&A surfaces constraints, tradeoffs, and context you can't infer. |
  | "I'll derive reasonable assumptions" | Assumptions in a design doc become wrong decisions. Ask, don't assume. |
  | "The user seems busy, let me just write it" | A bad design doc wastes more time than Q&A. Keep asking. |
  | "I can fill in the technical details myself" | You don't know the team's tech stack, infra constraints, or preferences. Ask. |
  | "This section is obvious, I'll skip the questions" | Every section has hidden constraints. Ask anyway. |
  | "I'll ask all questions at once to save time" | One question at a time. Batching overwhelms and gets shallow answers. |
  | "I can derive the trade-offs from the code" | Trade-offs are design decisions, not code facts. They must come from the user. Ask. |
  ```
- `new_string`:
  ```
  ## Red Flags — STOP If You Catch Yourself Thinking

  General red flags are defined in engine §7.4. Below are design-doc-specific additions only:

  | Thought | Reality |
  |---------|---------|
  | "I can derive the trade-offs from the code" | Trade-offs are design decisions, not code facts. They must come from the user. Ask. |
  | "I'll skip §5 Open Questions to look more decisive" | Forcing closure on open items causes hallucination later. List them honestly. |
  | "The Provenance block is paperwork, I'll skip it" | Provenance is the audit trail that prevents future drift. It is required. |
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "General red flags are defined in engine §7.4" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "Issue body has enough info to start drafting" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "I can derive the trade-offs from the code" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `0`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-design-doc/SKILL.md
git commit -m "refactor(beaver-design-doc): trim Red Flags to design-doc-specific rows; delegate generic rows to engine §7.4"
```

---

## Task 15: Update beaver-design-doc Constraints

**Files:**
- Modify: `plugins/beaver/skills/beaver-design-doc/SKILL.md` (Constraints section near EOF)

**Step 1: Verify**

Run:
```bash
grep -F "## Constraints" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected match the existing block:
```
## Constraints

- Argument is required (must provide owner/repo#issue-number)
- Issue must have `size/L` + `status/design-pending` labels
- One question at a time during Q&A
- All sections must be individually approved before submission
- Issue status stays at `design-pending` — no automatic transition
- Wiki repo cloned to fixed path `~/Code/wiki`
```

**Step 2: Replace Constraints**

Use Edit:
- `old_string`:
  ```
  ## Constraints

  - Argument is required (must provide owner/repo#issue-number)
  - Issue must have `size/L` + `status/design-pending` labels
  - One question at a time during Q&A
  - All sections must be individually approved before submission
  - Issue status stays at `design-pending` — no automatic transition
  - Wiki repo cloned to fixed path `~/Code/wiki`
  ```
- `new_string`:
  ```
  ## Constraints

  - Argument is required (must provide owner/repo#issue-number)
  - Issue must have `size/L` + `status/design-pending` labels
  - Phase 2 MUST run engine §8 Discovery Triad before the first question (HARD-GATE)
  - Q&A follows engine §7 (one question at a time per §7.3, approval per §7.5 grammar)
  - Each section approval MUST present engine §9.3 completeness checklist with all rows ☑
  - Doc must include a `<!-- provenance -->` block per §9.2 and a `## 5. Open Questions` section
  - Issue status stays at `design-pending` — no automatic transition
  - Phase 4 Step 7 prints a next-step hint pointing at `beaver-decompose`, but does NOT auto-invoke it
  - Wiki repo cloned to fixed path `~/Code/wiki`
  ```

**Step 3: Verify**

Run:
```bash
grep -cF "Phase 2 MUST run engine §8 Discovery Triad" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "next-step hint pointing at \`beaver-decompose\`" plugins/beaver/skills/beaver-design-doc/SKILL.md
grep -cF "engine §9.3 completeness checklist with all rows ☑" plugins/beaver/skills/beaver-design-doc/SKILL.md
```
Expected: `1`, `1`, `1`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-design-doc/SKILL.md
git commit -m "docs(beaver-design-doc): update Constraints to enforce §7/§8/§9 and decompose hint"
```

---

## Task 16: Final integration walkthrough + cross-skill JSON validity

**Files:** read-only

**Step 1: Validate plugin manifests still parse**

Run:
```bash
python -c "import json; json.load(open('.claude-plugin/marketplace.json'))" && echo OK
python -c "import json; json.load(open('plugins/beaver/.claude-plugin/plugin.json'))" && echo OK
```
Expected: `OK` printed twice.

**Step 2: Validate YAML frontmatter on all three modified SKILLs**

Run:
```bash
for f in plugins/beaver/skills/beaver-engine/SKILL.md plugins/beaver/skills/beaver-issue/SKILL.md plugins/beaver/skills/beaver-design-doc/SKILL.md; do
  echo "=== $f ==="
  head -5 "$f"
done
```
Expected: each file starts with `---`, contains `name:` and `description:` fields, ends frontmatter with `---` on the 4th or 5th line.

**Step 3: Cross-reference grep — every "engine §N" reference resolves**

Run:
```bash
for ref in "§7" "§8" "§9" "§7.4" "§7.5" "§9.3" "§9.4" "§9.5"; do
  echo "=== Sections naming '$ref' in engine ==="
  grep -F "$ref" plugins/beaver/skills/beaver-engine/SKILL.md | head -3
  echo "=== References to '$ref' in callers ==="
  grep -rF "$ref" plugins/beaver/skills/beaver-issue/SKILL.md plugins/beaver/skills/beaver-design-doc/SKILL.md | head -5
done
```
Expected: every `§N` referenced in callers exists in engine; engine defines all referenced sub-sections.

**Step 4: Manual walkthrough — issue Create / size S**

Read `plugins/beaver/skills/beaver-issue/SKILL.md` end-to-end. Trace:
- Step 0 type detection → user picks `feat`
- Step 1 load defaults
- Step 1.5 Discovery Triad runs → Brief printed
- Step 2 §7 Q&A: size question → S; then 3 questions (title, objective, acceptance)
- Step 3 preview + §9.4 checklist → user types `y`
- Step 4 create issue → Step 5 add to project → Step 6 link parent (skipped if no parent) → Step 7 transition triage → in-progress → Step 8 report

Confirm no orphan steps, no contradictions.

**Step 5: Manual walkthrough — issue Create / Bug / p0/blocker**

Trace:
- Step 0 → user picks `bug` → enter Bug Submode
- Bug Step 1 Discovery Triad
- Bug Step 2 forced size/S
- Bug Step 3 four-question Q&A
- Bug Step 4 priority → p0/blocker
- Bug Step 5 CODEOWNERS lookup → cc handles
- Bug Step 6 preview + §9.5 checklist
- Bug Step 7 create + direct status/triage → status/in-progress
- Bug Step 8 report

Confirm flow.

**Step 6: Manual walkthrough — design-doc**

Read `plugins/beaver/skills/beaver-design-doc/SKILL.md` end-to-end. Trace:
- Phase 1 fetch + validate
- Phase 2 Step 0 Discovery Triad → Brief printed
- Phase 2 Sections 1–4 each with §7/§9 discipline
- Phase 3 Step 1 draft → Step 2 sectional review with §9.3 + §7.5 → Step 3 final
- Phase 4 Step 1–6 wiki PR + comment → Step 7 report with decompose hint

Confirm flow.

**Step 7: No commit needed** — this task is verification only. If any step fails, return to the failing earlier task.

---

## Task 17: Update design doc status frontmatter to "implemented"

**Files:**
- Modify: `docs/plans/2026-04-20-beaver-qa-discovery-doc-quality-design.md:4`

**Step 1: Verify**

Run:
```bash
grep -F "status: approved" docs/plans/2026-04-20-beaver-qa-discovery-doc-quality-design.md
```
Expected: `status: approved`

**Step 2: Edit**

Use Edit:
- `old_string`: `status: approved`
- `new_string`: `status: implemented`

**Step 3: Commit**

```bash
git add docs/plans/2026-04-20-beaver-qa-discovery-doc-quality-design.md
git commit -m "docs(plans): mark beaver-qa-discovery-doc-quality design as implemented"
```

---

## Done

After Task 17, the worktree contains:
- 3 new sections (§7/§8/§9) on `beaver-engine`
- `beaver-issue` Create mode wired through §7/§8/§9 with Bug submode
- `beaver-design-doc` Phase 2/3/4 wired through §7/§8/§9 with Provenance + Open Questions + decompose hint
- Design doc marked implemented

Recommended next: open PR per `superpowers:finishing-a-development-branch`.
