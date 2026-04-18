# beaver-decompose Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `beaver-decompose` skill under `plugins/beaver/skills/` that takes a Goal/Task issue plus a design doc URL and creates child issues (Tasks or SubTasks) with per-child interactive confirmation.

**Architecture:** Single `SKILL.md` markdown file containing YAML frontmatter + 6-phase workflow. No external scripts. All logic embedded as `gh` CLI command templates following the conventions of `beaver-design-doc` and `beaver-issue`. References `beaver-engine` Sections 2/3/4 for label ops, state machine, and guardrails.

**Tech Stack:** Markdown + YAML frontmatter. `gh` CLI commands. No build, no test runner.

**Reference design doc:** `docs/plans/2026-04-18-beaver-decompose-design.md`

---

## Pre-flight: Conventions to follow

Before authoring, re-read these for style/format:
- `plugins/beaver/skills/beaver-design-doc/SKILL.md` — closest sibling (single-skill workflow with phases, Q&A, Red Flags table)
- `plugins/beaver/skills/beaver-audit/SKILL.md` — sibling that uses `gh api sub_issues` and posts comments
- `plugins/beaver/skills/beaver-issue/SKILL.md` — issue creation gh template
- `plugins/beaver/skills/beaver-engine/SKILL.md` — Sections 2, 3, 4 referenced

Match exactly:
- `name:` and `description:` frontmatter pattern
- "References beaver-engine for: ..." line right under the title
- Use `BEAVEREOF` heredoc terminator (per beaver-issue convention)
- Use `--method POST -F sub_issue_id=$CHILD_ID` (capital -F for integer typing per beaver-issue note)
- URL-encode label names with `/` for DELETE per beaver-engine Section 4
- Issue body in Chinese (目标 / 验收标准)

---

## Task 1: Create skill directory and stub frontmatter

**Files:**
- Create: `plugins/beaver/skills/beaver-decompose/SKILL.md`

**Step 1: Create directory**

Run:
```bash
mkdir -p plugins/beaver/skills/beaver-decompose
```
Expected: directory exists, no output.

**Step 2: Write frontmatter + title only**

Create `plugins/beaver/skills/beaver-decompose/SKILL.md` with EXACTLY this content:

```markdown
---
name: beaver-decompose
description: "Decompose a Beaver Goal into Tasks (size/L) or a Task into SubTasks (size/S), guided by a design doc. Trigger when the user wants to split, breakdown, or decompose an issue into sub-issues."
argument-hint: "<owner/repo#issue-number> --design-doc <url>"
---

# Beaver Decompose

Decompose a `Goal` Issue into `Task` sub-issues (size/L), or a `Task` Issue into `SubTask` sub-issues (size/S). Reads a required design doc, drafts the decomposition, confirms each child issue with the user one at a time, then creates them via `gh api`.

**References beaver-engine for:** label ops (Section 4), state machine (Section 2), guardrails G001 (Section 3), transition execution (Section 6).
```

**Step 3: Verify frontmatter is valid YAML**

Run:
```bash
python3 -c "
import sys, re
content = open('plugins/beaver/skills/beaver-decompose/SKILL.md').read()
m = re.match(r'---\n(.*?)\n---', content, re.DOTALL)
assert m, 'No frontmatter found'
import yaml
data = yaml.safe_load(m.group(1))
assert data.get('name') == 'beaver-decompose', f'name wrong: {data.get(\"name\")}'
assert 'description' in data
assert 'argument-hint' in data
print('OK:', data['name'])
"
```
Expected: `OK: beaver-decompose`

**Step 4: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "feat(beaver-decompose): scaffold skill with frontmatter"
```

---

## Task 2: Add Prerequisites and Workflow diagram

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Prerequisites and Workflow sections**

Append to `plugins/beaver/skills/beaver-decompose/SKILL.md`:

```markdown

## Prerequisites

- `gh auth status` must succeed
- Project scope: `gh auth refresh -s project` if missing
- Both arguments required:
  - `<owner/repo#issue-number>` — the parent Issue to decompose
  - `--design-doc <url>` — URL or local path to the design document

## Decomposition Mapping (fixed)

| Parent type | Child type | Child size | Child initial status |
|---|---|---|---|
| `Goal` | `Task` | `size/L` | `status/triage` → auto-transition to `status/design-pending` |
| `Task` | `SubTask` | `size/S` | `status/triage` (kept; no auto-transition) |

Level is determined by the GitHub native issue type field (`gh api ... --jq '.type.name'`).

## Workflow

```dot
digraph decompose {
    "Phase 1: Load & Validate" [shape=box];
    "Phase 2: Read Design Doc" [shape=box];
    "Phase 3: Draft Decomposition" [shape=box];
    "Phase 4: Per-child Q&A Confirm" [shape=box];
    "All confirmed?" [shape=diamond];
    "Phase 5: Create Sub-issues" [shape=box];
    "Phase 6: Offer Audit (Goal only)" [shape=box];

    "Phase 1: Load & Validate" -> "Phase 2: Read Design Doc";
    "Phase 2: Read Design Doc" -> "Phase 3: Draft Decomposition";
    "Phase 3: Draft Decomposition" -> "Phase 4: Per-child Q&A Confirm";
    "Phase 4: Per-child Q&A Confirm" -> "All confirmed?";
    "All confirmed?" -> "Phase 4: Per-child Q&A Confirm" [label="no, edit/add/remove"];
    "All confirmed?" -> "Phase 5: Create Sub-issues" [label="yes"];
    "Phase 5: Create Sub-issues" -> "Phase 6: Offer Audit (Goal only)";
}
```

---
```

**Step 2: Verify file still has valid frontmatter and is well-formed**

Run:
```bash
head -20 plugins/beaver/skills/beaver-decompose/SKILL.md
```
Expected: Frontmatter intact, title visible.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add prerequisites, mapping table, workflow diagram"
```

---

## Task 3: Phase 1 — Load & Validate

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Phase 1**

Append:

````markdown

## Phase 1: Load & Validate

### Step 1: Parse arguments

Extract `owner`, `repo`, `issue_number` from the first positional arg (format `owner/repo#number`). Extract `design_doc_url` from `--design-doc <url>`.

If either is missing or malformed, stop and inform user:
- "Usage: beaver-decompose <owner/repo#number> --design-doc <url>"

### Step 2: Fetch parent issue

```bash
gh api repos/{owner}/{repo}/issues/{number} \
  --jq '{title, body, state, type: (.type.name // null), labels: [.labels[].name]}'
```

### Step 3: Validate parent

Parse labels per engine Section 4. Verify:
- `state == "open"` — else stop: "Parent issue is closed; cannot decompose."
- `type` ∈ {`Goal`, `Task`} — else stop: "beaver-decompose only supports Goal or Task issues. This issue type is `{type}`."
- If `type == "Task"`: must have `status/ready-to-develop` label — else stop: "Task issues must be in status/ready-to-develop before decomposition. Current status: {status}. Run beaver-design-doc first if needed."

### Step 4: Fetch existing sub-issues (do not block)

```bash
gh api repos/{owner}/{repo}/issues/{number}/sub_issues \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  --jq '.[] | {number, title, state, body_summary: (.body | .[0:200])}'
```

Collect OPEN sub-issues into the "already covered" set. This set is passed to Phase 3 — existing sub-issues are NEVER modified, only the uncovered scope is generated.

If the API returns empty or errors with 404 (no sub-issues yet), continue with empty set.

---
````

**Step 2: Visual review**

Run:
```bash
sed -n '/^## Phase 1/,/^---$/p' plugins/beaver/skills/beaver-decompose/SKILL.md
```
Expected: Phase 1 section renders correctly, all 4 steps present.

**Step 3: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Phase 1 (load & validate)"
```

---

## Task 4: Phase 2 — Read Design Doc

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Phase 2**

Append:

````markdown

## Phase 2: Read Design Doc

Resolve `--design-doc <url>` based on form:

### Case A: GitHub PR URL (e.g. `https://github.com/primatrix/wiki/pull/123`)

```bash
# Get the PR's changed markdown files
gh pr view {url} --json files --jq '.files[] | select(.path | endswith(".md")) | .path'
# Then read the file from the PR's head ref
gh pr view {url} --json headRefName,headRepository --jq '.'
gh api repos/{head_owner}/{head_repo}/contents/{path}?ref={head_ref} --template '{{.content}}' | base64 -d
```

### Case B: GitHub blob URL (e.g. `https://github.com/primatrix/wiki/blob/main/docs/designs/X.md`)

Convert to API form:
```bash
gh api repos/{owner}/{repo}/contents/{path}?ref={branch} --template '{{.content}}' | base64 -d
```

### Case C: Local path (e.g. `~/Code/wiki/docs/designs/X.md`)

```bash
cat {expanded_path}
```

### Failure handling

If fetch fails (404, network error, file missing): stop with the specific error. Do NOT proceed to Phase 3 — drafting without a design doc is forbidden.

Store the full design doc text in memory; it is the primary input to Phase 3.

---
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Phase 2 (read design doc)"
```

---

## Task 5: Phase 3 — Draft Decomposition

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Phase 3**

Append:

````markdown

## Phase 3: Draft Decomposition

### Step 1: Build LLM input context

Concatenate:
- Parent Issue title + body (目标 + 验收标准)
- Full design doc text (from Phase 2)
- "Already covered" set from Phase 1, Step 4 — list of `{number, title, body_summary}`

### Step 2: Generate draft

Produce a draft list of children. Constraints:

- **Goal → Task**: each Task aligns to a component / module / phase named in the design doc. Use design doc section titles as the basis for Task scoping.
- **Task → SubTask**: each SubTask must have:
  - A complete functional description (what it does, inputs/outputs)
  - An end-to-end test plan (test path: input → execution → expected output, plus framework/command, plus test file location)
- **Coverage rule**: explicitly skip any scope that the "already covered" set addresses. If a draft Task overlaps with an existing OPEN sub-issue, drop it.
- **No LOC constraint at this phase** — `beaver-pr` G005 enforces 200 LOC at PR creation time.

### Step 3: Set defaults

For each draft child:
- `type/{label}`: inherit from parent (e.g. parent has `type/feat` → child has `type/feat`)
- `p*/{priority}`: inherit from parent
- `size/{L|S}`: from mapping table (Goal→L, Task→S)

User can override per item in Phase 4.

### Step 4: Render draft to user

Present in two parts:

```markdown
### Skipped (already covered by existing sub-issues)

- #{n} {title} — covers: {one-line summary}
- #{n} {title} — covers: {one-line summary}

### Drafted children

| # | Title | Type | Size | Priority | One-line scope |
|---|-------|------|------|----------|----------------|
| 1 | {title} | type/feat | size/L | p2/high | {scope} |
| 2 | ... | ... | ... | ... | ... |
```

Then announce: "Next, I will walk through each drafted child one at a time for your confirmation."

---
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Phase 3 (draft decomposition)"
```

---

## Task 6: Phase 4 — Per-child Q&A Confirm

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Phase 4**

Append:

````markdown

## Phase 4: Per-child Q&A Confirm

Iterate the draft list in order. For each item, render the FULL proposed issue body (using templates from "Issue Body Templates" section below) and prompt:

> Child #{i} of {N}: **{title}**
>
> [render full body]
>
> Choose action:
> - **accept** → keep as-is, move to next
> - **edit** → describe the change, I will rewrite and re-prompt
> - **delete** → drop this child entirely
> - **insert** → insert a new child at this position (collect title + body)

### Rules

- One item at a time. Do NOT batch multiple confirmations.
- After **edit**: rewrite the full body, re-render, re-prompt. Loop until accept/delete.
- After **insert**: collect new child via Q&A (title, type override?, priority override?, full body), then re-prompt the original item.
- After **delete**: do not store this item; move to the next.
- After all items processed, show final summary table:

```markdown
### Final decomposition (will be created)

| # | Title | Type | Size | Priority |
|---|-------|------|------|----------|
| ... | ... | ... | ... | ... |
```

Ask: "Create these {N} sub-issues now? (yes/no)"

If user says no: stop without creating anything. If yes: proceed to Phase 5.

---
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Phase 4 (per-child Q&A confirm)"
```

---

## Task 7: Phase 5 — Create Sub-issues

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Phase 5**

Append:

````markdown

## Phase 5: Create Sub-issues

For each confirmed child, sequentially execute:

### Step 1: Write body to temp file

```bash
BODY_FILE=$(mktemp)
cat > "$BODY_FILE" << 'BEAVEREOF'
{rendered_body}
BEAVEREOF
```

### Step 2: Create the issue

```bash
CHILD_URL=$(gh api repos/{owner}/{repo}/issues --method POST \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  -f title="{child_title}" -F body=@"$BODY_FILE" \
  -f type="{Task|SubTask}" \
  -f "labels[]=Control-By-Beaver" \
  -f "labels[]=type/{type}" \
  -f "labels[]=size/{L|S}" \
  -f "labels[]={priority}" \
  -f "labels[]=status/triage" \
  --template '{{.html_url}}')
```

If issue type API fails, retry without `-f type`.

### Step 3: Capture child id and link to parent

```bash
CHILD_NUMBER=$(echo "$CHILD_URL" | awk -F/ '{print $NF}')
CHILD_ID=$(gh api repos/{owner}/{repo}/issues/$CHILD_NUMBER --jq '.id')

gh api repos/{owner}/{repo}/issues/{parent_number}/sub_issues \
  --method POST -H "X-GitHub-Api-Version: 2026-03-10" \
  -F sub_issue_id=$CHILD_ID
```

> **NOTE:** Use `-F` (capital, `--field`) for `sub_issue_id` because it must be an integer. `-f` would send a string and 422.

### Step 4: Add to Project V2

```bash
gh project item-add {projectNumber} --owner {org} --url "$CHILD_URL" --format json
```

Then `gh project item-edit` to set Level={Task|SubTask}, Status=Not Started, Progress=0 (per `beaver-issue` Step 6).

### Step 5: Auto-transition (Goal→Task only)

If parent was a Goal (this child is a Task / size/L):
- Execute transition `status/triage` → `status/design-pending` per engine Section 6 (validates G001)

If parent was a Task (this child is a SubTask / size/S):
- Keep `status/triage` — do NOT auto-transition. Claim mode in `beaver-issue` will move to in-progress when a developer picks it up.

### Step 6: Cleanup

```bash
rm "$BODY_FILE"
```

### Failure handling

| Failure point | Action |
|---|---|
| Issue create fails | Stop the loop. Report: "Created {i-1}/{N}; failed at {i}: {error}". Do NOT roll back created issues. |
| Sub-issue link fails (issue created) | Report the orphan child URL. Continue to next item. User can manually link or rerun. |
| Project item-add fails | Warn, do not stop. Report URL with note "manually add to project". |
| Status transition fails | Warn, do not stop. Report which child needs manual transition. |

---
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Phase 5 (create sub-issues)"
```

---

## Task 8: Phase 6 — Offer Audit + Final Report

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append Phase 6**

Append:

````markdown

## Phase 6: Offer Audit & Report

### Step 1: Print summary

```markdown
## Decomposition Complete

Parent: {owner}/{repo}#{parent_number} ({Goal|Task})
Created: {N} {Task|SubTask} sub-issues

| # | URL | Status |
|---|-----|--------|
| 1 | {url} | created + linked + project added |
| ... | ... | ... |

Skipped (already covered): {count}
Failures: {count, with details}
```

### Step 2: Offer audit (Goal → Task only)

If parent was a `Goal`:

> Decomposed Goal #{n} into {N} Tasks. Run `beaver-audit {parent_number}` now to verify coverage, atomicity, and test definitions? (yes/no)

If yes: invoke beaver-audit on the parent issue number.
If no: stop; remind user they can run it manually later.

If parent was a `Task`: skip the audit prompt entirely (`beaver-audit` only targets size/L parents).

---
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Phase 6 (offer audit and report)"
```

---

## Task 9: Issue Body Templates

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append templates section**

Append:

````markdown

## Issue Body Templates

### Goal → Task child

```markdown
## 目标

{从 design doc 提炼的该 Task 的目标 — 完整一段}

## 验收标准

- {可验收要点 1}
- {可验收要点 2}
- ...

## 设计参考

- Parent: #{parent_number}
- Design Doc: {design_doc_url}
- 相关章节: {design doc 中与本 Task 对应的章节标题}

<!-- beaver-tracking
repos:
  - {repo}
paths:
  - {path1}
keywords:
  - {keyword1}
-->
```

### Task → SubTask child

```markdown
## 目标

{该 SubTask 完整的功能描述 — 输入、行为、输出}

## 验收标准

- {可验收要点 1}
- {可验收要点 2}

## 端到端测试方案

- 测试路径: {输入 → 执行 → 期望输出}
- 测试框架/命令: {pytest / go test / ...}
- 测试文件位置: {path/to/test_file}

## 设计参考

- Parent: #{parent_number}
- Design Doc: {design_doc_url}
- 相关章节: {对应章节}

<!-- beaver-tracking
repos:
  - {repo}
paths:
  - {path1}
keywords:
  - {keyword1}
-->
```

---
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add issue body templates"
```

---

## Task 10: Constraints + Red Flags

**Files:**
- Modify: `plugins/beaver/skills/beaver-decompose/SKILL.md` (append)

**Step 1: Append final sections**

Append:

````markdown

## Red Flags — STOP If You Catch Yourself Thinking

| Thought | Reality |
|---------|---------|
| "Design doc is optional, issue body is enough" | Design doc is the only reliable source for module boundaries. Issue body usually has only objective + acceptance criteria — not enough to scope sub-issues. |
| "I can modify or merge the existing sub-issue" | Never modify existing sub-issues. Only append uncovered scope. |
| "I'll batch all N children in one prompt" | One child at a time. Four actions only: accept / edit / delete / insert. |
| "Task without ready-to-develop is fine to decompose" | Decomposing without an approved design produces wrong boundaries. Stop and tell the user to complete design review first. |
| "SubTask body can skip the test section" | SubTask MUST have an end-to-end test plan, otherwise G004 will block at done time. |
| "On failure, I'll auto-rollback created issues" | Don't roll back. Report partial state and let the user choose retry / manual fix / delete. |
| "I'll auto-transition SubTask to in-progress" | SubTasks stay at `status/triage`. `beaver-issue` claim mode owns the transition to in-progress. |
| "I'll fabricate the design doc URL if user forgot it" | Both arguments are required. Stop and ask. Never invent a URL. |

## Constraints

- Both arguments required: `<owner/repo#number>` and `--design-doc <url>`
- Issue type must be `Goal` or `Task`
- Task issues must have `status/ready-to-develop` label
- Existing OPEN sub-issues are skipped, never modified
- Sub-issue creation failures do NOT auto-rollback
- Audit prompt only after Goal→Task decomposition
- Child issue body in Chinese (目标 / 验收标准 / 端到端测试方案)
- One child confirmation at a time during Phase 4
- LOC is NOT enforced at decompose time (G005 enforces at PR creation)
````

**Step 2: Commit**

```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): add Red Flags table and Constraints"
```

---

## Task 11: Validate final SKILL.md

**Step 1: Verify YAML frontmatter still valid**

Run:
```bash
python3 -c "
import re, yaml
content = open('plugins/beaver/skills/beaver-decompose/SKILL.md').read()
m = re.match(r'---\n(.*?)\n---', content, re.DOTALL)
data = yaml.safe_load(m.group(1))
assert data['name'] == 'beaver-decompose'
assert 'description' in data and len(data['description']) > 20
assert 'argument-hint' in data
print('Frontmatter OK')
"
```
Expected: `Frontmatter OK`

**Step 2: Check all 6 phases present**

Run:
```bash
grep -E "^## Phase [1-6]:" plugins/beaver/skills/beaver-decompose/SKILL.md
```
Expected: 6 lines, Phase 1 through Phase 6 in order.

**Step 3: Check required reference sections present**

Run:
```bash
grep -E "^## (Prerequisites|Workflow|Issue Body Templates|Red Flags|Constraints|Decomposition Mapping)" plugins/beaver/skills/beaver-decompose/SKILL.md
```
Expected: 6 section headers.

**Step 4: Confirm no script files were accidentally added**

Run:
```bash
ls plugins/beaver/skills/beaver-decompose/
```
Expected: only `SKILL.md` (no `scripts/` directory — design specifies no external scripts).

**Step 5: Confirm beaver plugin still loads (no marketplace.json change needed)**

Run:
```bash
python3 -c "
import json
m = json.load(open('.claude-plugin/marketplace.json'))
beaver = next((p for p in m['plugins'] if p['name'] == 'beaver'), None)
assert beaver, 'beaver plugin missing from marketplace'
print('Marketplace OK — beaver plugin entry:', beaver['name'], beaver.get('source', '?'))
"
```
Expected: `Marketplace OK — beaver plugin entry: beaver ...`

(Note: skills are auto-discovered from the `skills/` directory; no manifest update is required.)

---

## Task 12: End-to-end readability dry-run

**Step 1: Read the final SKILL.md as a fresh agent would**

Run:
```bash
cat plugins/beaver/skills/beaver-decompose/SKILL.md
```

**Verify by visual inspection:**

- Frontmatter at top, YAML valid
- Workflow diagram precedes Phase 1
- Each Phase has numbered steps with concrete `gh` commands
- Issue body templates have both Goal→Task and Task→SubTask variants
- Red Flags table present near the end
- Constraints listed last
- All `gh api` calls use `BEAVEREOF` heredoc terminator (consistent with `beaver-issue` and `beaver-audit`)
- No placeholder TODOs or `XXX` markers

**Step 2: Final commit if any cleanup needed, else nothing**

If the dry-run found issues, fix them and commit:
```bash
git add plugins/beaver/skills/beaver-decompose/SKILL.md
git commit -m "docs(beaver-decompose): polish based on dry-run review"
```

If clean, no further commit.

---

## Task 13: Open PR

**Step 1: Push branch**

Run:
```bash
git push -u origin HEAD
```

**Step 2: Open PR**

Run:
```bash
gh pr create --title "feat(beaver): add beaver-decompose skill" --body "$(cat <<'EOF'
## Summary

- New `beaver-decompose` skill that decomposes a Goal/Task issue into child issues using a required design doc
- Per-child interactive confirmation (accept / edit / delete / insert)
- Skips already-covered OPEN sub-issues; appends only the uncovered scope
- Goal→Task creates size/L children + auto-transitions to design-pending
- Task→SubTask creates size/S children, kept at status/triage
- Offers `beaver-audit` handoff after Goal→Task decomposition

## Design

See `docs/plans/2026-04-18-beaver-decompose-design.md` for the full design.

## Test plan

- [ ] Frontmatter validates as YAML
- [ ] All 6 phases present and numbered
- [ ] Manual dry-run on a Goal issue with a real design doc
- [ ] Manual dry-run on a Task issue (must require status/ready-to-develop)
- [ ] Verify created children have correct type/size/status labels and parent link
- [ ] Verify already-covered OPEN sub-issues are skipped, not modified

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Return the PR URL.
