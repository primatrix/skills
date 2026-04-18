---
date: 2026-04-18
topic: beaver-decompose skill
status: design-approved
---

# beaver-decompose Skill — Design Document

## 1. Context & Scope

The `beaver` plugin in this repo provides skills covering the GitHub-native Beaver issue lifecycle: creation (`beaver-issue`), design (`beaver-design-doc`), audit of existing decomposition (`beaver-audit`), PR with compliance (`beaver-pr`), focus dashboard (`beaver-focus`), and project report (`beaver-report`). All skills delegate to `beaver-engine` for label ops, state machine, and guardrails.

A gap exists between **design** and **audit**: there is no skill that takes a Goal or Task plus its design document and produces the child issues. Today this is done by a human in `beaver-issue` Create mode, one issue at a time. `beaver-decompose` fills this gap.

## 2. Design Goals

### 2.1 Goals

- Given a parent Issue (Goal or Task) and a design doc URL, produce the set of child issues that complete the parent's scope.
- Skip work already covered by existing OPEN sub-issues; only fill gaps.
- Interactively confirm each child issue with the user before any creation (per-child Q&A: accept / edit / delete / insert).
- Create child issues with correct type, size, status, parent link, project membership, and label set — matching `beaver-issue` Create mode output.
- Hand off to `beaver-audit` after Goal→Task decomposition.

### 2.2 Non-Goals

- Not a design-doc generator. Design must already exist.
- Not an auditor. Quality checks remain in `beaver-audit`.
- Not modifying existing sub-issues — only appending new ones.
- Not transitioning SubTask status to `in-progress` automatically — that is `beaver-issue` claim mode's job.
- Not enforcing 200 LOC at decomposition time — `beaver-pr` G005 enforces it at PR creation.
- Not rolling back partial creations on failure.

### 2.3 Success Metrics

- A user can decompose a typical Goal (3–6 Tasks) in one skill invocation with no manual `gh api` calls.
- Re-running on the same parent does not duplicate already-covered children.
- Created children pass `beaver-audit` (coverage / test definition checks) on the first attempt.

## 3. The Design

### 3.1 Skill Metadata

```yaml
---
name: beaver-decompose
description: "Decompose a Beaver Goal into Tasks (size/L) or a Task into SubTasks (size/S), guided by a design doc. Trigger when the user wants to split/breakdown an issue into sub-issues."
argument-hint: "<owner/repo#issue-number> --design-doc <url>"
---
```

### 3.2 Decomposition Mapping (fixed)

| Parent type | Child type | Child size | Child initial status |
|---|---|---|---|
| `Goal` | `Task` | `size/L` | `status/triage` → auto-transition `status/design-pending` |
| `Task` | `SubTask` | `size/S` | `status/triage` (kept; no auto-transition) |

Level is determined by reading the GitHub native issue type field (`gh api ... --jq '.type'`).

### 3.3 Workflow

```
Phase 1: Load & Validate
   ↓
Phase 2: Read Design Doc
   ↓
Phase 3: Draft decomposition (skip already-covered)
   ↓
Phase 4: Per-child Q&A confirm  ←─┐
   ↓                              │
All confirmed? ── no, edit/add/remove ┘
   ↓ yes
Phase 5: Create sub-issues (no rollback on failure)
   ↓
Phase 6: Offer beaver-audit (Goal→Task only)
```

#### Phase 1 — Load & Validate

1. Parse `owner/repo#number` and `--design-doc <url>`. Both required.
2. `gh api repos/{owner}/{repo}/issues/{number}` → fetch title, body, type, labels, state.
3. Validate:
   - state = OPEN
   - issue type ∈ {Goal, Task}
   - if type = Task: must have `status/ready-to-develop` label
4. Fetch existing sub-issues:
   ```bash
   gh api repos/{owner}/{repo}/issues/{number}/sub_issues \
     -H "X-GitHub-Api-Version: 2026-03-10"
   ```
   Existing OPEN sub-issues are NOT a blocker — collect their `{number, title, body summary}` as the "already covered" set for Phase 3.

#### Phase 2 — Read Design Doc

Resolve the URL based on form:
- GitHub PR URL → `gh pr view <url> --json files`, read changed design markdown
- GitHub blob URL → `gh api` for raw content
- Local path (e.g. `~/Code/wiki/...`) → read directly

Failure stops the skill before LLM drafting.

#### Phase 3 — Draft Decomposition

LLM input:
- Parent Issue body (Objective + Acceptance Criteria)
- Full design doc text
- "Already covered" set from Phase 1

LLM output:
- A "Skipped (already covered)" list — existing sub-issue numbers + titles
- A draft table:

```
| # | Title | Type | Size | Priority | One-line scope |
```

Constraints on draft items:
- Goal→Task: each Task aligns to a component / module / phase in the design doc
- Task→SubTask: each SubTask must have a complete functional description AND an end-to-end test plan
- LOC is NOT constrained at this phase (G005 enforces at PR time)

Defaults for type/priority labels: inherit from parent (user can override in Phase 4).

#### Phase 4 — Per-child Q&A Confirm

Iterate the draft list. For each item, render the full proposed issue body and prompt the user with four actions:

- **accept** → next item
- **edit** → user describes change, skill rewrites, re-prompts
- **delete** → skip this item entirely
- **insert at this position** → collect new item content

Loop until the user says "all OK, create".

#### Phase 5 — Create Sub-issues

For each confirmed item, sequentially:

```bash
BODY_FILE=$(mktemp)
cat > "$BODY_FILE" << 'BEAVEREOF'
{rendered_body}
BEAVEREOF

CHILD_URL=$(gh api repos/{owner}/{repo}/issues --method POST \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  -f title="{child_title}" -F body=@"$BODY_FILE" \
  -f type="{Task|SubTask}" \
  -f "labels[]=Control-By-Beaver" \
  -f "labels[]=type/{type}" \
  -f "labels[]=size/{L|S}" \
  -f "labels[]={priority}" \
  -f "labels[]=status/triage" \
  --jq '.html_url')

CHILD_NUMBER=$(echo "$CHILD_URL" | awk -F/ '{print $NF}')
CHILD_ID=$(gh api repos/{owner}/{repo}/issues/$CHILD_NUMBER --jq '.id')

gh api repos/{owner}/{repo}/issues/{parent_number}/sub_issues \
  --method POST -H "X-GitHub-Api-Version: 2026-03-10" \
  -F sub_issue_id=$CHILD_ID

gh project item-add {projectNumber} --owner {org} --url "$CHILD_URL" --format json
# Then gh project item-edit to set Level/Status/Progress

# Goal→Task only: auto-transition triage → design-pending per engine Section 6
# Task→SubTask: keep status/triage

rm "$BODY_FILE"
```

On failure of any step for any item: stop, report which items succeeded (1..N-1) and which failed (N). Do NOT roll back.

Sub-issue link failure but issue created: report URL, ask user to manually link or rerun.
Project item-add failure: warn, do not stop — issue body itself is created.

#### Phase 6 — Offer Audit

If parent was a Goal (children are size/L): prompt "Run `beaver-audit` on the parent now?" — only proceed on user confirmation.

If parent was a Task: skip — beaver-audit only targets size/L parents.

### 3.4 Issue Body Templates

#### Goal→Task child body

```markdown
## 目标

{从 design doc 提炼的该 Task 的目标}

## 验收标准

{该 Task 的可验收要点,bullet list}

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

#### Task→SubTask child body

```markdown
## 目标

{该 SubTask 完整的功能描述}

## 验收标准

{可验收要点}

## 端到端测试方案

{具体测试路径:输入 → 执行 → 期望输出。说明用什么测试框架/命令,以及测试文件位置}

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

### 3.5 Trade-offs

| Decision | Why this, not the alternative |
|---|---|
| Single skill with embedded full workflow | Matches `beaver-design-doc` / `beaver-audit` style. Splitting into draft+apply skills was rejected because per-child interactive confirmation already covers the editability need. |
| Fixed Level→Size mapping (Goal→size/L, Task→size/S) | Matches user's stated semantic: Task=large dev work needing design review, SubTask=small directly-implementable. LLM-judged mapping was rejected as unpredictable. |
| Design doc REQUIRED for both decomposition modes | Issue body alone is insufficient for module boundary decisions. Without a design doc, decomposition produces wrong boundaries. |
| Don't reject when OPEN sub-issues exist; auto-skip covered scope | Common case: re-running decompose after partial manual creation. Rejecting would force the user to delete or workaround. |
| Don't enforce 200 LOC at decompose time | G005 already enforces at PR time. Enforcing twice is redundant and over-restricts SubTask granularity. |
| No auto-rollback on Phase 5 partial failure | Partial state is recoverable by user (manual link / delete). Auto-rollback risks deleting work the user wanted to keep. |
| Children inherit parent type/priority by default | Reduces Phase 4 friction. User can override per item if needed. |
| SubTask kept at `status/triage` after creation | `beaver-issue` claim mode owns triage→in-progress transitions. decompose should not bypass that ownership. |

### 3.6 Test Strategy

This is a skill (markdown), not code, so no automated test suite. Verify by:

- JSON validity check: skill is registered correctly (no plugin.json change needed — plugin.json doesn't enumerate skills in this repo).
- YAML frontmatter validity in `SKILL.md`.
- Manual dry-run on a real Goal issue with a real design doc; verify each phase output matches this design.

### 3.7 Deployment & Dependencies

- New file: `plugins/beaver/skills/beaver-decompose/SKILL.md`
- No new external scripts. All logic is shell + `gh` commands embedded in the skill body.
- Depends on `gh` CLI with `repo` and `project` scopes.
- References `beaver-engine` Sections 2, 3, 4 (no engine changes needed).

## 4. Alternatives Considered

### Alt 1: Two skills — `beaver-decompose-draft` + `beaver-decompose-apply`

Draft writes a markdown file with the proposed children; apply reads the file and creates issues.

**Rejected because:** the user chose per-child interactive confirmation, which makes the file-handoff intermediate state unnecessary. Adds a file-passing step and breaks single-skill style of `beaver-design-doc`.

### Alt 2: Extend `beaver-issue` with a `--decompose` flag

Add decomposition as a mode of the existing issue creation skill.

**Rejected because:** `beaver-issue` is already two modes (create / claim). Adding a third mode bloats one skill that has clear single-purpose semantics. Single-purpose skills are the established convention in this repo (one skill per command).

### Alt 3: LLM-judged size per child (not fixed Goal→L, Task→S)

Let the LLM decide each child's size based on content.

**Rejected because:** the user wants predictable semantics — "Task means big dev work needing design review, SubTask means small directly-implementable." LLM judgment introduces variance the user explicitly does not want.

## 5. Constraints & Red Flags

### Constraints

- Both arguments required: `<owner/repo#number>` and `--design-doc <url>`
- Issue type ∈ {Goal, Task}
- If parent = Task, must have `status/ready-to-develop`
- Existing OPEN sub-issues are skipped, never modified
- Sub-issue creation failures do NOT auto-rollback
- Audit prompt only after Goal→Task
- Child issue body in Chinese (matches `beaver-issue` convention)

### Red Flags — STOP If You Catch Yourself Thinking

| Thought | Reality |
|---|---|
| "Design doc is optional, issue body is enough" | Design doc is the only reliable source for module boundaries. Issue body usually has only objective + acceptance criteria, not enough. |
| "I can modify/merge the existing sub-issue" | Never modify existing sub-issues. Only append uncovered scope. |
| "Batch all N children in one prompt" | User chose per-child confirmation. One at a time, four actions: accept / edit / delete / insert. |
| "Task without ready-to-develop is fine to decompose" | Decomposing without an approved design produces wrong boundaries. Wait for design review. |
| "SubTask body can skip the test section" | SubTask must have an end-to-end test plan, otherwise G004 will block at done time. |
| "On failure, auto-rollback created issues" | Don't roll back. Let the user see partial state and choose retry / manual fix / delete. |
