---
name: beaver-decompose
description: "Decompose a Beaver Goal into Tasks (size/L) or a Task into SubTasks (size/S), guided by a design doc. Trigger when the user wants to split, breakdown, or decompose an issue into sub-issues."
argument-hint: "<owner/repo#issue-number> --design-doc <url>"
---

# Beaver Decompose

Decompose a `Goal` Issue into `Task` sub-issues (size/L), or a `Task` Issue into `SubTask` sub-issues (size/S). Reads a required design doc, drafts the decomposition, confirms each child issue with the user one at a time, then creates them via `gh api`.

**References beaver-engine for:** label ops (Section 4), state machine (Section 2), guardrails G001 (Section 3), transition execution (Section 6).

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

## Phase 2: Read Design Doc

Resolve `--design-doc <url>` based on form:

### Case A: GitHub PR URL (e.g. `https://github.com/primatrix/wiki/pull/123`)

```bash
# Get the PR's changed markdown files
gh pr view {url} --json files --jq '.files[] | select(.path | endswith(".md")) | .path'
# Then read the file from the PR's head ref
gh pr view {url} --json headRefName,headRepository --jq '.'
gh api repos/{head_owner}/{head_repo}/contents/{path}?ref={head_ref} --jq '.content' | base64 -d
```

### Case B: GitHub blob URL (e.g. `https://github.com/primatrix/wiki/blob/main/docs/designs/X.md`)

Convert to API form:
```bash
gh api repos/{owner}/{repo}/contents/{path}?ref={branch} --jq '.content' | base64 -d
```

### Case C: Local path (e.g. `~/Code/wiki/docs/designs/X.md`)

```bash
cat {expanded_path}
```

### Failure handling

If fetch fails (404, network error, file missing): stop with the specific error. Do NOT proceed to Phase 3 — drafting without a design doc is forbidden.

Store the full design doc text in memory; it is the primary input to Phase 3.

---

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
