# Classification rubric (for the Stage 2 subagent)

A session is classified into ONE of these five mutually exclusive types. Pick the **primary outcome** — the dominant activity that defines the session — even if other activities also happened.

## `solved` — The user's problem was resolved

**Signals (positive):**
- The session ends with code changes that are committed, pushed, or PR'd.
- The user explicitly confirms ("好的", "完美", "搞定", "perfect", "thanks") near the end.
- Tool stats show meaningful Edit / Write activity, not just reads.
- A new file is created, or an existing bug is patched, with no signs of regression.

**Signals (negative — would push toward other types):**
- Only read-only tools used → likely `researched`.
- User frustrated at the end without resolution → likely `blocked`.

## `researched` — Investigated without changing code

**Signals (positive):**
- Tool stats heavily skewed toward Read / Grep / Glob / WebFetch.
- Few or zero Edit / Write calls.
- The user asks "为什么", "怎么", "看一下", "find out", "investigate".
- The session ends with the user saying "懂了", "明白了", or summarizing what they learned, but no code changed.

**Signals (negative):**
- Code was changed → likely `solved`.
- Mostly reading PR diffs of someone else's work → likely `reviewed`.

## `reviewed` — Looked at someone else's PR or code

**Signals (positive):**
- `gh pr view`, `gh pr diff`, `gh pr checkout` in Bash tool calls.
- The user explicitly asks to "review", "看一下 PR", "check the PR".
- Output ends with review comments (suggestions, approvals) rather than commits.

**Signals (negative):**
- Heavy Edit / Write on the local branch → it became `solved` work, not just review.

## `blocked` — Repeated attempts without resolution

**Signals (positive):**
- The same error / topic recurs across multiple turns without progress.
- User language shows frustration: "还是不行", "为什么没用", "卡住了", "再试一下", "still failing", "stuck".
- Multiple retries of the same Bash command with minor variations.
- The session ends without a fix or with the user giving up.

**Signals (negative):**
- The blocking issue was eventually fixed → `solved`.
- The user was deliberately exploring a hard problem with curiosity, not frustration → `researched`.

## `misc` — Housekeeping or low-value work

**Signals (positive):**
- File cleanup (deleting .DS_Store, removing untracked dirs).
- Environment / config tweaks not tied to a feature (renaming a key, updating a path).
- Personal experiments with no project artifact (trying out a CLI tool, asking conversational questions).
- Anything you would not want to write into a project status update.

**Signals (negative):**
- The cleanup was part of a bigger feature → roll it under that feature's `solved` entry.

## Issue reference extraction (`issue_ref`)

Only set `issue_ref` when there is a **strong signal**. Weak signals MUST return `null`.

**Strong signals (set `issue_ref`, confidence=high):**
- The user or assistant text contains an explicit `#<number>` token (e.g. `#1088`).
- A commit message in a Bash call contains `#<number>` or `Closes #<number>`.
- The git branch name starts with or contains an issue number (`fix/42-something`).

**Medium signals (set `issue_ref`, confidence=medium):**
- A commit message references a PR number (e.g. `(#1234)`) without explicit issue link.

**Weak signals → return `null`:**
- Topic seems related to an issue title you remember seeing somewhere.
- The repo has an open issue with similar wording.
- The branch name *could* be tied to an issue but doesn't include the number.

Format: `<owner>/<repo>#<number>` (e.g. `primatrix/skills#42`). When the repo cannot be confidently inferred from cwd, use just `#<number>` (the user will be asked to disambiguate).

## Output JSON contract (return ONLY this shape, nothing else)

```json
{
  "type": "solved" | "researched" | "reviewed" | "blocked" | "misc",
  "project": "<repo name from cwd, e.g. 'primatrix/skills' or 'sgl-jax'>",
  "topic": "<1 sentence in Chinese, ≤ 30 chars>",
  "issue_ref": "<owner/repo#N or #N or null>",
  "issue_ref_confidence": "high" | "medium",
  "evidence": ["<≤ 80-char fragment>", "<≤ 80-char fragment>"],
  "confidence": "high" | "medium" | "low"
}
```

Omit `issue_ref_confidence` when `issue_ref` is null.
