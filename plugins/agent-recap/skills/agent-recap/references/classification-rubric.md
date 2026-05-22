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

## Do NOT attempt to extract issue references

`issue_ref` has been removed from this contract. Even if you see explicit
`#<number>` tokens in the session, **do not** include them in the output JSON.
Issue linkage is decided interactively by the user in Stage 5.1 after they
review the Stage 3 recap. Trying to match issues here just causes false
positives and pre-empts a decision that belongs to the user.

## Output JSON contract (return ONLY this shape, nothing else)

```json
{
  "type": "solved" | "researched" | "reviewed" | "blocked" | "misc",
  "project": "<repo name from cwd, e.g. 'primatrix/skills' or 'sgl-jax'>",
  "topic": "<short Chinese title, ≤ 30 chars>",
  "purpose": "<Chinese: what the user was trying to do>",
  "process": "<Chinese: agent's key tools / commits / PRs / files touched>",
  "outcome": "<Chinese: current state / artifacts produced / what user said last>",
  "confidence": "high" | "medium" | "low"
}
```

`purpose` / `process` / `outcome` are reproduced verbatim by Stage 3 under
the headings "目的 / 过程 / 结果". No hard character/sentence limit — write
as much as the entry genuinely needs, but no more. Prefer one tight paragraph
over multiple loose sentences of filler. Be concrete: name the PR number,
branch, file path, commit message, or error you saw — but only as part of the
prose, never as a separate `issue_ref` field.
