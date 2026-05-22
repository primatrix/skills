# Classification rubric (for the Phase 2 Explore subagent)

Classify each session into exactly ONE of five mutually exclusive types. Pick the **primary outcome** — the activity that defines the session — even if other things also happened.

## The five types

### `solved` — The user's problem was resolved

Positive signals:
- Session ends with code changes that are committed, pushed, or PR'd.
- User explicitly confirms near the end (`好的` / `完美` / `搞定` / `perfect` / `thanks`).
- Tool stats show meaningful `Edit` / `Write` activity, not just reads.
- A new file is created or an existing bug is patched with no signs of regression.

Negative signals (push toward another type):
- Only read-only tools used → likely `researched`.
- User frustrated at the end without resolution → likely `blocked`.

### `researched` — Investigated without changing code

Positive:
- Tool stats heavily skewed to `Read` / `Grep` / `Glob` / `WebFetch`; few or zero `Edit` / `Write`.
- User asks `为什么` / `怎么` / `看一下` / `find out` / `investigate`.
- Session ends with the user saying `懂了` / `明白了` or summarizing findings, but no code changed.

Negative:
- Code was changed → `solved`.
- Mostly reading someone else's PR diff → `reviewed`.

### `reviewed` — Looked at someone else's PR or code

Positive:
- `gh pr view` / `gh pr diff` / `gh pr checkout` in Bash tool calls.
- User explicitly asks to "review" / `看一下 PR` / `check the PR`.
- Output ends with review comments / suggestions / approvals, not commits.

Negative:
- Heavy `Edit` / `Write` on the local branch → it became `solved` work, not just review.

### `blocked` — Repeated attempts without resolution

Positive:
- Same error / topic recurs across multiple turns without progress.
- User language shows frustration: `还是不行` / `为什么没用` / `卡住了` / `再试一下` / `still failing` / `stuck`.
- Multiple retries of the same Bash command with minor variations.
- Session ends without a fix or with the user giving up.

Negative:
- The blocker was eventually fixed → `solved`.
- User deliberately exploring a hard problem with curiosity, not frustration → `researched`.

### `misc` — Housekeeping or low-value work

Positive:
- File cleanup (`.DS_Store`, removing untracked dirs).
- Environment / config tweaks not tied to a feature.
- Personal experiments with no project artifact (trying out a CLI tool, conversational questions).
- Anything the user would not put into a status update.

Negative:
- The housekeeping was part of a bigger feature → roll it under that feature's `solved` entry.

## Output JSON contract (return ONLY this shape — no prose, no fences)

```json
{
  "type": "solved" | "researched" | "reviewed" | "blocked" | "misc",
  "project": "<repo name inferred from cwd, e.g. 'primatrix/skills' or 'sgl-jax'>",
  "topic": "<short Chinese title, ≤ 30 chars>",
  "purpose": "<Chinese: what the user was trying to do>",
  "process": "<Chinese: agent's key tools / commits / PRs / files touched>",
  "outcome": "<Chinese: current state / artifacts produced / what user said last>",
  "confidence": "high" | "medium" | "low"
}
```

Rules for the three Chinese fields:

- No hard character/sentence limit. Write as much as the entry genuinely needs and no more — prefer one tight paragraph over loose filler sentences.
- Be concrete: name the PR number, branch, file path, commit message, or error string you saw — only as part of the prose.
- Phase 3 reproduces these fields **verbatim** under `目的 / 过程 / 结果`. Anything you write here ends up in front of the user.

## Do NOT extract any issue / PR reference

This contract has no `issue_ref` field. Even if you see explicit `#<number>` tokens in the session, do NOT emit them as a separate field. Issue linkage is decided interactively by the user in Phase 5.1 after they review the Phase 3 recap. Auto-matching here only produces false positives and pre-empts a user decision.
