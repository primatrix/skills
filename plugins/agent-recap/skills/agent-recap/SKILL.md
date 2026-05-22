---
name: agent-recap
description: Mine local Claude/Codex session history to produce a structured work recap for the past 1-7 days, with optional sync to GitHub Issues. Trigger when the user asks to summarize their recent work, generate a daily/weekly report, or wants to see what they solved/researched/reviewed/was blocked on. Default range is 1 day.
argument-hint: "[time range, e.g. '今天' / '本周' / '3 days' / '7d']"
---

# agent-recap — Mine your AI agent history for a work recap

## What this skill does

You scan local Claude Code and Codex CLI session jsonl files, classify each session
into one of five work types (solved / researched / reviewed / blocked / misc), and
present the result as a Markdown checklist for the user to review and optionally
sync to GitHub Issues.

## Inputs

The user invokes you with an optional natural-language time range:

- "今天" / "today" / no argument → 1 day
- "昨天和今天" / "yesterday and today" → 2 days
- "本周" / "this week" → 7 days
- "过去 N 天" / "N days" / "Nd" → N days

**Always clamp N to the range [1, 7]**. If the user asks for more than 7 days,
silently cap at 7 and mention it in the output header.

## Five stages

### Stage 1 — Scan session metadata

Run the deterministic scanner:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/agent-recap/scripts/scan_sessions.py --since Nd
```

(Substitute the resolved N from the user's input.)

The output is a JSON document with `sessions[]` and `errors[]`. Each session
record contains: `id`, `source` (`claude` | `codex`), `path`, `subagent_paths[]`,
`cwd`, `git_branch`, `started_at`, `ended_at`, `user_msg_count`, `tool_stats`,
`first_user_msg`, `last_user_msg`, `has_compact_summary`, `size_bytes`.

**Filter out trivial sessions** before Stage 2:
- `user_msg_count == 0` (empty / aborted sessions)
- `size_bytes < 1024` (negligible content)

### Stage 2 — Dispatch one Explore subagent per session

For every remaining session, dispatch one `Agent` tool call with
`subagent_type: "Explore"`. **Dispatch all subagents in parallel** (multiple
Agent tool calls in a single message) so they run concurrently. The subagent
prompt template:

```
Read the session jsonl at <path> and any subagent paths: <subagent_paths>.
Also read these reference files first:
  - ${CLAUDE_PLUGIN_ROOT}/skills/agent-recap/references/jsonl-schema.md
  - ${CLAUDE_PLUGIN_ROOT}/skills/agent-recap/references/classification-rubric.md

Then classify this session per the rubric and return ONLY the JSON object
described at the end of classification-rubric.md (no other text).

The JSON MUST include `purpose` / `process` / `outcome` (in Chinese) so the
Stage 3 recap can show what the user wanted, how the agent worked, and the
current state — NOT just a one-line "evidence" snippet. There is no hard
length cap on these three fields; write as much as the entry genuinely
needs and no more (tight paragraphs over filler).

**Do NOT attempt to look up or guess any GitHub issue / PR number for the
`issue_ref` field — that field has been removed from the contract.** Issue
linkage is decided interactively by the user in Stage 5.1 after they review
the recap.

Session metadata for context:
  cwd: <cwd>
  git_branch: <git_branch>
  tool_stats: <tool_stats>
  user_msg_count: <user_msg_count>
  first_user_msg: <first_user_msg>
  last_user_msg:  <last_user_msg>
```

When a subagent returns:
- If the response parses as valid JSON matching the contract, store it.
- If the response is malformed or missing fields, mark this session as
  "parse failed" and continue. Do NOT retry.

### Stage 3 — Aggregate and print the recap

Group classified sessions by `type` in this fixed order:
`solved`, `researched`, `reviewed`, `blocked`, `misc`. Within each group,
sort by `ended_at` ascending. Print as Markdown:

```markdown
## Recap: 过去 <N> 天（<YYYY-MM-DD>）

### ✅ 解决（<count>）

**1. <topic>**（<session.id[:8]>, <project>, <ended_at>）
- **目的**: <中文：用户想做什么>
- **过程**: <中文：agent 关键步骤/工具/PR/commit>
- **结果**: <中文：当前状态/产出/用户最后确认>

(No hard length cap on these three fields — write as much as the entry
needs, no more. Tight paragraphs over filler.)

...

### 🔎 调研（<count>）
（同上格式）

### 👀 Review（<count>）
（同上格式）

### 🚧 被 Block（<count>）
（同上格式）

### 🗒️ 杂项（<count>）— 默认不进同步候选

- **<session.id[:8]>** (<ended_at>, <project>) — <一句杂项摘要>

### ⚠️ 解析失败（<count>）
- session <id> — <one-line reason>
```

Always include every section header, even if a section has zero items
(show `（0）` count and an empty body line). Always include the ⚠️ section
at the end; print "(空)" inside if there were no parse failures.

**Do NOT include any GitHub issue / PR reference in Stage 3.** Issue linkage
is decided interactively by the user in Stage 5.1 after they review the recap.

### Stage 4 — Human review

After printing the recap, say:

> 请审阅以上清单。可以说"删掉第 X 条"、"第 Y 条改成 ..."、"合并第 A 和 B"、"确认无误"。

Apply the user's edits in memory (do NOT write to disk yet). Repeat until
the user says "确认无误" or equivalent ("ok", "好"). If the user asks to stop
(refuses to sync), skip Stage 5 entirely and exit.

### Stage 5 — Sync to GitHub Issues (default dry-run)

**Step 5.0 — Run cleanup of expired intents files first:**

```bash
mkdir -p ~/.agent-recap
find ~/.agent-recap -name "*-intents.json" -type f -mtime +30 -delete 2>/dev/null || true
```

Files matching `~/.agent-recap/keep-*.json` are NEVER touched (the find
pattern `*-intents.json` excludes them by name).

**Step 5.1 — Ask the user to link each non-misc entry to an issue:**

After the user has confirmed the Stage 3 recap, present a per-entry decision
table for every entry where `type` ∈ {solved, researched, reviewed, blocked}
(misc is excluded by default — only include misc if user explicitly asks).

Present like this:

> 📌 issue 关联 / 创建 决策
>
> | 编号 | 摘要 | 选项 |
> |---|---|---|
> | ✅1 | <topic> | skip / 关联 issue # / 创建新 issue |
> | ✅2 | <topic> | skip / 关联 issue # / 创建新 issue |
> | ... | ... | ... |
>
> 请按编号告诉我每条的处理方式：
> - **skip** — 不进同步
> - **关联 <owner/repo>#<N>** — 会用 `gh issue comment` 在该 issue 下贴一条进展评论
> - **创建新 issue** — 会**主动调用 `/beaver-create` skill** 走完整 Beaver 流程
>   （含 brainstorming QA + 自动 Status / Type / Size 字段填写 + 自动 commit/PR 钩子）
>
> 回复格式举例：
> ```
> 1: skip
> 2: 关联 sgl-project/sglang-jax#1234
> 3: 创建新 issue
> ```
> 或一句话："全 skip" / "我先不同步，只看清单"

Based on user replies, map each entry to one of three `kind` values:
- `skip` — do nothing for this entry
- `comment_on_issue` — user supplied an `owner/repo#N`
- `create_issue` — user requested new issue creation (Stage 5.3 will
  **invoke the `/beaver-create` skill**, NOT raw `gh issue create`)

If a `misc` entry is opted in by the user, treat it the same way.

**Step 5.2 — Write intents.json and show the dry-run list:**

Write `~/.agent-recap/<ISO-8601-timestamp>-intents.json` with this shape:

```json
{
  "generated_at": "<ISO 8601>",
  "actions": [
    {
      "kind": "comment_on_issue",
      "issue": "<owner/repo#N>",
      "body": "<rendered comment body>",
      "source_sessions": ["<session id>", ...]
    },
    {
      "kind": "create_issue",
      "repo": "<owner/repo>",
      "title": "<title>",
      "body": "<body>",
      "source_sessions": [...]
    },
    {
      "kind": "skip",
      "reason": "user_skipped_unmatched",
      "topic": "<topic>",
      "source_sessions": [...]
    }
  ]
}
```

When multiple entries share the same `issue` (`comment_on_issue` kind),
**aggregate them into a single comment** with bullet points per entry.

Then print the dry-run list:

```
即将执行：
  1. gh issue comment primatrix/skills#42 --body "<first 80 chars>..."
  2. gh issue comment sgl-jax#1088 --body "..."
  3. /beaver-create primatrix/skills --title "..." --body "..."
  ...

⚠️ 这些内容将以你的 GitHub 身份发到对应 Issue/PR，请确认无误。
intents 已保存到 ~/.agent-recap/<filename>.json（30 天后自动清理；
改名为 keep-*.json 可永久保留）。

回复 "全部执行" / "只执行 1,3" / "取消"。
```

**Step 5.3 — Execute selected actions:**

For each chosen action:
- `kind == "comment_on_issue"` → **never inline the body in the shell command** (body is
  Markdown and almost certainly contains quotes, backticks, `$`, `\`, or newlines).
  Write the body to a temp file first, then pass it via `--body-file`:
  ```bash
  tmp=$(mktemp); printf '%s' "$BODY" > "$tmp"
  gh issue comment <owner/repo>#<N> --body-file "$tmp"
  rm -f "$tmp"
  ```
  Equivalently, you may use a stdin heredoc with a sentinel that does not appear in
  the body:
  ```bash
  gh issue comment <owner/repo>#<N> --body-file - <<'AGENT_RECAP_EOF'
  <body>
  AGENT_RECAP_EOF
  ```
- `kind == "create_issue"` → **always invoke the `/beaver-create` skill** (do NOT
  call `gh issue create` directly). `/beaver-create` runs the full Beaver issue
  lifecycle: brainstorming QA, Status / Type / Size field population on Project
  V2 #14, label hygiene, and any guardrail checks the Beaver engine enforces.
  Pass the repo, title, and pre-rendered body; if `/beaver-create` ultimately
  shells out to `gh`, use the same temp-file pattern above so the body is never
  inlined into the shell command.
- `kind == "skip"` → do nothing

Capture each action's exit status. After all actions run, summarize:

```
✅ 成功 X 条
❌ 失败 Y 条
  - 第 N 条: <reason from stderr>
```

Append a `result` field to each action in the on-disk intents.json so the user
can retry by hand if needed. Do NOT auto-retry.

## Error handling rules

1. **Stage 1 — scanner exits non-zero**: print the stderr to the user and stop.
2. **Stage 2 — subagent fails**: list that session in the ⚠️ section, continue.
3. **Stage 2 — ALL subagents fail**: still print the recap with all sections empty
   plus the ⚠️ section enumerating every failure. Do not stop.
4. **Stage 5 — gh / beaver call fails**: per-action failure, record reason,
   continue with remaining actions, do NOT retry.

Error message format:
```
⚠️ <stage>: <one-line reason> (<locator>)
```

## Working in Codex CLI

This skill works in both Claude Code and Codex. In Claude Code, Stage 2's
parallel `Agent` tool dispatch makes processing fast. In Codex, if subagent
dispatch isn't available, the runtime will degrade to having the main agent
read each session jsonl in turn — slower but the classification logic is the
same. Do not branch on the runtime; the same SKILL.md instructions cover both.

## Reference files

- `references/jsonl-schema.md` — Claude / Codex jsonl field reference
- `references/classification-rubric.md` — The 5-type rubric and JSON contract
