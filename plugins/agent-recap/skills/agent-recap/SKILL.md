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
1. [<project> <issue_ref or ⚠️未匹配issue>] <topic>
   - 证据: <evidence[0]>
   - 来源: <session.id[:8]>...
...

### 🔎 调研（<count>）
...

### 👀 Review（<count>）
...

### 🚧 被 Block（<count>）
...

### 🗒️ 杂项（<count>）— 默认不同步
...

### ⚠️ 解析失败（<count>）
- session <id> — <one-line reason>
```

Always include every section header, even if a section has zero items
(show `（0）` count and an empty body line). Always include the ⚠️ section
at the end; print "(空)" inside if there were no parse failures.

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

**Step 5.1 — Handle unmatched-issue entries:**

For every entry where `issue_ref` is null AND `type` ∈ {solved, researched, reviewed, blocked}, ask the user:

> 还有 N 条未匹配 issue（编号 3, 7, 8）。请逐条决定（或统一选项）：
>   a. 跳过（不进同步）
>   b. 手动指定 issue 号（如 "3: sglang/sgl-jax#999"）
>   c. 创建新 issue（走 /beaver-create）

Map each entry to one of three `kind` values: `skip`, `comment_on_issue`,
or `create_issue` based on the user's response.

`misc` entries are EXCLUDED from sync by default. Only include them if
the user explicitly says so.

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
- `kind == "comment_on_issue"` → run `gh issue comment <owner/repo>#<N> --body "<body>"` via Bash
- `kind == "create_issue"` → invoke the `/beaver-create` skill with the repo, title, and body
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
- `references/smoke-test.md` — End-to-end manual verification steps
