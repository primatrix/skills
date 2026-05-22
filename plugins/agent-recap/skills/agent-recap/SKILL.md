---
name: agent-recap
description: Mine local Claude/Codex session history to produce a structured work recap for the past 1-7 days, with optional sync to GitHub Issues. Trigger when the user asks to summarize their recent work, generate a daily/weekly report, or wants to see what they solved/researched/reviewed/was blocked on. Default range is 1 day.
argument-hint: "[time range, e.g. '今天' / '本周' / '3 days' / '7d']"
---

# /agent-recap — Local Claude/Codex session work recap

Scan local Claude Code and Codex CLI session jsonl files, classify each session into one of five work types (`solved` / `researched` / `reviewed` / `blocked` / `misc`), present the result as a Markdown checklist for the user to review, and optionally sync each entry to GitHub via `gh issue comment` or `/beaver-create`.

## Input

One optional natural-language time-range argument. Resolve to an integer `N` (days) and clamp to `[1, 7]`. If the user asks for > 7 days, silently cap at 7 and note it in the recap header.

| User input | N |
|---|---|
| (no argument) / `今天` / `today` | 1 |
| `昨天和今天` / `yesterday and today` | 2 |
| `本周` / `this week` | 7 |
| `过去 N 天` / `N days` / `Nd` | N |

## Workflow

### Phase 1: Scan session metadata

Run the deterministic scanner with the resolved N:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/agent-recap/scripts/scan_sessions.py --since <N>d
```

Output is one JSON document with `sessions[]` and `errors[]`. Per-session fields are documented in `references/jsonl-schema.md`.

Filter out trivial sessions before Phase 2:

- `user_msg_count == 0`
- `size_bytes < 1024`

If the scanner exits non-zero, print its stderr to the user and stop.

### Phase 2: Dispatch one Explore subagent per session

For each filtered session, dispatch one `Agent` tool call with `subagent_type: "Explore"`. Dispatch ALL subagents in parallel — multiple `Agent` tool calls in a single message — so they run concurrently.

Subagent prompt template:

```
Read these reference files first:
  - ${CLAUDE_PLUGIN_ROOT}/skills/agent-recap/references/jsonl-schema.md
  - ${CLAUDE_PLUGIN_ROOT}/skills/agent-recap/references/classification-rubric.md

Then analyze the session jsonl at <path> and any subagent paths: <subagent_paths>.

For jsonls larger than ~200 KB, use Bash `head` / `tail` / `grep` to sample
instead of `Read`-ing the whole file. Useful grep patterns: git commit /
gh pr / Edit / Write tool_use lines, plus tone cues like "完美" "搞定"
"还是不行" "为什么没用".

Classify per the rubric and return ONLY the JSON object specified at the
end of classification-rubric.md. Do NOT extract any GitHub issue / PR
reference — issue linkage is decided interactively in Phase 5.

Session metadata for context:
  cwd:             <cwd>
  git_branch:      <git_branch>
  tool_stats:      <tool_stats>
  user_msg_count:  <user_msg_count>
  first_user_msg:  <first_user_msg>
  last_user_msg:   <last_user_msg>
```

When a subagent returns:

- Valid JSON matching the contract → store it.
- Malformed / missing fields / timeout → mark the session as "parse failed", continue. Do NOT retry.
- If ALL subagents fail, still proceed to Phase 3 with empty groups plus a populated ⚠️ section, so the user can see what was scanned.

### Phase 3: Aggregate and print the recap

Group classified sessions by `type` in this fixed order: `solved`, `researched`, `reviewed`, `blocked`, `misc`. Within each group, sort by `ended_at` ascending. Render as Markdown:

```markdown
## Recap: 过去 <N> 天（<YYYY-MM-DD>）

### ✅ 解决（<count>）

**1. <topic>**（<session.id[:8]>, <project>, <ended_at>）
- **目的**: <subagent.purpose>
- **过程**: <subagent.process>
- **结果**: <subagent.outcome>

...

### 🔎 调研（<count>）
(same per-entry shape)

### 👀 Review（<count>）
(same per-entry shape)

### 🚧 被 Block（<count>）
(same per-entry shape)

### 🗒️ 杂项（<count>）— 默认不进同步候选

- **<session.id[:8]>** (<ended_at>, <project>) — <subagent.topic>

### ⚠️ 解析失败（<count>）
- session <id> — <one-line reason>
```

Rules:

- Always include every section header, even with zero items. Show `(0)` in the count and `(empty)` in the body.
- Always include the ⚠️ section. Print `(empty)` if no failures.
- `purpose` / `process` / `outcome` are reproduced **verbatim** from the subagent JSON. Do NOT paraphrase or compress.
- Do NOT print any GitHub issue / PR reference here. Issue linkage belongs to Phase 5.

### Phase 4: User review

After printing the recap, prompt the user:

> 请审阅以上清单。可以说"删掉第 X 条"、"第 Y 条改成 ..."、"合并第 A 和 B"、"确认无误"。

Apply edits in memory only (no disk writes yet). Loop until the user confirms or refuses sync.

If the user declines sync ("不同步" / "我先不发" / similar), skip Phase 5 entirely.

### Phase 5: Sync to GitHub (default dry-run, optional)

#### 5.0 — Cleanup expired intents files

Always run first:

```bash
mkdir -p ~/.agent-recap
find ~/.agent-recap -name "*-intents.json" -type f -mtime +30 -delete 2>/dev/null || true
```

Files named `keep-*.json` are exempt (the `*-intents.json` glob excludes them).

#### 5.1 — Per-entry issue-linkage decision

Present a per-entry decision table for every confirmed entry in `{solved, researched, reviewed, blocked}`. `misc` entries are excluded by default; only include them if the user explicitly opts them in.

```markdown
📌 issue 关联 / 创建 决策

| 编号 | 摘要 | 选项 |
|---|---|---|
| ✅1 | <topic> | skip / 关联 issue # / 创建新 issue |
| ✅2 | <topic> | skip / 关联 issue # / 创建新 issue |
| ... | ... | ... |

请按编号告诉我每条的处理方式：
- **skip** — 不进同步
- **关联 <owner/repo>#<N>** — 用 `gh issue comment` 在该 issue 下贴一条进展评论
- **创建新 issue** — 主动调用 `/beaver-create` skill 走完整 Beaver 流程
  （含 brainstorming QA + 自动 Status / Type / Size 字段填写 + 自动 commit/PR 钩子）

回复格式举例：
1: skip
2: 关联 sgl-project/sglang-jax#1234
3: 创建新 issue

或一句话："全 skip" / "我先不同步，只看清单"
```

Map each user reply to one `kind`:

- `skip` — no action.
- `comment_on_issue` — user supplied `owner/repo#N`. Phase 5.3 will `gh issue comment` there.
- `create_issue` — user wants a new issue. Phase 5.3 will invoke `/beaver-create` (NOT raw `gh issue create`).

#### 5.2 — Write intents.json and show the dry-run preview

Write `~/.agent-recap/<ISO-8601-timestamp>-intents.json`:

```json
{
  "generated_at": "<ISO 8601>",
  "actions": [
    {"kind": "comment_on_issue", "issue": "<owner/repo#N>", "body": "<...>", "source_sessions": ["<id>", ...]},
    {"kind": "create_issue", "repo": "<owner/repo>", "title": "<...>", "body": "<...>", "source_sessions": ["<id>", ...]},
    {"kind": "skip", "reason": "user_skipped", "topic": "<...>", "source_sessions": ["<id>", ...]}
  ]
}
```

When multiple entries share the same target issue (`comment_on_issue` kind), aggregate them into a single comment with bullet points per entry.

Print the dry-run preview and prompt for the final confirmation:

```
即将执行：
  1. gh issue comment <owner/repo>#<N>   (body 前 80 chars...)
  2. gh issue comment <owner/repo>#<N>   ...
  3. /beaver-create <owner/repo>         ...

⚠️ 这些内容将以你的 GitHub 身份发到对应 Issue/PR，请确认无误。
intents 已保存到 ~/.agent-recap/<filename>（30 天后自动清理；
改名为 keep-*.json 可永久保留）。

回复 "全部执行" / "只执行 1,3" / "取消"。
```

#### 5.3 — Execute selected actions

For each chosen action:

- **`kind == "comment_on_issue"`** — NEVER inline the body into the shell command (body is Markdown and almost certainly contains `"` / `` ` `` / `$` / `\` / newlines). Use a temp file:

  ```bash
  tmp=$(mktemp); printf '%s' "$BODY" > "$tmp"
  gh issue comment <owner/repo>#<N> --body-file "$tmp"
  rm -f "$tmp"
  ```

  Or equivalently, a stdin heredoc with a sentinel that does not appear in the body:

  ```bash
  gh issue comment <owner/repo>#<N> --body-file - <<'AGENT_RECAP_EOF'
  <body>
  AGENT_RECAP_EOF
  ```

- **`kind == "create_issue"`** — ALWAYS invoke the `/beaver-create` skill. `/beaver-create` runs the full Beaver lifecycle: brainstorming QA, Project V2 #14 Status / Type / Size field population, label hygiene, and engine guardrails. Pass the repo, title, and pre-rendered body. The same temp-file rule applies if `/beaver-create` shells out to `gh`. Do NOT call `gh issue create` directly.

- **`kind == "skip"`** — no action.

Capture each action's exit status. Append a `result` field to the on-disk `intents.json` per action so the user can retry failures by hand. Do NOT auto-retry.

Summarize at the end:

```
✅ 成功 X 条
❌ 失败 Y 条
  - 第 N 条: <stderr reason>
```

## Error handling

Format: `⚠️ <phase>: <one-line reason> (<locator>)`

| Phase | Failure | Action |
|---|---|---|
| 1 | scanner exits non-zero | Print stderr, stop. |
| 2 | one subagent fails | List in ⚠️ section, continue. |
| 2 | ALL subagents fail | Still emit recap (empty groups + populated ⚠️). |
| 5 | `gh` / `/beaver-create` non-zero | Per-action failure recorded in `intents.json`, continue, no retry. |

## Constraints

- Main agent NEVER reads session jsonls directly — only via the Phase 1 scanner output and the Phase 2 subagent summaries. This is the only thing keeping the main context from exploding on multi-MB sessions.
- `purpose` / `process` / `outcome` are reproduced verbatim from subagent JSON in Phase 3. Do not paraphrase.
- Issue linkage is interactive in Phase 5.1. NEVER auto-link from Phase 2 / Phase 3.
- Comment / issue bodies are NEVER inlined into shell commands — always via `--body-file` with `mktemp` or stdin heredoc.
- `~/.agent-recap/*-intents.json` files older than 30 days are auto-deleted at Phase 5.0. Rename to `keep-*.json` to preserve.
- Same instructions apply in Claude Code and Codex CLI. Runtime differences (parallel `Agent` dispatch vs sequential) are absorbed by the runtime, not by branching in this file.
