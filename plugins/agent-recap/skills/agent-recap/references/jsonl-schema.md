# Session jsonl schema reference

## Claude Code (`~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`)

Append-only jsonl. Each line is a JSON object. Top-level types observed:

| top-level `type` | meaning |
|---|---|
| `system` | initial system reminder |
| `user` | user message (or compact summary, see below) |
| `assistant` | assistant message (may contain `tool_use` content items) |
| `attachment` | file attachment |
| `file-history-snapshot` | snapshot of file state |
| `queue-operation` | internal queue event |
| `permission-mode` | mode changes |
| `last-prompt` | last prompt cache marker |

### Fields the parser must extract

- `cwd`: appears on most `user`/`assistant` lines; project working directory
- `gitBranch`: appears on most `user`/`assistant` lines; git branch at the time
- `sessionId`: session UUID
- `timestamp`: ISO 8601 UTC
- `version`: Claude Code CLI version (e.g. `"2.1.137"`)
- `message.content`: for `user` type, may be a string OR a list of items
  - When list: items have `type: "text" | "tool_result" | ...`
- `message.content[]` for `assistant` type: list of items
  - `type: "text"` — text response
  - `type: "tool_use"` — `{name: "Bash"|"Read"|"Edit"|"Write"|"Grep"|"Glob"|...}`
- `isCompactSummary: true` on a `user` line marks a compact rollup of pre-compact history

### Subagent files

Located at `<session-dir>/subagents/agent-<id>.jsonl`. Same schema. These MUST be excluded from the top-level session list but recorded under the parent session's `subagent_paths[]`.

## Codex CLI (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`)

Append-only jsonl. Each line has `timestamp` + `type` + `payload`. Top-level types:

| top-level `type` | `payload.type` (if any) | meaning |
|---|---|---|
| `session_meta` | (none) | first line; contains `id`, `cwd`, `originator`, `cli_version` |
| `turn_context` | (none) | `cwd`, `current_date`, sandbox/approval policies |
| `event_msg` | `task_started` | turn boundary |
| `event_msg` | `user_message` | `payload.message` is the user text |
| `event_msg` | `task_complete` | turn end, has `duration_ms` |
| `response_item` | `message` | assistant message; `payload.content` is a list |
| `response_item` | `function_call` | tool call (Codex equivalent of Claude's `tool_use`) |

### Fields the parser must extract

From `session_meta` (first line): `payload.id`, `payload.cwd`, `payload.cli_version`, `payload.timestamp`
From `event_msg`/`user_message`: `payload.message` (the user text)
From `event_msg`/`task_complete`: `payload.completed_at`
From `response_item`/`function_call`: `payload.name` (tool name)

Codex has no native "subagent" concept — there is no equivalent of Claude's `subagents/` subdirectory.

## Parser output schema (common, source-agnostic)

The parser MUST output the same shape regardless of `source`. Downstream code does not branch on source.

```json
{
  "id": "<session id or uuid>",
  "source": "claude" | "codex",
  "path": "<absolute path to jsonl>",
  "subagent_paths": ["..."],
  "cwd": "<absolute path or null>",
  "git_branch": "<branch or null>",
  "started_at": "<ISO 8601 or null>",
  "ended_at":   "<ISO 8601 or null>",
  "user_msg_count": <int>,
  "tool_stats": {"<ToolName>": <count>, ...},
  "first_user_msg": "<first 200 chars or null>",
  "last_user_msg":  "<first 200 chars or null>",
  "has_compact_summary": <bool>,
  "size_bytes": <int>
}
```

Comments on this shape:
- `subagent_paths` is claude-only; codex always `[]`
- `git_branch` is claude-only; codex always `null`
- `has_compact_summary` is claude-only; codex always `false`
