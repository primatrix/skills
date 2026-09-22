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

## Codex CLI and desktop (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`)

Append-only jsonl. Each line has `timestamp` + `type` + `payload`. Top-level types:

| top-level `type` | `payload.type` (if any) | meaning |
|---|---|---|
| `session_meta` | (none) | first line; contains `id`, `cwd`, `originator`, `cli_version` |
| `turn_context` | (none) | `cwd`, `current_date`, sandbox/approval policies |
| `event_msg` | `task_started` | turn boundary |
| `event_msg` | `user_message` | legacy user text in `payload.message` |
| `event_msg` | `item_completed` | modern UI item; `payload.item.type == "UserMessage"` carries user content |
| `event_msg` | `task_complete` | turn end, has `duration_ms` |
| `response_item` | `message` | `role` distinguishes user/assistant/developer; `content` contains text blocks |
| `response_item` | `function_call` / `custom_tool_call` | tool call; `name` is the tool and `call_id` identifies replays |
| `compacted` | (none) | compaction record, possibly containing replacement history |

### Fields the parser must extract

- Read identity and initial cwd from the **first** `session_meta`. A child can embed its parent's metadata later; that must not replace its identity.
- Group user observations by `turn_id` from `task_started` / `turn_context` and UI events. Prefer completed `UserMessage` items, then legacy `user_message` events, then response messages for each turn. Count a UI item ID once; preserve distinct messages with identical text. This avoids adding mirrored representations together.
- UI user content uses `text` blocks; response content commonly uses `input_text`. The response-only fallback strips known harness context wrappers, including environment/instructions/plugin listings and internal goal context, rather than treating them as user requests.
- Count both function and custom tool calls, deduplicating known `call_id`s. Detect top-level `compacted` records without recursively counting their replacement history.
- `ended_at` is the latest timestamp of user/assistant activity, reasoning, tool execution/results, or turn completion. Metadata, settings, token accounting, and compaction alone do not advance it. `started_at` remains the first record timestamp; a metadata-only log has `ended_at: null`.

### Subagents and duplicate exports

Codex child logs can live alongside root logs. The first header identifies them through `thread_source: "subagent"` or `source.subagent`. Resolve their parent from `source.subagent.thread_spawn.parent_thread_id`, then `forked_from_id`, then a distinct `session_id`. Attach descendants to a selected root's `subagent_paths`; exclude children from the top-level list, including children whose parent is outside the scan. A user-created fork is not automatically a subagent.

Within a source, duplicate session IDs retain the export with the latest activity, breaking ties by file size. Counts are not summed across exports. The scanner does not merge disjoint export histories.

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
- `subagent_paths` supports both sources; inherited history should not be summarized again as separate work.
- `git_branch` is claude-only; codex always `null`
- `has_compact_summary` detects Claude compact summaries and Codex `compacted` records.
- Message counts/previews and tool counts cover the retained file's full history, not only the requested recap window.
- `scan_directory` uses recent file mtime as a cheap candidate filter. Codex additionally requires recent `ended_at` activity, so touching settings in an old task does not include it. Claude retains its existing mtime selection and final-record `ended_at` behavior.
- Recap readers must restrict their conclusions to activity within the requested time window; a selected session can have begun much earlier.
