#!/usr/bin/env python3
"""Scan Claude and Codex session jsonl files for agent-recap.

Outputs a JSON document describing recent sessions; see references/jsonl-schema.md
for the output shape.
"""
import argparse
import datetime as _dt
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl_lines(path: Path):
    """Yield (lineno, parsed_obj) for valid lines; silently skip bad lines."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield lineno, entry


def _timestamp(value: str | None) -> float:
    """Compare ISO timestamps by time, not by their timezone spelling."""
    if not isinstance(value, str):
        return float("-inf")
    try:
        dt = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except ValueError:
        return float("-inf")


def _latest(current: str | None, candidate: str | None) -> str | None:
    return candidate if _timestamp(candidate) > _timestamp(current) else current


def _claude_user_text(entry: dict) -> str | None:
    """Extract user message text from a Claude `user`-type entry."""
    msg = entry.get("message") or {}
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    return t
    return None


def parse_claude_session(path: Path) -> dict[str, Any]:
    """Parse a Claude Code session jsonl into the common session-metadata shape."""
    session_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    user_msg_count = 0
    tool_stats: dict[str, int] = {}
    first_user_msg: str | None = None
    last_user_msg: str | None = None
    has_compact_summary = False

    for _lineno, entry in _read_jsonl_lines(path):
        if session_id is None:
            sid = entry.get("sessionId")
            if isinstance(sid, str):
                session_id = sid

        if cwd is None:
            c = entry.get("cwd")
            if isinstance(c, str):
                cwd = c

        if git_branch is None:
            b = entry.get("gitBranch")
            if isinstance(b, str):
                git_branch = b

        ts = entry.get("timestamp")
        if isinstance(ts, str):
            if started_at is None:
                started_at = ts
            ended_at = ts

        etype = entry.get("type")
        if etype == "user":
            if entry.get("isCompactSummary") is True:
                has_compact_summary = True
                continue  # do not count compact summary as a real user turn
            text = _claude_user_text(entry)
            if text is not None:
                user_msg_count += 1
                preview = text[:200]
                if first_user_msg is None:
                    first_user_msg = preview
                last_user_msg = preview

        elif etype == "assistant":
            msg = entry.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        name = item.get("name")
                        if isinstance(name, str):
                            tool_stats[name] = tool_stats.get(name, 0) + 1

    return {
        "id": session_id or path.stem,
        "source": "claude",
        "path": str(path),
        "subagent_paths": [],
        "cwd": cwd,
        "git_branch": git_branch,
        "started_at": started_at,
        "ended_at": ended_at,
        "user_msg_count": user_msg_count,
        "tool_stats": tool_stats,
        "first_user_msg": first_user_msg,
        "last_user_msg": last_user_msg,
        "has_compact_summary": has_compact_summary,
        "size_bytes": path.stat().st_size,
    }


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        text = "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
            and item.get("type") in ("text", "input_text", "output_text")
        )
        return text.strip() or None
    return None


def _response_user_text(content: Any) -> str | None:
    """Remove known harness context wrappers from response-only fallback logs."""
    text = _message_text(content) or ""
    text = re.sub(r"^# AGENTS\.md instructions\s*", "", text)
    wrapper = re.compile(
        r"^<(environment_context|recommended_plugins|user_instructions|INSTRUCTIONS|"
        r"codex_internal_context)(?:\s[^>]*)?>.*?</\1>\s*", re.DOTALL
    )
    while (match := wrapper.match(text)) is not None:
        text = text[match.end():]
    return text.strip() or None


def _codex_header(path: Path) -> dict[str, Any]:
    # Child logs can embed their parent's session_meta after their own header.
    for _, entry in _read_jsonl_lines(path):
        if entry.get("type") == "session_meta":
            payload = entry.get("payload")
            return payload if isinstance(payload, dict) else {}
    return {}


def _codex_parent(header: dict) -> tuple[bool, str | None]:
    source = header.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    is_child = header.get("thread_source") == "subagent" or subagent is not None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    parent = spawn.get("parent_thread_id") if isinstance(spawn, dict) else None
    if is_child:
        parent = parent or header.get("forked_from_id")
        if not parent and header.get("session_id") != header.get("id"):
            parent = header.get("session_id")
    return is_child, parent if isinstance(parent, str) else None


def parse_codex_session(path: Path) -> dict[str, Any]:
    """Parse a Codex CLI session jsonl into the common session-metadata shape."""
    session_id: str | None = None
    cwd: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    tool_stats: dict[str, int] = {}
    has_compact_summary = False
    header_seen = False
    turn = ""
    # Prefer the UI/legacy user event within each turn, never add its response mirror.
    messages: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen_messages: set[tuple[str, str, str]] = set()
    seen_calls: set[str] = set()

    def add_message(kind, text, ts, lineno, message_id=None, turn_id=None):
        if text is None:
            return
        key = (turn_id or turn, kind, message_id)
        if message_id and key in seen_messages:
            return
        if message_id:
            seen_messages.add(key)
        messages[turn_id or turn][kind].append((lineno, text, ts))

    for lineno, entry in _read_jsonl_lines(path):
        etype = entry.get("type")
        payload = entry.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        ts = entry.get("timestamp")
        if isinstance(ts, str):
            if started_at is None:
                started_at = ts

        if etype == "session_meta":
            if not header_seen:
                header_seen = True
                sid = payload.get("id")
                session_id = sid if isinstance(sid, str) else None
                c = payload.get("cwd")
                if isinstance(c, str):
                    cwd = c

        elif etype == "turn_context":
            turn = payload.get("turn_id") or turn
            c = payload.get("cwd")
            if isinstance(c, str) and cwd is None:
                cwd = c

        elif etype == "event_msg":
            ptype = payload.get("type")
            if ptype == "task_started":
                turn = payload.get("turn_id") or turn
            elif ptype == "user_message":
                msg = payload.get("message")
                if isinstance(msg, str):
                    add_message("legacy", msg, ts, lineno, payload.get("id"))
            elif ptype == "item_completed":
                item = payload.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "UserMessage":
                    add_message("ui", _message_text(item.get("content")), ts, lineno,
                                item.get("id"), payload.get("turn_id"))
                elif isinstance(item, dict) and item.get("type") in (
                    "AgentMessage", "Reasoning", "CommandExecution", "McpToolCall", "FileChange"
                ):
                    ended_at = _latest(ended_at, ts)
            elif ptype in ("agent_message", "task_complete"):
                ended_at = _latest(ended_at, ts)

        elif etype == "response_item":
            ptype = payload.get("type")
            if ptype == "message" and payload.get("role") == "user":
                add_message("response", _response_user_text(payload.get("content")),
                            ts, lineno, payload.get("id"))
            elif ptype == "message" and payload.get("role") == "assistant":
                ended_at = _latest(ended_at, ts)
            elif ptype in ("function_call", "custom_tool_call"):
                ended_at = _latest(ended_at, ts)
                call_id = payload.get("call_id")
                if call_id and call_id in seen_calls:
                    continue
                if call_id:
                    seen_calls.add(call_id)
                name = payload.get("name")
                if isinstance(name, str):
                    tool_stats[name] = tool_stats.get(name, 0) + 1
            elif ptype in ("function_call_output", "custom_tool_call_output", "reasoning"):
                ended_at = _latest(ended_at, ts)
        elif etype == "compacted":
            has_compact_summary = True

    user_messages = []
    for representations in messages.values():
        for kind in ("ui", "legacy", "response"):
            if representations.get(kind):
                user_messages.extend(representations[kind])
                break
    user_messages.sort(key=lambda item: item[0])
    for _, _, ts in user_messages:
        ended_at = _latest(ended_at, ts)

    return {
        "id": session_id or path.stem,
        "source": "codex",
        "path": str(path),
        "subagent_paths": [],
        "cwd": cwd,
        "git_branch": None,
        "started_at": started_at,
        "ended_at": ended_at,
        "user_msg_count": len(user_messages),
        "tool_stats": tool_stats,
        "first_user_msg": user_messages[0][1][:200] if user_messages else None,
        "last_user_msg": user_messages[-1][1][:200] if user_messages else None,
        "has_compact_summary": has_compact_summary,
        "size_bytes": path.stat().st_size,
    }


PARSERS = {
    "claude": parse_claude_session,
    "codex": parse_codex_session,
}


def scan_directory(
    root: Path,
    *,
    source: str,
    since_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Walk `root`, return (sessions, errors). Missing root → empty lists."""
    sessions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not root.exists():
        return sessions, errors

    parser = PARSERS[source]
    cutoff = time.time() - since_days * 86400

    # Collect all .jsonl files; classify each as top-level vs subagent
    all_files = list(root.rglob("*.jsonl"))
    subagent_files: list[Path] = []
    top_files: list[Path] = []
    for p in all_files:
        if p.stat().st_mtime < cutoff:
            continue
        if p.parent.name == "subagents":
            subagent_files.append(p)
        else:
            top_files.append(p)

    # Index subagents by their parent session directory
    subagents_by_parent_dir: dict[Path, list[Path]] = {}
    for sp in subagent_files:
        parent_dir = sp.parent.parent  # .../<session_uuid>/subagents/agent-x.jsonl → .../<session_uuid>
        subagents_by_parent_dir.setdefault(parent_dir, []).append(sp)

    codex_children: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    sessions_by_id: dict[str, dict] = {}
    for fp in sorted(top_files):
        try:
            if source == "codex":
                header = _codex_header(fp)
                is_child, parent_id = _codex_parent(header)
                if is_child:
                    if parent_id:
                        codex_children[parent_id].append((header.get("id") or fp.stem, fp))
                    continue
            meta = parser(fp)
        except Exception as exc:  # parser blew up entirely
            errors.append({"path": str(fp), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        # mtime is a cheap prefilter; settings/token bookkeeping isn't recent work.
        if source == "codex" and _timestamp(meta["ended_at"]) < cutoff:
            continue
        # Attach subagents whose parent dir matches this session's id.
        # Claude convention: <encoded-cwd>/<session_uuid>.jsonl as the main file,
        # with <encoded-cwd>/<session_uuid>/subagents/agent-X.jsonl as the children.
        # The anchor is `fp.parent / fp.stem` (the directory named after the session id).
        sib = fp.parent / fp.stem
        if sib in subagents_by_parent_dir:
            meta["subagent_paths"] = [str(p) for p in sorted(subagents_by_parent_dir[sib])]

        previous = sessions_by_id.get(meta["id"])
        if previous is None or (
            _timestamp(meta["ended_at"]), meta["size_bytes"]
        ) > (_timestamp(previous["ended_at"]), previous["size_bytes"]):
            sessions_by_id[meta["id"]] = meta

    for meta in sessions_by_id.values():
        pending = [meta["id"]]
        visited = set()
        children = set(meta["subagent_paths"])
        while pending:
            parent_id = pending.pop()
            if parent_id in visited:
                continue
            visited.add(parent_id)
            for child_id, child_path in codex_children.get(parent_id, []):
                children.add(str(child_path))
                pending.append(child_id)
        meta["subagent_paths"] = sorted(children)
        sessions.append(meta)

    return sessions, errors


DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"
DEFAULT_CODEX_ROOT = Path.home() / ".codex" / "sessions"


def _parse_since(value: str) -> int:
    """Accept '1d'..'7d' or plain '1'..'7'. Returns integer days."""
    m = re.fullmatch(r"(\d+)d?", value)
    if not m:
        raise argparse.ArgumentTypeError(f"--since must look like '3d' or '3', got {value!r}")
    n = int(m.group(1))
    if not (1 <= n <= 7):
        raise argparse.ArgumentTypeError(f"--since must be between 1 and 7 days, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scan_sessions",
        description="Scan local Claude/Codex session jsonl files and emit metadata JSON.",
    )
    p.add_argument("--since", type=_parse_since, default=1,
                   help="How many days back to include (1-7). Default: 1.")
    p.add_argument("--source", choices=["both", "claude", "codex"], default="both",
                   help="Which agent's sessions to scan. Default: both.")
    p.add_argument("--claude-root", type=Path, default=DEFAULT_CLAUDE_ROOT,
                   help="Override Claude session root (for tests).")
    p.add_argument("--codex-root", type=Path, default=DEFAULT_CODEX_ROOT,
                   help="Override Codex session root (for tests).")
    args = p.parse_args(argv)

    all_sessions: list[dict[str, Any]] = []
    all_errors: list[dict[str, str]] = []
    if args.source in ("both", "claude"):
        s, e = scan_directory(args.claude_root, source="claude", since_days=args.since)
        all_sessions.extend(s)
        all_errors.extend(e)
    if args.source in ("both", "codex"):
        s, e = scan_directory(args.codex_root, source="codex", since_days=args.since)
        all_sessions.extend(s)
        all_errors.extend(e)

    # Sort sessions by ended_at descending (most recent first); None at the end
    all_sessions.sort(key=lambda x: _timestamp(x.get("ended_at")), reverse=True)

    doc = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "since_days": args.since,
        "sessions": all_sessions,
        "errors": all_errors,
    }
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
