#!/usr/bin/env python3
"""Scan Claude and Codex session jsonl files for agent-recap.

Outputs a JSON document describing recent sessions; see references/jsonl-schema.md
for the output shape.
"""
import json
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
                yield lineno, json.loads(raw)
            except json.JSONDecodeError:
                continue


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
