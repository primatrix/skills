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


def parse_codex_session(path: Path) -> dict[str, Any]:
    """Parse a Codex CLI session jsonl into the common session-metadata shape."""
    session_id: str | None = None
    cwd: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    user_msg_count = 0
    tool_stats: dict[str, int] = {}
    first_user_msg: str | None = None
    last_user_msg: str | None = None

    for _lineno, entry in _read_jsonl_lines(path):
        etype = entry.get("type")
        payload = entry.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        ts = entry.get("timestamp")
        if isinstance(ts, str):
            if started_at is None:
                started_at = ts
            ended_at = ts

        if etype == "session_meta":
            sid = payload.get("id")
            if isinstance(sid, str):
                session_id = sid
            c = payload.get("cwd")
            if isinstance(c, str) and cwd is None:
                cwd = c

        elif etype == "turn_context":
            c = payload.get("cwd")
            if isinstance(c, str) and cwd is None:
                cwd = c

        elif etype == "event_msg":
            ptype = payload.get("type")
            if ptype == "user_message":
                msg = payload.get("message")
                if isinstance(msg, str):
                    user_msg_count += 1
                    preview = msg[:200]
                    if first_user_msg is None:
                        first_user_msg = preview
                    last_user_msg = preview

        elif etype == "response_item":
            ptype = payload.get("type")
            if ptype == "function_call":
                name = payload.get("name")
                if isinstance(name, str):
                    tool_stats[name] = tool_stats.get(name, 0) + 1

    return {
        "id": session_id or path.stem,
        "source": "codex",
        "path": str(path),
        "subagent_paths": [],
        "cwd": cwd,
        "git_branch": None,
        "started_at": started_at,
        "ended_at": ended_at,
        "user_msg_count": user_msg_count,
        "tool_stats": tool_stats,
        "first_user_msg": first_user_msg,
        "last_user_msg": last_user_msg,
        "has_compact_summary": False,
        "size_bytes": path.stat().st_size,
    }


import time

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

    for fp in sorted(top_files):
        try:
            meta = parser(fp)
        except Exception as exc:  # parser blew up entirely
            errors.append({"path": str(fp), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        # Attach subagents whose parent dir matches this session's id (Claude convention:
        # <encoded-cwd>/<session_uuid>/subagents/...). If a sibling dir of the same
        # basename as the file (minus .jsonl) exists with subagents, attach those.
        sib = fp.parent / fp.stem
        if sib in subagents_by_parent_dir:
            meta["subagent_paths"] = [str(p) for p in sorted(subagents_by_parent_dir[sib])]
        # Also handle the test layout: subagents/ directly under fp.parent
        flat_sub = fp.parent
        if flat_sub in subagents_by_parent_dir:
            meta["subagent_paths"] = sorted(
                set(meta["subagent_paths"]) | {str(p) for p in subagents_by_parent_dir[flat_sub]}
            )

        sessions.append(meta)

    return sessions, errors


import argparse
import datetime as _dt
import re

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
    all_sessions.sort(key=lambda x: x.get("ended_at") or "", reverse=True)

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
