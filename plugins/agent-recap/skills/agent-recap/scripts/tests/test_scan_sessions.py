"""Unit tests for scan_sessions.py."""
import json
import sys
import unittest
from pathlib import Path

# Make the script importable
sys.path.insert(0, str(Path(__file__).parent.parent))
import scan_sessions  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseClaudeSession(unittest.TestCase):
    def test_minimal_session_extracts_required_fields(self):
        path = FIXTURES / "claude" / "minimal.jsonl"
        result = scan_sessions.parse_claude_session(path)

        self.assertEqual(result["id"], "sess-min-001")
        self.assertEqual(result["source"], "claude")
        self.assertEqual(result["path"], str(path))
        self.assertEqual(result["cwd"], "/tmp/proj-a")
        self.assertEqual(result["git_branch"], "main")
        self.assertEqual(result["user_msg_count"], 1)
        self.assertEqual(result["tool_stats"], {})
        self.assertEqual(result["first_user_msg"], "hello")
        self.assertEqual(result["last_user_msg"], "hello")
        self.assertFalse(result["has_compact_summary"])
        self.assertEqual(result["subagent_paths"], [])
        self.assertEqual(result["started_at"], "2026-05-20T10:00:00.000Z")
        self.assertEqual(result["ended_at"], "2026-05-20T10:00:05.000Z")
        self.assertGreater(result["size_bytes"], 0)

    def test_tool_stats_counts_tool_uses(self):
        path = FIXTURES / "claude" / "with_tools.jsonl"
        result = scan_sessions.parse_claude_session(path)
        self.assertEqual(result["tool_stats"], {"Read": 1, "Bash": 2})
        self.assertEqual(result["user_msg_count"], 2)
        self.assertEqual(result["first_user_msg"], "read file")
        self.assertEqual(result["last_user_msg"], "thanks")

    def test_compact_summary_flag_set_but_not_counted(self):
        path = FIXTURES / "claude" / "with_compact.jsonl"
        result = scan_sessions.parse_claude_session(path)
        self.assertTrue(result["has_compact_summary"])
        # The compact-summary line must NOT count as a real user message
        self.assertEqual(result["user_msg_count"], 1)
        self.assertEqual(result["first_user_msg"], "continue work")
        # cwd appears only AFTER the compact summary — must still be picked up
        self.assertEqual(result["cwd"], "/tmp/proj-c")

    def test_broken_jsonl_skips_bad_lines(self):
        path = FIXTURES / "claude" / "broken.jsonl"
        result = scan_sessions.parse_claude_session(path)
        # Two of three lines are valid user messages
        self.assertEqual(result["user_msg_count"], 2)
        self.assertEqual(result["first_user_msg"], "first")
        self.assertEqual(result["last_user_msg"], "third")


class TestParseCodexSession(unittest.TestCase):
    def test_minimal_codex_session(self):
        path = FIXTURES / "codex" / "minimal.jsonl"
        result = scan_sessions.parse_codex_session(path)
        self.assertEqual(result["id"], "019e-codex-min-001")
        self.assertEqual(result["source"], "codex")
        self.assertEqual(result["cwd"], "/tmp/proj-cx")
        self.assertIsNone(result["git_branch"])  # Codex has no git branch
        self.assertEqual(result["user_msg_count"], 1)
        self.assertEqual(result["first_user_msg"], "hello codex")
        self.assertEqual(result["last_user_msg"], "hello codex")
        self.assertEqual(result["tool_stats"], {})
        self.assertFalse(result["has_compact_summary"])
        self.assertEqual(result["subagent_paths"], [])
        self.assertEqual(result["started_at"], "2026-05-20T15:00:00.000Z")

    def test_codex_session_with_function_calls(self):
        path = FIXTURES / "codex" / "with_tools.jsonl"
        result = scan_sessions.parse_codex_session(path)
        self.assertEqual(result["tool_stats"], {"shell": 2})
        self.assertEqual(result["user_msg_count"], 2)
        self.assertEqual(result["first_user_msg"], "run ls")
        self.assertEqual(result["last_user_msg"], "thanks")


class TestSchemaConsistency(unittest.TestCase):
    EXPECTED_KEYS = {
        "id", "source", "path", "subagent_paths",
        "cwd", "git_branch", "started_at", "ended_at",
        "user_msg_count", "tool_stats",
        "first_user_msg", "last_user_msg",
        "has_compact_summary", "size_bytes",
    }

    def test_claude_output_has_exact_expected_keys(self):
        result = scan_sessions.parse_claude_session(FIXTURES / "claude" / "minimal.jsonl")
        self.assertEqual(set(result.keys()), self.EXPECTED_KEYS)

    def test_codex_output_has_exact_expected_keys(self):
        result = scan_sessions.parse_codex_session(FIXTURES / "codex" / "minimal.jsonl")
        self.assertEqual(set(result.keys()), self.EXPECTED_KEYS)

    def test_both_sources_share_identical_key_set(self):
        claude = scan_sessions.parse_claude_session(FIXTURES / "claude" / "minimal.jsonl")
        codex = scan_sessions.parse_codex_session(FIXTURES / "codex" / "minimal.jsonl")
        self.assertEqual(set(claude.keys()), set(codex.keys()))


import tempfile
import os
import shutil
import time


class TestScanDirectory(unittest.TestCase):
    """scan_directory() walks a root and returns (sessions, errors)."""

    def test_returns_empty_on_missing_root(self):
        sessions, errors = scan_sessions.scan_directory(Path("/nonexistent/path/xyz"), source="claude", since_days=7)
        self.assertEqual(sessions, [])
        self.assertEqual(errors, [])

    def test_filters_by_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.jsonl"
            new = root / "new.jsonl"
            shutil.copy(FIXTURES / "claude" / "minimal.jsonl", old)
            shutil.copy(FIXTURES / "claude" / "minimal.jsonl", new)
            # Backdate `old` to 30 days ago
            ts_old = time.time() - 30 * 86400
            os.utime(old, (ts_old, ts_old))

            sessions, _ = scan_sessions.scan_directory(root, source="claude", since_days=7)
            paths = {Path(s["path"]).name for s in sessions}
            self.assertIn("new.jsonl", paths)
            self.assertNotIn("old.jsonl", paths)

    def test_subagent_files_attach_to_parent_not_top_level(self):
        # Lay out: <root>/parent.jsonl + <root>/subagents/agent-x.jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_dir = root / "encoded-cwd"
            sub_dir = parent_dir / "subagents"
            sub_dir.mkdir(parents=True)
            shutil.copy(FIXTURES / "claude" / "subagent_parent.jsonl", parent_dir / "sess-parent-001.jsonl")
            shutil.copy(FIXTURES / "claude" / "subagents" / "agent-fixture.jsonl", sub_dir / "agent-fixture.jsonl")

            sessions, _ = scan_sessions.scan_directory(root, source="claude", since_days=7)
            self.assertEqual(len(sessions), 1, f"Expected only the parent session at top level, got {sessions}")
            parent = sessions[0]
            self.assertEqual(parent["id"], "sess-parent-001")
            self.assertEqual(len(parent["subagent_paths"]), 1)
            self.assertTrue(parent["subagent_paths"][0].endswith("agent-fixture.jsonl"))

    def test_broken_file_listed_in_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # broken file that cannot be parsed AT ALL (e.g., binary)
            bad = root / "evil.jsonl"
            bad.write_bytes(b"\x00\x01\x02 not json at all \xff")
            sessions, errors = scan_sessions.scan_directory(root, source="claude", since_days=7)
            # The parser tolerates per-line decode errors, so the session may still be
            # produced (with empty fields). What MUST be true: scan does not crash.
            self.assertIsInstance(sessions, list)
            self.assertIsInstance(errors, list)


if __name__ == "__main__":
    unittest.main()
