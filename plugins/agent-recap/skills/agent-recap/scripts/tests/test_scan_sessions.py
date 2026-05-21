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


if __name__ == "__main__":
    unittest.main()
