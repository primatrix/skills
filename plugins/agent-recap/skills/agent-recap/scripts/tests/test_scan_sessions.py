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


if __name__ == "__main__":
    unittest.main()
