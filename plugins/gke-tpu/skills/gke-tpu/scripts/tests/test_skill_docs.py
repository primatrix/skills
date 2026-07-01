import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[2]
PLUGIN_DIR = SKILL_DIR.parents[1]
REPO_ROOT = PLUGIN_DIR.parents[1]
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCES = SKILL_DIR / "references"


class SkillDocTests(unittest.TestCase):
    def test_entrypoint_is_concise_and_routes_to_tool_commands(self):
        text = SKILL_MD.read_text()
        self.assertLess(len(text.split()), 500)
        for command in (
            "init",
            "validate",
            "plan-nodepool",
            "render-workload",
            "delete-workload-plan",
            "delete-nodepool-plan",
        ):
            self.assertIn(command, text)

    def test_removed_workflows_are_not_command_entries(self):
        text = SKILL_MD.read_text()
        command_section = text.split("## Commands", 1)[1].split("##", 1)[0]
        self.assertNotIn("`sync`", command_section)
        self.assertNotIn("`run`", command_section)
        self.assertNotIn("`status`", command_section)

    def test_docs_explicitly_forbid_agent_private_state(self):
        text = SKILL_MD.read_text()
        self.assertIn(".claude", text)
        self.assertIn(".codex", text)
        self.assertIn("~/.agents", text)
        self.assertIn("Do not write", text)

    def test_references_are_not_legacy_workflows(self):
        names = {path.name for path in REFERENCES.glob("*.md")}
        self.assertNotIn("create.md", names)
        self.assertNotIn("sync.md", names)
        self.assertNotIn("run.md", names)
        self.assertNotIn("status.md", names)
        self.assertIn("config.md", names)
        self.assertIn("workload.md", names)
        self.assertIn("nodepool.md", names)
        self.assertIn("cleanup.md", names)

    def test_marketplace_and_readme_do_not_advertise_removed_sync_run_scope(self):
        files = [
            PLUGIN_DIR / ".claude-plugin" / "plugin.json",
            REPO_ROOT / ".claude-plugin" / "marketplace.json",
            REPO_ROOT / "README.md",
        ]
        for path in files:
            text = path.read_text()
            self.assertNotIn("sync code", text)
            self.assertNotIn("syncing code", text)
            self.assertNotIn("run multi-process", text)


if __name__ == "__main__":
    unittest.main()
