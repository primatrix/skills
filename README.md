# Primatrix Skills Plugin Marketplace

A [Claude Code plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces) containing reusable Agent Skills. Also compatible with [Codex](https://developers.openai.com/codex/cli) and [Gemini CLI](https://geminicli.com/docs/cli/skills/).

## What is a Skill?

A Skill is a set of structured instructions (defined in a `SKILL.md` file) that teaches an AI coding agent how to perform a specific workflow. When installed, the agent loads relevant skills based on your request, giving it domain-specific knowledge and step-by-step procedures.

## Available Skills

| Skill | Description |
|-------|-------------|
| [exec-remote](#exec-remote) | Execute Python scripts on remote GPU/TPU clusters via SkyPilot |
| [linear](#linear) | Manage issues, projects & team workflows in Linear |
| [session-recorder](#session-recorder) | Records the complete session content to a daily work directory |
---

### exec-remote

Execute Python scripts on remote GPU/TPU clusters via [SkyPilot](https://skypilot.readthedocs.io/).

**Use when:** the user asks to run code on GPU, TPU, or any remote cluster.

This plugin contains three skills with a parent-child relationship:

```
exec-remote          ← Entry point: run scripts on a provisioned cluster
├── deploy-cluster   ← Deploy a SkyPilot-managed TPU cluster on GKE
└── apply-resource   ← Provision/manage the underlying GKE TPU cluster via xpk
```

`exec-remote` is the top-level skill. When a cluster doesn't exist yet, it delegates to `deploy-cluster`, which in turn delegates to `apply-resource` to create the GKE infrastructure.

| Skill | Description |
|-------|-------------|
| **exec-remote** | Executes Python scripts, tests, or benchmarks on a provisioned remote cluster (GPU or TPU). Entry point — delegates to the sub-skills as needed. |
| **deploy-cluster** | Deploys a SkyPilot-managed TPU cluster on GKE. Generates `~/.sky/config.yaml`, fetches GKE credentials, and runs `sky launch`. |
| **apply-resource** | Manages GKE TPU clusters using xpk. Creates, deletes, and lists TPU Nodepool resources. Multi-user safe — always queries GKE in real-time. |

**Capabilities:**
- Provision GPU clusters (H100, A100, L4, etc.) or TPU clusters (v4, v6e, etc.) on GCP
- Execute Python scripts and pytest tests on remote instances
- Automatically sync local working directory to the remote cluster
- Manage full cluster lifecycle (create GKE cluster → deploy SkyPilot → execute → teardown)

**Prerequisites:**
- [SkyPilot](https://skypilot.readthedocs.io/) installed and configured
- [xpk](https://github.com/AI-Hypercomputer/xpk) installed (for GKE TPU cluster management)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) with `gcloud auth login` completed
- [kubectl](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl) with `gke-gcloud-auth-plugin`
- [uv](https://github.com/astral-sh/uv) for dependency management

---

### linear

Manage issues, projects & team workflows in [Linear](https://linear.app/) through the Linear MCP server.

**Use when:** the user wants to read, create, or update tickets in Linear.

**Capabilities:**
- Issue management: create, update, list, search, triage
- Project & team operations: create projects, manage teams, view users
- Documentation & collaboration: manage documents, comments, cycles
- Workflow automation: sprint planning, bug triage, release planning, workload balancing, smart labeling

**Prerequisites:**
- Linear MCP server connected and accessible via OAuth
- Access to the relevant Linear workspace, teams, and projects

---

## Installation

### Claude Code

Install plugins via the [plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces):

```bash
# add this repository as a community marketplace
/plugin marketplace add primatrix/skills

# install plugins from the marketplace
/plugin install exec-remote@primatrix-skills
/plugin install linear@primatrix-skills

# project scope (default is user scope)
/plugin install exec-remote@primatrix-skills --scope project
```

### Codex

Install skills via [skills.sh](https://skills.sh):

```bash
npx skills add primatrix/skills@exec-remote -a codex
npx skills add primatrix/skills@linear -a codex
```

### Gemini CLI

Install skills via the built-in [skill commands](https://geminicli.com/docs/cli/skills/):

```bash
# install from GitHub (user scope by default: ~/.gemini/skills)
gemini skills install https://github.com/primatrix/skills.git --path plugins/exec-remote/skills/exec-remote
gemini skills install https://github.com/primatrix/skills.git --path plugins/linear/skills/linear

# project/workspace scope (.gemini/skills in current project)
gemini skills install https://github.com/primatrix/skills.git --path plugins/exec-remote/skills/exec-remote --scope workspace
```

### Cross-platform (skills.sh)

[skills.sh](https://skills.sh) is a universal package manager that works across Claude Code, Codex, and Gemini CLI:

```bash
npx skills add primatrix/skills@exec-remote
npx skills add primatrix/skills@linear

# install to a specific agent
npx skills add primatrix/skills -a claude-code
npx skills add primatrix/skills -a codex
```

### Verify installation

```bash
# Claude Code — start a session and run:
/exec-remote
/linear

# Codex — start a session and run:
/skills

# Gemini CLI
gemini skills list
```

## Repository Structure

This repository is a Claude Code [plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces). Each plugin wraps one or more skills.

```
.claude-plugin/
└── marketplace.json          # Marketplace definition (plugin registry)
plugins/
├── <plugin-name>/
│   ├── .claude-plugin/
│   │   └── plugin.json       # Plugin manifest (name, description, version)
│   └── skills/
│       └── <skill-name>/
│           ├── SKILL.md      # Skill definition (frontmatter + instructions)
│           └── scripts/      # Optional supporting scripts
```

The `SKILL.md` file contains:
- **YAML frontmatter** — `name`, `description`, and optional metadata
- **Markdown body** — detailed instructions the agent follows when the skill is activated

## Local Development

When developing or modifying skills locally, you need a way to test changes **before** pushing to GitHub. The plugin marketplace installs from the remote repository, so local edits won't take effect by default.

The solution is to symlink the plugin cache to your local working directory:

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/primatrix/skills.git
cd skills

# 2. Install the plugin from the marketplace first (this sets up the registry)
#    In a Claude Code session:
/plugin marketplace add primatrix/skills
/plugin install exec-remote@primatrix-skills

# 3. Replace the cache with a symlink to your local directory
rm -rf ~/.claude/plugins/cache/primatrix-skills/exec-remote/1.0.0
ln -s /path/to/skills/plugins/exec-remote \
      ~/.claude/plugins/cache/primatrix-skills/exec-remote/1.0.0

# 4. Restart Claude Code session, then verify
/plugin
```

Repeat step 3 for each plugin you want to develop locally:

```bash
rm -rf ~/.claude/plugins/cache/primatrix-skills/<plugin-name>/1.0.0
ln -s /path/to/skills/plugins/<plugin-name> \
      ~/.claude/plugins/cache/primatrix-skills/<plugin-name>/1.0.0
```

### Workflow

Once the symlink is in place, local edits are reflected immediately (after restarting the session):

```
Local edit → Restart Claude Code → /exec-remote → Verify
```

### Caveats

- **Do not run `/plugin install` again** — it will overwrite the symlink with a fresh copy from GitHub.
- **Restart the session** after making changes — plugins are loaded at session startup.
- `plugin.json` only needs `name`, `description`, and `version`. Do **not** add a `skills` field — Claude Code auto-discovers skills from the `skills/` subdirectory.

## Contributing

To add a new plugin:

1. Create a new directory under `plugins/` with your plugin name
2. Add `.claude-plugin/plugin.json` with `name`, `description`, and `version`
3. Add skills under `skills/<skill-name>/SKILL.md` with proper frontmatter and instructions
4. Include any supporting scripts or templates in the skill directory
5. Register the plugin in `.claude-plugin/marketplace.json`
6. Submit a pull request

## License

[Apache License 2.0](LICENSE)

---

### session-recorder

Records the complete session's content and logs it to a daily work directory with a dynamic filename.

**Use when:** the user wants to record their full progress for documentation purposes.

**Capabilities:**
- Record complete, unedited session history (questions and answers)
- Automatically exclude cancelled operations
- Generate dynamic log files named `{$cli-name}-session.md`
- Organize logs by date in a `YYYY-MM-DD` directory structure

**Prerequisites:**
- Python 3.x installed
- Base directory for logs set to `/Users/leos/python/daily_work`
