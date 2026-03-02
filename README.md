# Primatrix Skills Plugin Marketplace

A [Claude Code plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces) containing reusable Agent Skills. Also compatible with [Codex](https://developers.openai.com/codex/cli) and [Gemini CLI](https://geminicli.com/docs/cli/skills/).

## What is a Skill?

A Skill is a set of structured instructions (defined in a `SKILL.md` file) that teaches an AI coding agent how to perform a specific workflow. When installed, the agent loads relevant skills based on your request, giving it domain-specific knowledge and step-by-step procedures.

## Available Skills

| Skill | Description |
|-------|-------------|
| [exec-remote](#exec-remote) | Execute Python scripts on remote GPU/TPU clusters via SkyPilot |
| [linear](#linear) | Manage issues, projects & team workflows in Linear |
| [gke-tpu](#gke-tpu) | Manage GKE clusters and run TPU workloads on Google Kubernetes Engine |

---

### exec-remote

Executes Python scripts, tests, or benchmarks on a provisioned remote cluster (GPU or TPU) using [SkyPilot](https://skypilot.readthedocs.io/).

**Use when:** the user asks to run code on GPU, TPU, or any remote cluster.

**Capabilities:**
- Provision GPU clusters (H100, A100, L4, etc.) or TPU clusters (v4, v6e, etc.) on GCP via SkyPilot
- Execute Python scripts and pytest tests on remote instances
- Automatically sync local working directory to the remote cluster
- Manage cluster lifecycle (launch, execute, teardown)

**Prerequisites:**
- [SkyPilot](https://skypilot.readthedocs.io/) installed and configured
- GCP credentials set up
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

### gke-tpu

Manage GKE clusters and run TPU workloads on [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine).

**Use when:** the user wants to create GKE clusters, node pools, workload policies, or run TPU jobs on GKE.

**Capabilities:**
- Create GKE clusters for TPU workloads
- Create workload policies for multi-host TPU setups
- Create TPU node pools (single-host ≤4 chips, multi-host >4 chips)
- Create and submit Kubernetes Jobs on TPU nodes
- Interact with running pods via kubectl (exec, logs, port-forward)
- Run services, benchmarks, or tests on GKE TPU pods

**Prerequisites:**
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- `kubectl` installed
- GCP project with GKE API enabled

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
/plugin install gke-tpu@primatrix-skills

# project scope (default is user scope)
/plugin install exec-remote@primatrix-skills --scope project
/plugin install gke-tpu@primatrix-skills --scope project
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
