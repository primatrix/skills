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

#### GKE TPU Getting Started

For running code on TPU via GKE (the full `apply-resource → deploy-cluster → exec-remote` pipeline), additional tools are required:

| Tool | Install | Verify |
|------|---------|--------|
| [xpk](https://github.com/AI-Hypercomputer/xpk) | [Install guide](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/installation.md) | `xpk --help` |
| [kubectl](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl) | `gcloud components install kubectl` | `kubectl version --client` |

After installing the plugin (see [Installation](#installation)), open your AI agent in your project directory and paste the prompt below.

Replace **`YOUR_REPO_PATH/sglang-jax/benchmark/moe/bench_ep_moe.py`** with your actual script path.

##### COPY THIS PROMPT

> **[Context]**
> I'm working on a JAX-based ML project with `pyproject.toml` that has a `tpu` extra dependency group. No remote cluster exists yet — `.cluster_name_tpu` is absent. The `exec-remote` plugin provides a three-stage pipeline: `apply-resource` (creates GKE cluster via `xpk cluster create-pathways --spot`) → `deploy-cluster` (deploys SkyPilot on GKE via its `scripts/deploy.py`) → `exec-remote` (runs code via `sky exec`). The GCP project is `tpu-service-473302`. The deploy script writes `.cluster_name_tpu` as the integration point between stages.
>
> **[Objective]**
> Run `YOUR_REPO_PATH/sglang-jax/benchmark/moe/bench_ep_moe.py` on a TPU cluster by provisioning the full GKE infrastructure from scratch, following the complete `apply-resource → deploy-cluster → exec-remote` pipeline.
>
> **[Style]**
> Step-by-step automated execution. Collect all cluster parameters from me once upfront (cluster name, TPU type, number of slices, GCP zone), then carry them through every subsequent step — never re-ask. Auto-calculate `--num-nodes` from TPU type (total_chips / 4, e.g. v6e-8 = 2 nodes, v6e-4 = 1 node). After `xpk` creates the GKE cluster, poll `gcloud container clusters list` until status is `RUNNING` — do NOT proceed while `RECONCILING` or `PROVISIONING` (deploying SkyPilot in these states causes SSL errors).
>
> **[Tone]**
> Proactive — execute each pipeline step automatically and report progress. Only pause to collect user input during initial parameter gathering.
>
> **[Audience]**
> ML engineer who wants to run training code on cloud TPU without manually managing infrastructure.
>
> **[Response]**
> After each stage (GKE creation, SkyPilot deployment, code execution), briefly report the result and verify success before proceeding. Use `sky exec` with `--extra tpu` for dependencies and `--workdir .` to sync the local directory. If any step fails, diagnose the error and suggest a fix before continuing.

The agent will ask you for cluster parameters (cluster name, TPU type, number of slices, GCP zone) once, then execute the full pipeline automatically:

```
apply-resource   →  Creates GKE cluster with TPU nodepool via xpk
                     Polls until cluster status is RUNNING
deploy-cluster   →  Configures and launches SkyPilot on GKE
                     Writes .cluster_name_tpu
exec-remote      →  Syncs local directory, runs your script
                     with correct --num-nodes and --extra tpu
```

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
