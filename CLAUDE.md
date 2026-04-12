# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A Claude Code **plugin marketplace** containing reusable Agent Skills. Each skill is a `SKILL.md` file with YAML frontmatter + markdown instructions that teach AI coding agents how to perform specific workflows. Compatible with Claude Code, Codex, and Gemini CLI.

## Repository Layout

```
.claude-plugin/marketplace.json     # Marketplace registry — lists all plugins
plugins/<plugin>/
  .claude-plugin/plugin.json        # Plugin manifest (name, description, version, skills list)
  skills/<skill>/
    SKILL.md                        # Skill definition (frontmatter + agent instructions)
    scripts/                        # Supporting scripts (Python, Bash, YAML)
```

Four plugins exist: **exec-remote** (3 skills), **beaver** (6 skills), **session-recorder** (1 skill), **lint-fix** (1 skill).

## Architecture: exec-remote Plugin

The exec-remote plugin has a three-stage pipeline for GKE TPU provisioning:

```
apply-resource  →  deploy-cluster  →  exec-remote
(xpk/GKE)         (SkyPilot on GKE)   (sky exec)
```

**Defaults** (used unless user overrides): project=`tpu-service-473302`, cluster=`sglang-jax-agent-tests`, zone=`asia-northeast1-b`.

- **apply-resource**: Creates/deletes/lists GKE clusters with TPU nodepools via `xpk`. Scripts in `plugins/exec-remote/skills/apply-resource/scripts/` (Python: `main.py`, `cluster_manager.py`, `tpu_availability.py`).
- **deploy-cluster**: Deploys SkyPilot on top of the GKE cluster. Uses `scripts/deploy.py` to generate `~/.sky/config.yaml`, fetch GKE credentials, and run `sky launch`. Each TPU type gets its own SkyPilot cluster named `<cluster>-<username>-<tpu_type>`, allowing parallel execution across topologies. References `config.yaml` and `setup.yaml` as templates. Writes `.cluster_name_tpu` as integration point.
- **exec-remote**: Entry point skill. Uses per-TPU-type cluster names (e.g. `sglang-jax-agent-tests-hongmao-v6e-1`) or reads `.cluster_name_gpu`/`.cluster_name_tpu` for provisioned clusters. Runs code via `sky exec`. Delegates to deploy-cluster/apply-resource when no cluster exists. Also supports standalone GPU/TPU provisioning via `launch_gpu.sh` and `launch_tpu.sh`.

Key integration file: `.cluster_name_tpu` (or `_gpu`) in the plugin root — written by provisioning, read by exec-remote.

Fixed GCP project: `tpu-service-473302`.

## Adding a New Plugin

1. Create `plugins/<name>/.claude-plugin/plugin.json` with `name`, `description`, `version`
2. Create `plugins/<name>/skills/<skill>/SKILL.md` with YAML frontmatter (`name`, `description`) and markdown body
3. Register in `.claude-plugin/marketplace.json`

## SKILL.md Format

```markdown
---
name: skill-name
description: What this skill does and when to activate it.
argument-hint: "[optional usage hint]"
---

# Skill Title

Instructions the agent follows when the skill is activated...
```

The `description` field in frontmatter controls when agents load the skill. The markdown body contains step-by-step procedures, commands, and parameters.

## Testing / Verification

No build system, linter, or test suite. Verify changes by:
- Checking JSON validity of `marketplace.json` and `plugin.json` files
- Ensuring SKILL.md frontmatter has valid YAML between `---` fences
- For exec-remote scripts: `python scripts/deploy.py --help` or `python scripts/main.py --help`
