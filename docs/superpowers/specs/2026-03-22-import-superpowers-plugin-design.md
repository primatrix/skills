# Import Superpowers Plugin into Local Repository

**Date:** 2026-03-22
**Status:** Approved

## Goal

Copy the superpowers plugin (v5.0.2) into this repository as a locally maintained plugin, giving full ownership and the ability to customize skills.

## Scope

Copy the superpowers plugin directory structure into `plugins/superpowers/`, including only the runtime-relevant directories (skills/, agents/, commands/, hooks/, multi-platform integration files). Exclude upstream development artifacts. Modify `plugin.json` to match the repository's minimal format. Register in `marketplace.json`.

## Directory Structure

```
plugins/superpowers/
├── .claude-plugin/
│   └── plugin.json                     # Minimal format: name, description, version
├── agents/
│   └── code-reviewer.md
├── commands/
│   ├── brainstorm.md
│   ├── execute-plan.md
│   └── write-plan.md
├── hooks/
│   ├── hooks.json
│   ├── run-hook.cmd
│   └── session-start
├── skills/
│   ├── brainstorming/                  # SKILL.md + visual-companion.md + spec-document-reviewer-prompt.md + scripts/
│   ├── dispatching-parallel-agents/    # SKILL.md
│   ├── executing-plans/                # SKILL.md
│   ├── finishing-a-development-branch/ # SKILL.md
│   ├── receiving-code-review/          # SKILL.md
│   ├── requesting-code-review/         # SKILL.md + code-reviewer.md
│   ├── subagent-driven-development/    # SKILL.md + prompt templates
│   ├── systematic-debugging/           # SKILL.md + techniques + scripts
│   ├── test-driven-development/        # SKILL.md + anti-patterns.md
│   ├── using-git-worktrees/            # SKILL.md
│   ├── using-superpowers/              # SKILL.md + references/
│   ├── verification-before-completion/ # SKILL.md
│   ├── writing-plans/                  # SKILL.md + reviewer prompt
│   └── writing-skills/                 # SKILL.md + guides + scripts
├── .cursor-plugin/plugin.json          # Cursor editor integration
├── .codex/INSTALL.md                   # Codex integration
├── .opencode/                          # OpenCode integration
│   ├── INSTALL.md
│   └── plugins/superpowers.js
├── gemini-extension.json               # Gemini CLI integration
└── GEMINI.md                           # Gemini CLI instructions
```

## Changes

### 1. plugin.json

Simplified to match repository conventions:

```json
{
  "name": "superpowers",
  "description": "Core skills library for Claude Code: TDD, debugging, collaboration patterns, and proven techniques",
  "version": "1.0.0"
}
```

### 2. marketplace.json

Add entry:

```json
{
  "name": "superpowers",
  "source": "./plugins/superpowers",
  "description": "Core skills library: TDD, debugging, brainstorming, collaboration patterns, and proven techniques",
  "category": "workflow",
  "version": "1.0.0",
  "license": "Apache-2.0",
  "keywords": ["skills", "tdd", "debugging", "brainstorming", "collaboration", "workflows"]
}
```

### 3. SKILL.md files

All 14 SKILL.md files copied without modification. Internal relative path references remain valid since the directory structure is preserved.

### 4. Supporting files

All supporting files (agents/, commands/, hooks/, multi-platform integration files) copied without modification.

## What is NOT copied

- `docs/` — upstream development documentation
- `tests/` — upstream test suites
- `README.md` — upstream project README
- `RELEASE-NOTES.md` — upstream release notes
- `LICENSE` — upstream MIT license file (this repo uses Apache-2.0)
- `.github/` — upstream GitHub config (FUNDING.yml)
- `.gitignore`, `.gitattributes` — upstream git config
- `.claude-plugin/marketplace.json` — upstream's own marketplace registry (only `plugin.json` is kept and rewritten)

## Skills Included (14 total)

| Skill | Purpose |
|-------|---------|
| brainstorming | Collaborative design and spec creation before implementation |
| dispatching-parallel-agents | Pattern for running independent tasks in parallel |
| executing-plans | Execute implementation plans with review checkpoints |
| finishing-a-development-branch | Guide completion of development work (merge/PR/cleanup) |
| receiving-code-review | Handle code review feedback with technical rigor |
| requesting-code-review | Verify work meets requirements before merging |
| subagent-driven-development | Execute plans using parallel subagents |
| systematic-debugging | 4-phase debugging process |
| test-driven-development | RED-GREEN-REFACTOR cycle |
| using-git-worktrees | Workspace isolation for feature work |
| using-superpowers | Meta-skill for discovering and using other skills |
| verification-before-completion | Evidence-based verification before completion claims |
| writing-plans | Create detailed implementation plans from specs |
| writing-skills | Create and test new skills |

## Implementation

This is a pure file operation task:

1. Copy files from superpowers cache directory to `plugins/superpowers/`
2. Write simplified `plugin.json`
3. Update `marketplace.json`
4. Commit
