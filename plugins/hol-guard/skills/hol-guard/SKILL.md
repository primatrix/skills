---
name: hol-guard
description: Install and operate HOL Guard local runtime security for supported AI coding agents before tool execution, including setup, dry runs, status checks, approvals, receipts, and diagnostics.
license: Apache-2.0
---

# HOL Guard

Use HOL Guard when the user wants local runtime protection for an AI coding agent. Guard is an independent safety layer; it does not replace the agent's own permissions, sandboxing, provider policy, or explicit user authorization.

## Safety rules

- Never bypass Guard approvals or weaken an existing policy to make a command pass.
- Do not claim a harness is protected until Guard reports that state.
- Preserve existing agent configuration and use Guard-owned install/bootstrap commands rather than editing harness configuration by hand.
- Treat blocked actions as blocked until the user reviews the reason and approves them through Guard.
- Never read `.env` files or expose credentials while troubleshooting.

## Install and verify

First check whether the CLI is available:

```bash
hol-guard --version
```

If it is not installed and the user asked to set up Guard, prefer an isolated install:

```bash
pipx install hol-guard
```

Then inspect the environment:

```bash
hol-guard status
hol-guard detect --json
```

## Protect a supported harness

Use the exact harness reported or requested. Supported harness names include `codex`, `claude-code`, `copilot`, `cursor`, `gemini`, `hermes`, `openclaw`, `opencode`, and `antigravity`.

For most harnesses:

```bash
hol-guard bootstrap
hol-guard install <harness>
hol-guard run <harness> --dry-run
hol-guard doctor <harness> --json
hol-guard run <harness>
hol-guard status
```

For Hermes, use its dedicated bootstrap when appropriate:

```bash
hol-guard hermes bootstrap
```

Do not infer protection of the current agent merely because some other detected harness is healthy. Verify the intended harness explicitly.

## Review blocked or sensitive work

```bash
hol-guard approvals
hol-guard approvals open <request-id>
hol-guard receipts
hol-guard diff <harness>
```

Only approve a request after the user has reviewed the risk reason and intended scope. Terminal approval commands are available when the user explicitly decides:

```bash
hol-guard approvals approve <request-id>
hol-guard approvals deny <request-id>
```

## Evidence and diagnostics

```bash
hol-guard doctor
hol-guard doctor <harness> --json
hol-guard inventory
hol-guard receipts
hol-guard events
hol-guard explain <artifact-id>
```

When reporting results, state which Guard command actually ran, what Guard reported, what remains blocked or risky, and the exact next action. Never claim protection or approval without command output that proves it.
