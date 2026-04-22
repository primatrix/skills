---
allowed-tools: Bash(gh api:*), Bash(gh project:*)
description: Claim a Beaver-tracked GitHub Issue. Trigger when the user wants to claim, start, or pick up a task.
argument-hint: "<issue-number>"
---

# /beaver-claim — 认领任务

Phase 3 of the Beaver development lifecycle.

## Workflow

Argument is required: the issue number to claim.

1. **Load Issue**:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-claim.sh fetch {org} {issueRepo} {number}
   ```

1. **Validate claimable status**:

   | Issue Type | Required Status | Notes |
   |---|---|---|
   | Feature (size/S) | `status/ready-to-claim` | Must be in Iteration |
   | Feature (size/L) | `status/ready-to-claim` | Must be in Iteration |
   | Bug (non-p/0) | `status/triage` | Bugs skip Iteration, G007 exempt |
   | Bug (p/0-blocker) | Already `status/in-progress` | No claim needed, warn user |

1. **Guardrail checks**:
   - G001: Verify `size/S` or `size/L` label exists
   - G007: Verify Iteration associated (exempt for `type/bug`)

1. **Assign current user**:

   ```bash
   CURRENT_USER=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-claim.sh whoami)
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-claim.sh assign {org} {issueRepo} {number} "$CURRENT_USER"
   ```

1. **Status transition** (engine §6):
   - size/S (including Bug) → `status/in-progress`
   - size/L → `status/design-pending`
   Execute atomic label swap per engine §4.

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-claim.sh swap-status {org} {issueRepo} {number} status/ready-to-claim status/in-progress
   ```

1. **Report**: Print updated Issue URL, new status, and next-step hint:
   - size/S: "Ready to develop. Use `/beaver-dev {number}` to start TDD development."
   - size/L: "Design review required. Use `/beaver-design {number}` to write the design doc."
   - Bug: "Ready to fix. Use `/beaver-dev {number}` to start TDD development."

## Constraints

- §7 QA loop is NOT required (claim is a pure label transition on existing content)
- §7.2 HARD-GATE still applies: confirm with user before executing label swap
