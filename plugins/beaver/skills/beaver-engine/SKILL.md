---
name: beaver-engine
description: "Internal engine for Beaver workflow skills. DO NOT trigger directly. Provides state machine rules, guardrail checks, label operations, and project config reading used by beaver-issue, beaver-pr, beaver-audit, beaver-report, and beaver-focus."
---

# Beaver Engine

Internal skill. Do not invoke directly. Other beaver skills reference these rules and command templates.

## 1. Label Taxonomy

All labels use a `prefix/name` format:

### Type labels (`type/`)
- `type/feat` — New feature
- `type/bug` — Bug fix
- `type/refactor` — Refactoring
- `type/docs` — Documentation
- `type/chore` — Infrastructure, build, misc

### Priority labels (`p/`)
- `p0/blocker` — Blocking, top of daily report
- `p1/urgent` — Urgent, top of daily report
- `p2/high` — High priority
- `p3/normal` — Normal

### Size labels (`size/`)
- `size/S` — Small, fast-track SOP
- `size/L` — Large, full lifecycle SOP

### Status labels (`status/`)
- `status/triage` — Initial state, awaiting triage
- `status/requirements-gathering` — (size/L only) Requirements refinement
- `status/design-pending` — (size/L only) Design review in progress
- `status/ready-to-develop` — (size/L only) Design approved, ready to code
- `status/in-progress` — Active development
- `status/blocked` — Blocked (must note reason)
- `status/review-needed` — Awaiting code/design review
- `status/done` — Completed and merged

### Beaver agent labels (`beaver/`)
- `beaver/needs-split` — PR LOC exceeds 200 in core dirs
- `beaver/missing-test` — No test evidence found before done
- `beaver/missing-context` — Incomplete labels or description
- `beaver/stale` — Stuck in same status > 3 days
- `beaver/overdue` — Past DDL and not done
- `beaver/upstream-blocked` — Upstream dependency is blocked
- `beaver/wontfix` — Will not fix, skip stale/overdue detection

## 2. State Machine

### size/S Fast Track
```
triage → in-progress → review-needed → done
```

### size/L Standard SOP
```
triage → requirements-gathering → design-pending → ready-to-develop → in-progress → review-needed → done
```

### Universal transitions
- Any status → `blocked` (must note reason in Issue comment)
- `blocked` → restore to previous status

### Legal next-states lookup

| Current Status | size/S next | size/L next |
|---|---|---|
| triage | in-progress | requirements-gathering |
| requirements-gathering | N/A | design-pending |
| design-pending | N/A | ready-to-develop |
| ready-to-develop | N/A | in-progress |
| in-progress | review-needed | review-needed |
| review-needed | done | done |
| blocked | (previous) | (previous) |

## 3. Guardrail Rules

### G001: Size required before leaving triage
- **Check:** Issue has a `size/S` or `size/L` label
- **When:** Any transition FROM `status/triage`
- **Fail action:** Block transition, comment on Issue requesting size classification

### G002: size/L must not skip stages
- **Check:** Target status is the legal next state per size/L SOP
- **When:** Any transition of a `size/L` Issue
- **Fail action:** Block transition, comment listing required intermediate stages

### G003: Cannot skip review
- **Check:** `in-progress` cannot go directly to `done`
- **When:** Transition to `status/done`
- **Fail action:** Block transition, require `status/review-needed` first

### G004: Test evidence required for done
- **Check:** Find test evidence from (in priority order):
  1. Current session context — scan conversation for test runner output (pytest, go test, npm test, cargo test, etc.)
  2. PR diff — new/modified test files (`*_test.*`, `test_*.*`, `tests/**`)
  3. CI status — GitHub Actions / Check Runs on associated PR
- **When:** Transition to `status/done` or PR creation
- **Fail action:** Add `beaver/missing-test` label, comment requesting evidence
- **On success:** Write test summary to PR body's Test Plan section or Issue comment

### G005: LOC limit on PR
- **Check:** Count added lines in core directories, excluding:
  - `**/*_test.*`, `**/test_*.*`, `**/tests/**`
  - `**/*.md`, `**/docs/**`
  - `*.pb.go`, `*_generated.*`, `*.lock`
- **Threshold:** 200 lines
- **When:** PR creation
- **Fail action:** Add `beaver/needs-split` label, warn developer (do not block — let developer confirm)

### G006: PR must have complete labels
- **Check:** Associated Issue has at least one `type/` label AND one `size/` label
- **When:** PR creation
- **Fail action:** Add `beaver/missing-context` label, list missing labels

## 4. Label Operations (gh command templates)

### Read all labels on an Issue
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --jq '.[].name'
```
Parse into structured data:
- `type`: first label matching `type/*`
- `size`: first label matching `size/*`
- `status`: first label matching `status/*`
- `priority`: first label matching `p*/*`
- `beaver_flags`: all labels matching `beaver/*`

### Set status label (atomic swap)
Remove all existing `status/*` labels, then add the new one:
```bash
# Remove current status label
gh api repos/{owner}/{repo}/issues/{number}/labels/{current_status_label} --method DELETE

# Add new status label
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={new_status_label}"
```

### Add beaver flag label
```bash
gh api repos/{owner}/{repo}/issues/{number}/labels --method POST -f "labels[]={beaver_label}"
```

## 5. Project Config Reading

Read from Project V2 README's `beaver-config` YAML block:

```bash
gh project view {number} --owner {org} --format json --jq '.readme'
```

Parse the ` ```yaml beaver-config ` fenced block for:
- `repositories`: list of observed repos
- `issueRepo`: the repo hosting Beaver issues
- `customFields`: field name overrides (default: Level, Status, Progress)

## 6. Transition Execution Template

When a command-layer skill needs to transition an Issue's status:

1. Read current labels (Section 4)
2. Determine `size` from labels
3. Look up legal next states (Section 2)
4. Validate target state against guardrails (Section 3)
5. If all checks pass: execute atomic label swap (Section 4)
6. If any check fails: report failure, do NOT swap labels
