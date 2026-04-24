#!/usr/bin/env bash
#
# Acceptance test for Issue #118:
#   "B3: beaver-design 迁移 + 创建 spec-document-reviewer subagent 模板"
#
# Static assertions (run by default):
#   AC1.precheck   : beaver-design.md asserts前置校验 Type=Task ∧ Size=L ∧
#                    Status=Design Pending + assignee == gh user.
#   AC2.qa-order   : beaver-design.md enumerates the 5 QA dimensions in spec
#                    order (Context & Scope / Design Goals / The Design /
#                    Implementation Plan / Alternatives Considered) and forbids
#                    cross-dimension jumps.
#   AC2.impl-plan  : beaver-design.md says Implementation Plan dimension output
#                    is written to RFC '## 实施计划' segment.
#   AC3.reviewer   : spec-document-reviewer template exists (SKILL.md with valid
#                    frontmatter) and beaver-design.md schedules it for at most
#                    5 review rounds, gating push on PASS.
#   AC4.draft-pr   : beaver-design.md says the PR is created via
#                    `gh pr create --draft`, file lives at
#                    docs/rfc/NNNN-<slug>.md and docs/rfc/index.md gets a new
#                    line; the command also comments the PR URL on the original
#                    Task Issue.
#   AC5.no-status  : beaver-design.md / beaver-design.sh do not write Project V2
#                    fields; in particular do not invoke beaver-lib.sh
#                    set_status / set_size / set_type / set_iteration nor
#                    raw `gh api` paths matching `updateProjectV2ItemFieldValue`
#                    / `labels` Status mutations.
#   AC6.tempfile   : every `--body-file` callsite in beaver-design.sh is fed by
#                    a unique tempfile (mktemp or $$/$RANDOM/timestamp suffix).
#
# Live assertions (BEAVER_LIVE=1):
#   AC5.live       : Pick the issue used by tests, snapshot Status, run a no-op
#                    of beaver-design.sh comment-issue, re-read Status, assert
#                    unchanged. (We do NOT run the full /beaver-design QA loop
#                    in CI; only the bash-script side-effects are exercised.)

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
design_md="$repo_root/plugins/beaver/commands/beaver-design.md"
design_sh="$repo_root/plugins/beaver/scripts/beaver-design.sh"
reviewer_md="$repo_root/plugins/beaver/skills/spec-document-reviewer/SKILL.md"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

# ---------- AC1.precheck ----------
if [ -f "$design_md" ]; then
  for needle in 'Type=Task' 'Size=L' 'Design Pending' 'assignee'; do
    if grep -qF "$needle" "$design_md"; then
      report_pass "AC1.precheck.${needle}: present"
    else
      report_fail "AC1.precheck.${needle}: missing in beaver-design.md"
    fi
  done
else
  report_fail "AC1.precheck: $design_md missing"
fi

# ---------- AC2.qa-order ----------
if [ -f "$design_md" ]; then
  # The 5 dimensions must appear in this exact order.
  expected_order=(
    'Context & Scope'
    'Design Goals'
    'The Design'
    'Implementation Plan'
    'Alternatives Considered'
  )
  prev_line=0
  order_ok=1
  for dim in "${expected_order[@]}"; do
    line=$(grep -nF "$dim" "$design_md" | head -1 | cut -d: -f1 || true)
    if [ -z "$line" ]; then
      report_fail "AC2.qa-order: dimension '$dim' missing from beaver-design.md"
      order_ok=0
      continue
    fi
    if [ "$line" -le "$prev_line" ]; then
      report_fail "AC2.qa-order: dimension '$dim' appears at line $line, not after previous (line $prev_line)"
      order_ok=0
    fi
    prev_line=$line
  done
  if [ "$order_ok" -eq 1 ]; then
    report_pass "AC2.qa-order: 5 dimensions appear in spec order"
  fi

  # Forbid cross-dimension jumps — the doc must say so explicitly.
  if grep -qE '禁止跨维度|不得跨维度|no.*cross.*dimension' "$design_md"; then
    report_pass "AC2.qa-order.no-jump: cross-dimension jumps explicitly forbidden"
  else
    report_fail "AC2.qa-order.no-jump: beaver-design.md must explicitly forbid cross-dimension QA jumps"
  fi

  # ---------- AC2.impl-plan ----------
  if grep -qE '## 实施计划|实施计划.*段|Implementation Plan.*实施计划' "$design_md"; then
    report_pass "AC2.impl-plan: Implementation Plan output writes to '## 实施计划'"
  else
    report_fail "AC2.impl-plan: beaver-design.md must say Implementation Plan dimension produces '## 实施计划' content"
  fi
fi

# ---------- AC3.reviewer ----------
if [ -f "$reviewer_md" ]; then
  report_pass "AC3.reviewer.exists: spec-document-reviewer template present"
  # Extract frontmatter (between the first two `---` lines).
  fm=$(awk 'BEGIN{in_fm=0} NR==1 && /^---$/ {in_fm=1; next} in_fm && /^---$/ {exit} in_fm{print}' "$reviewer_md")
  if [ -n "$fm" ]; then
    if printf '%s\n' "$fm" | grep -qE '^name:[[:space:]]+spec-document-reviewer'; then
      report_pass "AC3.reviewer.frontmatter.name: 'name: spec-document-reviewer' present"
    else
      report_fail "AC3.reviewer.frontmatter.name: missing or wrong name field"
    fi
    if printf '%s\n' "$fm" | grep -qE '^description:[[:space:]]+'; then
      report_pass "AC3.reviewer.frontmatter.description: present"
    else
      report_fail "AC3.reviewer.frontmatter.description: missing"
    fi
  else
    report_fail "AC3.reviewer.frontmatter: file does not have a YAML frontmatter block"
  fi
else
  report_fail "AC3.reviewer.exists: $reviewer_md missing"
fi

if [ -f "$design_md" ]; then
  if grep -qE '最多 5 轮|最多5轮|max(imum)? 5 (rounds|iterations)|5 轮' "$design_md"; then
    report_pass "AC3.reviewer.5-rounds: 5-round cap mentioned in beaver-design.md"
  else
    report_fail "AC3.reviewer.5-rounds: beaver-design.md must cap reviewer at 5 rounds"
  fi
  if grep -qE 'PASS|通过' "$design_md" && grep -qE 'BLOCK|阻断|阻止' "$design_md"; then
    report_pass "AC3.reviewer.gate: PASS / BLOCK gating mentioned"
  else
    report_fail "AC3.reviewer.gate: PASS / BLOCK gating must appear in beaver-design.md"
  fi
fi

# ---------- AC4.draft-pr ----------
if [ -f "$design_md" ]; then
  if grep -qE 'gh pr create.*--draft' "$design_md" || grep -qE 'gh pr create.*--draft' "$design_sh"; then
    report_pass "AC4.draft-pr.cmd: 'gh pr create --draft' used"
  else
    report_fail "AC4.draft-pr.cmd: 'gh pr create --draft' must be used"
  fi
  if grep -qE 'docs/rfc/NNNN-' "$design_md"; then
    report_pass "AC4.draft-pr.path: docs/rfc/NNNN-<slug>.md path mentioned"
  else
    report_fail "AC4.draft-pr.path: beaver-design.md must reference docs/rfc/NNNN-<slug>.md"
  fi
  if grep -qE 'docs/rfc/index\.md' "$design_md"; then
    report_pass "AC4.draft-pr.index: docs/rfc/index.md append mentioned"
  else
    report_fail "AC4.draft-pr.index: beaver-design.md must mention appending to docs/rfc/index.md"
  fi
  if grep -qE 'comment(-| )issue|评论.*PR|PR.*评论' "$design_md"; then
    report_pass "AC4.draft-pr.comment: command comments PR URL on original Issue"
  else
    report_fail "AC4.draft-pr.comment: beaver-design.md must comment the PR URL on the original Task Issue"
  fi
fi

# ---------- AC5.no-status ----------
forbidden_writes='set_status|set_size|set_type|set_iteration|updateProjectV2ItemFieldValue|updateIssueIssueType'
if [ -f "$design_md" ]; then
  if grep -nE "$forbidden_writes" "$design_md" >/dev/null; then
    report_fail "AC5.no-status.md: beaver-design.md must not invoke field-mutation APIs"
    grep -nE "$forbidden_writes" "$design_md" >&2
  else
    report_pass "AC5.no-status.md: beaver-design.md performs no field mutation"
  fi
fi
if [ -f "$design_sh" ]; then
  if grep -nE "$forbidden_writes" "$design_sh" >/dev/null; then
    report_fail "AC5.no-status.sh: beaver-design.sh must not invoke field-mutation APIs"
    grep -nE "$forbidden_writes" "$design_sh" >&2
  else
    report_pass "AC5.no-status.sh: beaver-design.sh performs no field mutation"
  fi
  # The script must not source beaver-lib.sh (no field ops needed).
  # Match only actual source/. statements, not comments.
  if grep -qE '^[[:space:]]*(source[[:space:]]+|\.[[:space:]]+).*beaver-lib\.sh' "$design_sh"; then
    report_fail "AC5.no-status.sh.lib: beaver-design.sh must not source beaver-lib.sh"
  else
    report_pass "AC5.no-status.sh.lib: beaver-design.sh does not source beaver-lib.sh"
  fi
fi

# ---------- AC6.tempfile ----------
# beaver-design.sh must (a) use --body-file (not --raw-field) for both
# create-pr and comment-issue subcommands, and (b) feed --body-file from a
# unique tempfile (mktemp / $$ / $RANDOM / timestamp).
if [ -f "$design_sh" ]; then
  if grep -qE '\-\-body-file' "$design_sh"; then
    report_pass "AC6.tempfile.body-file: --body-file used"
  else
    report_fail "AC6.tempfile.body-file: beaver-design.sh must use --body-file for create-pr / comment-issue"
  fi
  if grep -qE 'mktemp|\$\$|\$RANDOM|date \+%s' "$design_sh"; then
    report_pass "AC6.tempfile.unique: unique tempfile pattern present"
  else
    report_fail "AC6.tempfile.unique: beaver-design.sh must uniquify tempfile names (mktemp / \$\$ / \$RANDOM)"
  fi
  # comment-issue must accept either a body-file or a body string but route
  # through a tempfile internally so callers don't have to.
  if awk '/comment-issue\)/{flag=1; next} flag && /;;/{flag=0} flag' "$design_sh" \
       | grep -qE '\-\-body-file'; then
    report_pass "AC6.tempfile.comment-issue: comment-issue uses --body-file"
  else
    report_fail "AC6.tempfile.comment-issue: comment-issue must use --body-file"
  fi
fi

# ---------- Live (BEAVER_LIVE=1) ----------
if [ "${BEAVER_LIVE:-0}" = "1" ]; then
  echo "--- BEAVER_LIVE=1 — running live gh API assertions ---"
  : "${BEAVER_LIVE_ISSUE:?BEAVER_LIVE_ISSUE must be set to a sandbox Issue number on primatrix/projects}"

  status_before=$(gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) {
            nodes {
              project { number }
              fieldValues(first: 30) {
                nodes {
                  ... on ProjectV2ItemFieldSingleSelectValue {
                    name
                    field { ... on ProjectV2SingleSelectField { name } }
                  }
                }
              }
            }
          }
        }
      }
    }' -f owner=primatrix -f repo=projects -F number="$BEAVER_LIVE_ISSUE" \
    --jq '.data.repository.issue.projectItems.nodes
          | map(select(.project.number == 14))[0].fieldValues.nodes
          | map(select(.field.name == "Status"))[0].name // "<empty>"')

  comment_url=$(bash "$repo_root/plugins/beaver/scripts/beaver-design.sh" \
    comment-issue primatrix projects "$BEAVER_LIVE_ISSUE" \
    "[live-test] beaver-design test_beaver_design_migration.sh — Status snapshot probe @ $(date -u +%FT%TZ)" \
    | grep -oE 'https://github.com/primatrix/projects/issues/[0-9]+#issuecomment-[0-9]+' \
    | head -1 || true)

  status_after=$(gh api graphql -f query='
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        issue(number: $number) {
          projectItems(first: 20) {
            nodes {
              project { number }
              fieldValues(first: 30) {
                nodes {
                  ... on ProjectV2ItemFieldSingleSelectValue {
                    name
                    field { ... on ProjectV2SingleSelectField { name } }
                  }
                }
              }
            }
          }
        }
      }
    }' -f owner=primatrix -f repo=projects -F number="$BEAVER_LIVE_ISSUE" \
    --jq '.data.repository.issue.projectItems.nodes
          | map(select(.project.number == 14))[0].fieldValues.nodes
          | map(select(.field.name == "Status"))[0].name // "<empty>"')

  if [ "$status_before" = "$status_after" ]; then
    report_pass "AC5.live: Status unchanged across comment-issue ($status_before)"
  else
    report_fail "AC5.live: Status changed: '$status_before' -> '$status_after'"
  fi
  echo "  (left a probe comment: $comment_url — please clean up manually)"
fi

# ---------- summary ----------
echo
if [ "$failures" -eq 0 ]; then
  echo "All acceptance assertions passed."
  exit 0
else
  echo "$failures assertion(s) failed."
  exit 1
fi
