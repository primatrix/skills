#!/usr/bin/env bash
#
# Acceptance test for Issue #122:
#   "B7: beaver-focus 迁移到字段语义 + 零写入断言"
#
# Static assertions (run by default) — verify the 5 acceptance criteria
# from the Issue body by inspecting beaver-focus.sh + beaver-focus.md.
#
#   AC1: zero mutation strings in script + md (verbatim AC5 grep regex).
#   AC2.fields: source uses Project V2 Status field via GraphQL
#               (fieldValueByName(name: "Status") or equivalent).
#   AC2.order:  source/md mentions all 6 status names in the required order:
#               In Progress, Blocked, Design Pending, Ready to Develop,
#               Ready to Claim, Triage. (First-occurrence line ordering.)
#   AC2.recency: source mentions sorting by last commit / comment timestamp.
#   AC3.p0:     source/md mentions p/0-blocker AND 24h AND ⚠️.
#   AC4.llm:    command md describes ONE LLM call producing
#               "Today's Top 3 Priorities" framed as actionable next steps.
#   AC5:        verbatim grep from AC5 produces zero matches across both files.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
focus_sh="$repo_root/plugins/beaver/scripts/beaver-focus.sh"
focus_md="$repo_root/plugins/beaver/commands/beaver-focus.md"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

[ -f "$focus_sh" ] || { echo "FAIL: $focus_sh missing" >&2; exit 1; }
[ -f "$focus_md" ] || { echo "FAIL: $focus_md missing" >&2; exit 1; }

# ---------- AC1 / AC5 (same regex) ----------
ac5_hits=$(grep -nE "(mutation|--method (POST|PATCH|PUT)|gh issue edit|gh issue comment|gh pr ready)" \
  "$focus_sh" "$focus_md" || true)
if [ -z "$ac5_hits" ]; then
  report_pass "AC1/AC5: zero mutation calls in beaver-focus.sh + beaver-focus.md"
else
  report_fail "AC1/AC5: mutation calls present (must be zero):"
  echo "$ac5_hits" >&2
fi

# ---------- AC2.fields ----------
# Source must read Project V2 Status field via GraphQL (fieldValueByName Status).
if grep -qE 'fieldValueByName.*Status|"Status"|name:[[:space:]]*"Status"' "$focus_sh"; then
  report_pass "AC2.fields: source reads Project V2 Status field via GraphQL"
else
  report_fail "AC2.fields: source must read Project V2 Status field via GraphQL"
fi

# Should also read Priority + Type fields for grouping/highlighting.
if grep -qE 'Priority' "$focus_sh"; then
  report_pass "AC2.fields.priority: source reads Project V2 Priority field"
else
  report_fail "AC2.fields.priority: source must read Project V2 Priority field"
fi

# ---------- AC2.order ----------
# All 6 status names must appear, and their first occurrences in the
# command doc must be in the required order.
required_order=("In Progress" "Blocked" "Design Pending" "Ready to Develop" "Ready to Claim" "Triage")
prev_line=0
order_ok=1
for name in "${required_order[@]}"; do
  # First-line occurrence in focus_md.
  line=$(grep -nF "$name" "$focus_md" | head -1 | cut -d: -f1 || true)
  if [ -z "$line" ]; then
    report_fail "AC2.order: status name '$name' missing from beaver-focus.md"
    order_ok=0
    continue
  fi
  if [ "$line" -le "$prev_line" ]; then
    report_fail "AC2.order: status '$name' (line $line) appears before previous status (line $prev_line)"
    order_ok=0
  fi
  prev_line=$line
done
if [ "$order_ok" = "1" ]; then
  report_pass "AC2.order: all 6 status names present in required order"
fi

# ---------- AC2.recency ----------
if grep -qE 'updatedAt|comments.*createdAt|last.*comment|最后.*(评论|commit|提交)|last_activity' \
   "$focus_sh" "$focus_md"; then
  report_pass "AC2.recency: source mentions sorting by last commit/comment timestamp"
else
  report_fail "AC2.recency: source must mention sorting by last commit/comment timestamp"
fi

# ---------- AC3.p0 ----------
for needle in 'p/0-blocker' '24h' '⚠️'; do
  if grep -qF "$needle" "$focus_sh" "$focus_md"; then
    report_pass "AC3.p0.${needle}: '${needle}' present"
  else
    report_fail "AC3.p0.${needle}: '${needle}' missing"
  fi
done

# ---------- AC4.llm ----------
# Command md must describe ONE LLM call producing actionable Top 3 priorities.
if grep -qE "Today'?s Top 3 Priorities" "$focus_md"; then
  report_pass "AC4.llm.heading: 'Today's Top 3 Priorities' heading present"
else
  report_fail "AC4.llm.heading: command md must contain 'Today's Top 3 Priorities' heading"
fi

if grep -qE 'actionable|actionable next step|可执行|下一步' "$focus_md"; then
  report_pass "AC4.llm.actionable: doc instructs actionable next steps (not issue list)"
else
  report_fail "AC4.llm.actionable: doc must frame priorities as actionable next steps"
fi

if grep -qE 'one[- ]shot|once|一次|single LLM|一次性' "$focus_md"; then
  report_pass "AC4.llm.once: doc specifies a single LLM call"
else
  report_fail "AC4.llm.once: doc must specify ONE LLM call"
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
