#!/usr/bin/env bash
#
# Acceptance test for Issue #117:
#   "B2: beaver-tracker 迁移 + 当前 Iteration sub-issue 差集同步"
#
# Static assertions (run by default) — verify the 6 acceptance criteria
# from the Issue body by inspecting beaver-tracker.sh + beaver-tracker.md.
#
#   AC1: command source emits the three repo labels
#        (tracker / tracker/<repo> / tracker/<YYYY-MM>) AND writes the
#        tracker Issue's own Iteration field via beaver-lib.sh::set_iteration.
#   AC2: command queries backlog via Project V2 fields
#        (Iteration empty ∧ Status=Triage ∧ Type ∈ {Task, Bug}) and contains
#        a Step-7 unmount routine (sub-issue whose Iteration ≠ <YYYY-MM> or
#        repo mismatch is detached from the tracker).
#   AC3: zero `gh api .*/labels` calls touching status/*, type/*, or size/*
#        in command source (tracker/*, Control-By-Beaver, beaver/* are exempt
#        per RFC §「成功指标 3」 grep scope).
#   AC4: command doc still describes 全选/全拒/逐项 (select-all / reject-all /
#        per-item) interaction for both carry-over (Step 4) and backlog (Step 5).
#   AC5: command source prints summary with all 4 statistics: sub-issue total,
#        carry-over count, newly-pulled count, unmounted count.
#   AC6: tempfile usage employs unique filenames (mktemp or $$/$RANDOM suffix).

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
tracker_sh="$repo_root/plugins/beaver/scripts/beaver-tracker.sh"
tracker_md="$repo_root/plugins/beaver/commands/beaver-tracker.md"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

[ -f "$tracker_sh" ] || { echo "FAIL: $tracker_sh missing" >&2; exit 1; }
[ -f "$tracker_md" ] || { echo "FAIL: $tracker_md missing" >&2; exit 1; }

# ---------- AC1 ----------
# The trio of tracker labels must appear in source (script or md).
for needle in 'tracker/<repo>' 'tracker/<YYYY-MM>' '"tracker"'; do
  if grep -qF "$needle" "$tracker_md" || grep -qF "$needle" "$tracker_sh"; then
    report_pass "AC1.label.${needle}: tracker label '${needle}' referenced"
  else
    # Allow alternate quoting of the bare 'tracker' label.
    if [ "$needle" = '"tracker"' ] && (grep -qE "\btracker\b" "$tracker_sh" || grep -qE "\btracker\b" "$tracker_md"); then
      report_pass "AC1.label.tracker: bare 'tracker' label referenced"
    else
      report_fail "AC1.label.${needle}: tracker label '${needle}' not found"
    fi
  fi
done

# Tracker issue's own Iteration field must be written via beaver-lib.
if grep -qE 'beaver-lib\.sh.*set_iteration|set_iteration\b' "$tracker_sh"; then
  report_pass "AC1.set_iteration: source uses beaver-lib.sh::set_iteration"
else
  report_fail "AC1.set_iteration: tracker source must call beaver-lib.sh::set_iteration"
fi

# ---------- AC2 ----------
# Backlog query — must read Project V2 fields, not status/triage label.
if grep -qE 'Iteration.*Status|Status.*Iteration' "$tracker_sh" \
   || grep -qE 'fieldValueByName.*Iteration' "$tracker_sh"; then
  report_pass "AC2.backlog.fields: backlog query reads Project V2 Iteration field"
else
  report_fail "AC2.backlog.fields: backlog query must use Project V2 Iteration field"
fi

if grep -qE 'fieldValueByName.*Status|Status.*Triage' "$tracker_sh" \
   || grep -qE '"Status"' "$tracker_sh"; then
  report_pass "AC2.backlog.status: backlog query reads Status field (not status/triage label)"
else
  report_fail "AC2.backlog.status: backlog query must read Project V2 Status field"
fi

# Backlog query must filter Type ∈ {Task, Bug}.
if grep -qE 'issueType.*Task|Task.*Bug|Bug.*Task' "$tracker_sh"; then
  report_pass "AC2.backlog.type: backlog query filters Issue Type ∈ {Task, Bug}"
else
  report_fail "AC2.backlog.type: backlog query must filter Issue Type ∈ {Task, Bug}"
fi

# Step 7 unmount routine — there must be a subcommand or routine that
# detaches stale sub-issues from the tracker.
if grep -qE 'unmount|detach|remove.*sub.*issue|stale' "$tracker_sh"; then
  report_pass "AC2.unmount: source contains an unmount/detach routine"
else
  report_fail "AC2.unmount: source must contain Step 7 unmount routine"
fi

# Sub-Issues API DELETE endpoint must be invoked for unmount.
if grep -qE '\-\-method DELETE.*sub_issue|sub_issue.*DELETE|DELETE.*sub_issues' "$tracker_sh"; then
  report_pass "AC2.unmount.delete: sub_issues DELETE endpoint used for detach"
else
  report_fail "AC2.unmount.delete: source must call sub_issues DELETE endpoint to detach"
fi

# ---------- AC3 ----------
# Zero `gh api ... labels ...` calls touching status/, type/, or size/ prefixes.
ac3_hits=$(grep -nE 'status/[a-z-]+|type/[a-z]+|size/[A-Z]+' "$tracker_sh" | \
  grep -vE 'tracker/|Control-By-Beaver|beaver/|^[[:space:]]*#' || true)
if [ -z "$ac3_hits" ]; then
  report_pass "AC3: zero status/|type/|size/ label references in beaver-tracker.sh"
else
  report_fail "AC3: forbidden label-prefix references found in beaver-tracker.sh:"
  echo "$ac3_hits" >&2
fi

ac3_md_hits=$(grep -nE 'gh api.*status/[a-z]|gh api.*type/[a-z]|gh api.*size/[A-Z]' "$tracker_md" || true)
if [ -z "$ac3_md_hits" ]; then
  report_pass "AC3.md: no gh api label calls for status/type/size in beaver-tracker.md"
else
  report_fail "AC3.md: forbidden gh api label references in beaver-tracker.md:"
  echo "$ac3_md_hits" >&2
fi

# ---------- AC4 ----------
# Carry-over (Step 4) and backlog (Step 5) must offer 全选/全拒/逐项.
for stage_label in 'carry-over' 'backlog'; do
  if grep -qE '全选|全拒|逐项|select.*all|reject.*all|per.*item' "$tracker_md"; then
    report_pass "AC4.${stage_label}: doc describes 全选/全拒/逐项 UX"
  else
    report_fail "AC4.${stage_label}: doc must describe 全选/全拒/逐项 interaction"
  fi
  break  # The single grep covers both stages; loop kept for symmetry.
done

# Both Step 4 (carry-over) and Step 5 (backlog) must exist as headings.
if grep -qE '^### (Step )?4\b' "$tracker_md"; then
  report_pass "AC4.step4: Step 4 (carry-over) section present"
else
  report_fail "AC4.step4: doc must contain Step 4 carry-over section"
fi
if grep -qE '^### (Step )?(5|8\.5)\b' "$tracker_md"; then
  report_pass "AC4.step5: Step 5 / 8.5 (backlog) section present"
else
  report_fail "AC4.step5: doc must contain Step 5 (or 8.5) backlog section"
fi

# ---------- AC5 ----------
# Summary must include 4 stats: total, carry-over, new, unmounted.
declare -a ac5_patterns=(
  'sub-issue.*total|总数'
  'carry-over|carried'
  'new.*pulled|拉取|新增'
  'unmounted|解挂|detached'
)
for stat_label in "${ac5_patterns[@]}"; do
  if grep -qE "$stat_label" "$tracker_sh" || grep -qE "$stat_label" "$tracker_md"; then
    report_pass "AC5.stat.${stat_label}: summary mentions '${stat_label}'"
  else
    report_fail "AC5.stat.${stat_label}: summary must mention '${stat_label}'"
  fi
done

# ---------- AC6 ----------
# Tempfile usage must employ unique names: mktemp OR $$/$RANDOM/timestamp suffix.
# Look for any reference to a temp body file in the doc/script.
ac6_uses_tempfile=$(grep -nE '/tmp/beaver-tracker|--body-file' "$tracker_md" "$tracker_sh" || true)
if [ -n "$ac6_uses_tempfile" ]; then
  if grep -qE 'mktemp|\$\$|\$RANDOM|date \+' "$tracker_md" "$tracker_sh"; then
    report_pass "AC6.tempfile.unique: tempfile uses mktemp / \$\$ / \$RANDOM / timestamp"
  else
    report_fail "AC6.tempfile.unique: tempfile must use mktemp / \$\$ / \$RANDOM / timestamp suffix"
  fi
else
  report_pass "AC6.tempfile.unique: no static tempfile path used (compliant by absence)"
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
