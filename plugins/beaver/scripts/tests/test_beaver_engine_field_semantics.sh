#!/usr/bin/env bash
#
# Acceptance test for Issue #114:
#   "重写 beaver-engine 为字段语义 (G008 删除 / G011 新增)"
#
# Asserts the 4 acceptance criteria from the Issue body:
#   AC1: zero `status/|type/|size/` hits outside the "废弃说明" block
#   AC2: G008 absent; G011 present and aligned with RFC algorithm;
#        G001/G002/G006/G009 read Project V2 fields (no label-API references)
#   AC3: §4 renamed "Field Operations"; references beaver-lib.sh functions
#   AC4: §2 state-machine table lists the 7 Status values

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
skill="$repo_root/plugins/beaver/skills/beaver-engine/SKILL.md"

if [ ! -f "$skill" ]; then
  echo "FAIL: SKILL.md not found at $skill" >&2
  exit 1
fi

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

# ---------- AC1 ----------
# Zero hits for status/|type/|size/ in the file, except inside a fenced
# "废弃说明" (deprecation) block. Strip that block before grepping.
ac1_filtered=$(awk '
  /<!-- BEGIN 废弃说明 -->/ { skip=1; next }
  /<!-- END 废弃说明 -->/   { skip=0; next }
  !skip { print }
' "$skill")

ac1_hits=$(echo "$ac1_filtered" | grep -nE 'status/|type/|size/' || true)
if [ -z "$ac1_hits" ]; then
  report_pass "AC1: zero status/|type/|size/ hits outside 废弃说明 block"
else
  report_fail "AC1: forbidden label-prefix references found:"
  echo "$ac1_hits" >&2
fi

# ---------- AC2 ----------
# G008 must be absent. G011 must be present with key algorithm landmarks.
# G001/G002/G006/G009 must reference Project V2 fields (no `gh api repos/.../labels`).
if grep -qE '^### G008\b' "$skill"; then
  report_fail "AC2a: G008 still present (must be deleted)"
else
  report_pass "AC2a: G008 deleted"
fi

if grep -qE '^### G011\b' "$skill"; then
  report_pass "AC2b: G011 added"
else
  report_fail "AC2b: G011 missing"
fi

# Helper: extract block belonging to one ### G### heading until the next ### G### heading.
extract_g_block() {
  local g=$1
  awk -v g="$g" '
    {
      if ($0 ~ "^### " g "([^0-9]|$)") { flag = 1; next }
      else if (flag && $0 ~ /^### G[0-9]+/) { flag = 0 }
      if (flag) print
    }
  ' "$skill"
}

# G011 body should mention the algorithm's key landmarks and the lib helper.
g011_block=$(extract_g_block G011)
for needle in 'latest_iteration_for_repo' 'startDate' 'duration' 'UTC' '/beaver-tracker'; do
  if echo "$g011_block" | grep -q "$needle"; then
    report_pass "AC2b.${needle}: G011 references '${needle}'"
  else
    report_fail "AC2b.${needle}: G011 must reference '${needle}'"
  fi
done

# G001/G002/G006/G009 must refer to Project V2 / native Issue Type, not labels.
for g in G001 G002 G006 G009; do
  block=$(extract_g_block "$g")
  if [ -z "$block" ]; then
    report_fail "AC2c.${g}: not found"
    continue
  fi
  if echo "$block" | grep -qE 'gh api .*labels|status/[a-z]|type/[a-z]|size/[A-Z]'; then
    report_fail "AC2c.${g}: still references labels (forbidden)"
  else
    report_pass "AC2c.${g}: no label references"
  fi
  if echo "$block" | grep -qE 'Project V2|projectV2|Issue Type|beaver-lib'; then
    report_pass "AC2c.${g}: references Project V2 / Issue Type / beaver-lib"
  else
    report_fail "AC2c.${g}: must reference Project V2 / Issue Type / beaver-lib"
  fi
done

# ---------- AC3 ----------
# §4 must be "Field Operations" and reference beaver-lib.sh functions.
if grep -qE '^## 4\. Field Operations\b' "$skill"; then
  report_pass "AC3a: §4 renamed to 'Field Operations'"
else
  report_fail "AC3a: §4 must be 'Field Operations'"
fi

if grep -qE '^## 4\. Label Operations\b' "$skill"; then
  report_fail "AC3b: legacy '§4 Label Operations' heading still present"
else
  report_pass "AC3b: legacy '§4 Label Operations' removed"
fi

field_ops_block=$(awk '
  {
    if ($0 ~ /^## 4\. Field Operations/) { flag = 1; next }
    else if (flag && $0 ~ /^## [0-9]+\./) { flag = 0 }
    if (flag) print
  }
' "$skill")
if [ -z "$field_ops_block" ]; then
  report_fail "AC3c: §4 body empty"
else
  for fn in set_status set_size set_type get_iteration set_iteration latest_iteration_for_repo; do
    if echo "$field_ops_block" | grep -q "$fn"; then
      report_pass "AC3c.${fn}: §4 references beaver-lib.sh::${fn}"
    else
      report_fail "AC3c.${fn}: §4 must reference beaver-lib.sh::${fn}"
    fi
  done
  if echo "$field_ops_block" | grep -q 'beaver-lib.sh'; then
    report_pass "AC3d: §4 mentions beaver-lib.sh by path"
  else
    report_fail "AC3d: §4 must mention beaver-lib.sh by path"
  fi
fi

# ---------- AC4 ----------
# §2 state-machine table must list all 7 Status values verbatim.
sec2=$(awk '/^## 2\. /{flag=1; next} /^## 3\./{flag=0} flag{print}' "$skill")
for v in 'Triage' 'Ready to Claim' 'Design Pending' 'Ready to Develop' 'In Progress' 'Blocked' 'Done'; do
  if echo "$sec2" | grep -q "$v"; then
    report_pass "AC4.${v}: §2 mentions '${v}'"
  else
    report_fail "AC4.${v}: §2 must list '${v}'"
  fi
done

# §2 must contain a markdown table (transition table).
if echo "$sec2" | grep -qE '^\|.*\|.*\|'; then
  report_pass "AC4.table: §2 contains a markdown table"
else
  report_fail "AC4.table: §2 must contain a transition table (markdown |...|...|)"
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
