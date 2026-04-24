#!/usr/bin/env bash
#
# Acceptance test for Issue #116:
#   "B1: beaver-create 迁移到字段语义 + Type 分支 (Task/SubTask/Bug)"
#
# Static assertions (run by default):
#   AC1 : Type inference (task/subtask/bug) + explicit --type override;
#         Bug path collects Priority (P0/P1/P2) and skips Size.
#   AC2 : Step-9 ordering create-issue → link-parent → add-to-project →
#         set fields (via beaver-lib.sh); Task/SubTask write
#         Type/Size/Status=Triage/Iteration; Bug writes
#         Type=Bug/Priority/Status (P0→In Progress, P1|P2→Ready to Claim)/
#         Iteration; @CODEOWNERS only inside the P0/Bug context;
#         beaver-create.sh delegates Project V2 field writes to beaver-lib.sh.
#   AC3 : Bug path calls latest_iteration_for_repo and on G011 fail prints
#         the /beaver-tracker hint.
#   AC4 : beaver-create.sh and beaver-create.md contain no
#         `gh api .../labels` calls and no `status/|type/|size/` literals
#         (matches the RFC success-metric grep).
#   AC6 : All `gh ... --body-file` paths in beaver-create.sh and
#         beaver-create.md use `mktemp` (preferred) or
#         `$$`/`$RANDOM`-suffixed unique paths. No reused fixed paths
#         like `/tmp/beaver-issue-body.md`.
#
# Live assertions (BEAVER_LIVE=1):
#   AC5 : Create a P0 Bug in primatrix/projects, assert Status=In Progress
#         and Iteration is set; clean up sandbox issue.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
create_sh="$repo_root/plugins/beaver/scripts/beaver-create.sh"
create_md="$repo_root/plugins/beaver/commands/beaver-create.md"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

if [ ! -f "$create_sh" ]; then
  report_fail "beaver-create.sh not found at $create_sh"
fi
if [ ! -f "$create_md" ]; then
  report_fail "beaver-create.md not found at $create_md"
fi

# ---------- AC4: no label-API calls or status/type/size literals ----------

# (a) `gh api .../labels` calls used to write lifecycle metadata are
# forbidden. The labels API is allowed only for the `Control-By-Beaver`
# / `beaver/*` flag labels, but the Cycle-2 implementation removes the
# entire add-labels surface from beaver-create, so any `labels` API hit
# inside the two source files is a regression.
for f in "$create_sh" "$create_md"; do
  hits=$(grep -nE 'gh api[^|]*labels' "$f" || true)
  if [ -z "$hits" ]; then
    report_pass "AC4.labels-api: $(basename "$f") has no \`gh api .../labels\` call"
  else
    report_fail "AC4.labels-api: $(basename "$f") still uses \`gh api .../labels\`:"
    echo "$hits" >&2
  fi
done

# (b) status/* / type/* / size/* literal strings are forbidden.
for f in "$create_sh" "$create_md"; do
  hits=$(grep -nE 'status/[a-z]|type/[a-z]|size/[A-Z]' "$f" || true)
  if [ -z "$hits" ]; then
    report_pass "AC4.literals: $(basename "$f") has no status/|type/|size/ literals"
  else
    report_fail "AC4.literals: $(basename "$f") still contains label literals:"
    echo "$hits" >&2
  fi
done

# ---------- AC6: body-file paths must be unique per invocation ----------

# (a) The legacy fixed path /tmp/beaver-issue-body.md must not appear.
for f in "$create_sh" "$create_md"; do
  hits=$(grep -nE '/tmp/beaver-issue-body\.md' "$f" || true)
  if [ -z "$hits" ]; then
    report_pass "AC6.fixed: $(basename "$f") does not use the fixed /tmp/beaver-issue-body.md path"
  else
    report_fail "AC6.fixed: $(basename "$f") still uses /tmp/beaver-issue-body.md:"
    echo "$hits" >&2
  fi
done

# (b) Every `--body-file <path>` reference must use either `mktemp`
# (path is computed from a mktemp variable) or contain `$$`/`$RANDOM`
# in the path literal. We accept the requirement satisfied if the file
# contains `mktemp` AND every `--body-file` reference is to a variable
# (so $-prefixed) — this avoids false positives on doc prose.
for f in "$create_sh" "$create_md"; do
  body_refs=$(grep -nE -- '--body-file[ =][^ ]+' "$f" || true)
  if [ -z "$body_refs" ]; then
    report_pass "AC6.unique: $(basename "$f") has no --body-file references"
    continue
  fi
  # Each reference must point to a $-variable (e.g. $BODY_FILE) or to
  # a literal containing $$ / $RANDOM / mktemp output.
  bad=$(echo "$body_refs" \
    | grep -vE -- '--body-file[ =]("?\$[A-Za-z_]|"?\$\{|/tmp/[^ ]*(\$\$|\$RANDOM)|"?\$\(mktemp)' \
    || true)
  if [ -z "$bad" ]; then
    report_pass "AC6.unique: $(basename "$f") --body-file refs all use unique paths"
  else
    report_fail "AC6.unique: $(basename "$f") has --body-file refs with non-unique paths:"
    echo "$bad" >&2
  fi
done

# (c) The source must mention `mktemp` somewhere as the canonical
# unique-temp-file recipe (sanity check for the rewrite).
if grep -qE '\bmktemp\b' "$create_sh" "$create_md"; then
  report_pass "AC6.mktemp: mktemp is referenced in beaver-create source"
else
  report_fail "AC6.mktemp: neither beaver-create.sh nor beaver-create.md references mktemp"
fi

# ---------- AC1: Type inference + --type override + Bug branch (no Size, P0/P1/P2) ----------

# (a) beaver-create.md must mention Type inference covering task/subtask/bug.
if grep -qE 'task[ /].*subtask[ /].*bug|task / subtask / bug|task/subtask/bug' "$create_md"; then
  report_pass "AC1.inference: beaver-create.md mentions task / subtask / bug Type inference"
else
  report_fail "AC1.inference: beaver-create.md must describe task / subtask / bug Type inference"
fi

# (b) beaver-create.md must mention an explicit `--type` override.
if grep -qE -- '--type\b' "$create_md"; then
  report_pass "AC1.override: beaver-create.md describes explicit --type override"
else
  report_fail "AC1.override: beaver-create.md must describe an explicit --type override"
fi

# (c) Bug path must collect Priority P0/P1/P2.
if grep -qE 'P0[ /].*P1[ /].*P2|P0 / P1 / P2|P0/P1/P2' "$create_md"; then
  report_pass "AC1.priority: beaver-create.md Bug path lists P0 / P1 / P2"
else
  report_fail "AC1.priority: beaver-create.md Bug path must list P0 / P1 / P2"
fi

# (d) Bug path explicitly does NOT write Size.
# Look for a sentence mentioning Bug + (no Size | does not write Size | NO Size | skip Size).
if grep -qiE '(bug.*(no |not |skip |without ).*size|size.*(not|no).*written.*bug|bug path.*no size)' "$create_md"; then
  report_pass "AC1.no-size: beaver-create.md Bug path explicitly skips Size"
else
  report_fail "AC1.no-size: beaver-create.md must explicitly state Bug path does not write Size"
fi

# ---------- AC2: §Step-9 ordering — create → link-parent → add-to-project → set fields ----------

# (a) The step-9 section must enumerate the four ordered actions
#     (create issue, link parent, add to project, set fields) — order matters.
md_text=$(cat "$create_md")
order_create=$(echo "$md_text"   | grep -nE '^\s*1\.\s+\*\*9a' | head -1 | cut -d: -f1)
order_link=$(echo "$md_text"     | grep -nE '^\s*2\.\s+\*\*9b' | head -1 | cut -d: -f1)
order_project=$(echo "$md_text"  | grep -nE '^\s*3\.\s+\*\*9c' | head -1 | cut -d: -f1)
order_fields=$(echo "$md_text"   | grep -nE '^\s*4\.\s+\*\*9d' | head -1 | cut -d: -f1)

if [ -n "$order_create" ] && [ -n "$order_link" ] && [ -n "$order_project" ] && [ -n "$order_fields" ]; then
  if [ "$order_create" -lt "$order_link" ] \
     && [ "$order_link" -lt "$order_project" ] \
     && [ "$order_project" -lt "$order_fields" ]; then
    report_pass "AC2.order: 9a create-issue → 9b link-parent → 9c add-to-project → 9d set fields order preserved"
  else
    report_fail "AC2.order: ordering wrong (9a=$order_create 9b=$order_link 9c=$order_project 9d=$order_fields)"
  fi
else
  report_fail "AC2.order: one or more 9a/9b/9c/9d step markers missing (9a=$order_create 9b=$order_link 9c=$order_project 9d=$order_fields)"
fi

# (b) The Task/SubTask field-write section must list Type, Size, Status=Triage, Iteration.
for needle in 'Type' 'Size' 'Triage' 'Iteration'; do
  if grep -q "$needle" "$create_md"; then
    report_pass "AC2.task-fields.${needle}: beaver-create.md mentions '${needle}'"
  else
    report_fail "AC2.task-fields.${needle}: beaver-create.md must mention '${needle}'"
  fi
done

# (c) The Bug field-write section must list Priority, Status mapping P0→In Progress, P1/P2→Ready to Claim.
if grep -qE 'P0.*(In Progress|in progress)' "$create_md"; then
  report_pass "AC2.bug-status.p0: beaver-create.md maps P0 → In Progress"
else
  report_fail "AC2.bug-status.p0: beaver-create.md must map P0 → In Progress"
fi
if grep -qE '(P1|P2|P1[/ ]?P2|P1\|P2).*(Ready to Claim|ready to claim)' "$create_md"; then
  report_pass "AC2.bug-status.p1p2: beaver-create.md maps P1/P2 → Ready to Claim"
else
  report_fail "AC2.bug-status.p1p2: beaver-create.md must map P1/P2 → Ready to Claim"
fi

# (d) @CODEOWNERS must be mentioned only in the P0 Bug context, not in the
#     Task/SubTask body template. We assert: every @CODEOWNERS occurrence
#     appears within ~10 lines of the word "P0" or "Bug".
codeowners_lines=$(grep -nE '@CODEOWNERS' "$create_md" || true)
if [ -z "$codeowners_lines" ]; then
  report_fail "AC2.codeowners: @CODEOWNERS must be mentioned in beaver-create.md (P0 Bug context)"
else
  bad_codeowners=0
  while IFS= read -r line_ref; do
    [ -z "$line_ref" ] && continue
    line_no=$(echo "$line_ref" | cut -d: -f1)
    start=$((line_no > 10 ? line_no - 10 : 1))
    end=$((line_no + 10))
    ctx=$(sed -n "${start},${end}p" "$create_md")
    if echo "$ctx" | grep -qE '\bP0\b|\bBug\b|p0/blocker'; then
      :
    else
      bad_codeowners=$((bad_codeowners + 1))
      echo "  unexpected @CODEOWNERS at line $line_no without P0/Bug context" >&2
    fi
  done <<< "$codeowners_lines"
  if [ "$bad_codeowners" -eq 0 ]; then
    report_pass "AC2.codeowners: @CODEOWNERS only appears in P0/Bug context"
  else
    report_fail "AC2.codeowners: $bad_codeowners @CODEOWNERS occurrence(s) outside P0/Bug context"
  fi
fi

# ---------- AC3: Bug path uses latest_iteration_for_repo + G011 + tracker hint ----------

if grep -q 'latest_iteration_for_repo' "$create_md"; then
  report_pass "AC3.lib-call: beaver-create.md references beaver-lib.sh::latest_iteration_for_repo"
else
  report_fail "AC3.lib-call: beaver-create.md must reference beaver-lib.sh::latest_iteration_for_repo for the Bug path"
fi

if grep -qE '\bG011\b' "$create_md"; then
  report_pass "AC3.g011: beaver-create.md mentions G011"
else
  report_fail "AC3.g011: beaver-create.md must mention G011"
fi

if grep -qE '/beaver-tracker' "$create_md"; then
  report_pass "AC3.hint: beaver-create.md mentions the /beaver-tracker hint"
else
  report_fail "AC3.hint: beaver-create.md must point users to /beaver-tracker on G011 fail"
fi

# ---------- AC2.sh: beaver-create.sh must NOT duplicate beaver-lib.sh field writes ----------
# i.e., no raw updateProjectV2ItemFieldValue mutations in beaver-create.sh.
if grep -qE 'updateProjectV2ItemFieldValue' "$create_sh"; then
  report_fail "AC2.sh: beaver-create.sh contains raw updateProjectV2ItemFieldValue (must delegate to beaver-lib.sh)"
else
  report_pass "AC2.sh: beaver-create.sh does not duplicate beaver-lib.sh field writes"
fi

# ---------- AC5 (LIVE): create P0 Bug in primatrix/projects, assert Status=In Progress + Iteration set ----------
if [ "${BEAVER_LIVE:-0}" = "1" ]; then
  echo "--- BEAVER_LIVE=1 — running live AC5 P0 Bug round-trip ---"
  source "$repo_root/plugins/beaver/scripts/beaver-lib.sh"

  body_file=$(mktemp -t beaver-create-test.XXXXXX)
  cat >"$body_file" <<'BODYEOF'
## 复现步骤
[beaver-create AC5 live test] sandbox P0 Bug, will be deleted on success.

@CODEOWNERS

<!-- beaver-create-ac5-test -->
BODYEOF
  title="[beaver-create AC5 live] $(date -u +%Y-%m-%dT%H:%M:%SZ) $$"
  num=$(bash "$create_sh" create-issue primatrix projects "$title" "$body_file")
  rm -f "$body_file"

  # Capture node_id for cleanup.
  node_id=$(gh api "repos/primatrix/projects/issues/${num}" --jq '.node_id')
  cleanup_live() {
    if [ -n "${node_id:-}" ]; then
      echo "AC5.live: cleanup deleting sandbox issue #${num}..."
      gh api graphql -f query='
        mutation($issueId: ID!) {
          deleteIssue(input: { issueId: $issueId }) { repository { name } }
        }' -f issueId="$node_id" >/dev/null 2>&1 \
        || echo "AC5.live: WARN failed to delete #${num}, please remove manually" >&2
    fi
  }
  trap cleanup_live EXIT

  # Field writes via beaver-lib.sh (the canonical Cycle-2 path).
  set_type "$num" Bug
  iter_title=$(latest_iteration_for_repo projects)
  if [ -z "$iter_title" ]; then
    report_fail "AC5.live.iter: latest_iteration_for_repo returned empty (G011 fail)"
  else
    set_iteration "$num" "$iter_title"
  fi
  set_status "$num" "In Progress"

  actual_type=$(get_type "$num")
  actual_status=$(_get_single_select_value "$num" "Status")
  actual_iter=$(get_iteration "$num")

  [ "$actual_type" = "Bug" ] \
    && report_pass "AC5.live.type: get_type == 'Bug'" \
    || report_fail "AC5.live.type: expected 'Bug', got '$actual_type'"
  [ "$actual_status" = "In Progress" ] \
    && report_pass "AC5.live.status: Status == 'In Progress'" \
    || report_fail "AC5.live.status: expected 'In Progress', got '$actual_status'"
  [ -n "$actual_iter" ] \
    && report_pass "AC5.live.iteration: Iteration is non-empty ('$actual_iter')" \
    || report_fail "AC5.live.iteration: Iteration is empty"
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
