#!/usr/bin/env bash
#
# Acceptance test for Issue #119:
#   "B4: beaver-decompose 迁移到字段语义 + audit 警告改为 body 注释"
#
# Static assertions (run by default) cover the 10 acceptance criteria:
#   AC1 : Pre-check reads parent Type=Task ∧ Status=Ready to Develop via
#         beaver-lib.sh (no status/* label reads).
#   AC2 : Step ordering — create child → link parent → add to Project V2
#         → write fields (Type=SubTask, Size=S, Status=Triage,
#         Iteration=parent's). All field writes via beaver-lib.sh.
#   AC3 : Each child body inserts `> Design Doc: <url>` block at top,
#         passing through the --design-doc value verbatim.
#   AC4 : Default child assignees = parent assignees; empty parent
#         assignees stay empty (no fallback to current `gh` user); per-child
#         override allowed.
#   AC5 : Per-child unique tempfile via `mktemp` (or $$/$RANDOM literal).
#   AC6 : Failed audits append `<!-- audit-warnings -->` block to child
#         body listing (missing-test|needs-split|missing-context); no
#         `gh api .../labels` calls for `beaver/*` labels.
#   AC7 : Parent comment lists every child + audit result + a dependency
#         graph (#A blocked by #B, #C); failed dependency writes are
#         called out separately.
#   AC8 : No `gh api .../labels` calls involving status/|type/|size/
#         literals in source files.
#   AC9 : Per-child QA collects blockers as relative refs (child#N);
#         in-memory map; DFS cycle detection runs before any landing.
#   AC10: After all children created, dependency landing iterates the map
#         and calls the GitHub Issue Dependencies API (REST endpoint
#         `dependencies/blocked_by` or GraphQL `addIssueDependency`).
#         Per-row failure does NOT abort, but is surfaced in the AC7
#         summary comment.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
decompose_sh="$repo_root/plugins/beaver/scripts/beaver-decompose.sh"
decompose_md="$repo_root/plugins/beaver/commands/beaver-decompose.md"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

if [ ! -f "$decompose_sh" ]; then
  report_fail "beaver-decompose.sh not found at $decompose_sh"
fi
if [ ! -f "$decompose_md" ]; then
  report_fail "beaver-decompose.md not found at $decompose_md"
fi

# ---------- AC8: no `gh api .../labels` involving status/|type/|size/ literals ----------
# Stronger than AC4 of beaver-create's test: we forbid only the lifecycle-label
# label-API hits (status/* / type/* / size/*). beaver/* labels are still
# allowed via the labels API for the audit-warnings replacement check (which
# AC6 separately forbids inside this command's source).
for f in "$decompose_sh" "$decompose_md"; do
  hits=$(grep -nE 'gh api[^|]*labels' "$f" || true)
  bad=$(echo "$hits" | grep -E 'status/[a-z]|type/[a-z]|size/[A-Z]' || true)
  if [ -z "$bad" ]; then
    report_pass "AC8.lifecycle-labels: $(basename "$f") has no \`gh api .../labels\` write of status/|type/|size/ values"
  else
    report_fail "AC8.lifecycle-labels: $(basename "$f") still uses \`gh api .../labels\` with lifecycle literals:"
    echo "$bad" >&2
  fi
done

# Stricter: status/|type/|size/ string literals in either source file.
for f in "$decompose_sh" "$decompose_md"; do
  hits=$(grep -nE 'status/[a-z]|type/[a-z]|size/[A-Z]' "$f" || true)
  if [ -z "$hits" ]; then
    report_pass "AC8.literals: $(basename "$f") has no status/|type/|size/ literals"
  else
    report_fail "AC8.literals: $(basename "$f") still contains lifecycle label literals:"
    echo "$hits" >&2
  fi
done

# ---------- AC1: pre-check reads field semantics ----------

# (a) The pre-check section must mention reading the parent's native Issue
# Type AND Status field via beaver-lib.sh (or its `_get_single_select_value`
# helper), and must require Type=Task ∧ Status="Ready to Develop".
if grep -qE 'beaver-lib\.sh' "$decompose_md"; then
  report_pass "AC1.lib: beaver-decompose.md references beaver-lib.sh"
else
  report_fail "AC1.lib: beaver-decompose.md must reference beaver-lib.sh for field reads"
fi

if grep -qE 'get_type|_get_single_select_value' "$decompose_md"; then
  report_pass "AC1.read-type: beaver-decompose.md reads Type via beaver-lib helper"
else
  report_fail "AC1.read-type: beaver-decompose.md must read Type via beaver-lib (get_type / _get_single_select_value)"
fi

if grep -qE 'Type[^A-Za-z].*Task' "$decompose_md" || grep -qE 'Task[^A-Za-z].*Type' "$decompose_md"; then
  report_pass "AC1.task-required: beaver-decompose.md requires parent Type=Task"
else
  report_fail "AC1.task-required: beaver-decompose.md must require parent Type=Task"
fi

if grep -qE 'Ready to Develop' "$decompose_md"; then
  report_pass "AC1.status-required: beaver-decompose.md requires parent Status=Ready to Develop"
else
  report_fail "AC1.status-required: beaver-decompose.md must require parent Status=Ready to Develop"
fi

# ---------- AC2: step ordering create → link → add-to-project → set fields ----------

md_text=$(cat "$decompose_md")
order_create=$(echo "$md_text"  | grep -nE '^\s*[0-9]+\.\s+\*\*6a' | head -1 | cut -d: -f1)
order_link=$(echo "$md_text"    | grep -nE '^\s*[0-9]+\.\s+\*\*6b' | head -1 | cut -d: -f1)
order_project=$(echo "$md_text" | grep -nE '^\s*[0-9]+\.\s+\*\*6c' | head -1 | cut -d: -f1)
order_fields=$(echo "$md_text"  | grep -nE '^\s*[0-9]+\.\s+\*\*6d' | head -1 | cut -d: -f1)

if [ -n "$order_create" ] && [ -n "$order_link" ] && [ -n "$order_project" ] && [ -n "$order_fields" ]; then
  if [ "$order_create" -lt "$order_link" ] \
     && [ "$order_link" -lt "$order_project" ] \
     && [ "$order_project" -lt "$order_fields" ]; then
    report_pass "AC2.order: 6a create → 6b link → 6c add-to-project → 6d set-fields order preserved"
  else
    report_fail "AC2.order: ordering wrong (6a=$order_create 6b=$order_link 6c=$order_project 6d=$order_fields)"
  fi
else
  report_fail "AC2.order: one or more 6a/6b/6c/6d step markers missing (6a=$order_create 6b=$order_link 6c=$order_project 6d=$order_fields)"
fi

# (b) The fields written must include Type=SubTask, Size=S, Status=Triage,
# Iteration=parent's.
if grep -qE 'set_type[^A-Za-z]+.*SubTask|Type[^A-Za-z]+.*SubTask' "$decompose_md"; then
  report_pass "AC2.field.type: beaver-decompose.md writes Type=SubTask"
else
  report_fail "AC2.field.type: beaver-decompose.md must write Type=SubTask via beaver-lib"
fi

if grep -qE 'set_size[^A-Za-z]+.*S\b|Size[^A-Za-z]+.*=\s*S\b|Size[^A-Za-z]+.*"S"' "$decompose_md"; then
  report_pass "AC2.field.size: beaver-decompose.md writes Size=S"
else
  report_fail "AC2.field.size: beaver-decompose.md must write Size=S via beaver-lib"
fi

if grep -qE 'set_status[^A-Za-z]+.*Triage|Status[^A-Za-z]+.*Triage' "$decompose_md"; then
  report_pass "AC2.field.status: beaver-decompose.md writes Status=Triage"
else
  report_fail "AC2.field.status: beaver-decompose.md must write Status=Triage via beaver-lib"
fi

if grep -qE 'set_iteration|Iteration.*(parent|inherited)|inherits.*Iteration' "$decompose_md"; then
  report_pass "AC2.field.iteration: beaver-decompose.md writes inherited Iteration"
else
  report_fail "AC2.field.iteration: beaver-decompose.md must write Iteration inherited from parent"
fi

# (c) beaver-decompose.sh must NOT duplicate Project V2 field-write mutations.
if grep -qE 'updateProjectV2ItemFieldValue|updateIssueIssueType' "$decompose_sh"; then
  report_fail "AC2.sh: beaver-decompose.sh contains raw Project V2 / Issue Type mutations (must delegate to beaver-lib.sh)"
else
  report_pass "AC2.sh: beaver-decompose.sh delegates field writes to beaver-lib.sh"
fi

# ---------- AC3: design doc reference inserted at top of child body ----------

# Either the body template literal contains `> Design Doc: ` at the very top,
# or the doc/sh explicitly states that this prefix is prepended for every
# child (verbatim, no normalization).
if grep -qE '^\s*>?\s*Design Doc:' "$decompose_md"; then
  report_pass 'AC3.template: beaver-decompose.md template contains the "Design Doc:" blockquote line'
else
  report_fail 'AC3.template: beaver-decompose.md must show a "Design Doc: <url>" blockquote at top of each child body'
fi

if grep -qiE 'verbatim|原值|no normalization|不做规范化' "$decompose_md"; then
  report_pass "AC3.verbatim: beaver-decompose.md says the URL is written verbatim"
else
  report_fail "AC3.verbatim: beaver-decompose.md must state the --design-doc value is written verbatim"
fi

# ---------- AC4: assignee inheritance from parent, no fallback to gh user ----------

if grep -qE 'assignees|assignee' "$decompose_md"; then
  report_pass "AC4.mentions: beaver-decompose.md mentions assignees"
else
  report_fail "AC4.mentions: beaver-decompose.md must describe assignee inheritance"
fi

if grep -qE 'parent.*assignee|assignee.*parent|inherit.*assignee|assignees.*inherit' "$decompose_md"; then
  report_pass "AC4.inherit: beaver-decompose.md says child assignees default to parent's"
else
  report_fail "AC4.inherit: beaver-decompose.md must say child assignees default to parent assignees"
fi

# Must explicitly forbid the fallback to current gh user when parent has no assignees.
if grep -qiE '(no fallback|do not fallback|不回退|不退化|不 fallback|no auto.?assign).*gh user|当前.*gh' "$decompose_md" \
   || grep -qiE '父无 assignee.*child.*(无|no)|empty.*parent.*empty.*child|父.*无.*assignee.*保持' "$decompose_md"; then
  report_pass "AC4.no-fallback: beaver-decompose.md forbids fallback to current gh user"
else
  report_fail "AC4.no-fallback: beaver-decompose.md must explicitly state: empty parent assignees → empty child assignees (no fallback to gh user)"
fi

if grep -qiE 'override.*assignee|assignee.*override|覆盖.*assignee|assignee.*覆盖' "$decompose_md"; then
  report_pass "AC4.override: beaver-decompose.md allows per-child assignee override"
else
  report_fail "AC4.override: beaver-decompose.md must allow per-child assignee override in QA"
fi

# beaver-decompose.sh must own a subcommand that reads parent assignees
# and one that writes them (assignees REST endpoint or `gh issue edit --add-assignee`).
if grep -qE 'parent-fields|parent-assignees|fetch-parent.*assignees' "$decompose_sh"; then
  report_pass "AC4.sh-read: beaver-decompose.sh exposes a parent-fields/assignees read subcommand"
else
  report_fail "AC4.sh-read: beaver-decompose.sh must expose a subcommand that reads parent assignees"
fi

if grep -qE 'set-assignees|add-assignee|--add-assignee|assignees\b' "$decompose_sh"; then
  report_pass "AC4.sh-write: beaver-decompose.sh exposes assignee write surface"
else
  report_fail "AC4.sh-write: beaver-decompose.sh must expose a subcommand that writes child assignees"
fi

# ---------- AC5: per-child unique tempfile (mktemp / $$ / $RANDOM) ----------

# Forbid the fixed legacy tempfile path used by the pre-migration spec.
for f in "$decompose_sh" "$decompose_md"; do
  hits=$(grep -nE '/tmp/beaver-sub-issue\.md\b' "$f" || true)
  if [ -z "$hits" ]; then
    report_pass "AC5.fixed: $(basename "$f") does not use the legacy /tmp/beaver-sub-issue.md path"
  else
    report_fail "AC5.fixed: $(basename "$f") still uses /tmp/beaver-sub-issue.md:"
    echo "$hits" >&2
  fi
done

# Every `--body-file <path>` reference must point to a $-variable (e.g.
# $BODY_FILE) or to a literal containing $$ / $RANDOM / mktemp output.
for f in "$decompose_sh" "$decompose_md"; do
  body_refs=$(grep -nE -- '--body-file[ =][^ ]+' "$f" || true)
  if [ -z "$body_refs" ]; then
    report_pass "AC5.unique: $(basename "$f") has no --body-file references"
    continue
  fi
  bad=$(echo "$body_refs" \
    | grep -vE -- '--body-file[ =]("?\$[A-Za-z_]|"?\$\{|/tmp/[^ ]*(\$\$|\$RANDOM)|"?\$\(mktemp)' \
    || true)
  if [ -z "$bad" ]; then
    report_pass "AC5.unique: $(basename "$f") --body-file refs all use unique paths"
  else
    report_fail "AC5.unique: $(basename "$f") has --body-file refs with non-unique paths:"
    echo "$bad" >&2
  fi
done

# The script (which loops to create N children) must reference mktemp.
if grep -qE '\bmktemp\b' "$decompose_sh"; then
  report_pass "AC5.mktemp.sh: beaver-decompose.sh references mktemp"
else
  report_fail "AC5.mktemp.sh: beaver-decompose.sh must use mktemp for per-child body files"
fi
if grep -qE '\bmktemp\b' "$decompose_md"; then
  report_pass "AC5.mktemp.md: beaver-decompose.md references mktemp"
else
  report_fail "AC5.mktemp.md: beaver-decompose.md must reference mktemp for the per-child body recipe"
fi

# ---------- AC6: audit warnings as body comment, no `beaver/*` label writes ----------

if grep -qE '<!--\s*audit-warnings\s*-->' "$decompose_md"; then
  report_pass "AC6.marker: beaver-decompose.md mentions the <!-- audit-warnings --> body marker"
else
  report_fail "AC6.marker: beaver-decompose.md must show the <!-- audit-warnings --> body comment marker"
fi

for needle in missing-test needs-split missing-context; do
  if grep -qE "\b${needle}\b" "$decompose_md"; then
    report_pass "AC6.category.${needle}: beaver-decompose.md lists '${needle}' as an audit category"
  else
    report_fail "AC6.category.${needle}: beaver-decompose.md must list '${needle}' as an audit category"
  fi
done

# Forbid any `--add-label`/`gh issue edit ... --add-label` that adds a beaver/* label.
for f in "$decompose_sh" "$decompose_md"; do
  bad=$(grep -nE 'beaver/(missing-test|needs-split|missing-context)' "$f" \
        | grep -E 'add-label|--add-label|labels[^ ]*POST|gh api[^|]*labels' || true)
  if [ -z "$bad" ]; then
    report_pass "AC6.no-label-write: $(basename "$f") does not write beaver/{missing-test|needs-split|missing-context} labels"
  else
    report_fail "AC6.no-label-write: $(basename "$f") still writes beaver/* audit labels:"
    echo "$bad" >&2
  fi
done

# ---------- AC7: parent audit summary comment with dependency graph ----------

if grep -qE 'audit summary|审计汇总|审计总结|summary comment' "$decompose_md"; then
  report_pass "AC7.summary: beaver-decompose.md describes the audit summary parent comment"
else
  report_fail "AC7.summary: beaver-decompose.md must describe an audit summary parent comment"
fi

if grep -qE 'blocked by|依赖关系图|依赖图|dependency graph' "$decompose_md"; then
  report_pass "AC7.depgraph: beaver-decompose.md mentions the dependency graph in the parent comment"
else
  report_fail "AC7.depgraph: beaver-decompose.md must include a dependency graph in the parent comment"
fi

if grep -qiE '依赖写入失败|dependency.*(failed|failure)|手动补登|manual' "$decompose_md"; then
  report_pass "AC7.depfail: beaver-decompose.md notes how dependency-write failures are reported"
else
  report_fail "AC7.depfail: beaver-decompose.md must report dependency-write failures separately"
fi

# ---------- AC9: dependency QA + cycle detection ----------

if grep -qE 'child#[0-9]+|child#N|relative ref|相对引用' "$decompose_md"; then
  report_pass "AC9.relref: beaver-decompose.md uses child#N relative refs for dependencies"
else
  report_fail "AC9.relref: beaver-decompose.md must use child#N relative refs for inter-child dependencies"
fi

if grep -qiE '环检测|cycle detection|DFS' "$decompose_md"; then
  report_pass "AC9.cycle: beaver-decompose.md describes cycle detection (DFS)"
else
  report_fail "AC9.cycle: beaver-decompose.md must describe cycle detection (DFS) before landing"
fi

if grep -qiE 'blocker|阻塞|被.*阻塞' "$decompose_md"; then
  report_pass "AC9.qa: beaver-decompose.md asks per-child blocker question"
else
  report_fail "AC9.qa: beaver-decompose.md must ask per-child blocker question in QA"
fi

# ---------- AC10: dependency landing via Issue Dependencies API ----------

# The doc must mention either the REST endpoint or the GraphQL mutation.
if grep -qE 'dependencies/blocked_by|addIssueDependency' "$decompose_md"; then
  report_pass "AC10.api: beaver-decompose.md references the Issue Dependencies API"
else
  report_fail "AC10.api: beaver-decompose.md must reference dependencies/blocked_by or addIssueDependency"
fi

# The script must own a subcommand for landing dependencies.
if grep -qE 'add-blocked-by|add-dependency|blocked_by' "$decompose_sh"; then
  report_pass "AC10.sh: beaver-decompose.sh exposes a dependency-landing subcommand"
else
  report_fail "AC10.sh: beaver-decompose.sh must expose a dependency-landing subcommand"
fi

# Sub-issue link step (AC2 step 6b) and the dependency landing step are
# distinct — the doc must distinguish them so users know they don't
# conflict.
if grep -qiE 'sub-?issue.*(separate|distinct|independent|不影响|不冲突).*depend|depend.*(separate|distinct|independent).*sub-?issue' "$decompose_md"; then
  report_pass "AC10.distinct: beaver-decompose.md distinguishes sub-issue links from dependencies"
else
  report_fail "AC10.distinct: beaver-decompose.md must distinguish sub-issue links from Issue dependencies"
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
