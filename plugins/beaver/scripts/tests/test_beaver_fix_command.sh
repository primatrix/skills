#!/usr/bin/env bash
#
# Acceptance test for Issue primatrix/projects#123:
#   "B8: 实现新命令 /beaver-fix：批量回应 PR review comments"
#
# RFC-0013 §10 — RED-phase static + structural assertions on:
#   plugins/beaver/commands/beaver-fix.md
#   plugins/beaver/scripts/beaver-fix.sh
#   plugins/beaver/.claude-plugin/plugin.json
#
# Acceptance criteria asserted (one or more grep/structural check per AC):
#   AC1: PR author == current `gh` user; abort with literal Chinese message.
#   AC2: Filter review threads by isResolved; bail with literal「无待处理评论」.
#   AC3: Per-comment 4-option prompt (4 literal substrings).
#   AC4: HARD-GATE summary; "yes" confirmation; rollback path on no/Ctrl-C.
#   AC5: resolveReviewThread mutation; conventional-commit prefix
#        `fix(<scope>): address review comments`; explicit Project V2
#        untouched assertion.
#   AC6: every --body-file uses mktemp or $$/$RANDOM unique tmp file.
#   FRONTMATTER: command md has name/description/argument-hint mentioning
#        <pr-number>.
#   PLUGIN: plugin.json registers beaver-fix (description bump or version bump).
#
# Convention: bash, set -uo pipefail (no -e — accumulate failures), failures
# counter, exit code = number of FAIL lines.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
fix_sh="$repo_root/plugins/beaver/scripts/beaver-fix.sh"
fix_md="$repo_root/plugins/beaver/commands/beaver-fix.md"
plugin_json="$repo_root/plugins/beaver/.claude-plugin/plugin.json"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

# Existence pre-checks — every later assertion is guarded so we still print
# all FAILs (helpful for the GREEN-phase implementer).
if [ ! -f "$fix_sh" ]; then
  report_fail "EXIST.sh: $fix_sh missing"
fi
if [ ! -f "$fix_md" ]; then
  report_fail "EXIST.md: $fix_md missing"
fi
if [ ! -f "$plugin_json" ]; then
  report_fail "EXIST.plugin: $plugin_json missing"
fi

# ---------- AC1.static: author check ----------
if [ -f "$fix_sh" ]; then
  # gh pr view ... --json author somewhere
  if grep -qE 'gh pr view.*--json[^|]*author' "$fix_sh"; then
    report_pass "AC1.sh.prview: beaver-fix.sh calls 'gh pr view --json author'"
  else
    report_fail "AC1.sh.prview: beaver-fix.sh must call 'gh pr view ... --json ...author...'"
  fi
  # gh api user --jq .login (current user)
  if grep -qE 'gh api user.*\.login' "$fix_sh"; then
    report_pass "AC1.sh.user: beaver-fix.sh resolves current user via 'gh api user --jq .login'"
  else
    report_fail "AC1.sh.user: beaver-fix.sh must resolve current user via 'gh api user --jq .login'"
  fi
fi

# Literal Chinese error string — accept it in either the script or the .md spec.
ac1_msg='只能对自己发起的 PR 运行 /beaver-fix'
if { [ -f "$fix_sh" ] && grep -qF "$ac1_msg" "$fix_sh"; } || \
   { [ -f "$fix_md" ] && grep -qF "$ac1_msg" "$fix_md"; }; then
  report_pass "AC1.msg: literal '${ac1_msg}' present"
else
  report_fail "AC1.msg: literal '${ac1_msg}' must appear in beaver-fix.sh or beaver-fix.md"
fi

# ---------- AC2.static: open-thread filter + empty bail ----------
if [ -f "$fix_sh" ]; then
  if grep -qE 'gh api graphql' "$fix_sh"; then
    report_pass "AC2.sh.graphql: beaver-fix.sh uses 'gh api graphql'"
  else
    report_fail "AC2.sh.graphql: beaver-fix.sh must use 'gh api graphql' for thread fetch"
  fi
  # Either GraphQL `isResolved` field or a jq filter on RESOLVED state.
  if grep -qE 'isResolved|RESOLVED' "$fix_sh"; then
    report_pass "AC2.sh.filter: beaver-fix.sh filters by isResolved/RESOLVED"
  else
    report_fail "AC2.sh.filter: beaver-fix.sh must filter threads by isResolved (or RESOLVED state)"
  fi
fi
ac2_msg='无待处理评论'
if { [ -f "$fix_sh" ] && grep -qF "$ac2_msg" "$fix_sh"; } || \
   { [ -f "$fix_md" ] && grep -qF "$ac2_msg" "$fix_md"; }; then
  report_pass "AC2.msg: literal '${ac2_msg}' present"
else
  report_fail "AC2.msg: literal '${ac2_msg}' must appear in beaver-fix.sh or beaver-fix.md"
fi

# ---------- AC3.static: 4-option per-comment prompt ----------
if [ -f "$fix_md" ]; then
  for opt in '[接受修复]' '[修改建议]' '[跳过]' '[仅 resolve]'; do
    if grep -qF "$opt" "$fix_md"; then
      report_pass "AC3.opt: beaver-fix.md lists option '${opt}'"
    else
      report_fail "AC3.opt: beaver-fix.md must list per-comment option '${opt}'"
    fi
  done
fi

# ---------- AC4.static: HARD-GATE / yes / rollback ----------
if [ -f "$fix_md" ]; then
  if grep -qF 'HARD-GATE' "$fix_md"; then
    report_pass "AC4.md.hardgate: beaver-fix.md mentions HARD-GATE"
  else
    report_fail "AC4.md.hardgate: beaver-fix.md must mention 'HARD-GATE'"
  fi
  # require some explicit "yes" confirmation phrasing
  if grep -qE 'yes' "$fix_md"; then
    report_pass "AC4.md.yes: beaver-fix.md mentions 'yes' confirmation"
  else
    report_fail "AC4.md.yes: beaver-fix.md must mention 'yes' confirmation token"
  fi
  if grep -qE '回滚|rollback' "$fix_md"; then
    report_pass "AC4.md.rollback: beaver-fix.md mentions 回滚 / rollback"
  else
    report_fail "AC4.md.rollback: beaver-fix.md must mention 回滚 / rollback"
  fi
fi
if [ -f "$fix_sh" ]; then
  # Some recognized rollback mechanism (checkout HEAD, reset, restore, stash pop, etc.).
  if grep -qE 'git checkout --|git checkout HEAD|git restore|git reset --hard|git stash' "$fix_sh"; then
    report_pass "AC4.sh.rollback: beaver-fix.sh contains a git rollback path"
  else
    report_fail "AC4.sh.rollback: beaver-fix.sh must contain a rollback path (git checkout/restore/reset/stash)"
  fi
  # A trap on INT for Ctrl-C handling.
  if grep -qE 'trap[[:space:]]+.*INT' "$fix_sh"; then
    report_pass "AC4.sh.trap: beaver-fix.sh installs an INT trap (Ctrl-C handler)"
  else
    report_fail "AC4.sh.trap: beaver-fix.sh must install a 'trap ... INT' handler for Ctrl-C rollback"
  fi
fi

# ---------- AC5.static: resolveReviewThread + conventional commit + Project V2 ----------
if [ -f "$fix_sh" ]; then
  if grep -qF 'resolveReviewThread' "$fix_sh"; then
    report_pass "AC5.sh.mutation: beaver-fix.sh calls resolveReviewThread"
  else
    report_fail "AC5.sh.mutation: beaver-fix.sh must call GraphQL mutation 'resolveReviewThread'"
  fi
  # Conventional commit message template — accept any scope.
  if grep -qE 'fix\([^)]+\): address review comments' "$fix_sh"; then
    report_pass "AC5.sh.commit: beaver-fix.sh contains 'fix(<scope>): address review comments' template"
  else
    report_fail "AC5.sh.commit: beaver-fix.sh must contain a conventional-commit template matching 'fix(<scope>): address review comments'"
  fi
fi
# Project V2 untouched assertion — accept either spelling, in script or md.
if { [ -f "$fix_sh" ] && grep -qE '不修改 Project V2 字段|Project V2 fields untouched' "$fix_sh"; } || \
   { [ -f "$fix_md" ] && grep -qE '不修改 Project V2 字段|Project V2 fields untouched' "$fix_md"; }; then
  report_pass "AC5.projectv2: explicit 'Project V2 untouched' assertion present"
else
  report_fail "AC5.projectv2: must contain '不修改 Project V2 字段' or 'Project V2 fields untouched' in script or md"
fi

# ---------- AC6.static: every --body-file refers to a unique tmp file ----------
# Intent (per RFC §命令规约 preamble + Issue AC6): the path passed to
# `--body-file` must be a unique tmp file (mktemp-backed, or built from
# $$+$RANDOM). The path is typically passed via a variable like `--body-file
# "$qf"`, where `$qf` was assigned from `mktemp` somewhere earlier in the
# script. So for each --body-file invocation we:
#   1. Reject literal path arguments (e.g. `--body-file /tmp/foo.md`) —
#      inherently non-unique.
#   2. Extract the variable name (`qf` from `"$qf"`) and confirm the script
#      contains an assignment of that variable from `mktemp` or a path built
#      from `$$`+`$RANDOM`.
if [ -f "$fix_sh" ]; then
  body_lines=$(grep -nE '\-\-body-file' "$fix_sh" || true)
  if [ -z "$body_lines" ]; then
    report_fail "AC6.usage: beaver-fix.sh must use '--body-file' at least once (per RFC §命令规约)"
  else
    bad=0
    while IFS= read -r entry; do
      line=${entry#*:}
      # Reject literal path arguments to --body-file.
      if echo "$line" | grep -qE -- '--body-file[[:space:]]+/[^"$]'; then
        bad=$((bad + 1))
        echo "  unsafe --body-file (literal path): $line" >&2
        continue
      fi
      # Extract the variable name from `--body-file "$VAR"` (or `$VAR`).
      var=$(echo "$line" | sed -nE 's/.*--body-file[[:space:]]+"?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"?.*/\1/p')
      if [ -z "$var" ]; then
        bad=$((bad + 1))
        echo "  unparseable --body-file argument: $line" >&2
        continue
      fi
      # Confirm an assignment of $var from mktemp or $$+$RANDOM exists.
      if grep -qE "^[[:space:]]*${var}=.*mktemp" "$fix_sh"; then
        :
      elif grep -E "^[[:space:]]*${var}=" "$fix_sh" | grep -qE '\$\$' && \
           grep -E "^[[:space:]]*${var}=" "$fix_sh" | grep -qE '\$RANDOM'; then
        :
      else
        bad=$((bad + 1))
        echo "  variable '\$${var}' is not assigned from mktemp or \$\$+\$RANDOM: $line" >&2
      fi
    done <<< "$body_lines"
    if [ "$bad" -eq 0 ]; then
      report_pass "AC6.unique: every '--body-file' refers to a mktemp / \$\$+\$RANDOM unique tmp file"
    else
      report_fail "AC6.unique: ${bad} '--body-file' invocation(s) lack a unique-name construction"
    fi
  fi
fi

# ---------- AC2.extra: PR-level issue comments are ALSO collected ----------
if [ -f "$fix_sh" ]; then
  if grep -qF 'issueComments' "$fix_sh"; then
    report_pass "AC2.sh.issuecomments: beaver-fix.sh collects PR-level issueComments"
  else
    report_fail "AC2.sh.issuecomments: beaver-fix.sh must reference 'issueComments' (PR-level top-level comments) in its graphql query"
  fi
fi
if [ -f "$fix_md" ]; then
  # Require BOTH a mention of review threads AND a mention of issue comments
  has_threads=0
  has_issue=0
  if grep -qiE 'review thread|review threads|reviewThreads|审阅线程|评论线程' "$fix_md"; then
    has_threads=1
  fi
  if grep -qiE 'issue comment|issueComments|issue-level|PR-level top|PR 级|顶级评论|顶层评论|PR 顶层' "$fix_md"; then
    has_issue=1
  fi
  if [ "$has_threads" -eq 1 ] && [ "$has_issue" -eq 1 ]; then
    report_pass "AC2.md.bothsources: beaver-fix.md mentions both review threads AND issue comments"
  else
    report_fail "AC2.md.bothsources: beaver-fix.md must mention BOTH review threads AND issue comments (got threads=${has_threads} issue=${has_issue})"
  fi
fi

# ---------- AC3.md.serial: explicit per-comment serialization ----------
if [ -f "$fix_md" ]; then
  has_serial=0
  has_immediate=0
  if grep -qF '逐条' "$fix_md"; then
    has_serial=1
  fi
  if grep -qF '立即写入' "$fix_md" || grep -qiE 'immediate' "$fix_md"; then
    has_immediate=1
  fi
  if [ "$has_serial" -eq 1 ] && [ "$has_immediate" -eq 1 ]; then
    report_pass "AC3.md.serial: beaver-fix.md mandates per-comment serialization with immediate write"
  else
    report_fail "AC3.md.serial: beaver-fix.md must mandate '逐条' AND ('立即写入' OR 'immediate') (got serial=${has_serial} immediate=${has_immediate})"
  fi
fi

# ---------- AC4.sh.scoped_rollback: rollback must be scoped, not blanket ----------
if [ -f "$fix_sh" ]; then
  # Reject the literal blanket form
  if grep -qE 'git checkout[[:space:]]+--[[:space:]]+\.[[:space:]]*$' "$fix_sh" || \
     grep -qE 'git checkout[[:space:]]+--[[:space:]]+\.[[:space:]]+' "$fix_sh"; then
    report_fail "AC4.sh.scoped_rollback: beaver-fix.sh must NOT use blanket 'git checkout -- .' (clobbers unrelated work)"
  else
    # Require some scoped form: git restore -- <args>, OR git checkout HEAD -- "$var"
    if grep -qE 'git restore[[:space:]]+--[[:space:]]+"?\$' "$fix_sh" || \
       grep -qE 'git checkout[[:space:]]+HEAD[[:space:]]+--[[:space:]]+"?\$' "$fix_sh"; then
      report_pass "AC4.sh.scoped_rollback: beaver-fix.sh uses scoped rollback (per-file restore/checkout)"
    else
      report_fail "AC4.sh.scoped_rollback: beaver-fix.sh must use a scoped rollback ('git restore -- \"\$f\"' or 'git checkout HEAD -- \"\$f\"')"
    fi
  fi
fi

# ---------- AC4.sh.trap_err: trap must include ERR (not only INT) ----------
if [ -f "$fix_sh" ]; then
  if grep -qE 'trap[[:space:]]+[^#]*\bERR\b' "$fix_sh"; then
    report_pass "AC4.sh.trap_err: beaver-fix.sh installs an ERR trap (covers mid-script crash with set -e)"
  else
    report_fail "AC4.sh.trap_err: beaver-fix.sh must include 'ERR' in a trap line (otherwise set -e exits skip rollback)"
  fi
fi

# ---------- AC5.sh.snapshot: Project V2 snapshot subcommand ----------
if [ -f "$fix_sh" ]; then
  if grep -qE 'snapshot-projectv2-fields|snapshot_projectv2_fields' "$fix_sh"; then
    report_pass "AC5.sh.snapshot: beaver-fix.sh defines a snapshot-projectv2-fields path"
  else
    report_fail "AC5.sh.snapshot: beaver-fix.sh must define a 'snapshot-projectv2-fields' subcommand/function"
  fi
  # Must read Project V2 field values via graphql
  if grep -qE 'projectItems|ProjectV2Item|fieldValues' "$fix_sh"; then
    report_pass "AC5.sh.snapshot.gql: beaver-fix.sh reads Project V2 fieldValues via graphql"
  else
    report_fail "AC5.sh.snapshot.gql: beaver-fix.sh snapshot must query 'projectItems'/'ProjectV2Item'/'fieldValues' via graphql"
  fi
fi

# ---------- AC5.sh.compare: snapshot before/after diff with non-zero exit ----------
if [ -f "$fix_sh" ]; then
  has_verify=0
  if grep -qE 'verify-projectv2-fields|verify_projectv2_fields' "$fix_sh"; then
    has_verify=1
  fi
  has_cmp=0
  # Accept diff/cmp invocation, OR an explicit string comparison.
  if grep -qE '\bdiff[[:space:]]+' "$fix_sh" || \
     grep -qE '\bcmp[[:space:]]+' "$fix_sh" || \
     grep -qE '\[[[:space:]]+"\$[A-Za-z_]+"[[:space:]]*=[[:space:]]*"\$[A-Za-z_]+"[[:space:]]+\]' "$fix_sh"; then
    has_cmp=1
  fi
  has_exit=0
  if grep -qE 'exit[[:space:]]+1' "$fix_sh"; then
    has_exit=1
  fi
  if [ "$has_verify" -eq 1 ] && [ "$has_cmp" -eq 1 ] && [ "$has_exit" -eq 1 ]; then
    report_pass "AC5.sh.compare: beaver-fix.sh has verify-projectv2-fields with diff/cmp + exit 1 on mismatch"
  else
    report_fail "AC5.sh.compare: beaver-fix.sh must have verify-projectv2-fields with diff/cmp comparison + exit 1 on mismatch (verify=${has_verify} cmp=${has_cmp} exit=${has_exit})"
  fi
fi

# ---------- AC5.md.assertion: explicit Phase calling snapshot + verify ----------
if [ -f "$fix_md" ]; then
  has_snap_call=0
  has_verify_call=0
  has_phrase=0
  if grep -qE 'snapshot-projectv2-fields' "$fix_md"; then
    has_snap_call=1
  fi
  if grep -qE 'verify-projectv2-fields' "$fix_md"; then
    has_verify_call=1
  fi
  if grep -qE 'Project V2 字段未被修改|Project V2 untouched' "$fix_md"; then
    has_phrase=1
  fi
  if [ "$has_snap_call" -eq 1 ] && [ "$has_verify_call" -eq 1 ] && [ "$has_phrase" -eq 1 ]; then
    report_pass "AC5.md.assertion: beaver-fix.md calls snapshot + verify subcommands with explicit phrase"
  else
    report_fail "AC5.md.assertion: beaver-fix.md must call snapshot-projectv2-fields AND verify-projectv2-fields AND mention 'Project V2 字段未被修改' or 'Project V2 untouched' (snap=${has_snap_call} verify=${has_verify_call} phrase=${has_phrase})"
  fi
fi

# ---------- AC5.sh.empty_diff_guard: skip commit when nothing staged ----------
if [ -f "$fix_sh" ]; then
  if grep -qE 'git diff --cached --quiet([^a-zA-Z0-9_-]|$)|git diff --quiet HEAD' "$fix_sh"; then
    report_pass "AC5.sh.empty_diff_guard: beaver-fix.sh guards commit step on empty staged diff"
  else
    report_fail "AC5.sh.empty_diff_guard: beaver-fix.sh must guard commit on 'git diff --cached --quiet' (or 'git diff --quiet HEAD')"
  fi
fi

# ---------- AC5.sh.push_upstream: git push -u for fresh branches ----------
if [ -f "$fix_sh" ]; then
  if grep -qE 'git push[[:space:]]+-u\b' "$fix_sh"; then
    report_pass "AC5.sh.push_upstream: beaver-fix.sh uses 'git push -u' (matches beaver-pr.sh)"
  else
    report_fail "AC5.sh.push_upstream: beaver-fix.sh must use 'git push -u origin HEAD' (or branch) for fresh PR branches"
  fi
fi

# ---------- FRONTMATTER: command md ----------
if [ -f "$fix_md" ]; then
  # Frontmatter must include name/description/argument-hint inside the leading --- fence.
  fm=$(awk 'BEGIN{n=0} /^---[[:space:]]*$/{n++; next} n==1{print} n>1{exit}' "$fix_md")
  if echo "$fm" | grep -qE '^name:[[:space:]]*beaver-fix[[:space:]]*$'; then
    report_pass "FM.name: frontmatter has 'name: beaver-fix'"
  else
    report_fail "FM.name: frontmatter must have 'name: beaver-fix'"
  fi
  if echo "$fm" | grep -qE '^description:'; then
    report_pass "FM.desc: frontmatter has 'description:' field"
  else
    report_fail "FM.desc: frontmatter must have a 'description:' field"
  fi
  if echo "$fm" | grep -qE '^argument-hint:.*<pr-number>'; then
    report_pass "FM.hint: frontmatter argument-hint mentions <pr-number>"
  else
    report_fail "FM.hint: frontmatter must have 'argument-hint:' mentioning <pr-number>"
  fi
fi

# ---------- PLUGIN: plugin.json registration ----------
if [ -f "$plugin_json" ]; then
  # Accept either a description string mentioning beaver-fix OR a version > 3.2.0.
  hit_desc=0
  hit_ver=0
  if grep -qE '"description"[[:space:]]*:[[:space:]]*"[^"]*beaver-fix' "$plugin_json"; then
    hit_desc=1
  fi
  # Version bump above 3.2.0 — accept anything matching 3.[3-9]+ or 4+ or 3.2.[1-9]+
  ver=$(grep -E '"version"[[:space:]]*:' "$plugin_json" | head -n1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  if [ -n "$ver" ]; then
    # naive semver check: split major.minor.patch
    major=$(echo "$ver" | awk -F. '{print $1+0}')
    minor=$(echo "$ver" | awk -F. '{print $2+0}')
    patch=$(echo "$ver" | awk -F. '{print $3+0}')
    if [ "$major" -gt 3 ] || \
       { [ "$major" -eq 3 ] && [ "$minor" -gt 2 ]; } || \
       { [ "$major" -eq 3 ] && [ "$minor" -eq 2 ] && [ "$patch" -gt 0 ]; }; then
      hit_ver=1
    fi
  fi
  if [ "$hit_desc" -eq 1 ] || [ "$hit_ver" -eq 1 ]; then
    report_pass "PLUGIN.register: plugin.json registers beaver-fix (description mention or version > 3.2.0)"
  else
    report_fail "PLUGIN.register: plugin.json must mention beaver-fix in description OR bump version above 3.2.0 (current: ${ver:-<missing>})"
  fi
fi

# ---------- Live stub (BEAVER_LIVE=1) ----------
# Author-check is only meaningfully testable against a real PR. Provide a stub
# that the orchestrator can wire up later.
if [ "${BEAVER_LIVE:-0}" = "1" ]; then
  echo "--- BEAVER_LIVE=1 — live assertions for /beaver-fix would go here ---"
  echo "  (stub; requires a real PR number and gh auth to exercise the author-check)"
fi

# ---------- summary ----------
echo
if [ "$failures" -eq 0 ]; then
  echo "All acceptance assertions passed."
  exit 0
else
  echo "$failures assertion(s) failed."
  exit "$failures"
fi
