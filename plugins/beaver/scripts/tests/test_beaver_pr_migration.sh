#!/usr/bin/env bash
#
# Acceptance test for Issue #121:
#   "B6: beaver-pr 迁移：G004/G006 audit 改为 PR body warning + 自动补齐 Type/Size"
#
# Static assertions (run by default):
#   AC1.closes      : beaver-pr.md describes inferring issue number from branch
#                     prefix or recent commit, prompting the user when missing,
#                     and embedding `Closes #<n>` in the PR body.
#   AC2.g004-warn   : G004 (no test files) appends a warning line to the PR body
#                     ("⚠️ Beaver audit: 本次 PR 未包含 test 文件改动") and does
#                     NOT post `beaver/missing-test` on the Issue.
#   AC3.g006-fill   : G006 calls beaver-lib.sh::set_type / set_size to auto-fill
#                     fields (Size defaults to S); only on auto-fill failure does
#                     it append a warning to the PR body. It does NOT post
#                     `beaver/missing-context` on the Issue.
#   AC4.finishing   : The command presents exactly 4 mutually-exclusive options
#                     (keep Draft / mark-ready / keep branch / discard), with
#                     discard requiring a typed second confirmation.
#   AC5.no-status   : Source files contain no `gh api .../labels` references
#                     mentioning `status/* / type/* / size/*` strings; outside
#                     of the G006 Type/Size auto-fill, the command does not
#                     mutate any Project V2 fields (no set_status, no
#                     set_iteration, no updateProjectV2ItemFieldValue, no
#                     updateIssueIssueType outside the G006 path).
#   AC6.tempfile    : Every `--body-file` callsite in beaver-pr.sh is fed by a
#                     unique tempfile (mktemp / $$ / $RANDOM / timestamp) to
#                     avoid name collisions inside one command invocation.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
pr_md="$repo_root/plugins/beaver/commands/beaver-pr.md"
pr_sh="$repo_root/plugins/beaver/scripts/beaver-pr.sh"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

# ---------- preflight ----------
[ -f "$pr_md" ] || { report_fail "beaver-pr.md missing at $pr_md"; }
[ -f "$pr_sh" ] || { report_fail "beaver-pr.sh missing at $pr_sh"; }

# ---------- AC1.closes ----------
if [ -f "$pr_md" ]; then
  # Require the full cross-repo form `Closes {org}/{issueRepo}#<n>` because
  # Beaver Issues live in `primatrix/projects` while code PRs are usually in
  # other repos; GitHub's bare `Closes #N` only auto-closes same-repo issues.
  if grep -qE 'Closes \{org\}/\{issueRepo\}#' "$pr_md"; then
    report_pass "AC1.closes.body: PR body contains 'Closes {org}/{issueRepo}#<n>' (cross-repo form)"
  else
    report_fail "AC1.closes.body: beaver-pr.md must say 'Closes {org}/{issueRepo}#<n>' in PR body (cross-repo auto-close requires full owner/repo form)"
  fi
  # Guard against regression to the bare same-repo form on a template line.
  if grep -nE '^[^#>]*Closes #\{issue_number\}' "$pr_md" >/dev/null; then
    report_fail "AC1.closes.no-bare: beaver-pr.md must not use bare 'Closes #{issue_number}' (cross-repo close will silently fail)"
    grep -nE '^[^#>]*Closes #\{issue_number\}' "$pr_md" >&2
  else
    report_pass "AC1.closes.no-bare: beaver-pr.md uses no bare 'Closes #{issue_number}' template"
  fi
  # Branch prefix inference + commit fallback + ask-user fallback all mentioned.
  if grep -qE '分支前缀|branch.*prefix|branch.*name' "$pr_md"; then
    report_pass "AC1.closes.branch-prefix: branch-prefix inference mentioned"
  else
    report_fail "AC1.closes.branch-prefix: beaver-pr.md must describe inferring issue # from branch prefix"
  fi
  if grep -qE 'commit|提交信息|commit message' "$pr_md"; then
    report_pass "AC1.closes.commit: commit-message fallback mentioned"
  else
    report_fail "AC1.closes.commit: beaver-pr.md must describe commit-message fallback"
  fi
  if grep -qE '提示用户|ask.*user|prompt.*user' "$pr_md"; then
    report_pass "AC1.closes.prompt: prompt-user fallback mentioned"
  else
    report_fail "AC1.closes.prompt: beaver-pr.md must describe prompting user when issue # missing"
  fi
fi

# ---------- AC2.g004-warn ----------
if [ -f "$pr_md" ]; then
  if grep -qE 'Beaver audit.*test|未包含 test|missing.*test.*audit' "$pr_md"; then
    report_pass "AC2.g004-warn.text: G004 warning text described in beaver-pr.md"
  else
    report_fail "AC2.g004-warn.text: beaver-pr.md must describe '⚠️ Beaver audit ...test' warning line"
  fi
  if grep -qE 'beaver/missing-test' "$pr_md"; then
    report_fail "AC2.g004-warn.no-label: beaver-pr.md must not mention applying 'beaver/missing-test' label"
  else
    report_pass "AC2.g004-warn.no-label: beaver-pr.md does not mention 'beaver/missing-test' label"
  fi
fi
if [ -f "$pr_sh" ]; then
  if grep -qE 'beaver/missing-test' "$pr_sh"; then
    report_fail "AC2.g004-warn.no-label.sh: beaver-pr.sh must not write 'beaver/missing-test' label"
  else
    report_pass "AC2.g004-warn.no-label.sh: beaver-pr.sh does not write 'beaver/missing-test' label"
  fi
fi

# ---------- AC3.g006-fill ----------
if [ -f "$pr_md" ]; then
  if grep -qE 'set_type|set_size' "$pr_md"; then
    report_pass "AC3.g006-fill.lib: beaver-pr.md references set_type / set_size"
  else
    report_fail "AC3.g006-fill.lib: beaver-pr.md must reference beaver-lib.sh::set_type / set_size for G006 auto-fill"
  fi
  if grep -qE 'Size.*默认.*S|默认.*Size.*S|default.*Size.*S|Size.*default.*S' "$pr_md"; then
    report_pass "AC3.g006-fill.size-default: Size default 'S' mentioned"
  else
    report_fail "AC3.g006-fill.size-default: beaver-pr.md must say 'Size defaults to S' on G006 auto-fill"
  fi
  if grep -qE 'beaver/missing-context' "$pr_md"; then
    report_fail "AC3.g006-fill.no-label: beaver-pr.md must not mention applying 'beaver/missing-context' label"
  else
    report_pass "AC3.g006-fill.no-label: beaver-pr.md does not mention 'beaver/missing-context' label"
  fi
  # Auto-fill failure must result in a PR body warning.
  if grep -qE 'auto-fill.*fail|补齐失败|fill.*fail' "$pr_md"; then
    report_pass "AC3.g006-fill.fail-warn: auto-fill failure triggers PR body warning"
  else
    report_fail "AC3.g006-fill.fail-warn: beaver-pr.md must say 'auto-fill failure -> PR body warning'"
  fi
fi
if [ -f "$pr_sh" ]; then
  if grep -qE 'beaver/missing-context' "$pr_sh"; then
    report_fail "AC3.g006-fill.no-label.sh: beaver-pr.sh must not write 'beaver/missing-context' label"
  else
    report_pass "AC3.g006-fill.no-label.sh: beaver-pr.sh does not write 'beaver/missing-context' label"
  fi
fi

# ---------- AC4.finishing ----------
if [ -f "$pr_md" ]; then
  declare -a needles=(
    'Draft'
    'Ready for Review'
    '保留分支|keep.*branch|Keep the branch'
    'discard|Discard'
  )
  for needle in "${needles[@]}"; do
    if grep -qE "$needle" "$pr_md"; then
      report_pass "AC4.finishing.${needle}: option present"
    else
      report_fail "AC4.finishing.${needle}: option missing"
    fi
  done
  # Discard must require a typed second confirmation.
  if grep -qE '二次确认|typed.*confirm|confirmation|输入.*discard|type.*discard' "$pr_md"; then
    report_pass "AC4.finishing.discard-confirm: discard requires typed confirmation"
  else
    report_fail "AC4.finishing.discard-confirm: beaver-pr.md must require typed confirmation for discard"
  fi
fi

# ---------- AC5.no-status ----------
# Forbidden in the source: gh api ... /labels with a status/|type/|size/ argument.
forbidden_label_writes='gh api .*labels.*(status/|type/|size/)'
for f in "$pr_md" "$pr_sh"; do
  [ -f "$f" ] || continue
  if grep -nE "$forbidden_label_writes" "$f" >/dev/null; then
    report_fail "AC5.no-status.${f##*/}: legacy status/|type/|size/ label-API references found"
    grep -nE "$forbidden_label_writes" "$f" >&2
  else
    report_pass "AC5.no-status.${f##*/}: no legacy status/|type/|size/ label-API references"
  fi
done

# Outside G006 auto-fill (set_type/set_size), no other Project V2 mutations.
forbidden_writes='set_status|set_iteration|updateProjectV2ItemFieldValue|updateIssueIssueType'
for f in "$pr_md" "$pr_sh"; do
  [ -f "$f" ] || continue
  if grep -nE "$forbidden_writes" "$f" >/dev/null; then
    report_fail "AC5.no-status.${f##*/}.fields: forbidden Project V2 mutations found"
    grep -nE "$forbidden_writes" "$f" >&2
  else
    report_pass "AC5.no-status.${f##*/}.fields: no forbidden Project V2 mutations"
  fi
done

# ---------- AC6.tempfile ----------
if [ -f "$pr_sh" ]; then
  if grep -qE '\-\-body-file' "$pr_sh"; then
    report_pass "AC6.tempfile.body-file: --body-file used"
  else
    report_fail "AC6.tempfile.body-file: beaver-pr.sh must use --body-file for create-pr"
  fi
  if grep -qE 'mktemp|\$\$|\$RANDOM|date \+%s' "$pr_sh"; then
    report_pass "AC6.tempfile.unique: unique tempfile pattern present"
  else
    report_fail "AC6.tempfile.unique: beaver-pr.sh must uniquify tempfile names (mktemp / \$\$ / \$RANDOM)"
  fi
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
