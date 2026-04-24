#!/usr/bin/env bash
#
# Acceptance test for Issue #120:
#   "B5: beaver-dev 迁移到字段语义 + 仅接受 Size=S + 完成分支询问 /beaver-pr"
#
# Static assertions (run by default):
#   AC1 : Preflight reads Size/Status/assignee via beaver-lib.sh field
#         helpers (NOT labels). Size != S is rejected with the message
#         "本命令仅处理 Size=S".
#   AC2 : Worktree branch naming pattern <type>/<n>-<short_desc>.
#   AC3 : Markdown invokes the three superpowers skills by name:
#         test-driven-development, systematic-debugging,
#         requesting-code-review.
#   AC4 : Verification Iron Law — full test suite is run before
#         "completion" branch; 0 failures required; forbidden words
#         ("should", "probably", "seems to") guidance present.
#   AC5 : Completion branch asks user "是否直接 /beaver-pr <n>? (y/N)";
#         on `y` invokes /beaver-pr with the issue number.
#   AC6 : Source files contain no `status/` / `type/` / `size/` literal
#         strings; the script does NOT mutate Status (no set_status,
#         no swap-status, no swap-to-in-progress subcommand).

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
dev_sh="$repo_root/plugins/beaver/scripts/beaver-dev.sh"
dev_md="$repo_root/plugins/beaver/commands/beaver-dev.md"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

if [ ! -f "$dev_sh" ]; then
  report_fail "beaver-dev.sh not found at $dev_sh"
fi
if [ ! -f "$dev_md" ]; then
  report_fail "beaver-dev.md not found at $dev_md"
fi

# ---------- AC6: no label literals, no status mutations ----------

for f in "$dev_sh" "$dev_md"; do
  hits=$(grep -nE 'status/[a-z]|type/[a-z]|size/[A-Z]' "$f" || true)
  if [ -z "$hits" ]; then
    report_pass "AC6.literals: $(basename "$f") has no status/|type/|size/ literals"
  else
    report_fail "AC6.literals: $(basename "$f") still contains label literals:"
    echo "$hits" >&2
  fi
done

# Script must not contain set_status invocation, swap-status, or
# swap-to-in-progress subcommand — Status is not mutated by this command.
for needle in 'set_status' 'swap-status' 'swap-to-in-progress'; do
  if grep -qE "\\b${needle}\\b" "$dev_sh"; then
    report_fail "AC6.no-status-mutation: beaver-dev.sh must not reference '${needle}'"
  else
    report_pass "AC6.no-status-mutation: beaver-dev.sh does not reference '${needle}'"
  fi
done

# Markdown must not instruct status mutation either.
for needle in 'set_status' 'swap-status' 'swap-to-in-progress'; do
  if grep -qE "\\b${needle}\\b" "$dev_md"; then
    report_fail "AC6.no-status-mutation: beaver-dev.md must not reference '${needle}'"
  else
    report_pass "AC6.no-status-mutation: beaver-dev.md does not reference '${needle}'"
  fi
done

# ---------- AC1: preflight reads fields via beaver-lib.sh ----------

# (a) Script must expose a `preflight` subcommand.
if grep -qE '^\s*preflight\)' "$dev_sh"; then
  report_pass "AC1.preflight-subcmd: beaver-dev.sh exposes a 'preflight' subcommand"
else
  report_fail "AC1.preflight-subcmd: beaver-dev.sh must expose a 'preflight' subcommand"
fi

# (b) Script must source beaver-lib.sh (for field reads).
if grep -qE 'beaver-lib\.sh' "$dev_sh"; then
  report_pass "AC1.lib-source: beaver-dev.sh references beaver-lib.sh"
else
  report_fail "AC1.lib-source: beaver-dev.sh must source/reference beaver-lib.sh for field reads"
fi

# (c) Script must read Size, Status, assignee in preflight.
preflight_block=$(awk '/^[[:space:]]*preflight\)/,/^[[:space:]]*;;/' "$dev_sh")
for keyword in 'Size' 'Status' 'assignee'; do
  if echo "$preflight_block" | grep -qiE "$keyword"; then
    report_pass "AC1.preflight-reads: preflight references '$keyword'"
  else
    report_fail "AC1.preflight-reads: preflight must reference '$keyword'"
  fi
done

# (d) The Size != S rejection message must appear in either source.
if grep -qE '本命令仅处理 ?Size=S|本命令仅处理.*Size.*S' "$dev_sh" "$dev_md"; then
  report_pass "AC1.reject-msg: 'Size != S' rejection message present"
else
  report_fail "AC1.reject-msg: missing rejection message '本命令仅处理 Size=S'"
fi

# (e) Markdown must mention reading fields via beaver-lib.sh
if grep -qE 'beaver-lib\.sh' "$dev_md"; then
  report_pass "AC1.md-lib: beaver-dev.md references beaver-lib.sh"
else
  report_fail "AC1.md-lib: beaver-dev.md must reference beaver-lib.sh"
fi

# ---------- AC2: worktree branch naming <type>/<n>-<short_desc> ----------

# Script's add-worktree subcommand still exists (the workflow uses it).
if grep -qE '^\s*add-worktree\)' "$dev_sh"; then
  report_pass "AC2.add-worktree: beaver-dev.sh retains add-worktree subcommand"
else
  report_fail "AC2.add-worktree: beaver-dev.sh must retain add-worktree subcommand"
fi

# Markdown must describe the branch naming pattern.
if grep -qE '<type>/<(n|number|issue_number|issue-number)>-<short[_-]desc>' "$dev_md"; then
  report_pass "AC2.branch-pattern: beaver-dev.md documents <type>/<n>-<short_desc> branch pattern"
else
  report_fail "AC2.branch-pattern: beaver-dev.md must document <type>/<n>-<short_desc> branch naming"
fi

# ---------- AC3: superpowers skill names invoked ----------

for skill in 'test-driven-development' 'systematic-debugging' 'requesting-code-review'; do
  if grep -qE "superpowers:${skill}" "$dev_md"; then
    report_pass "AC3.skill: beaver-dev.md invokes superpowers:${skill}"
  else
    report_fail "AC3.skill: beaver-dev.md must invoke superpowers:${skill}"
  fi
done

# ---------- AC4: Verification Iron Law ----------

if grep -qE 'Verification Iron Law' "$dev_md"; then
  report_pass "AC4.iron-law: beaver-dev.md mentions Verification Iron Law"
else
  report_fail "AC4.iron-law: beaver-dev.md must mention Verification Iron Law"
fi

if grep -qE '0 failures?|zero failures?' "$dev_md"; then
  report_pass "AC4.zero-failures: beaver-dev.md requires 0 failures"
else
  report_fail "AC4.zero-failures: beaver-dev.md must require 0 failures"
fi

# Forbidden-words guidance present.
forbidden_count=0
for word in 'should' 'probably' 'seems to'; do
  if grep -qE "\"${word}\"|'${word}'" "$dev_md"; then
    forbidden_count=$((forbidden_count + 1))
  fi
done
if [ "$forbidden_count" -ge 2 ]; then
  report_pass "AC4.forbidden-words: beaver-dev.md flags ≥2 forbidden hedge words"
else
  report_fail "AC4.forbidden-words: beaver-dev.md must flag forbidden hedge words (should/probably/seems to)"
fi

# ---------- AC5: completion branch asks /beaver-pr ----------

if grep -qE '是否直接 ?\`?/beaver-pr' "$dev_md"; then
  report_pass "AC5.ask-pr: beaver-dev.md asks '是否直接 /beaver-pr ...'"
else
  report_fail "AC5.ask-pr: beaver-dev.md must ask '是否直接 /beaver-pr <n>?'"
fi

if grep -qE '\(y/N\)|\(Y/n\)' "$dev_md"; then
  report_pass "AC5.yn-prompt: beaver-dev.md uses (y/N) prompt format"
else
  report_fail "AC5.yn-prompt: beaver-dev.md must include (y/N) prompt"
fi

# Issue number must be passed to /beaver-pr.
if grep -qE '/beaver-pr [{<]?(n|number|issue[-_]?number)[}>]?' "$dev_md"; then
  report_pass "AC5.passes-number: beaver-dev.md passes issue number to /beaver-pr"
else
  report_fail "AC5.passes-number: beaver-dev.md must pass issue number to /beaver-pr"
fi

# ---------- AC6 (extra): assert Status field invariant ----------

# Markdown must explicitly state Status is not modified at end.
if grep -qE 'Status.*(unchanged|不变|保持|未修改|not modified)' "$dev_md"; then
  report_pass "AC6.status-invariant: beaver-dev.md asserts Status field unchanged"
else
  report_fail "AC6.status-invariant: beaver-dev.md must assert Status field is unchanged at completion"
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
