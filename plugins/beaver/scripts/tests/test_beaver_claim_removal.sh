#!/usr/bin/env bash
#
# Acceptance test for Issue #124:
#   "C1: 删除 /beaver-claim 命令与相关脚本"
#
# Static assertions:
#   AC1 : plugins/beaver/commands/beaver-claim.md and
#         plugins/beaver/scripts/beaver-claim.sh no longer exist.
#   AC2 : `git grep -nE "/beaver-claim|beaver-claim\.(md|sh)" plugins/beaver/`
#         contains no command-invocation references — only documentation /
#         transitional notes are allowed (lines that mention RFC-0013 §3 or
#         「在 GitHub UI assign + 手动切 Status」 transitional message).
#   AC3 : .claude-plugin/plugin.json (root and plugins/beaver/) has no
#         `beaver-claim` entry in any commands list.
#   AC4 : beaver-create.md and beaver-decompose.md do NOT contain
#         `/beaver-claim` next-step hints; their next-step text instead
#         describes the GitHub-UI manual transition.

set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
beaver_dir="$repo_root/plugins/beaver"

failures=0
report_fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
report_pass() { echo "PASS: $*"; }

# ---------------------------------------------------------------- AC1
if [ -e "$beaver_dir/commands/beaver-claim.md" ]; then
  report_fail "AC1: $beaver_dir/commands/beaver-claim.md still exists"
else
  report_pass "AC1: beaver-claim.md is gone"
fi

if [ -e "$beaver_dir/scripts/beaver-claim.sh" ]; then
  report_fail "AC1: $beaver_dir/scripts/beaver-claim.sh still exists"
else
  report_pass "AC1: beaver-claim.sh is gone"
fi

# ---------------------------------------------------------------- AC2
# Allowed forms: lines that explicitly cite RFC-0013 §3 OR the transitional
# message about GitHub-UI assign + manual Status switch. Anything else that
# references /beaver-claim or beaver-claim.{md,sh} is a residual command call.
hits=$(cd "$repo_root" && \
  git grep -nE "/beaver-claim|beaver-claim\.(md|sh)" -- plugins/beaver/ \
  || true)

bad=0
if [ -n "$hits" ]; then
  while IFS= read -r line; do
    case "$line" in
      *"RFC-0013"*|*"已删除"*|*"removed"*|*"deprecated"*|*"deleted"*)
        # Documented transitional/historical reference — allowed.
        ;;
      *)
        report_fail "AC2: residual reference -> $line"
        bad=$((bad + 1))
        ;;
    esac
  done <<< "$hits"
fi
if [ "$bad" -eq 0 ]; then
  report_pass "AC2: no live command references to /beaver-claim"
fi

# ---------------------------------------------------------------- AC3
for pj in "$repo_root/.claude-plugin/plugin.json" \
          "$repo_root/plugins/beaver/.claude-plugin/plugin.json"; do
  [ -f "$pj" ] || continue
  # Look for a JSON commands list value of "beaver-claim" specifically.
  if grep -nE '"(beaver-claim)"' "$pj" >/dev/null 2>&1; then
    report_fail "AC3: $pj still lists beaver-claim as a command"
  else
    report_pass "AC3: $pj has no beaver-claim command entry"
  fi
done

# ---------------------------------------------------------------- AC4
# Next-step hints in beaver-create.md and beaver-decompose.md must not
# instruct users to invoke /beaver-claim. A line that documents the removal
# (cites RFC-0013 §3 or 「已删除」) is allowed.
for f in "$beaver_dir/commands/beaver-create.md" \
         "$beaver_dir/commands/beaver-decompose.md"; do
  bad_lines=$(grep -nE '/beaver-claim' "$f" || true)
  bad_count=0
  if [ -n "$bad_lines" ]; then
    while IFS= read -r line; do
      case "$line" in
        *"RFC-0013"*|*"已删除"*|*"removed"*|*"deprecated"*|*"deleted"*)
          ;;
        *)
          report_fail "AC4: $f live /beaver-claim hint -> $line"
          bad_count=$((bad_count + 1))
          ;;
      esac
    done <<< "$bad_lines"
  fi
  if [ "$bad_count" -eq 0 ]; then
    report_pass "AC4: $(basename "$f") has no /beaver-claim invocation hint"
  fi
done

# ---------------------------------------------------------------- summary
echo ""
if [ "$failures" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  exit 0
else
  echo "$failures CHECK(S) FAILED"
  exit 1
fi
