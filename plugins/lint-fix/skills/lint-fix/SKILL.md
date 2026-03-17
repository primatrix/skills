---
name: lint-fix
description: Check and fix lint issues for changed Python files. Supports single commit, commit range, and unstaged/staged working tree changes. Use when the user wants to verify or fix lint compliance.
argument-hint: "<commit> | <from>..<to> | unstaged | staged | help"
---

# Lint Check & Fix

Target: `$ARGUMENTS`

## 1. Handle Help

If `$ARGUMENTS` is `help` (case-insensitive), output the following usage guide exactly as-is and stop:

```
/lint-fix - Check and fix lint issues for changed Python files.

Usage:
  /lint-fix <commit>          Check files changed in a single commit
  /lint-fix <from>..<to>      Check files changed across a commit range
  /lint-fix unstaged          Check unstaged working tree changes
  /lint-fix staged            Check staged (added but not committed) changes
  /lint-fix help              Show this help message

Examples:
  /lint-fix HEAD              Lint files changed in the latest commit
  /lint-fix abc1234           Lint files changed in commit abc1234
  /lint-fix main..HEAD        Lint all files changed since branching from main
  /lint-fix HEAD~5..HEAD      Lint files changed in the last 5 commits
  /lint-fix unstaged          Lint files you've modified but not yet git-added
  /lint-fix staged            Lint files you've git-added but not yet committed

Linters (run in order): isort -> ruff -> black -> codespell
```

## 2. Determine Mode and Get Changed Files

Parse `$ARGUMENTS` to determine which mode to use:

| Argument | Mode | Command to get changed `.py` files |
|---|---|---|
| `unstaged` | Unstaged working tree changes | `git diff --name-only -- '*.py'` |
| `staged` | Staged (added but not committed) | `git diff --cached --name-only -- '*.py'` |
| Contains `..` (e.g. `main..HEAD`, `HEAD~5..HEAD`) | Commit range | `git diff --name-only <from>..<to> -- '*.py'` |
| Anything else (e.g. `HEAD`, `abc1234`) | Single commit | `git diff --name-only <commit>^..<commit> -- '*.py'` (fall back to `git diff-tree --no-commit-id --name-only -r <commit> -- '*.py'` for initial commits) |

After getting the file list, filter out any files that no longer exist on disk (deleted files).

If no Python files are found, report that and stop.

## 3. Run Lint Checks

Run each linter on the changed files. The project's `.pre-commit-config.yaml` defines the canonical tool versions and flags. Use these defaults if no project config is found:

### a. isort (import sorting)

Check:
```bash
isort --check-only --diff --profile=black <files>
```
Fix if needed:
```bash
isort --profile=black <files>
```

### b. ruff (linting)

```bash
ruff check --output-format=github --fix <files>
```

### c. black (formatting)

Check:
```bash
black --check <files>
```
Fix if needed:
```bash
black <files>
```

### d. codespell (spelling)

```bash
codespell <files>
```

If a linter is not installed, skip it and note the skip in the report.

## 4. Report Results

Summarize:
- Number of files checked and their paths
- Issues found per linter
- Issues auto-fixed vs remaining
- If any issues could not be auto-fixed, read the file and fix them manually using the Edit tool.

## 5. Stage Fixes

If fixes were applied, show the diff summary and stage with `git add <files>` so the user can review before committing.
