---
name: beaver-pr
description: Commit, push, and open a PR linked to a Beaver issue. Trigger this skill whenever the user wants to create a GitHub pull request.
---

# Beaver PR

Commit staged changes, push the branch, and open a GitHub pull request with optional Beaver issue association. The workflow gathers repository context, handles branching, and links the PR to a Beaver-tracked issue.

## Prerequisites

- `gh auth status` must succeed (run `gh auth login` if not)
- Working directory must be inside a git repository

## Workflow

1. **Gather context** -- Run the following commands to understand the current repository state:
   - `git status` to see staged/unstaged changes
   - `git diff HEAD` to see the full diff
   - `git branch --show-current` to identify the current branch
   - `git log --oneline -10` to review recent commits

2. **Create a new branch** if currently on main/master. Use a descriptive branch name based on the changes.

3. **Stage and commit** all changes with an appropriate commit message.

4. **Push** the branch to origin. If a new branch was created, use `git push -u origin <branch-name>` to set the upstream.

5. **Ask about Beaver issue association.** Present this question to the user:

   > "Associate this PR with a Beaver issue?
   > - Enter an issue number (e.g., `#42` or `42`) or full issue URL
   > - Type `new` to create a new issue via `create-beaver-issue`
   > - Type `skip` to create the PR without issue association"

   Handle the response:
   - **Issue number/URL provided:** Parse the input. If a full URL like `https://github.com/org/repo/issues/42` is given, extract `ISSUE_OWNER`, `ISSUE_REPO`, and `ISSUE_NUMBER`. If just a number like `#42` or `42`, record only `ISSUE_NUMBER`.
   - **`new`:** Tell the user to run `create-beaver-issue` first, then come back with the issue number. Stop here -- do NOT create the PR yet. Say: "Please run `create-beaver-issue` to create the issue, then run `beaver-pr` again or tell me the issue number to continue."
   - **`skip`:** Set `ISSUE_NUMBER` to empty.

6. **Create the PR** using `gh pr create`:
   - Generate a concise PR title from the commit(s)
   - Build the PR body:
     - A brief summary section describing the changes
     - If `ISSUE_NUMBER` is set, add a line: `Relates to #ISSUE_NUMBER`. If `ISSUE_OWNER/ISSUE_REPO` differ from the current PR's repo, use `Relates to ISSUE_OWNER/ISSUE_REPO#ISSUE_NUMBER` instead.
   - Use a HEREDOC for the body to preserve formatting

7. **Report** the PR URL to the user.

## Execution Notes

- Steps 1-4 should be executed in a single message, using parallel tool calls where possible (e.g., gather context commands in parallel, then branch/commit/push sequentially).
- Step 5 requires user input -- pause and wait for the response.
- After receiving the answer, execute steps 6-7 in a single message.
