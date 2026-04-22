# Beaver Commands Bash Extraction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move all `gh api` / `git` / GraphQL bash blocks out of `plugins/beaver/commands/*.md` into per-command `plugins/beaver/scripts/*.sh`, leaving commands with workflow logic + invocation lines only.

**Architecture:** One bash script per command, subcommand-dispatched via `case "$1"`. Scripts are pure executors (no `read`, no prompts) — Claude collects all interactive input. Body templates remain in commands; scripts accept body content as file paths. No shared `lib/` (YAGNI).

**Tech Stack:** Bash 3.2+ (macOS default), `gh` CLI, `jq`, `git`. No new dependencies.

**Reference design:** `docs/plans/2026-04-22-beaver-commands-script-extraction-design.md`

---

## Conventions for every script

- Shebang: `#!/usr/bin/env bash`
- First line after: `set -euo pipefail`
- Subcommand dispatch: `case "${1:-}" in ...; --help|"") usage; exit 0;; *) echo "unknown subcommand: $1" >&2; usage >&2; exit 1;; esac`
- `usage()` prints subcommand list to stdout
- Stdout = data, stderr = human messages
- All `gh api` calls preserve original flags/headers verbatim
- Body content always passed as file path, never inline string
- Path used in commands: `${CLAUDE_PLUGIN_ROOT}/scripts/<name>.sh`

## Verification commands

After every script created/edited:
```bash
bash -n plugins/beaver/scripts/<name>.sh
bash plugins/beaver/scripts/<name>.sh --help
```
Expected: zero output from `-n`, subcommand list from `--help`.

After every command file edited:
```bash
diff <(grep -E '^(##|###|####)' plugins/beaver/commands/<name>.md.bak) \
     <(grep -E '^(##|###|####)' plugins/beaver/commands/<name>.md)
```
Expected: identical headers (workflow narrative preserved).

---

## Task 1: Create scripts/ directory and skeleton

**Files:**
- Create: `plugins/beaver/scripts/.gitkeep`

**Step 1: Create the directory**

```bash
mkdir -p plugins/beaver/scripts
touch plugins/beaver/scripts/.gitkeep
```

**Step 2: Verify**

```bash
ls -la plugins/beaver/scripts/
```
Expected: directory exists with `.gitkeep`.

**Step 3: Commit**

```bash
git add plugins/beaver/scripts/.gitkeep
git commit -m "chore(beaver): scaffold scripts/ directory"
```

---

## Task 2: Extract beaver-claim.sh (smallest — pilot)

**Files:**
- Create: `plugins/beaver/scripts/beaver-claim.sh`
- Modify: `plugins/beaver/commands/beaver-claim.md`

**Step 1: Read the source**

Read `plugins/beaver/commands/beaver-claim.md` and identify the bash blocks:
- `gh api repos/.../issues/{number} --jq '{number, title, state, labels: ..., assignees: ...}'`
- `gh api user --jq '.login'`
- `gh api repos/.../issues/{number}/assignees --method POST -f "assignees[]=${CURRENT_USER}"`
- (status swap is described in workflow text but the actual label DELETE/POST commands are inherited from beaver-engine — leave a `swap-status` subcommand)

**Step 2: Write `beaver-claim.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-claim.sh <subcommand> [args]

Subcommands:
  fetch <org> <repo> <number>           Print issue summary as JSON
  whoami                                Print current gh user
  assign <org> <repo> <number> <user>   Assign user to issue
  swap-status <org> <repo> <number> <from-label> <to-label>
                                        Atomic label swap (DELETE old, POST new)
EOF
}

case "${1:-}" in
  fetch)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, state, labels: [.labels[].name], assignees: [.assignees[].login]}'
    ;;
  whoami)
    gh api user --jq '.login'
    ;;
  assign)
    org=$2; repo=$3; num=$4; user=$5
    gh api "repos/${org}/${repo}/issues/${num}/assignees" --method POST \
      -f "assignees[]=${user}"
    ;;
  swap-status)
    org=$2; repo=$3; num=$4; from=$5; to=$6
    # URL-encode forward slash in label name
    from_enc=${from//\//%2F}
    gh api "repos/${org}/${repo}/issues/${num}/labels/${from_enc}" --method DELETE
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST \
      -f "labels[]=${to}"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Step 3: Make executable and syntax-check**

```bash
chmod +x plugins/beaver/scripts/beaver-claim.sh
bash -n plugins/beaver/scripts/beaver-claim.sh
bash plugins/beaver/scripts/beaver-claim.sh --help
```
Expected: no syntax errors; help text printed.

**Step 4: Update `beaver-claim.md`**

Replace each bash block with a script invocation. Preserve all workflow narrative, validation tables, and constraints. Example replacement:

Old:
````
   ```bash
   gh api repos/{org}/{issueRepo}/issues/{number} --jq '{number, title, state, labels: [.labels[].name], assignees: [.assignees[].login]}'
   ```
````

New:
````
   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-claim.sh fetch {org} {issueRepo} {number}
   ```
````

Repeat for `whoami`/`assign`/`swap-status`.

**Step 5: Verify command file structurally unchanged**

```bash
git diff plugins/beaver/commands/beaver-claim.md
```
Expected: only bash block lines changed; all `##`/`###` headers, validation tables, constraint bullets unchanged.

**Step 6: Commit**

```bash
git add plugins/beaver/scripts/beaver-claim.sh plugins/beaver/commands/beaver-claim.md
git commit -m "refactor(beaver): extract beaver-claim bash into scripts/beaver-claim.sh"
```

---

## Task 3: Extract beaver-focus.sh

**Files:**
- Create: `plugins/beaver/scripts/beaver-focus.sh`
- Modify: `plugins/beaver/commands/beaver-focus.md`

**Step 1: Identify bash blocks in `beaver-focus.md`**

- `gh api user --jq '.login'`
- Big GraphQL `projectV2.items` query filtered to current user
- `gh api search/issues?q=is:pr+is:open+review-requested:$CURRENT_USER ...`

**Step 2: Write `beaver-focus.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-focus.sh <subcommand> [args]

Subcommands:
  whoami                              Print current gh user
  fetch-my-issues <user>              Print my open Beaver issues from project #14 (JSON array)
  fetch-review-prs <user>             Print PRs awaiting my review (JSON array)
EOF
}

case "${1:-}" in
  whoami)
    gh api user --jq '.login'
    ;;
  fetch-my-issues)
    user=$2
    gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) {
          projectV2(number: $number) {
            items(first: 100) {
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    state
                    labels(first: 30) { nodes { name } }
                    assignees(first: 10) { nodes { login } }
                  }
                }
                fieldValueByName(name: "Iteration") {
                  ... on ProjectV2ItemFieldIterationValue {
                    title
                    startDate
                    duration
                  }
                }
              }
            }
          }
        }
      }' -f owner=primatrix -F number=14 \
      --jq --arg user "$user" '.data.organization.projectV2.items.nodes
            | map(select(.content != null and .content.state == "OPEN"))
            | map(select(.content.assignees.nodes | map(.login) | index($user)))
            | map(select(.content.labels.nodes | map(.name) | index("Control-By-Beaver")))
            | map({number: .content.number, title: .content.title,
                   labels: [.content.labels.nodes[].name],
                   iteration: (if .fieldValueByName then
                     {title: .fieldValueByName.title,
                      startDate: .fieldValueByName.startDate,
                      duration: .fieldValueByName.duration} else null end)})'
    ;;
  fetch-review-prs)
    user=$2
    gh api "search/issues?q=is:pr+is:open+review-requested:${user}" \
      --jq '.items[] | {number, title, repository_url, created_at, user: .user.login}'
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Note:** The `--jq` original used shell interpolation `'"$CURRENT_USER"'` which jumbles quote nesting. The script uses `--arg user "$user"` for safe injection.

**Step 3: Verify script**

```bash
chmod +x plugins/beaver/scripts/beaver-focus.sh
bash -n plugins/beaver/scripts/beaver-focus.sh
bash plugins/beaver/scripts/beaver-focus.sh --help
```

**Step 4: Update `beaver-focus.md`**

Replace each bash block:
- Step 1 → `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh whoami`
- Step 3 → `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh fetch-my-issues "$CURRENT_USER"`
- Step 4 → `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-focus.sh fetch-review-prs "$CURRENT_USER"`

Preserve all narrative, dashboard markdown template, constraints.

**Step 5: Commit**

```bash
git add plugins/beaver/scripts/beaver-focus.sh plugins/beaver/commands/beaver-focus.md
git commit -m "refactor(beaver): extract beaver-focus bash into scripts/beaver-focus.sh"
```

---

## Task 4: Extract beaver-pr.sh

**Files:**
- Create: `plugins/beaver/scripts/beaver-pr.sh`
- Modify: `plugins/beaver/commands/beaver-pr.md`

**Step 1: Identify blocks**

- Phase 1 ctx: `git status / git diff --stat HEAD / git branch --show-current / git log --oneline -10`
- Phase 3: `git checkout -b ${BRANCH_NAME} 2>/dev/null || true`
- Phase 3: `git add / git commit -m / git push -u origin ${BRANCH_NAME}`
- Phase 4 G004: `git diff --name-only origin/main...HEAD | grep -E '(test_|_test\.|/tests/)'`
- Phase 4 G006: `gh api repos/.../issues/{n}/labels --jq '.[].name'`
- Phase 5: `gh pr create --draft --title --body "$(cat <<'EOF' ... EOF)"`
- Phase 6 option 2: `gh pr ready {pr_number}`

**Step 2: Write `beaver-pr.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-pr.sh <subcommand> [args]

Subcommands:
  ctx                                       Print git status, diff stat, branch, recent log
  create-branch <branch>                    git checkout -b (idempotent)
  commit-push <branch> <message-file> <file> [<file> ...]
                                            Stage files, commit (message from file), push -u
  check-tests                               List test files changed vs origin/main (exit 0 if any)
  check-labels <org> <repo> <number>        List labels on issue
  add-label <org> <repo> <number> <label>   POST a single label
  create-pr <title> <body-file>             gh pr create --draft, echo PR URL
  mark-ready <pr-number>                    gh pr ready
EOF
}

case "${1:-}" in
  ctx)
    echo "=== git status ==="
    git status
    echo "=== git diff --stat HEAD ==="
    git diff --stat HEAD
    echo "=== current branch ==="
    git branch --show-current
    echo "=== recent commits ==="
    git log --oneline -10
    ;;
  create-branch)
    branch=$2
    git checkout -b "$branch" 2>/dev/null || git checkout "$branch"
    ;;
  commit-push)
    branch=$2; msg_file=$3; shift 3
    git add "$@"
    git commit -F "$msg_file"
    git push -u origin "$branch"
    ;;
  check-tests)
    git diff --name-only origin/main...HEAD | grep -E '(test_|_test\.|/tests/)' || true
    ;;
  check-labels)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/labels" --jq '.[].name'
    ;;
  add-label)
    org=$2; repo=$3; num=$4; label=$5
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST \
      -f "labels[]=${label}"
    ;;
  create-pr)
    title=$2; body_file=$3
    gh pr create --draft --title "$title" --body-file "$body_file"
    ;;
  mark-ready)
    pr=$2
    gh pr ready "$pr"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Note:** `commit-push` takes the commit message as a file (not inline) to avoid escaping issues with multi-line conventional commit bodies. Claude writes the message to a temp file via `Write`.

**Step 3: Verify**

```bash
chmod +x plugins/beaver/scripts/beaver-pr.sh
bash -n plugins/beaver/scripts/beaver-pr.sh
bash plugins/beaver/scripts/beaver-pr.sh --help
```

**Step 4: Update `beaver-pr.md`**

Replace each phase's bash block with the corresponding script call. Keep tables, completion-options text, code-review-reception narrative, constraints.

**Step 5: Commit**

```bash
git add plugins/beaver/scripts/beaver-pr.sh plugins/beaver/commands/beaver-pr.md
git commit -m "refactor(beaver): extract beaver-pr bash into scripts/beaver-pr.sh"
```

---

## Task 5: Extract beaver-dev.sh

**Files:**
- Create: `plugins/beaver/scripts/beaver-dev.sh`
- Modify: `plugins/beaver/commands/beaver-dev.md`

**Step 1: Identify blocks**

- Phase 1: fetch issue, fetch sub-issues
- Phase 3: `git worktree add .claude/worktrees/${BRANCH_NAME} -b ${BRANCH_NAME}`
- Phase 3: status swap (DELETE ready-to-develop, POST in-progress)

**Step 2: Write `beaver-dev.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-dev.sh <subcommand> [args]

Subcommands:
  fetch-issue <org> <repo> <number>     Print issue with body and labels
  fetch-sub-issues <org> <repo> <number>
                                        Print sub-issues array
  add-worktree <branch>                 Create git worktree at .claude/worktrees/<branch>
  swap-to-in-progress <org> <repo> <number>
                                        DELETE status/ready-to-develop, POST status/in-progress
EOF
}

case "${1:-}" in
  fetch-issue)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, body, labels: [.labels[].name]}'
    ;;
  fetch-sub-issues)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/sub_issues" \
      --jq '[.[] | {number, title, body, labels: [.labels[].name]}]'
    ;;
  add-worktree)
    branch=$2
    git worktree add ".claude/worktrees/${branch}" -b "$branch"
    ;;
  swap-to-in-progress)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/labels/status%2Fready-to-develop" --method DELETE
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST \
      -f "labels[]=status/in-progress"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Step 3: Verify and commit**

```bash
chmod +x plugins/beaver/scripts/beaver-dev.sh
bash -n plugins/beaver/scripts/beaver-dev.sh
bash plugins/beaver/scripts/beaver-dev.sh --help
```

Update `beaver-dev.md` accordingly. Preserve TDD narrative, debugging narrative, all phase descriptions.

```bash
git add plugins/beaver/scripts/beaver-dev.sh plugins/beaver/commands/beaver-dev.md
git commit -m "refactor(beaver): extract beaver-dev bash into scripts/beaver-dev.sh"
```

---

## Task 6: Extract beaver-decompose.sh

**Files:**
- Create: `plugins/beaver/scripts/beaver-decompose.sh`
- Modify: `plugins/beaver/commands/beaver-decompose.md`

**Step 1: Identify blocks**

- Phase 1: fetch parent issue, list existing sub-issues
- Phase 5: create child (POST issue with body file), add labels, link to parent (sub_issues), add to project

**Step 2: Write `beaver-decompose.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-decompose.sh <subcommand> [args]

Subcommands:
  fetch-parent <org> <repo> <number>           Print parent issue summary
  list-sub-titles <org> <repo> <number>        Print existing sub-issue titles
  create-child <org> <repo> <title> <body-file>
                                               Create child issue, echo "number=N id=ID"
  add-labels <org> <repo> <number> <label> [<label> ...]
                                               POST labels
  link-parent <org> <repo> <parent-number> <child-id>
                                               Attach child as sub-issue
  add-to-project <project-number> <org> <child-url>
                                               Add issue to project, echo item id
EOF
}

case "${1:-}" in
  fetch-parent)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}" \
      --jq '{number, title, body, labels: [.labels[].name]}'
    ;;
  list-sub-titles)
    org=$2; repo=$3; num=$4
    gh api "repos/${org}/${repo}/issues/${num}/sub_issues" --jq '.[].title'
    ;;
  create-child)
    org=$2; repo=$3; title=$4; body_file=$5
    num=$(gh api "repos/${org}/${repo}/issues" --method POST \
      -f title="$title" \
      -F body=@"$body_file" \
      --jq '.number')
    id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.id')
    echo "number=${num} id=${id}"
    ;;
  add-labels)
    org=$2; repo=$3; num=$4; shift 4
    args=()
    for label in "$@"; do
      args+=(-f "labels[]=${label}")
    done
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  link-parent)
    org=$2; repo=$3; parent=$4; child_id=$5
    gh api "repos/${org}/${repo}/issues/${parent}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  add-to-project)
    project=$2; org=$3; url=$4
    gh project item-add "$project" --owner "$org" --url "$url" \
      --format json --jq '.id'
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Step 3: Verify and commit**

```bash
chmod +x plugins/beaver/scripts/beaver-decompose.sh
bash -n plugins/beaver/scripts/beaver-decompose.sh
bash plugins/beaver/scripts/beaver-decompose.sh --help
```

Update `beaver-decompose.md`: replace bash blocks; preserve QA flow, audit table, constraints. The `cat > /tmp/beaver-sub-issue.md << 'BODY'` pattern stays in narrative as "Claude writes the rendered template via the Write tool to a temp file".

```bash
git add plugins/beaver/scripts/beaver-decompose.sh plugins/beaver/commands/beaver-decompose.md
git commit -m "refactor(beaver): extract beaver-decompose bash into scripts/beaver-decompose.sh"
```

---

## Task 7: Extract beaver-design.sh

**Files:**
- Create: `plugins/beaver/scripts/beaver-design.sh`
- Modify: `plugins/beaver/commands/beaver-design.md`

**Step 1: Identify blocks**

- Phase 4: clone-or-pull wiki repo, branch checkout, commit+push, gh pr create --draft, comment on issue

**Step 2: Write `beaver-design.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-design.sh <subcommand> [args]

Subcommands:
  prepare-wiki [<wiki-dir>]                    Clone or pull wiki repo (default ~/Code/wiki)
  create-branch <wiki-dir> <branch>            git checkout -b in wiki dir
  commit-push <wiki-dir> <file> <message> <branch>
                                               Add+commit+push from wiki dir
  create-pr <repo> <title> <body>              gh pr create --draft against repo, echo URL
  comment-issue <org> <repo> <number> <body>   POST comment to issue
EOF
}

WIKI_REPO=primatrix/wiki
WIKI_DEFAULT=~/Code/wiki

case "${1:-}" in
  prepare-wiki)
    dir=${2:-$WIKI_DEFAULT}
    if [ -d "$dir" ]; then
      cd "$dir" && git checkout main && git pull
    else
      gh repo clone "$WIKI_REPO" "$dir"
    fi
    ;;
  create-branch)
    dir=$2; branch=$3
    cd "$dir" && git checkout -b "$branch"
    ;;
  commit-push)
    dir=$2; file=$3; msg=$4; branch=$5
    cd "$dir"
    git add "$file"
    git commit -m "$msg"
    git push -u origin "$branch"
    ;;
  create-pr)
    repo=$2; title=$3; body=$4
    gh pr create --repo "$repo" --draft --title "$title" --body "$body"
    ;;
  comment-issue)
    org=$2; repo=$3; num=$4; body=$5
    gh api "repos/${org}/${repo}/issues/${num}/comments" --method POST \
      --raw-field body="$body"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Step 3: Verify and commit**

```bash
chmod +x plugins/beaver/scripts/beaver-design.sh
bash -n plugins/beaver/scripts/beaver-design.sh
bash plugins/beaver/scripts/beaver-design.sh --help
```

Update `beaver-design.md`: replace bash blocks; preserve RFC template, QA narrative, constraints.

```bash
git add plugins/beaver/scripts/beaver-design.sh plugins/beaver/commands/beaver-design.md
git commit -m "refactor(beaver): extract beaver-design bash into scripts/beaver-design.sh"
```

---

## Task 8: Extract beaver-create.sh (heavy GraphQL)

**Files:**
- Create: `plugins/beaver/scripts/beaver-create.sh`
- Modify: `plugins/beaver/commands/beaver-create.md`

**Step 1: Identify blocks**

- Issue creation (POST), id refetch (.id, .node_id, .html_url)
- Add labels
- Add to project (item-add)
- Project field-edit (Level/Status/Progress)
- Iteration resolve (GraphQL query for project_id, field_id, iteration_id)
- Iteration set (GraphQL mutation `updateProjectV2ItemFieldValue`)
- Link to parent (sub_issues POST)

**Step 2: Write `beaver-create.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-create.sh <subcommand> [args]

Subcommands:
  create-issue <org> <repo> <title> <body-file>
                                Create issue, echo .number
  fetch-ids <org> <repo> <number>
                                Echo "id=N node_id=NID html_url=URL"
  add-labels <org> <repo> <number> <label> [<label> ...]
                                POST labels
  add-to-project <project-number> <org> <issue-url>
                                Add issue to project, echo item id
  set-field <item-id> <project-id> <field-id> <option-id>
                                Set single-select field
  resolve-iteration <org> <project-number> <yyyymm>
                                Echo "project_id=ID field_id=ID iteration_id=ID"
                                Iteration ID is empty if not found.
  set-iteration <project-id> <item-id> <field-id> <iteration-id>
                                Set Iteration field via GraphQL mutation
  link-parent <org> <repo> <parent-number> <child-id>
                                Attach child as sub-issue
EOF
}

case "${1:-}" in
  create-issue)
    org=$2; repo=$3; title=$4; body_file=$5
    gh api "repos/${org}/${repo}/issues" --method POST \
      -f title="$title" \
      -F body=@"$body_file" \
      --jq '.number'
    ;;
  fetch-ids)
    org=$2; repo=$3; num=$4
    id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.id')
    node_id=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.node_id')
    url=$(gh api "repos/${org}/${repo}/issues/${num}" --jq '.html_url')
    echo "id=${id} node_id=${node_id} html_url=${url}"
    ;;
  add-labels)
    org=$2; repo=$3; num=$4; shift 4
    args=()
    for label in "$@"; do
      args+=(-f "labels[]=${label}")
    done
    gh api "repos/${org}/${repo}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  add-to-project)
    project=$2; org=$3; url=$4
    gh project item-add "$project" --owner "$org" --url "$url" \
      --format json --jq '.id'
    ;;
  set-field)
    item=$2; project=$3; field=$4; option=$5
    gh project item-edit --id "$item" --project-id "$project" \
      --field-id "$field" --single-select-option-id "$option"
    ;;
  resolve-iteration)
    org=$2; project=$3; yyyymm=$4
    info=$(gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) {
          projectV2(number: $number) {
            id
            field(name: "Iteration") {
              ... on ProjectV2IterationField {
                id
                configuration { iterations { id title } }
              }
            }
          }
        }
      }' -f owner="$org" -F number="$project")
    project_id=$(echo "$info" | jq -r '.data.organization.projectV2.id')
    field_id=$(echo "$info" | jq -r '.data.organization.projectV2.field.id')
    iteration_id=$(echo "$info" | jq -r --arg yyyymm "$yyyymm" \
      '.data.organization.projectV2.field.configuration.iterations
       | map(select(.title | startswith($yyyymm))) | .[0].id // ""')
    echo "project_id=${project_id} field_id=${field_id} iteration_id=${iteration_id}"
    ;;
  set-iteration)
    project_id=$2; item_id=$3; field_id=$4; iteration_id=$5
    read -r -d '' MUT <<'GRAPHQL' || true
mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
  updateProjectV2ItemFieldValue(input: {
    projectId: $projectId
    itemId: $itemId
    fieldId: $fieldId
    value: { iterationId: $iterationId }
  }) { projectV2Item { id } }
}
GRAPHQL
    gh api graphql \
      -f query="$MUT" \
      -f projectId="$project_id" \
      -f itemId="$item_id" \
      -f fieldId="$field_id" \
      -f iterationId="$iteration_id"
    ;;
  link-parent)
    org=$2; repo=$3; parent=$4; child_id=$5
    gh api "repos/${org}/${repo}/issues/${parent}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Note:** `read -r -d '' MUT <<'GRAPHQL'` exits non-zero on EOF; the `|| true` keeps `set -e` happy. Same pattern reused in tasks 9 and 10.

**Step 3: Verify and commit**

```bash
chmod +x plugins/beaver/scripts/beaver-create.sh
bash -n plugins/beaver/scripts/beaver-create.sh
bash plugins/beaver/scripts/beaver-create.sh --help
```

Update `beaver-create.md`: replace the long Issue creation block, the GraphQL iteration resolution block, and the iteration mutation block. Preserve QA narrative, Bug Submode, body templates, constraints.

```bash
git add plugins/beaver/scripts/beaver-create.sh plugins/beaver/commands/beaver-create.md
git commit -m "refactor(beaver): extract beaver-create bash into scripts/beaver-create.sh"
```

---

## Task 9: Extract beaver-tracker.sh (longest)

**Files:**
- Create: `plugins/beaver/scripts/beaver-tracker.sh`
- Modify: `plugins/beaver/commands/beaver-tracker.md`

**Step 1: Identify blocks**

- Step 2: idempotent label-create (tracker, tracker/<repo>, tracker/<yyyymm>, tracker/<prev>)
- Step 3 / Step 5: search/issues for prior + current tracker
- Step 4: list open sub-issues of prior tracker
- Step 7: create tracker issue (POST with body file), label it
- Step 8: per-task `gh api repos/.../issues/<n> --jq '.id'`, then sub_issues POST
- Step 8.5: GraphQL fetch triage backlog
- Step 8.6: resolve iteration (same as create), resolve item id (per-issue), addProjectV2ItemById fallback, set iteration mutation
- list sub-issues of tracker (`.[].number`)

**Step 2: Write `beaver-tracker.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-tracker.sh <subcommand> [args]

Subcommands:
  ensure-label <repo> <name> <color> <description>
                                Idempotent label-create (skip on dup)
  find-tracker <repo> <yyyymm>  Search for tracker issue, echo JSON {count, items:[{number,title}]}
  list-carried <prev-number>    Open sub-issues of prior tracker, echo [{number, title}]
  create <repo> <title> <body-file>
                                Create tracker issue, echo .number
  add-labels <repo> <number> <label> [<label> ...]
  resolve-issue-id <repo> <number>
                                Echo numeric DB .id of issue
  attach-sub <tracker-number> <child-id>
                                Attach via sub_issues API
  fetch-triage-backlog          GraphQL fetch primatrix/projects status/triage no-iteration items
  list-tracker-subs <tracker-number>
                                Echo .[].number of sub-issues
  resolve-iteration <yyyymm>    Echo "project_id=ID field_id=ID iteration_id=ID"
  resolve-item-id <repo> <issue-number>
                                Echo ProjectV2Item id for project #14 (empty if not on project)
  add-to-project <project-id> <issue-number>
                                addProjectV2ItemById, echo new item id
  set-iteration <project-id> <item-id> <field-id> <iteration-id>
                                updateProjectV2ItemFieldValue
EOF
}

ORG=primatrix
PROJECT_REPO=projects
PROJECT_NUM=14

case "${1:-}" in
  ensure-label)
    repo=$2; name=$3; color=$4; desc=$5
    gh label create "$name" --repo "${ORG}/${repo}" --color "$color" --description "$desc" 2>/dev/null || true
    ;;
  find-tracker)
    repo=$2; yyyymm=$3
    gh api -X GET search/issues \
      -f q="repo:${ORG}/${PROJECT_REPO} is:issue label:\"tracker/${repo}\" label:\"tracker/${yyyymm}\"" \
      --jq '{count: (.items | length), items: [.items[] | {number, state, title}]}'
    ;;
  list-carried)
    prev=$2
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${prev}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '[.[] | select(.state=="open") | {number, title}]'
    ;;
  create)
    repo=$2; title=$3; body_file=$4
    gh api "repos/${ORG}/${PROJECT_REPO}/issues" --method POST \
      -f title="$title" \
      -F body=@"$body_file" \
      --jq '.number'
    ;;
  add-labels)
    repo=$2; num=$3; shift 3
    args=()
    for label in "$@"; do
      args+=(-f "labels[]=${label}")
    done
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  resolve-issue-id)
    repo=$2; num=$3
    gh api "repos/${ORG}/${repo}/issues/${num}" --jq '.id'
    ;;
  attach-sub)
    tracker=$2; child_id=$3
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  fetch-triage-backlog)
    gh api graphql -f query='
      query {
        organization(login: "primatrix") {
          projectV2(number: 14) {
            items(first: 100) {
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    repository { nameWithOwner }
                    labels(first: 30) { nodes { name } }
                  }
                }
                fieldValueByName(name: "Iteration") {
                  ... on ProjectV2ItemFieldIterationValue { title }
                }
              }
            }
          }
        }
      }' --jq '.data.organization.projectV2.items.nodes
                | map(select(.content != null and .content.repository.nameWithOwner == "primatrix/projects"))
                | map(select(.content.labels.nodes | map(.name) | index("status/triage")))
                | map(select(.fieldValueByName == null))
                | map({number: .content.number, title: .content.title})'
    ;;
  list-tracker-subs)
    tracker=$2
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '.[].number'
    ;;
  resolve-iteration)
    yyyymm=$2
    info=$(gh api graphql -f query='
      query {
        organization(login: "primatrix") {
          projectV2(number: 14) {
            id
            field(name: "Iteration") {
              ... on ProjectV2IterationField {
                id
                configuration { iterations { id title } }
              }
            }
          }
        }
      }')
    project_id=$(echo "$info" | jq -r '.data.organization.projectV2.id')
    field_id=$(echo "$info" | jq -r '.data.organization.projectV2.field.id')
    iteration_id=$(echo "$info" | jq -r --arg yyyymm "$yyyymm" \
      '.data.organization.projectV2.field.configuration.iterations
       | map(select(.title | startswith($yyyymm))) | .[0].id // ""')
    echo "project_id=${project_id} field_id=${field_id} iteration_id=${iteration_id}"
    ;;
  resolve-item-id)
    repo=$2; num=$3
    read -r -d '' Q <<'GRAPHQL' || true
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      projectItems(first: 10) { nodes { id project { number } } }
    }
  }
}
GRAPHQL
    gh api graphql \
      -f query="$Q" \
      -f owner="$ORG" \
      -f repo="$repo" \
      -F number="$num" \
      --jq '.data.repository.issue.projectItems.nodes
            | map(select(.project.number == 14)) | .[0].id // ""'
    ;;
  add-to-project)
    project_id=$2; num=$3
    content_id=$(gh api "repos/${ORG}/${PROJECT_REPO}/issues/${num}" --jq '.node_id')
    read -r -d '' M <<'GRAPHQL' || true
mutation($projectId: ID!, $contentId: ID!) {
  addProjectV2ItemById(input: { projectId: $projectId, contentId: $contentId }) {
    item { id }
  }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f projectId="$project_id" \
      -f contentId="$content_id" \
      --jq '.data.addProjectV2ItemById.item.id'
    ;;
  set-iteration)
    project_id=$2; item_id=$3; field_id=$4; iteration_id=$5
    read -r -d '' M <<'GRAPHQL' || true
mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $iterationId: String!) {
  updateProjectV2ItemFieldValue(input: {
    projectId: $projectId
    itemId: $itemId
    fieldId: $fieldId
    value: { iterationId: $iterationId }
  }) { projectV2Item { id } }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f projectId="$project_id" \
      -f itemId="$item_id" \
      -f fieldId="$field_id" \
      -f iterationId="$iteration_id"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Step 3: Verify and commit**

```bash
chmod +x plugins/beaver/scripts/beaver-tracker.sh
bash -n plugins/beaver/scripts/beaver-tracker.sh
bash plugins/beaver/scripts/beaver-tracker.sh --help
```

Update `beaver-tracker.md`. The body template (`## 月度 Tracker — ...`) stays in the command file as a fenced markdown block — Claude renders placeholders, writes via `Write` to a temp path, passes that path to `bash scripts/beaver-tracker.sh create`. Preserve all step narrative, HARD-GATE preview, error handling rules.

```bash
git add plugins/beaver/scripts/beaver-tracker.sh plugins/beaver/commands/beaver-tracker.md
git commit -m "refactor(beaver): extract beaver-tracker bash into scripts/beaver-tracker.sh"
```

---

## Task 10: Extract beaver-setup.sh (heaviest GraphQL)

**Files:**
- Create: `plugins/beaver/scripts/beaver-setup.sh`
- Modify: `plugins/beaver/commands/beaver-setup.md`

**Step 1: Identify blocks**

- Context: `gh auth status`, `date +%Y-%m-%d`
- Field-create (Level/Progress)
- Status field GraphQL replace (full options replacement)
- Field-create fallback for Status
- README write (`gh project edit --readme "$(cat /tmp/...)"`)
- List + create issue types
- Label create (~25 labels with idempotent skip)
- Iteration field create (GraphQL with iterationConfiguration)
- Iteration field append-missing via `updateProjectV2IterationField`

**Step 2: Write `beaver-setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-setup.sh <subcommand> [args]

Subcommands:
  auth-status                              gh auth status
  today                                    Print YYYY-MM-DD
  project-id <org> <project-number>        Echo node id of projectV2
  field-list <org> <project-number>        List custom fields (JSON)
  field-create <org> <project-number> <name> <type> [<options-csv>]
                                           Create single-select or number field
  status-replace <field-id>                GraphQL: replace 7 status options
  edit-readme <org> <project-number> <readme-file>
                                           gh project edit --readme
  list-issue-types <org>                   List org issue types
  create-issue-type <org> <name> <color> <desc>
                                           POST issue type, skip on 422
  ensure-label <repo> <name> <color> <desc>
                                           gh label create, skip on dup
  iteration-field-create <project-id> <start-date> <duration> <iterations-json>
                                           Create Iteration field with initial entries
  iteration-field-list <org> <project-number>
                                           Read existing Iteration entries
  iteration-field-append <field-id> <additions-json>
                                           Append missing iteration entries
EOF
}

case "${1:-}" in
  auth-status)
    gh auth status
    ;;
  today)
    date +%Y-%m-%d
    ;;
  project-id)
    org=$2; num=$3
    gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) { projectV2(number: $number) { id } }
      }' -f owner="$org" -F number="$num" \
      --jq '.data.organization.projectV2.id'
    ;;
  field-list)
    org=$2; num=$3
    gh project field-list "$num" --owner "$org" --format json
    ;;
  field-create)
    org=$2; num=$3; name=$4; type=$5
    if [ "$type" = "SINGLE_SELECT" ]; then
      opts=$6
      gh project field-create "$num" --owner "$org" --name "$name" \
        --data-type SINGLE_SELECT --single-select-options "$opts"
    else
      gh project field-create "$num" --owner "$org" --name "$name" --data-type "$type"
    fi
    ;;
  status-replace)
    field_id=$2
    gh api graphql -f query='
      mutation($fieldId: ID!) {
        updateProjectV2Field(input: {
          fieldId: $fieldId
          singleSelectOptions: [
            {name: "Triage", color: GRAY, description: "Awaiting triage"},
            {name: "Ready to Claim", color: BLUE, description: "Added to Iteration, awaiting claim"},
            {name: "Design Pending", color: PURPLE, description: "Design review in progress (size/L)"},
            {name: "Ready to Develop", color: ORANGE, description: "Ready to code (size/L, design approved)"},
            {name: "In Progress", color: YELLOW, description: "Active development"},
            {name: "Blocked", color: RED, description: "Blocked"},
            {name: "Done", color: GREEN, description: "Completed and merged"}
          ]
        }) {
          projectV2Field {
            ... on ProjectV2SingleSelectField { name options { name } }
          }
        }
      }' -f fieldId="$field_id"
    ;;
  edit-readme)
    org=$2; num=$3; file=$4
    gh project edit "$num" --owner "$org" --readme "$(cat "$file")"
    ;;
  list-issue-types)
    org=$2
    gh api "orgs/${org}/issue-types" -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '.[].name'
    ;;
  create-issue-type)
    org=$2; name=$3; color=$4; desc=$5
    gh api "orgs/${org}/issue-types" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -f name="$name" -f color="$color" -f description="$desc" \
      -F is_enabled=true 2>/dev/null || true
    ;;
  ensure-label)
    repo=$2; name=$3; color=$4; desc=$5
    gh label create "$name" --repo "$repo" --color "$color" --description "$desc" 2>/dev/null || true
    ;;
  iteration-field-create)
    project_id=$2; start=$3; duration=$4; iterations=$5
    read -r -d '' M <<'GRAPHQL' || true
mutation($projectId: ID!, $startDate: Date!, $duration: Int!, $iterations: [ProjectV2IterationFieldIterationInput!]!) {
  createProjectV2Field(input: {
    projectId: $projectId
    dataType: ITERATION
    name: "Iteration"
    iterationConfiguration: {
      startDate: $startDate
      duration: $duration
      iterations: $iterations
    }
  }) { projectV2Field { ... on ProjectV2IterationField { id name } } }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f projectId="$project_id" \
      -f startDate="$start" \
      -F duration="$duration" \
      -f iterations="$iterations"
    ;;
  iteration-field-list)
    org=$2; num=$3
    gh api graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) { projectV2(number: $number) {
          field(name: "Iteration") {
            ... on ProjectV2IterationField {
              id
              configuration { iterations { title startDate duration } }
            }
          }
        } }
      }' -f owner="$org" -F number="$num"
    ;;
  iteration-field-append)
    field_id=$2; additions=$3
    read -r -d '' M <<'GRAPHQL' || true
mutation($fieldId: ID!, $additions: [ProjectV2IterationFieldIterationInput!]!) {
  updateProjectV2IterationField(input: {
    iterationFieldId: $fieldId
    additions: $additions
  }) { iterationField { id } }
}
GRAPHQL
    gh api graphql \
      -f query="$M" \
      -f fieldId="$field_id" \
      -f additions="$additions"
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
```

**Step 3: Verify and commit**

```bash
chmod +x plugins/beaver/scripts/beaver-setup.sh
bash -n plugins/beaver/scripts/beaver-setup.sh
bash plugins/beaver/scripts/beaver-setup.sh --help
```

Update `beaver-setup.md`: replace bash blocks. The big constants table, README/yaml block, label/issue-type tables, and all narrative stay. The label-creation loop becomes "for each row in the label tables, run `bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-setup.sh ensure-label projects <name> <color> <desc>`".

```bash
git add plugins/beaver/scripts/beaver-setup.sh plugins/beaver/commands/beaver-setup.md
git commit -m "refactor(beaver): extract beaver-setup bash into scripts/beaver-setup.sh"
```

---

## Task 11: Final verification

**Step 1: Syntax-check all scripts**

```bash
for f in plugins/beaver/scripts/*.sh; do
  echo "=== $f ==="
  bash -n "$f"
  bash "$f" --help >/dev/null
done
echo "All scripts OK"
```
Expected: "All scripts OK", no errors, no missing subcommand output.

**Step 2: JSON validity**

```bash
python3 -c 'import json; json.load(open("plugins/beaver/.claude-plugin/plugin.json"))' && echo "plugin.json OK"
python3 -c 'import json; json.load(open(".claude-plugin/marketplace.json"))' && echo "marketplace.json OK"
```
Expected: both "OK".

**Step 3: Confirm no orphan bash blocks remain in commands**

```bash
# Look for gh api or graphql calls still embedded in command markdown
grep -nE '(gh api |gh project |gh label create|graphql)' plugins/beaver/commands/*.md | grep -v 'scripts/' | grep -v '^[^:]*:[0-9]*:[[:space:]]*-' || echo "No raw gh calls found in commands"
```
Expected: "No raw gh calls found in commands" (the grep filters out script invocation lines and bullet text describing them).

If matches surface: review case-by-case — some narrative descriptions of `gh api` are intentional (e.g. "the underlying call is `gh api`...").

**Step 4: Confirm script invocation lines exist**

```bash
grep -c 'CLAUDE_PLUGIN_ROOT/scripts/' plugins/beaver/commands/*.md
```
Expected: every command file shows ≥1 invocation.

**Step 5: Final commit if any cleanup needed**

```bash
git status
git add -A
git commit -m "chore(beaver): finalize bash extraction" || echo "nothing to commit"
```

---

## Notes for Claude executing this plan

- Each task is one command + its script. Do them in order (claim → focus → pr → dev → decompose → design → create → tracker → setup) so the easier extractions build muscle memory before the GraphQL-heavy ones.
- **Read the command file first** before writing the script — the bash block inventory in this plan was derived from a single read; subtle flags or headers may have been missed.
- **Preserve flags verbatim**: `-f` vs `-F`, `--method POST`, all `-H "X-GitHub-Api-Version: 2026-03-10"` headers, all `--jq` filters. A missing `-F` (vs `-f`) causes integer→string GraphQL errors.
- **Templates stay in commands**: any `cat > /tmp/... << 'BODY'` block is replaced by narrative like "Render the body template to a temp file via the Write tool, then run `bash scripts/<name>.sh create-issue ... <body-file>`".
- **`allowed-tools` frontmatter**: leave as-is. The bash scripts still call `gh api` etc., so the existing permissions remain valid.
- **No behavior changes**. If you spot a bug while extracting, note it — do not fix it in the same commit.
