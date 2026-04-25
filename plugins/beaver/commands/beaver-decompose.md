---
allowed-tools: Bash(gh api:*), Bash(gh project:*), Bash(gh issue:*), Bash(mktemp:*), Bash(bash:*), Bash(jq:*)
description: Decompose a Beaver Task into SubTasks (Size=S child each), guided by a design doc. Trigger when the user wants to split, breakdown, or decompose an issue into sub-issues.
argument-hint: "<issue-number> --design-doc <url-or-path>"
---

# /beaver-decompose — 任务拆解

Phase 5 of the Beaver development lifecycle (Task → SubTask only). All lifecycle metadata for the children is written to **Project V2 #14** custom fields and the **native GitHub Issue Type** through `beaver-lib.sh` — never via repository labels. Audit failures (Coverage / Atomicity / Tests) are surfaced as `<!-- audit-warnings -->` blocks in each child's Issue body, not as `beaver/*` labels. See `plugins/beaver/skills/beaver-engine/SKILL.md` §1 (Field Taxonomy) and §4 (Field Operations).

## Workflow

Arguments required: parent issue number AND design doc reference (`--design-doc <url-or-path>`).

### Phase 1: Load & Validate (field-semantics pre-check)

1. Parse arguments: extract `<issue-number>` and the `--design-doc <url-or-path>` value.

   The `--design-doc` value (PR URL / blob URL / local path) is captured **verbatim**. No normalization is performed; it is stored as-is for use in step 6a (each child body's top-of-document Design Doc reference).

1. Fetch parent Issue:

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh fetch-parent {org} {issueRepo} {number}
   ```

1. Pre-check via `beaver-lib.sh` (no `status/*` label reads):

   ```bash
   source ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-lib.sh

   PARENT_TYPE=$(get_type "{number}")                                    # Bug / Task / SubTask / ""
   PARENT_STATUS=$(_get_single_select_value "{number}" "Status")         # Project V2 Status
   PARENT_ITERATION=$(get_iteration "{number}")                          # Iteration title or ""
   ```

   Equivalent one-shot via the script (which itself calls `beaver-lib.sh`):

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh parent-fields {org} {issueRepo} {number}
   # echoes JSON: {"issueType": "...", "status": "...", "iteration": "...", "assignees": [...]}
   ```

   Pick ONE of the two paths (inline `source` OR `parent-fields`) — do not call both.

   The parent MUST satisfy:

   - `Type = Task` (read via `beaver-lib.sh::get_type`). Goal is no longer a native Issue Type per RFC-0013; only Task → SubTask decomposition is supported.
   - `Status = Ready to Develop` (read via `_get_single_select_value`). This confirms the design doc PR has merged.

   If either check fails, abort with a clear message naming the actual values and pointing the user at `/beaver-design` (Status not yet `Ready to Develop`) or `/beaver-create --type task` (Type not `Task`).

1. Fetch existing sub-issues (avoid duplication) and parent assignees (default child assignees):

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh list-sub-titles {org} {issueRepo} {number}

   # parent-fields above already returned `assignees` — capture as PARENT_ASSIGNEES.
   ```

### Phase 2: Read Design Doc

Support three reference formats:

- **GitHub PR URL** (`https://github.com/.../pull/N`): fetch files from PR.
- **GitHub blob URL** (`https://github.com/.../blob/...`): fetch file content via API.
- **Local path**: read directly.

Extract the design doc content for decomposition analysis. The verbatim `--design-doc` value (not the fetched content) is what gets embedded in each child body in step 6a.

### Phase 3: Draft Decomposition + Initial Dependency Suggestions

1. Analyze the design doc to identify logical work units. Each child gets `Type=SubTask`, `Size=S`, and should be independently deliverable.

1. For each child, draft:

   - Title (imperative, specific)
   - Objective (one sentence)
   - Acceptance criteria (≥ 2 verifiable items)
   - Estimated scope

1. Skip any scope already covered by an existing sub-issue title (from step 4 of Phase 1).

1. Generate **initial dependency suggestions** between children using **relative refs** of the form `child#N` (where `N` is the 1-based index in the proposed list). For example: `child#3 blocked by child#1, child#2`. These are SUGGESTIONS — they will be confirmed (or overridden) per child during Phase 4 QA.

### Phase 4: Per-Child QA Confirm

Engine §7 applies. For each proposed child, ask exactly two QA rounds:

1. **Content round** — present the child's title / objective / acceptance criteria / suggested assignee set (defaults to PARENT_ASSIGNEES). User can: accept / edit / delete / insert / **override assignees**. Assignee override accepts ANY GitHub login set (not constrained to the parent assignee set; can also be empty to leave the child unassigned).

   - If `PARENT_ASSIGNEES` is empty, the child default is also empty. **Do not fall back to the current `gh` user** — an empty parent assignee set is preserved as an empty child assignee set.
   - The user override at this round can add or remove logins freely.

1. **Blocker round** — explicitly ask: "Is this child blocked by any of the other children we are about to create?" The user may specify zero or more blockers, each as a relative ref `child#N` referring to **another child in this current batch** (not external Issue numbers; cross-batch dependencies must be added manually in the GitHub UI after landing). The pre-suggested blockers from Phase 3 step 4 are pre-filled and the user can accept / edit / clear them.

One child per turn, wait for §7.5 approval before advancing.

After all children are confirmed, the command holds in memory:

- `CHILDREN[i]`: title, objective, acceptance, assignees (resolved login set)
- `BLOCKERS[i]`: list of relative refs (`child#N`) that block child `i`

Then present the **assembled dependency graph** in one shot for explicit user approval (AC9 "用户确认后"):

```text
Dependency graph:
- child#1 (no blockers)
- child#2 blocked by child#1
- child#3 blocked by child#1, child#2
Approved? (y/revise)
```

Only on §7.5 approval does the workflow advance to Phase 4.5. A `revise` answer drops the user back into Phase 4 to edit per-child blockers.

#### Phase 4.5: Cycle detection (DFS)

Before any landing step (Phase 6), run depth-first cycle detection over the `BLOCKERS` map (each `child#A → [child#B, child#C, ...]` edge means "A is blocked by B/C"). Algorithm:

```text
state = {n: WHITE for n in CHILDREN}        # WHITE | GRAY | BLACK
cycle_path = []

def dfs(node, stack):
  state[node] = GRAY
  stack.append(node)
  for blocker in BLOCKERS[node]:
    if state[blocker] == GRAY:
      # cycle found — extract from stack[stack.index(blocker):] + [blocker]
      cycle_path = stack[stack.index(blocker):] + [blocker]
      raise CycleError
    if state[blocker] == WHITE:
      dfs(blocker, stack)
  state[node] = BLACK
  stack.pop()

for n in CHILDREN:
  if state[n] == WHITE:
    dfs(n, [])
```

On any cycle:

- Surface the cycle as an arrow path (e.g. `child#1 → child#3 → child#2 → child#1`).
- Refuse to enter Phase 6.
- Drop the user back into Phase 4 (re-doing the per-child blocker round, then the graph approval gate above).

This guard runs locally on the in-memory map; it precedes any `gh` POST.

### Phase 5: Auto-Audit

For each confirmed child (independent of dependency landing), classify against three categories:

- **Coverage** — does this child clearly contribute to the parent's objective + acceptance criteria? If not, mark `missing-context`.
- **Atomicity** — is the scope plausibly independently deliverable? If too large, mark `needs-split`.
- **Test Definition** — does the body contain testable acceptance criteria? If not, mark `missing-test`.

A child may receive zero, one, or multiple categories. Categories accumulate and feed into the body comment in step 6a.

This command does NOT call `gh issue edit --add-label` or `gh api .../labels` for any of the three audit categories — they live exclusively in the body comment.

### Phase 6: 落库 (Issue creation + dependency landing)

Per RFC-0013 §5 step 6, perform exactly this ordered sequence per child. Use `mktemp` to allocate a **unique** body file per child so that the loop over N children does not clobber a single shared path. The script's `create-child` subcommand internally also routes its body through a `mktemp` file as a defense-in-depth measure.

1. **6a — Render and create the child Issue.**

   Build the child body string in this order (top-down):

   ```markdown
   > Design Doc: <verbatim --design-doc value>

   ## 目标
   {objective}

   ## 验收标准
   {acceptance_criteria}

   <!-- audit-warnings
   missing-test
   needs-split
   missing-context
   -->

   <!-- beaver-tracking
   issue-type: SubTask
   size: S
   parent: #{parent_number}
   created-by: beaver-decompose
   -->
   ```

   - The `> Design Doc: <url>` blockquote line is ALWAYS present at the very top, followed by one blank line. The `<url>` is the verbatim `--design-doc` argument from Phase 1 step 1 — **no normalization** (no URL canonicalization, no path resolution).
   - The `<!-- audit-warnings -->` block lists ONLY the categories that failed in Phase 5. If a child passed all three audits, the entire `<!-- audit-warnings -->` block is OMITTED from the body. The block sits between the user-facing description and the `beaver-tracking` block.
   - Render the body to a unique tempfile via `mktemp` and pass it through the `create-child` script:

     ```bash
     BODY=$(printf '%s\n' "$rendered_body")
     CHILD_OUT=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh \
       create-child {org} {issueRepo} "$child_title" "$BODY")
     # CHILD_OUT="number=N id=ID"
     eval "$CHILD_OUT"
     CHILD_NUM=$number; CHILD_ID=$id
     ```

   The script internally allocates a per-call mktemp file as `/tmp/beaver-decompose-child-$$-$RANDOM-<epoch_ns>.XXXXXX`, satisfying the AC5 uniqueness requirement even when a single shell invocation creates N children back-to-back.

1. **6b — Link to parent (Sub-Issues API).**

   Run BEFORE add-to-project so Projects V2 emits the parent→child rollup at the moment the project item is created (otherwise the parent's project card shows "1 sub-issue not in this project").

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh link-parent \
     {org} {issueRepo} {parent_number} "$CHILD_ID"
   ```

1. **6c — Add child to Project V2 #14.**

   ```bash
   CHILD_URL="https://github.com/{org}/{issueRepo}/issues/${CHILD_NUM}"
   ITEM_ID=$(bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh \
     add-to-project {project_number} {org} "$CHILD_URL")
   ```

1. **6d — Write Project V2 fields via `beaver-lib.sh`.**

   ```bash
   source ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-lib.sh

   set_type   "$CHILD_NUM" "SubTask"   # native Issue Type
   set_size   "$CHILD_NUM" "S"          # Project V2 Size
   set_status "$CHILD_NUM" "Triage"     # Project V2 Status

   # Iteration is INHERITED from the parent (may be empty — set_iteration
   # is skipped in that case so the child stays Iteration-unassigned).
   if [ -n "$PARENT_ITERATION" ]; then
     set_iteration "$CHILD_NUM" "$PARENT_ITERATION"
   fi
   ```

   Then write assignees (the resolved per-child set from Phase 4 round 1, which defaults to `PARENT_ASSIGNEES` and may be overridden):

   ```bash
   # Pass the resolved login list as positional args. Empty list clears.
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh set-assignees \
     {org} {issueRepo} "$CHILD_NUM" "${CHILD_ASSIGNEES[@]}"
   ```

1. **6e — Loop.** Repeat 6a–6d for each remaining child, recording the real Issue number assigned by GitHub into `CHILD_NUMS[i]` (where `i` matches the Phase 4 index used by the `child#i` relative refs):

   ```bash
   CHILD_NUMS[$i]="$CHILD_NUM"
   ```

   After every child has been created, build the resolved blocker map by translating each `child#N` ref to the actual Issue number:

   ```bash
   for i in "${!CHILDREN[@]}"; do
     resolved=()
     for ref in "${BLOCKERS[$i]}"; do        # e.g. "child#2"
       j="${ref#child#}"                      # 1-based index from Phase 4
       resolved+=("${CHILD_NUMS[$j]}")
     done
     RESOLVED_BLOCKERS[$i]="${resolved[*]}"
   done
   ```

1. **6f — Land dependencies.**

   GitHub Issue Dependencies are a **separate** relationship type from Sub-Issues — landing them does NOT replace, undo, or affect the sub-issue link from step 6b.

   For each `(child #A, blocker #B)` pair in the `RESOLVED_BLOCKERS` map (built in 6e), call:

   ```bash
   # Resolve the blocker's numeric repo-issue id (NOT its issue number).
   BLOCKER_ID=$(gh api "repos/{org}/{issueRepo}/issues/${blocker_num}" --jq '.id')

   if ! bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-decompose.sh add-blocked-by \
        {org} {issueRepo} "$child_num" "$BLOCKER_ID" 2>err; then
     DEPENDENCY_FAILURES+=("${child_num}|${blocker_num}|$(cat err)")
   fi
   ```

   The script wraps the REST endpoint `POST /repos/{owner}/{repo}/issues/{issue_number}/dependencies/blocked_by` (`X-GitHub-Api-Version: 2026-03-10`). The GraphQL `addIssueDependency` mutation provides the same semantics if the REST endpoint is later deprecated; switching the script to GraphQL would be a one-place change in `add-blocked-by`.

   - A failure on one dependency MUST NOT abort subsequent dependency writes. Capture each `(child, blocker, error_message)` triple into a `DEPENDENCY_FAILURES` list for inclusion in the Phase 7 parent comment.

### Phase 7: Audit Summary Comment on Parent

Post one audit summary comment on the parent Issue summarizing the decomposition. Structure:

```markdown
## Decomposition summary (beaver-decompose)

| Child | Coverage | Atomicity | Tests | Status |
|---|---|---|---|---|
| #N title | ✅/⚠️ | ✅/⚠️ | ✅/⚠️ | PASS / WARN |

### Dependency graph

- #A blocked by #B, #C
- #D blocked by #B
- #E (no blockers)

### 依赖写入失败，需手动补登

- #A blocked by #B → <error message>
```

The `依赖写入失败` section appears ONLY when `DEPENDENCY_FAILURES` is non-empty; it lists each failed pair so the user can manually add the dependency in the GitHub UI. The dependency graph section uses real Issue numbers (no longer relative refs) since landing is complete by this point.

### Phase 8: Report

Print to the user:

- Number of children created.
- Audit results table (same as Phase 7's first table).
- Dependency landing results: `M succeeded, K failed`.
- Next-step hint:
  - "Children are at `Status = Triage` and inherit the parent's Iteration. `/beaver-claim` 已删除（见 RFC-0013 §3）：team members 请在 GitHub UI assign 自己后手动将 Status 切到对应值。"
  - If `K > 0`: "Add the K failed dependencies manually in the GitHub UI (issue → ⋯ → Dependencies → Add 'blocked by')."

## Constraints

- Engine §7 QA applies to Phase 4 (per-child confirmation, including the blocker round).
- Engine §7.2 HARD-GATE applies to Phase 6 (issue creation + landing).
- All lifecycle-metadata writes (Type, Size, Status, Iteration) flow through `beaver-lib.sh`. This command does not write `status/* / type/* / size/*` repository labels.
- Audit failures are body comments (`<!-- audit-warnings -->`), never `beaver/missing-test`/`beaver/needs-split`/`beaver/missing-context` labels.
- Each child body file is allocated via `mktemp` (per-call unique path) so a single decompose invocation that creates N children does not clobber a shared path.
- Sub-Issue links (step 6b) and Issue Dependencies (step 6f) are independent relationship types; both are written and neither replaces the other.
