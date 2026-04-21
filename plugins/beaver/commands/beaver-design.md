---
allowed-tools: Bash(gh api:*), Bash(gh repo clone:*), Bash(gh pr create:*), Bash(git:*), Bash(cat > /tmp/*)
description: Write and submit a design document for a Beaver size/L issue in status/design-pending. Trigger when the user wants to write a design doc, start design review, or work on a design-pending issue.
argument-hint: "<issue-number>"
---

# /beaver-design — 设计评审

Phase 4 of the Beaver development lifecycle (size/L only).

## Workflow

Argument is required: the issue number.

### Phase 1: Load & Validate

1. Fetch Issue and verify:
   - Has `size/L` label
   - Has `status/design-pending` label
   - Fail with clear message if either check fails

1. Extract objective and acceptance criteria from Issue body as design starting point.

### Phase 2: Context Collection

1. **Discovery Triad** (engine §8): D1 recent activity, D2 keyword search, D3 project conventions. Print Discovery Brief.

1. **Iterative QA** (engine §7) across 4 sections, each requiring §7.5 approval:

   **Section 1: Context & Scope**

   - Technical environment, system boundaries, objective factual background
   - Ask one question at a time, prefer multiple choice

   **Section 2: Design Goals**

   - Goals (what to achieve)
   - Non-Goals (explicitly out of scope)
   - Success Metrics (quantifiable, verifiable)

   **Section 3: The Design**

   - System context diagram
   - Core architecture
   - Interfaces & data flow
   - Key trade-offs and their rationale
   - Test strategy
   - Deployment dependencies

   **Section 4: Alternatives Considered**

   - Other viable approaches
   - Why each was rejected

1. Each section uses §9.3 completeness checklist before approval.

### Phase 3: Spec Review Loop

1. After all 4 sections approved, dispatch spec-document-reviewer subagent with:
   - The complete design doc content
   - The original Issue objective and acceptance criteria
   - Instructions to check completeness, consistency, and feasibility

1. If reviewer finds issues: fix and re-dispatch (max 5 iterations).

1. If reviewer approves: proceed to Phase 4.

### Phase 4: Submit to Wiki

1. **Prepare wiki repo**:

   ```bash
   # Clone or pull wiki repo
   if [ -d ~/Code/wiki ]; then
     cd ~/Code/wiki && git checkout main && git pull
   else
     gh repo clone primatrix/wiki ~/Code/wiki
     cd ~/Code/wiki
   fi
   ```

1. **Determine RFC number**: Read `docs/rfc/index.md`, find next available NNNN.

1. **Create branch**:

   ```bash
   git checkout -b design/{issue_number}-{slug}
   ```

1. **Write design doc** to `docs/rfc/NNNN-{slug}.md` following wiki RFC template:

   ```markdown
   ---
   title: "RFC-NNNN: {title}"
   status: draft
   author: {gh_username}
   date: {YYYY-MM-DD}
   reviewers: []
   ---

   # RFC-NNNN: {title}

   ## 概述
   {one-line summary}

   ## 背景
   {context and scope from Section 1}

   ## 方案
   {design from Section 3}

   ### 备选方案
   {alternatives from Section 4}

   ## 影响范围
   {derived from design goals and interfaces}

   ## 实施计划
   {high-level phases derived from architecture}

   ## 风险
   {derived from trade-offs}

   <!-- provenance
   {fact-to-source mapping per engine §9.2}
   -->
   ```

1. **Commit and push**:

   ```bash
   git add docs/rfc/NNNN-{slug}.md
   git commit -m "docs(rfc): add RFC-NNNN {title}"
   git push -u origin design/{issue_number}-{slug}
   ```

1. **Create Draft PR**:

   ```bash
   gh pr create --repo primatrix/wiki --draft \
     --title "RFC-NNNN: {title}" \
     --body "Design doc for {org}/{issueRepo}#{issue_number}"
   ```

1. **Comment on original Issue**:

   ```bash
   gh api repos/{org}/{issueRepo}/issues/{issue_number}/comments --method POST \
     --raw-field body="Design Doc PR: {pr_url}"
   ```

1. **Report**: Print PR URL and next-step hint:
   > "Design Doc submitted as Draft PR at {pr_url}. Self-review the Draft, then mark it Open for team review. When the PR is merged, the Issue will transition to `status/ready-to-develop`. Then use `/beaver-decompose {issue_number} --design-doc {pr_url}` to split into SubTasks."

### Status Transition

- Status stays at `status/design-pending` during this command
- Transition to `status/ready-to-develop` happens when the Design Doc PR is merged (manual or automated)

## Constraints

- Engine §7.2 HARD-GATE applies throughout Phase 2-3
- Engine §9.1-9.3 apply to all design doc content
- Engine §9.2 anti-hallucination: every fact must be traceable
- Provenance block required at end of design doc
