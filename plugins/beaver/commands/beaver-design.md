---
allowed-tools: Bash(gh api:*), Bash(gh repo clone:*), Bash(gh pr create:*), Bash(git:*), Bash(bash:*)
description: Write and submit a design document for a Beaver size/L issue in status/design-pending. Trigger when the user wants to write a design doc, start design review, or work on a design-pending issue.
argument-hint: "<issue-number>"
---

# /beaver-design — 设计评审

Phase 4 of the Beaver development lifecycle. Per RFC-0013 §4 #4, this command targets size/L Tasks only and **never modifies any Project V2 field**. Status stays at `Design Pending` throughout the run; the system migration to `Ready to Develop` happens after the Design Doc PR is merged (out-of-scope for this command).

> 所有交互式 QA 与终端输出统一使用中文（RFC-0013 §命令规约「语言约定」）。

## Workflow

Argument is required: the issue number (Project V2 #14 上的 Task Issue 编号)。

### Phase 1: 前置校验（HARD-GATE，任一失败即中止）

读取 Project V2 #14 上该 Issue 的字段并断言：

- `Type=Task`
- `Size=L`
- `Status=Design Pending`
- 当前 `gh` 用户 (`gh api user --jq .login`) ∈ Issue 的 assignees 集合

任一不满足即立即中止，并打印失败原因。命令本阶段**仅读不写**。

### Phase 2: wiki 工作树准备

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-design.sh prepare-wiki /tmp/wiki
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-design.sh create-branch /tmp/wiki design/{issue_number}-{slug}
```

`<slug>` 由 Issue 标题派生为 kebab-case。

### Phase 3: 设计资料采集

读取 Issue body（objective + 验收标准 + 已知约束）作为意图主源，并在当前 worktree 内根据关键词主动检索相关代码（模块、接口、现有实现），将合并后的内容作为后续 QA 的「已有上下文」展示给用户。

### Phase 4: 五维度结构化 QA

按下列顺序逐维度进行；**禁止跨维度跳问**——任一维度未结束之前不得开启下一维度的问题。每维度在该维度内部可有多轮 Q&A，命令负责判断本维度信息是否足够完整后才进入下一维度。

#### 4.1 Context & Scope

- 技术现状、系统边界、客观背景事实
- 与现有模块的关系
- 关键约束与依赖

#### 4.2 Design Goals

- 可量化的目标 (Goals)
- 明确不做的非目标 (Non-Goals)
- 成功指标 (Success Metrics, 可验证)

#### 4.3 The Design

- 系统上下文图 / 核心架构
- 接口与数据流
- **端到端数据流图（HARD-GATE）**
- 关键 trade-off 与理由
- 测试策略
- 部署 / 依赖

##### 4.3.1 端到端数据流图

在讨论「接口与数据流」之后、进入 trade-off 之前，命令**必须**执行以下步骤：

1. **深度代码阅读**：基于 Phase 3 已采集的关键词与模块信息，在当前 worktree 内递归检索所有涉及的源文件——包括但不限于：入口函数、调用链上下游、数据模型/DTO 定义、中间件/拦截器、持久层/数据库操作、外部 API 调用、消息队列生产者/消费者、配置加载路径。对每个关键模块至少读取其公开接口签名与核心处理逻辑。

2. **绘制完整数据流图**：以 Mermaid `flowchart` 语法生成一张端到端数据流图，要求：
   - **不仅限于本次改动范围**——必须包含数据从源头到终点经过的**所有已有组件**（即使本次不修改），以展示改动在整个系统中的位置与影响。
   - 用不同样式区分：`本次新增/修改的节点`（加粗边框或高亮色）vs `现有不变的节点`（普通样式）。
   - 标注每条边上的数据格式/协议（如 HTTP JSON、gRPC protobuf、SQL query、事件 payload 等）。
   - 标注关键节点的源文件路径（`file:line` 格式），便于 reviewer 跳转验证。

3. **展示并确认**：将数据流图展示给用户，逐一确认：
   - 是否遗漏了上下游组件；
   - 数据格式标注是否准确；
   - 本次改动的边界标识是否正确。
   用户确认通过后方可进入 trade-off 讨论。

> **产物要求**：最终确认的数据流图**必须**原样写入 RFC 模板的 `## 方案` 段内，作为 `### 数据流` 子章节。

#### 4.4 Implementation Plan

命令基于 4.3 的产物自动草拟「分阶段 SubTask 候选 + 依赖顺序 + 每个 SubTask 的预期交付物」清单，逐项与用户确认；用户可增、删、改、合并、拆细。本维度的最终产物**必须**写入 RFC 模板的 `## 实施计划` 段，作为后续 `/beaver-decompose` 摄取 design doc 的依据。

#### 4.5 Alternatives Considered

命令从已有上下文中识别核心决策点的主要替代方案，逐一询问「为什么不采用」；用户的回答即为「被否决的方案 + 否决理由」。

### Phase 5: RFC 草稿生成与逐段确认

把五维度收集的内容拼装为完整 RFC，遵循 wiki RFC 模板（`docs/projects/{project}/rfc/NNNN-<slug>.md`），含：

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
{Context & Scope}

## 方案
{The Design}

### 数据流
{端到端数据流图 — Mermaid flowchart，含现有组件与本次改动标识}

### 备选方案
{Alternatives Considered}

## 影响范围
{derived from Design Goals + interfaces}

## 实施计划
{Implementation Plan 维度产物}

## 风险
{derived from trade-offs}

<!-- provenance
{fact-to-source mapping，每条事实标注来源}
-->
```

逐段展示给用户审批；用户可对任一段落要求修改，命令重新生成该段，直到用户全部满意。

### Phase 6: spec-document-reviewer 评审循环（push 前 HARD-GATE）

调度 `spec-document-reviewer` subagent（位于 `plugins/beaver/skills/spec-document-reviewer/SKILL.md`）对 Phase 5 的草稿做迭代评审，**最多 5 轮**：

1. 调度 reviewer 并传入：完整 RFC 草稿、原 Issue objective、原 Issue 验收标准列表、当前轮次编号、上一轮 BLOCK 反馈（轮次 ≥ 2 时）。
2. 解析 reviewer 的 verdict：
   - `PASS` → 退出循环，进入 Phase 7。
   - `BLOCK` → 把 reviewer 的 `Required fixes` 列表逐条转给用户，收集用户回答并合入草稿；轮次 +1，回到第 1 步。
3. 如第 5 轮仍返回 `BLOCK`：命令中止，打印最终 BLOCK 反馈并提示用户人工介入；**不允许 push**。

> **门控规则**：reviewer 任一轮 BLOCK 即继续；`PASS` 才允许进入 Phase 7。

### Phase 7: 提交 Draft PR

1. **写入 RFC 文件**到 `/tmp/wiki/docs/projects/{project}/rfc/NNNN-<slug>.md`（`NNNN` 由读取 `docs/projects/{project}/rfc/index.md` 找到的下一个可用编号决定）。
1. **追加 index 行**到 `/tmp/wiki/docs/projects/{project}/rfc/index.md` 末尾，格式遵循该文件现有约定（典型为 `- [RFC-NNNN: {title}](NNNN-<slug>.md)`）。
1. **commit + push**：

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-design.sh commit-push /tmp/wiki docs/projects/{project}/rfc/NNNN-{slug}.md "docs(rfc): add RFC-NNNN {title}" design/{issue_number}-{slug}
   ```

   注意：`commit-push` 内部会将 `docs/projects/{project}/rfc/index.md` 一并 stage。

1. **创建 Draft PR**（必须用 `gh pr create --draft`）。命令负责把 PR body 写入唯一命名的临时文件后调用 `--body-file`：

   ```bash
   bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-design.sh create-pr primatrix/wiki "RFC-NNNN: {title}" "Design doc for {org}/{issueRepo}#{issue_number}"
   ```

   PR body 至少包含：原 Task Issue 链接（不写 `Closes #N`，因 Task 还要继续开发）、五维度产物索引、reviewer 通过的轮次。

### Phase 8: 回写原 Issue

在原 Task Issue 上评论 PR 链接（命令负责把评论正文写入唯一命名的临时文件后调用 `--body-file`，以与 RFC §命令规约「临时文件命名约定」一致）：

```bash
bash ${CLAUDE_PLUGIN_ROOT}/scripts/beaver-design.sh comment-issue {org} {issueRepo} {issue_number} "Design Doc PR: {pr_url}"
```

### Phase 9: 收尾断言与下一步指引

1. **断言 Project V2 字段未变**（命令侧自检）：重新读取 Issue 在 Project V2 #14 上的 `Status / Type / Size / Iteration` 四个字段，确认与 Phase 1 校验时的快照完全一致；不一致即报错（应为不可能路径）。
1. 打印 PR URL 与下一步说明：
   > Design Doc 已作为 Draft PR 提交至 {pr_url}。请自审 Draft → 转 Open → 等待 Reviewer 通过 → 合并；之后用 `/beaver-decompose {issue_number} --design-doc {pr_url}` 拆解 SubTask。Status 仍为 `Design Pending`，由后续系统迁移推进到 `Ready to Develop`（过渡期内由用户在 Project #14 手动切换）。

## Constraints

- **本命令不修改任何 Project V2 字段**（`Status` / `Type` / `Size` / `Iteration` 等）；Phase 9.1 自检断言之。
- 五维度 QA 严格按 Phase 4 顺序进行，**禁止跨维度跳问**。
- spec-document-reviewer 评审循环最多 5 轮；任一轮 BLOCK 则继续，PASS 才允许 push；5 轮未通过则中止。
- 所有写入 `gh` CLI `--body-file` 的临时文件必须使用唯一文件名（`mktemp` 或 `/tmp/beaver-design-body-$$-$RANDOM.md` 等）；该约束由 `beaver-design.sh` 在内部统一处理，命令本身只传 body 字符串。
- PR 提交走 `gh pr create --draft`；RFC 文件落在 `docs/projects/{project}/rfc/NNNN-<slug>.md`，`docs/projects/{project}/rfc/index.md` 追加索引行；同时在原 Task Issue 上评论 PR 链接。
