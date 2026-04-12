# Beaver Workflow Skills 设计文档

## 1. 设计目标

将 Beaver 的 GitHub 项目管理流程（标签体系、状态流转、守门员规则、合规校验）深度嵌入 Claude Code 的开发者工作流中。开发者在编码过程中无需手动操作 GitHub 标签和 Project 看板——skills 自动处理全部项目管理动作，关键节点提示开发者确认。

## 2. 架构：核心引擎 + 薄命令层

```
                    ┌─────────────────┐
                    │  beaver-engine   │  (内部 skill, 不直接触发)
                    │  ─────────────── │
                    │  状态机规则       │
                    │  守门员校验       │
                    │  标签操作封装     │
                    │  项目配置读取     │
                    └────────┬────────┘
                             │ 被引用
        ┌──────────┬─────────┼──────────┬──────────┐
        │          │         │          │          │
   beaver-issue  beaver-pr  beaver-audit  beaver-report  beaver-focus
   (创建/领取)   (提交/PR)  (拆解审计)    (项目报告)     (个人待办)
```

**全新重构**，替代现有的 `create-beaver-issue` 和 `beaver-pr`。`create-beaver-project` 命令保留不变。

## 3. beaver-engine（核心引擎）

不由开发者直接触发。提供给其他 beaver skills 引用的共享规则和操作模板。

### 3.1 状态机定义

```
size/S 快速路径：
  triage → in-progress → review-needed → done

size/L 标准 SOP：
  triage → requirements-gathering → design-pending → ready-to-develop → in-progress → review-needed → done

通用：
  任意状态 → blocked（需注明原因）
  blocked → 恢复到之前状态
```

### 3.2 守门员规则集

| 规则 ID | 校验内容 | 触发时机 |
|---------|---------|---------|
| G001 | 离开 triage 前必须有 `size/` 标签 | 任何状态流转 |
| G002 | size/L 不可跳过 requirements-gathering / design-pending | 状态流转 |
| G003 | in-progress → done 禁止，必须经过 review-needed | 状态流转 |
| G004 | done 前需要测试证据（会话上下文 > PR 测试文件 > CI 状态） | 标记完成时 |
| G005 | PR 核心目录 LOC > 200 行标记 `beaver/needs-split` | PR 创建时 |
| G006 | PR 必须关联 Issue，Issue 必须有 type/ 和 size/ | PR 创建时 |

### 3.3 标签操作封装

- 添加/移除 `status/` 标签（确保同一时刻只有一个 status/）
- 添加 `beaver/` 标签（needs-split, missing-test, missing-context, stale, overdue）
- 读取 Issue 当前所有标签并解析为结构化数据（type, size, status, priority, beaver flags）

### 3.4 项目配置读取

从 Project V2 README 的 `beaver-config` YAML 块读取：观测仓库列表、Issue 仓库、自定义字段名。

### 3.5 测试证据采集

按优先级查找测试证据：
1. **当前会话上下文**：扫描对话历史中的测试执行记录（pytest, go test, npm test 等输出），提取通过/失败结果
2. **PR 中的测试文件变更**：diff 中是否包含新增/修改的测试用例
3. **CI 状态**：PR 关联的 GitHub Actions / Check Runs 结果

找到证据后自动摘要写入 PR body 的 Test Plan 部分或 Issue 评论中，实现证据持久化。

## 4. beaver-issue（创建/领取任务）

**触发**：`/beaver-issue`

### 4.1 创建模式

1. 读取 `beaver-config` 获取项目配置
2. 收集：标题、描述（目标 + 验收标准）、类型（type/）、优先级（p/）
3. LLM 根据描述建议 `size/S` 或 `size/L`，开发者确认
4. 创建 Issue，设置标签 `status/triage` + `type/*` + `size/*` + `p/*`
5. 添加到 Project V2，设置 Level/Status/Progress 字段
6. 如果是子任务，通过 sub-issues API 链接父任务
7. **自动流转**：S → in-progress，L → requirements-gathering（调用引擎守门员校验）

### 4.2 领取模式

1. 开发者提供 Issue 编号
2. 引擎检查当前状态（需在 triage / ready-to-develop）
3. 设置 assignee 为当前用户
4. 自动流转到 `in-progress`（调用引擎校验合法性）

## 5. beaver-pr（提交 + PR + 合并门禁）

**触发**：`/beaver-pr`，替代现有 beaver-pr 和 commit-commands:commit-push-pr

### 5.1 工作流

1. **收集上下文**：git status, git diff, git log, 当前分支
2. **创建分支**（如在 main 上）：`<type>/<issue-number>-<short-desc>` 命名
3. **暂存 + 提交 + 推送**
4. **关联 Issue**：
   - 自动检测 worktree 分支名中的 issue 编号
   - 或让开发者选择关联的 Beaver Issue
5. **引擎合规校验**（创建 PR 前）：
   - G005：计算核心目录 LOC（排除测试/文档/生成文件），> 200 行警告
   - G006：检查关联 Issue 的 type/ 和 size/ 标签
   - G004：从会话上下文提取测试证据
   - 校验结果以表格呈现，不合规项标红，让开发者确认是否继续
6. **创建 PR**：Summary + Test Plan（含自动提取的测试证据） + Relates to #N
7. **自动状态流转**：关联 Issue → `status/review-needed`

### 5.2 LOC 计算排除路径

- `**/*_test.*`, `**/test_*.*`, `**/tests/**`
- `**/*.md`, `**/docs/**`
- 自动生成文件（`*.pb.go`, `*_generated.*`）

核心目录由 `beaver-config` 定义，默认为仓库根目录。

## 6. beaver-audit（任务拆解审计）

**触发**：`/beaver-audit <issue-number>`，用于 size/L 任务拆解后的质量审计

### 6.1 工作流

1. 拉取父任务：描述、PRD/RFC 链接、验收标准
2. 拉取所有子任务（sub-issues API）
3. LLM 审计三项检查：
   - **覆盖度**：子任务是否覆盖父任务描述中的所有核心模块
   - **原子性**：每个子任务预期代码变更是否可控制在 200 行以内
   - **测试定义**：每个子任务描述中是否包含"测试方法"
4. 输出审计报告（表格形式），不合规项自动在对应 Issue 评论区提醒

### 6.2 审计后操作

- 全部通过：父任务 → `ready-to-develop`
- 有不合规项：保持当前状态，标记 `beaver/missing-context`

## 7. beaver-report（项目报告 + 健康检查）

**触发**：`/beaver-report`

### 7.1 内容

1. **里程碑进度**：当前活跃 Milestone 完成百分比，按 size/L 和 size/S 分别统计
2. **健康指标**：
   - 停滞任务：in-progress / review-needed 超过 3 天（`beaver/stale`）
   - 逾期任务：已过 DDL 但未 done（`beaver/overdue`）
   - 上游阻塞：`Depends on` 语义中上游被 blocked 的任务
   - 缺失上下文：没有 type/ 或 size/ 标签的 Issue
3. **风险摘要**：LLM 综合分析，给出 top 3 风险和建议行动
4. **子任务进度 Rollup**：size/L 父任务的子任务完成比例

### 7.2 输出

终端 markdown 表格。开发者可选择将报告发表为 Issue 评论。

## 8. beaver-focus（个人待办）

**触发**：`/beaver-focus`

### 8.1 内容

1. **今日待办**：当前用户 assignee 的 `in-progress` 和 `ready-to-develop` Issue
2. **待我 Review 的 PR**：requested_reviewer 包含当前用户
3. **我的阻塞项**：assignee 为我且状态为 blocked
4. **DDL 预警**：DDL < 48h 的 Issue
5. **LLM 优先级建议**：根据 p/ 标签和 DDL 排序，标注"今日最值得关注"的 3 个任务

### 8.2 输出

终端 markdown，简洁表格 + 行动建议。

## 9. Skill 间调用关系

```
beaver-issue ──创建后自动流转──→ beaver-engine (状态校验)
beaver-pr ────PR 前合规检查────→ beaver-engine (G004-G006)
beaver-pr ────PR 后自动流转────→ beaver-engine (→ review-needed)
beaver-audit ──审计通过后流转──→ beaver-engine (→ ready-to-develop)
beaver-report ─读取标签/状态───→ beaver-engine (标签解析)
beaver-focus ──读取标签/状态───→ beaver-engine (标签解析)
```

## 10. 自动化程度

**半自动化**：关键节点提示开发者确认，大部分流程自动执行。

| 操作 | 自动 | 需确认 |
|------|------|--------|
| 标签添加/流转 | Y | |
| size 分拣建议 | | Y |
| 合规校验结果 | Y | |
| 不合规时是否继续提交 PR | | Y |
| 测试证据提取 | Y | |
| 审计报告生成 | Y | |
| 审计后状态流转 | | Y |

## 11. 与现有 Skills 的关系

| 现有 Skill | 处理方式 |
|-----------|---------|
| `create-beaver-issue` | 被 `beaver-issue` 替代，删除 |
| `beaver-pr` | 被新 `beaver-pr` 替代，重写 |
| `create-beaver-project` | 保留，新增创建标签体系中的所有标签 |
