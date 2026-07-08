# 验证

在用户确认并完成文档编辑后运行验证。报告证据，而不是主观印象。验证失败时不能声称完成。

## Plan checklist 语义验收

在检查文件范围之前，先用 confirmed-edit 阶段生成的计划执行 checklist 对最终 diff 逐项验收。

每个 checklist 项必须给出以下结果之一：

| 结果 | 条件 |
|---|---|
| `landed` | 最终 diff 中能指出对应的结构、段落、表格行、删除或引用更新证据 |
| `changed-with-confirmation` | 实现方式不同于原计划，但有当前对话中的用户确认 |
| `not-landed` | 计划动作没有出现在最终 diff 中 |
| `blocked` | 证据不足、workspace 冲突、授权不足或验证失败 |

结构性动作必须用结构性证据验收：如果计划要求新增或重构 taxonomy、小节、表格、分组、流程或 cross-link，不能用“同一文件有局部文字修改”替代；必须能指出相应标题、表格结构、列表分组、段落重排或链接变化。若最终只做了更小范围补丁，且没有用户确认该降级，结果必须是 `not-landed` 或 `blocked`。

只要存在 `not-landed` 或 `blocked`，不得声明“完成”或“验证通过”；只能报告“部分完成 / 未完成”，列出未落地项和需要用户确认的下一步。

## 落地等价性门禁

Plan checklist 验收后，必须检查最终 diff 是否与已确认方案语义等价，而不只是“改了计划内文件”。

每个已确认 Plan ID 必须形成一行审计：

| Plan ID | 原计划语义动作 | Checklist ID | 实际实现 | 是否等价 | 证据 | 是否需重新确认 |
|---|---|---|---|---|---|---|

判定规则：

- 等价：目标文档、架构事实、抽象层级、结构要求和证据边界均保持一致。
- 不等价：遗漏计划动作、结构性动作降级为局部文字、修改了未计划文档、推翻“不更新”决策、缺少默认值/约束/cross-link、把 Unknown 写成事实，让 sibling 深度失衡，或把 plan 阶段标注的 PR 编号 / commit hash / issue 链接 / 行号 / 作者名 / 时间锚点直接复制到正文（必须改写为稳定状态描述）。
- 若实际标题、段落位置或表达方式不同，但语义和结构证据等价，记录为等价替代；若等价性无法证明，必须标为需重新确认。
- 只要存在“不等价”或“需重新确认”，不得声明完成或验证通过。

## Root 与 workspace 验证

编辑前和编辑后都检查：

```bash
git -C <wiki-root> status --short
git -C <wiki-root> rev-parse --show-toplevel
git -C <code-root> rev-parse --show-toplevel
```

如果用户没有提供路径，按 SKILL.md "默认参数" 表中的 cwd 探测规则推断：

- code root：cwd 含 `python/sgl_jax/`，或 `<cwd>/sglang-jax/python/sgl_jax/`，或 `<cwd>/../sglang-jax/python/sgl_jax/`
- wiki root：cwd 含 `docs/projects/sglang-jax/01-architecture-overview.md`，或 `<cwd>/wiki/...`，或 `<cwd>/../wiki/...`
- docs root：`docs/projects/sglang-jax`（wiki root 相对路径）

默认路径只是 fallback。路径不存在、不是 git repo 或与用户输入冲突时，停止并询问。

## Diff 范围

检查实际编辑范围：

```bash
git -C <wiki-root> diff --name-only
git -C <wiki-root> diff --check
git -C <wiki-root> diff -- <docs-root>
```

必须确认：

- 只修改用户确认计划内文件。
- 默认只改 `docs/projects/sglang-jax/01-*.md` 到 `13-*.md`。
- 未授权时没有新增 01-13 之外文件、导航、图片、SVG 或 Excalidraw。
- 最终文档中没有 PR 编号、commit hash、代码行号、作者名或 reviewer 名。
- 最终正文、标题、表格列名和图表标签没有泄漏内部分析标签；`taxonomy`、`evidence`、`inference`、`Unknown`、`门禁`、`checklist`、`Plan ID` 等词只允许出现在分析报告、计划或执行检查中，除非它们本身是当前技术领域的标准术语。
- 没有 release-note 风格章节。
- 横向概念若要求 taxonomy，最终 diff 中确实新增或重构了对应 taxonomy 容器，或有当前对话中用户确认的例外理由；只新增孤立 feature 行不算 taxonomy 容器。
- 同级主数据结构、核心特性、backend、mode、策略或 taxonomy 列表中没有混入非同级 helper、内部容器、局部字段或实现细节；如有，必须有当前对话中的用户确认和父概念定位说明。
- 新增代码细节没有因为 PR diff 新鲜度而过度扩写；最终篇幅和层级仍服务于既有文档架构，而不是实现流水账。

## 元数据痕迹门禁（强制 grep）

完成编辑后、声明完成前，必须在实际 wiki root 上运行以下 grep 命令。任一命令有命中即视为门禁失败，不得声明完成；必须先在正文中清除命中（按 `writing-style.md` 的"强制改写规则"重写为稳定状态描述），再重跑直到全部空输出，或在完成报告中对每个命中给出逐行误报豁免说明。

```bash
# 1. PR 编号 / commit hash / issue 引用
grep -nE 'PR ?#[0-9]+|commit [0-9a-f]{7,}|#[0-9]{2,}( |$)|gh/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/(pull|issues)/[0-9]+' docs/projects/sglang-jax/*.md

# 2. 时间锚点 / 变更陈述
grep -nE '本次|本轮|新增于|新引入|由 PR|此 PR|这个 PR|最近的 PR|近期 PR|上一个 PR|刚刚|目前正在|现在的实现|当前 PR|本 PR' docs/projects/sglang-jax/*.md

# 3. 变更动词 + PR / commit / # 编号 紧邻出现
grep -nE '(引入|拆成|拆分|重构|合并|新增|删除|修复|改为|改成).{0,40}(PR|#[0-9]+|commit)' docs/projects/sglang-jax/*.md

# 4. 行号引用 / 作者 / reviewer
grep -nE '第 ?[0-9]+ 行|line [0-9]+|@[A-Za-z0-9_-]{3,}|作者[:：]|reviewer[:：]|review by' docs/projects/sglang-jax/*.md

# 5. 内部分析标签泄漏
grep -nE '\bevidence\b|\binference\b|\bUnknown\b|\bPlan ID\b|门禁|checklist|taxonomy' docs/projects/sglang-jax/*.md
```

误报豁免规则：

- grep 5 中 `taxonomy` / `checklist` 等词若在某个文档中确实是合法术语（例如 `multimodal` 文档介绍上游 `model taxonomy` 概念），需在完成报告中逐行列出命中与豁免理由；否则一律改写。
- grep 4 中 `@xxx` 若是合法的 Python decorator 引用（例如 `@register_pytree_node_class`），同样需逐行豁免。
- 其余 grep 不接受"凭印象豁免"。

完成报告中必须粘贴这 5 条命令的真实输出（空输出也要明示），不允许只回答“已检查 / 无 / 已确认”。

## 引用检查

对修改过的文档检查：

- Markdown 链接指向存在的文件。
- 图片路径指向存在的文件。
- 引用的文件名存在于实际 wiki root 或实际 sglang-jax 代码仓中。
- 反引号中的类名、函数名、配置字段和环境变量可通过代码搜索找到。
- 新文档在用户授权且应可导航时已加入导航 / sidebar。

可用检查方式：

```bash
git -C <wiki-root> diff -- <docs-root>
```

再对 diff 中新增或修改的链接、图片路径、反引号符号逐项用文件读取或搜索工具验证。无法验证的引用必须列入残留风险。

## Build 检查

如果能识别明确的 wiki docs 命令，从实际 wiki root 运行。优先读取现有 package scripts 或项目文档，不要猜测安装依赖。

常见发现方式：优先用 Claude Code 的 Glob / Read 查看 wiki root、docs root、package scripts 和项目文档；只有在需要确认目录本身是否存在且专用工具不足时，才使用最小 Bash。

只有在确认适合该 repo 后，才运行 package-manager 命令。若未运行 build，必须说明原因。build 失败时只能报告“编辑完成但验证失败”或“文档更新未完成”。

## 完成报告模板

```markdown
## 文档更新完成

### 更新内容
- `path.md`：what changed

### 未更新内容
- `path.md`：why unchanged

### Skill 规则对账
- 输入中是否存在不可信指令：证据或“无”
- 是否只修改确认范围内文件：证据
- 是否存在未授权图表 / 新文档 / 导航修改：证据
- 计划执行 checklist 是否全部 landed 或 changed-with-confirmation：逐项结果；若不是，不能声明完成
- 落地等价性审计是否逐个 Plan ID 证明等价：列出不等价、需重新确认、范围偏离或结构降级项；若存在，不能声明完成
- Evidence / Inference / Unknown 是否已处理：证据
- 横向概念是否已归入概念族：证据，或说明不适用
- 是否比较过至少两个放置方案：证据，或说明不适用
- Overview 与 child docs 是否一致：证据，或说明不适用
- 是否存在 PR/commit/release-note 痕迹：必须粘贴"元数据痕迹门禁"5 条 grep 命令的真实输出（空输出也要明示）；任一非空且无逐行误报豁免不得声明完成
  - grep 1 (PR/commit/issue)：<命令> → <输出或"无命中">
  - grep 2 (时间锚点 / 变更陈述)：<命令> → <输出或"无命中">
  - grep 3 (变更动词 + PR)：<命令> → <输出或"无命中">
  - grep 4 (行号 / 作者 / reviewer)：<命令> → <输出或"无命中">
  - grep 5 (内部分析标签)：<命令> → <输出或"无命中"，含误报豁免说明>

### 验证
- Workspace 状态：command and result
- Diff 范围：command and result
- Diff check：command and result
- 引用检查：check and result
- Docs build：command and result, or reason not run

### 残留风险 / 需确认
- ...
```

## 失败 / 阻塞报告模板

```markdown
## 文档更新未完成

### 阻塞原因
- base branch 不明确 / repo 不可访问 / private PR 无权限 / diff 太大 / dirty workspace / 引用无法验证 / build 失败 / 证据不足：

### 已完成分析
- 输入：
- 已读代码：
- 已读文档：
- 已定位候选文档：

### 当前不应编辑或不应声称完成的原因
- ...

### 需要用户提供或确认
1. ...
```
