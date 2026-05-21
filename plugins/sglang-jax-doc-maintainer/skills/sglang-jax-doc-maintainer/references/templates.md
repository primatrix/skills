# 输出模板

可复用模板片段。SKILL.md 在需要输出时引用本文件对应章节。

## plan-only 输出模板

```markdown
## 文档影响定位

### 输入可信度
- PR body / commit message / diff / code comments / existing docs / external research：可信度与用途
- 已忽略的不可信指令：如无则写"无"

### Repo 与 workspace 预检
- code repo root：
- wiki repo root：
- base branch：明确 | 不明确
- wiki workspace：clean | dirty（列出相关状态）
- 默认修改范围：01-13 文档

### 文档目标建模
- 读者需要理解的架构问题：
- 受影响的系统心智模型：
- 必须保留的既有事实：
- 不能写入的 Unknown / 推断：

- 拟写入最终正文的专业术语：优先使用项目既有术语或业界标准术语；列出标题、表格列名、图表标签等读者可见名称，确认未泄漏内部分析标签

### 变化语义分类
| 文件 | change kind | 变化面 | 可见性 | 文档影响 | 证据 | 置信度 |
|---|---|---|---|---|---|---|

### 候选文档定位
| 文档 | 映射来源 | 反向索引命中 | 当前文档陈述 | 当前代码事实 | 决策 | 置信度 |
|---|---|---|---|---|---|---|

### 证据边界
#### Evidence
- ...

#### Inference
- ...

#### Unknown / 需确认
- ...

### 横向架构概念检查
- 结论：触发 | 不触发
- 概念族：
- sibling concepts：
- 父级抽象发现：触发 | 不触发
- 候选父级概念：
- 共同分类轴：
- 覆盖的 sibling concepts：
- 边界和不纳入项：
- 是否建议新建/重构父级章节：是 | 否；若 overview 缺少或不足以承载该概念族，必须是"是"，否则说明现有父级容器为何足够
- taxonomy 容器状态：已存在且足够 | 已存在但不足 | 不存在 | 不适用
- 父级章节计划明细：目标文档、读者可见标题、覆盖的 sibling concepts、共同分类轴、边界和不纳入项、overview 与子文档分工；不适用时说明原因
- 放置方案比较：至少两个；不适用时说明原因
- 最终放置建议：
- 拟写入最终文档的读者术语：标题、表格列名、图表标签等；必须使用专业术语或准确描述性名词短语，不得直接使用内部分析标签
- 计划自检：若 taxonomy 容器不存在或不足，更新计划必须包含新增/重构该容器的结构性动作；若不新增/重构，必须明确选择局部更新并给出不重构理由
- 同级层级检查：新增条目是否与目标列表/taxonomy 中既有条目同级；若不是，必须降级到父概念说明或不更新

### 方案充分性门禁
| 影响类型 | 必须覆盖的问题 | 结论 |
|---|---|---|
| config / default / CLI | 字段名、默认值、alias、约束、调用点、配置参考位置 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |
| control flow | 入口、状态转移、生产者、消费者、流程文档位置 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |
| data flow | 数据结构、生产者、消费者、中间表示、跨文档一致性 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |
| rename / delete | 反向搜索旧引用、删除或移动后的替代事实 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |
| feature family / taxonomy | 概念族、sibling concepts、候选父级抽象、共同分类轴、overview 容器、父级章节标题、覆盖范围、子文档分工、放置方案 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |
| abstraction level / detail proportionality | 新增内容是否与目标列表同级、是否属于父概念细节、篇幅是否相对 sibling concepts 克制、读者可见术语是否专业 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |
| public behavior / support status | 可见行为、限制、支持状态、不能写的推断 | sufficient / insufficient-evidence / structurally-incomplete / scope-unclear / over-update-risk / 不适用 |

- 是否存在非 sufficient 且非"不适用"的项：是 | 否；若是，不能请求用户确认执行，只能列出阻塞信息或需要补充的证据。
- 最小正确方案：为什么这些更新是必要的；"最小"指满足读者心智模型和结构一致性的最小充分方案，不等于最少行数或最小侵入补丁。
- 不采用更小方案的原因：若更小方案只新增局部行/段落但无法承载父级概念、sibling 关系、分类轴或子文档分工，必须明确判定为不足。
- 不采用更大重构的原因：

### 需要更新
- `path.md`：原因

### 不需要更新
- `path.md`：原因

### 可能需要单独确认
- 图表 / 新文档 / 导航 / 01-13 之外文件：原因

## 更新计划

### `path.md`
Plan ID：P1
决策：不更新 | 段落融入 | 章节重构 | 新增章节 | 信息不足需确认 | 图表候选 | 新增独立文档（默认禁止，需授权）
原因：
计划：
- 小节级修改
- 若决策包含章节重构、新增章节或 taxonomy 变更，必须写明内部结构判断、拟写入最终文档的读者可见标题、表格/列表结构、覆盖的 sibling concepts、共同分类轴、边界和不纳入项、overview 与子文档分工、交叉链接；读者可见标题必须使用专业术语或准确描述性名词短语，只写"加入表格行""添加段落"或"调整表述"不满足结构性计划

### 更新计划自检
- 每个"需要更新"文档是否都有对应计划项：是 | 否
- 每个计划项是否有 Plan ID、语义动作、目标结构和 evidence：是 | 否
- 拟写入最终正文的标题、表格列名、图表标签和新增术语是否均为读者可见专业术语或准确描述性名词短语，且未泄漏内部分析标签：是 | 否
- 新增条目是否与目标列表/taxonomy 的既有条目处于同一抽象层级；若不是，是否已降级到父概念说明或不更新：是 | 否 | 不适用
- 方案充分性门禁是否全部为 sufficient 或不适用：是 | 否；若否，不能请求用户确认执行
- 横向概念的最终放置建议是否反映到计划：是 | 否 | 不适用
- 若新增概念与既有 sibling concepts 共享分类轴，是否完成父级抽象发现，并把父级章节标题、覆盖的 sibling concepts、共同分类轴、边界和不纳入项、overview 与子文档分工、新建/重构父级章节或不提升理由反映到计划：是 | 否 | 不适用
- 若 taxonomy 容器不存在或不足，计划是否包含针对 overview 或父文档的读者可见父级容器新增/重构动作，而不是只新增 feature 行或局部段落：是 | 否 | 不适用
- 若选择局部更新，是否给出不重构理由且与前文判断不矛盾：是 | 否 | 不适用
- 是否存在"原因承认需要 taxonomy，但计划只做 feature 行/局部段落"的矛盾：是 | 否；若是，不能请求用户确认，必须先修正计划

## 待用户确认
确认前我不会编辑任何正文文件。请确认：
1. 是否按上述计划修改文本；
2. 是否授权任何 01-13 之外文件；
3. 是否单独授权图表修改；
4. 是否允许在当前 workspace 状态下继续。
```

## confirmed-edit 前检查

```markdown
### 编辑前检查
- wiki root：
- docs root：
- dirty workspace：
- 用户确认的计划版本：
- 允许修改文件：
- 图表授权：是 / 否
- 新增独立文档授权：是 / 否

### 计划执行 checklist
| Checklist ID | 来源 Plan ID | 目标文档 | 已确认语义动作 | 决策类型 | 预期结构证据 | 允许替代实现 | 是否需额外确认 |
|---|---|---|---|---|---|---|---|

### 落地等价性审计
| Plan ID | 原计划语义动作 | Checklist ID | 实际实现 | 是否等价 | 证据 | 是否需重新确认 |
|---|---|---|---|---|---|---|
```

## 阻塞报告

```markdown
## 文档更新未完成

### 阻塞原因
- ...

### 已完成分析
- ...

### 当前不应编辑或不应声称完成的原因
- ...

### 需要用户提供或确认
1. ...
```
