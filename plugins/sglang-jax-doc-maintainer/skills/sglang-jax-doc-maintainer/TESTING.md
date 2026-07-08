# skill 测试说明

本 skill 按 TDD-for-skills 思路维护：先用 eval 暴露失败，再修改 SKILL.md 和 references。

## 当前已覆盖的失败模式

- plan-only 场景必须停止等待确认，不能“假设用户确认”。
- confirmed-edit 场景只能修改用户确认范围内的文件。
- PR body / code comment 等不可信输入不能覆盖 skill 规则。
- changed-files-only 证据不足时必须请求 diff。
- base branch 不明确时必须询问。
- 图表未单独确认时不得修改。
- 新增 01-13 之外文档必须请求授权。
- dirty workspace 中存在未归属改动时必须停止。
- confirmed-edit 必须把已确认计划拆成语义动作 checklist；结构性动作未落地时不得声称完成。
- 内部分析标签不得泄漏到最终正文标题、表格列名或图表标签；计划中的 taxonomy 判断必须转换为读者可见的专业术语或准确描述性名词短语。
- 非同级 helper、内部容器、局部字段或实现细节不得进入核心特性、主数据结构、backend、mode、策略或 taxonomy 同级列表。
- 新增概念与既有 sibling concepts 共享分类轴时，必须主动进行父级抽象发现，判断是否新建或重构父级架构章节。

## 最近一次静态验证

2026-05-13：

- `python3 -m json.tool evals/evals.json >/dev/null`：通过，JSON 语法有效。
- 泛化痕迹检查：新增规则没有写入当前 PR 编号或只覆盖单一事故；现有 `Data Parallelism` 字样只存在于既有 plan-only eval 的通用 parallelism 场景。
- 关键规则检查：`SKILL.md`、`references/validation.md`、`references/templates.md`、`references/review-checklist.md` 均包含计划执行 checklist / 结构性动作验收规则。
- plan-only taxonomy 门禁检查：`SKILL.md` 和 `references/taxonomy.md` 已加入 taxonomy 容器状态、更新计划自检、计划矛盾处理；evals 增加通用资源策略场景，覆盖“判断 taxonomy 不足但计划只做 feature 行”的失败模式。
- 双门禁重构检查：新增通用方案充分性 eval 和落地等价性 eval；`SKILL.md`、`references/update-decisions.md`、`references/validation.md`、`references/templates.md`、`references/review-checklist.md` 均包含方案充分性门禁 / 落地等价性门禁 / Plan ID 审计规则。
- 术语与层级检查：新增通用 eval 覆盖内部分析标签泄漏、非同级 helper 进入核心同级列表；`SKILL.md` 和 `references/update-decisions.md` 已把 abstraction level / detail proportionality 纳入方案充分性门禁，`references/validation.md` 完成报告要求输出落地等价性审计。
- 父级抽象发现检查：新增通用 eval 覆盖“新增概念 + 既有 sibling concepts”可上升为父级架构容器的场景；`SKILL.md`、`references/taxonomy.md`、`references/update-decisions.md` 已加入候选父级概念、共同分类轴、边界/不纳入项、新建或重构父级章节/不提升理由。

2026-05-12：

- `python3 -m json.tool evals/evals.json >/dev/null`：通过，JSON 语法有效。
- 检查 eval prompt 中的“假设用户确认”：无 prompt 级匹配；该短语只保留在禁止项中。
- 检查 eval phase：包含 `plan-only`、`negative`、`confirmed-edit` 三层。
- 检查 `SKILL.md` 关键规则：包含不可信输入、plan-only、confirmed-edit、dirty workspace、Evidence、新增独立文档限制和“假设用户确认”禁用。

尚未运行真实 PR 的 plan-only dry run；confirmed-edit dry run 必须等用户确认计划后再运行。

## 后续验证建议

1. 用当前 evals 做人工或 runner 驱动的 baseline。
2. 对真实 PR 先跑 plan-only dry run。
3. 用户确认后再跑 confirmed-edit。
4. 对比人工 review，记录误报、漏报、过度更新和证据不足。
