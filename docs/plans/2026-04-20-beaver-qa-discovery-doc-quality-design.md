---
date: 2026-04-20
topic: 把 superpowers 的 QA / 信息收集 / 文档编写融入 beaver-issue 与 beaver-design-doc
status: approved
related:
  - plugins/beaver/skills/beaver-engine/SKILL.md
  - plugins/beaver/skills/beaver-issue/SKILL.md
  - plugins/beaver/skills/beaver-design-doc/SKILL.md
  - https://github.com/primatrix/wiki/blob/main/docs/onboarding/project-management.md (本地副本: ~/Code/wiki/docs/onboarding/project-management.md)
---

# Beaver Skills QA / Discovery / Doc Quality 改造设计

## 1. Context & Scope

本仓库是 Claude Code Plugin Marketplace，其中 `beaver` 插件已包含 8 个 skill：
`beaver-engine`、`beaver-issue`、`beaver-pr`、`beaver-design-doc`、`beaver-decompose`、`beaver-focus`、`beaver-report`、`beaver-audit`。

`beaver-engine` 是不直接触发的 internal engine，已对外提供 6 个 Section（Label Taxonomy / State Machine / Guardrails / Label Ops / Project Config / Transition Execution）。其它 SKILL 通过 "References beaver-engine for: ..." 形式引用。

`beaver-issue` 当前 Create 模式以"一次性收集 6 字段 + size 自动分类"驱动，没有 QA 对话循环、没有 codebase 探索、没有写作质量约束。
`beaver-design-doc` 已有 4 段 Q&A + HARD-GATE + Sectional Review，与 superpowers `brainstorming` 的精神同源，但 codebase 探索仅靠自然语言提示，未工具化；写作约束（中英规范、反幻觉、checklist）缺失；与 decompose 衔接未明确。

外部约束来自 `~/Code/wiki/docs/onboarding/project-management.md`（primatrix/wiki）："**Beaver Skills 中 TDD 驱动开发与 Review 能力从 Superpowers 迁移而来**"。本次改造延续此趋势——把 superpowers 的 brainstorming（QA loop）、信息收集（Discovery Triad）、writing-skills/writing-plans 的写作纪律下沉到 beaver。

本次改造**不涉及** Worker、不涉及 beaver-pr / decompose / focus / report / audit、不涉及 Claim mode、不涉及 GitHub Project V2 字段语义。

## 2. Design Goals

### 2.1 Goals

1. 把"QA 对话循环 + HARD-GATE + Sectional Review"提炼到 `beaver-engine` 的可复用 Section。
2. 把"git/grep/docs 三项必跑信息收集动作"提炼到 `beaver-engine` 的可复用 Section，并强制为 Q&A 第一问的前置条件。
3. 把"中英规范 + 反幻觉 + 完整性 checklist"提炼到 `beaver-engine` 的可复用 Section。
4. `beaver-issue` Create 模式接入上述三节；按 size 区分 QA 强度；新增 Bug 模式（type/bug 强制 size/S、p0/blocker 直 in-progress + @CODEOWNERS）。
5. `beaver-design-doc` Phase 2/3/4 接入上述三节；保留 4 段语义；末尾追加 Provenance 区块；提示但不自动调用 beaver-decompose。

### 2.2 Non-Goals

- 不实现 Worker 自动化（如 Milestone 加入后自动 design-pending、PR 合并自动状态流转）。
- 不实现 beaver-pr / beaver-decompose / beaver-focus / beaver-report / beaver-audit 的 QA 化（待后续单独开 Goal）。
- 不为 `beaver-issue` Claim 模式新增 QA（已有充分上下文，加 QA 反而冗余）。
- 不引入新的依赖插件（不绑定 `superpowers:brainstorming` 版本号）。
- 不修改 GitHub Project V2 字段、Issue/Label 命名约定、guardrail 编号体系（G001-G006）。

### 2.3 Success Metrics

- **可复用性**：`beaver-engine` 新增 Section 7/8/9 后，其它 6 个 beaver SKILL 至少有一个（issue / design-doc）通过引用方式接入，无内联复制。
- **HARD-GATE 有效性**：在 issue Create 与 design-doc Phase 2/3 中，未通过 §9.3 checklist 的内容**不能**进入提交步骤（gh api POST / gh pr create）；可由 SKILL.md 中是否包含 "Approved? (y/revise)" gate 文本验证。
- **反幻觉可追溯**：design doc 末尾必有 `<!-- provenance ... -->` 块；issue body 不出现 Discovery Brief 之外的文件路径。
- **Bug 通道一致性**：type/bug + p0/blocker 的 issue 创建后状态为 `status/in-progress`，body 含 `cc @owner`；与 wiki Phase 2 描述一致。

## 3. The Design

### 3.1 System Context Diagram

```text
                +-------------------------+
                | superpowers (upstream)  |
                |  - brainstorming        |
                |  - writing-skills       |
                |  - writing-plans        |
                +-----------+-------------+
                            | (借鉴，不绑定)
                            v
+--------------+    refs   +-------------------------+
| beaver-issue |---------> |  beaver-engine          |
+--------------+           |  §1-6 (existing)        |
                           |  §7 QA & HARD-GATE (new)|
+------------------+ refs  |  §8 Discovery Triad(new)|
| beaver-design-doc|-----> |  §9 Doc Quality   (new) |
+------------------+       +-------------------------+
                                       ^
                                       | (future)
                            beaver-decompose / pr / ...
```

### 3.2 Core Architecture

**改造原则**：方案 A —— engine 集中、SKILL 引用。

**变更范围**：3 个文件
- `plugins/beaver/skills/beaver-engine/SKILL.md`：追加 Section 7、8、9。
- `plugins/beaver/skills/beaver-issue/SKILL.md`：扩展 Mode Detect、Create 流程接入 §7/§8/§9、新增 Bug submode。
- `plugins/beaver/skills/beaver-design-doc/SKILL.md`：Phase 2/3/4 改造、模板加 §5 Open Questions + Provenance。

**关键约束**（§7 / §8 / §9 三节均为 HARD-GATE）：
- 调用方在主流程开始前必须串行执行 §8 → §7 →（每节）§9.3 checklist；否则禁止 `gh api ... POST` / `gh pr create` / `git commit`。
- engine §1-6 编号不变（兼容现有 G001-G006 引用）。

### 3.3 Interfaces & Data Flow

**§7 QA Loop & HARD-GATE 接口**

输入：Discovery Brief（来自 §8）+ 调用方定义的"段语义清单"（issue 是 6 字段，design-doc 是 4 段）。
输出：每段 ≥ 1 轮 Q&A 后产出的"已确认要点"段，配 §9.3 checklist 表。
契约：
- 一次一问；优先多选；
- approval grammar：必须显式 "approve / ok / 继续 / y"；模糊回答 = revise；
- §7.4 Skip-detection 红旗表，调用方共享；
- §7.5 approval 字面规则。

**§8 Discovery Triad 接口**

输入：issue title + objective（自由文本）。
输出：固定格式 "Discovery Brief"（见 §8.3），含 D1/D2/D3 + Open questions surfaced。
契约：
- D1 `git log --oneline -20` + `git log --all --since="14 days ago" --oneline`；
- D2 关键词从 issue 文本字面抽取，≤ 5 个；用 `Glob` + `Grep`；
- D3 `Read` 仓库根 README/CLAUDE.md + 命中目录的 `*/README.md`；
- 任何"似乎/可能/应该"等推测语禁用；0 命中必须明示 "0 files / absent"。

**§9 Doc Quality 接口**

输入：调用方在每段 approval 前的草稿。
输出：5 行 checklist 表（Why / Verifiable / No invented facts / Bilingual / Length scaled）+ Provenance 区块。
契约：任一项 ☐ 未勾选 → 必须先修该节再请求 approval。

**调用方数据流（issue Create / Feature submode 为例）**

```text
user 触发 beaver-issue
  ↓
Mode Detect: arg 为空 → Create
  ↓
Step 0 问 type → feat
  ↓
Step 1 加载 defaults
  ↓
Step 1.5 执行 engine §8 → 输出 Discovery Brief
  ↓
Step 2 进入 engine §7 Q&A loop
   先问 size → S？L？
     ├─ size/S：3 个最少必要问题（title / objective / acceptance）
     └─ size/L：完整 4 维 Q&A（objective / scope / acceptance / parent stakeholder），逐段 §9.3 checklist
  ↓
Step 4 Preview + §9.4 issue body checklist
  ↓
Step 5 写 body 到临时文件 → gh api POST 创建
  ↓
Step 6-9 项目添加、父链接、状态流转、保存默认值
```

**调用方数据流（design-doc Phase 2 改造）**

```text
Phase 1 fetch + 校验 size/L + status/design-pending
  ↓
Phase 2 进入前先执行 engine §8 → Discovery Brief
  ↓
逐段（Context / Goals / Design / Alternatives）走 engine §7 Q&A loop
   每段 approval 前展示 §9.3 checklist 表
  ↓
Phase 3 Step 1 写完整 doc + 末尾追加 <!-- provenance --> + § 5 Open Questions
       Step 2 逐段呈现 + §9.3 checklist + approval grammar
  ↓
Phase 4 写到 ~/Code/wiki，提交 PR，issue 评论附 Next-step 提示（不调用 decompose）
```

### 3.4 Trade-offs

| 决策 | 取舍 | 理由 |
|---|---|---|
| 选方案 A（engine 集中） | 改 engine 新增 3 节 vs 在两个 SKILL 内联（B）vs 新建独立 qa-loop SKILL（C） | A 与 engine "shared logic" 的现状一致；未来 decompose/pr 可复用；改动定位清晰，不影响 G001 编号。 |
| HARD-GATE 全开（无 escape hatch） | 用户体验 vs 防"agent 自我合理化"跳过 | 现有 design-doc 已采用此风格；wiki 强调"反幻觉"；放宽则失去价值。 |
| Discovery Triad 三项全跑 | 起手成本 vs 信息完备 | git log 几乎无成本；Glob/Grep 只跑 ≤ 5 关键词；Read README/CLAUDE.md 阅读量小。收益（消除幻觉）远大于成本。 |
| 中英规范"中文叙述+英文术语" | 一致性 vs 母语友好 | 与现有 beaver-issue 模板一致（"目标 / 验收标准"中文）；术语保留英文避免歧义。 |
| Bug 模式 + Feature QA 同批做 | 工作量 vs 与 wiki 一致 | wiki Phase 2 显式要求 Bug 模板 + p/0 跳 triage + @CODEOWNERS；分两次做易遗漏；同批做避免后续重写 issue Create 流程。 |
| design-doc → decompose 仅提示不自动跳 | 无缝 vs 越界 | wiki Phase 3→4 是人工切换；自动跳超出本次需求；提示文本足以引导。 |
| 不绑定 superpowers 插件版本 | 复用 vs 独立 | 把规则文本下沉到 beaver-engine 而非通过 `Skill('superpowers:brainstorming')` 调用，避免插件版本耦合（用户决策）。 |
| size/S 走"轻 QA"（3 问），size/L 走"重 QA"（4 维分段 approval） | 速度 vs 一致性 | size/S 本身就是 fast track；强迫 4 段 approval 与 SOP 矛盾；按 size 路由保持精神一致同时尊重粒度差异。 |

### 3.5 Test Strategy

无运行时；通过对 SKILL.md 文本与示例对话的人工 walkthrough 验证：

1. **Discovery Brief 必出**：在 issue Create 与 design-doc Phase 2 起手，walkthrough 确认 SKILL.md 包含执行 §8 的明确指令与"Discovery Brief"输出模板。
2. **HARD-GATE 文本可定位**：grep 三个 SKILL.md 是否含 "Approved? (y/revise)" / "approve / ok / 继续 / y"。
3. **Provenance 块**：design-doc 模板末尾必含 `<!-- provenance` 标记。
4. **Bug 模式分支**：issue SKILL.md 包含 type 检测 → Bug submode 跳转，含强制 size/S、p0/blocker 跳 triage、cc @CODEOWNERS 步骤。
5. **JSON 合规**：`marketplace.json` / `plugin.json` 不需修改；用 `python -c "import json; json.load(open(...))"` 验证。
6. **mock 走查**：用一个真实的小 size/S issue 创建场景人工跑一遍 issue 流程，确认 §8 Brief、§7 Q&A、§9.4 checklist 都被触发。

### 3.6 Deployment & Dependencies

- **部署方式**：纯文本改动，无需 release。修改后通过 git commit 进入插件仓库；用户通过 `/plugin update` 拉取最新版本即可生效。
- **依赖**：保持现有依赖（`gh` CLI、`git`）。不引入对 `superpowers` 插件的运行时依赖（仅借鉴文本规则，下沉到 beaver-engine）。
- **兼容性**：engine §1-6 编号不变；G001-G006 引用不变；其它 4 个 beaver SKILL（pr / focus / report / audit / decompose）本次不改、行为不变。

## 4. Alternatives Considered

**方案 B — 在两个 SKILL 中各自内联**
- 优点：不动 engine，影响面最小；改动可独立 review。
- 劣势：未来 decompose/pr 复用时需复制；与"engine 是共享底座"架构相左；维护时多处需同步。
- 拒绝原因：用户在澄清问题中明确选 "SKILL.md + 共享子模块"。

**方案 C — 把 QA loop 拆成新独立 SKILL `beaver-qa-loop`**
- 优点：复用粒度最大，可被任何调用方 `Skill()` 触发。
- 劣势：与 beaver-engine "internal engine, do not trigger directly" 模式重复；用户可能误调；Skill 加载链变深；对 Claude Code 的 Skill tool 调用模型增加一层。
- 拒绝原因：与现有 engine 角色冲突，且本仓库未定义"内部子 SKILL 之间互相调用"的模式。

**方案 D — 直接 `Skill('superpowers:brainstorming')` 调用**
- 优点：复用最强，无需复制规则。
- 劣势：绑定 superpowers 插件及其版本；脱离本插件自洽性；wiki 明确说"从 Superpowers 迁移"——是迁移而非依赖。
- 拒绝原因：与"迁移"语义不符；引入跨插件版本耦合。

## 5. Open Questions

- 是否要为 §7 approval grammar 增加多语言变体（日语 "はい"、英文 "lgtm"）？当前先支持中英主流词，后续按使用反馈扩展。owner: 实现者；解决时机: 实现 Phase 1 完成后。
- §8 D2 关键词抽取若 issue title 全是中文，`Grep` 的命中率可能低；是否需要 Python 分词？当前先按字面 token 抽取（中文按"/"和空格切），观察实际效果。owner: 实现者；解决时机: 第一次实际 issue Create 后回顾。
- 未来 `beaver-decompose` 是否也接入 §7/§8/§9？本次设计已为可复用做好准备，但留作后续 Goal。

<!-- provenance
- "current beaver-issue is one-shot 6-field collect" ← plugins/beaver/skills/beaver-issue/SKILL.md L33-46
- "current beaver-design-doc has 4-section Q&A + HARD-GATE + Sectional Review" ← plugins/beaver/skills/beaver-design-doc/SKILL.md L73-141
- "engine has 6 sections, G001-G006 guardrails" ← plugins/beaver/skills/beaver-engine/SKILL.md L10-167
- "wiki: TDD 驱动开发与 Review 能力从 Superpowers 迁移而来" ← ~/Code/wiki/docs/onboarding/project-management.md L71
- "wiki Phase 2 Bug 模式：type/bug 强制 size/S、p/0-blocker 直 in-progress、@CODEOWNERS" ← ~/Code/wiki/docs/onboarding/project-management.md L88-103, L130-132
- "wiki Phase 3→4 是人工切换" ← ~/Code/wiki/docs/onboarding/project-management.md L134-156
- 用户 QA 决策（共 8 轮）：方案 A / 硬闸 HARD-GATE / 三项必跑 / 中英规范+反幻觉+checklist / 按 size 区分强度 / 同时加 Bug 模式 / 只提示不自动跳 / approve 全部 5 个 section
-->
