---
name: gcloud-support-case
description: >-
  Use when a production job on Google Cloud (Cloud TPU, GKE, or GCS) is failing
  or stuck and the user needs to open or check a Google Cloud support case —
  e.g. "open a support case", "开个 case", "提交 GCP 工单", TPU node pool stuck,
  GKE workload failing, GCS upload/download errors. Covers listing existing
  cases, viewing case details, and creating new ones via the Cloud Support API v2.
---

# 开 Google Cloud Support Case（Cloud TPU / GKE / GCS）

## Overview

Google Cloud 没有 `gcloud support cases create` 这个子命令。开 case 走 **Cloud Support API v2**（`cloudsupport.googleapis.com`），`gcloud` 只负责出 access token。本 skill 把团队约定固化下来，引导你完成：**环境预检 → 收集信息 → 选 classification → 用户确认（真实/测试）→ 提交**。其中**可选项（产品 / 优先级 / 项目）开场一次性弹框选完**、确认门也用**对话框**（`AskUserQuestion`），标题/描述等自由文本集中收集（见〈交互方式〉）。

聚焦三个产品：**Cloud TPU、GKE、GCS**。

**核心原则：绝不替用户拍板 priority；自动推荐 `classification.id` 但必须经用户在对话框确认；绝不直接提交未经用户确认的 case；最终必须让用户明确选「真实 case / 测试 case」（默认真实，`testCase: false`）。**

## 交互方式：开场一次性选完，自由文本集中放最后

把「点选类」问题**合并成一次 `AskUserQuestion` 调用**（该工具单次支持多个问题，用户一屏选完），别拆成多轮、别一个个等用户打字。顺序固定：

**A. 开场对话框（一次性，多问题合并）—— 第 ① 步之前就弹**：

1. **受影响产品**：Cloud TPU / GKE / GCS。
2. **优先级 priority**（必问，绝不自己定）：列 **P1 / P2 / P3 / P4** 四个选项（团队约定不开 P0），`description` 写清含义与紧急程度。正好 4 项、贴合 `AskUserQuestion` 上限，无需 Other。
3. **目标项目**：列两个选项——**`tpu-for-training`（标 `(默认)`）** + **「其它项目」**（提示用户选后填 `PROJECT_ID`，或直接用「Other」）。不能只列 1 个,否则报 `too_small`（见下）。

> 这三问**放在同一个 `AskUserQuestion` 调用里**一起弹，用户一次选完。**每个问题的 `options` 数组必须是 2–4 项**（工具硬限制：少于 2 报 `too_small`，多于 4 报 `too_big`）。所以：priority 正好列 P1–P4 四项；只有一个默认值的问题（如项目）也要补第 2 个选项（如「其它项目」），别只放 1 个。`(默认)` 之外的取值都可由用户选「Other」自填。不要因为「想先 `gcloud` 预检」就推迟——预检命令在用户选择期间/之后跑都行。

**B. 自由文本集中收集（开场选完之后，用普通对话一次性问）**：标题 + 描述所需的现象/影响/报错/复现等，**列成一个清单一次性向用户要**，别挤牙膏分多轮问。日志、报错、复现步骤无法做成选项，故用自由文本。

**C. 两道确认门（流程中按需弹对话框）**：

- **分类确认门**（第 ③ 步）：推荐分类作**第一个选项并标 `(推荐)`** + 备选 + 「都不对，重新查询」。
- **最终确认门**（第 ④ 步）：让用户明确这是**真实 case（默认，`testCase: false`，路由给工程师）/ 测试 case（`testCase: true`，不路由）/ 取消**。

用户在任一对话框都可选「Other」自行输入。

> 对话框只能由主对话弹给用户；这是交互呈现方式，不改变流程纪律——该问的仍要问、该确认的仍要确认。

## 团队固定约定（不要每次问用户，直接用）

| 字段 | 固定值 |
|---|---|
| `PROJECT_ID`（默认项） | `tpu-for-training`（开场对话框里作为默认/首选项，用户可改） |
| `timeZone` | `Asia/Shanghai` |
| `languageCode` | `zh-CN` |
| `subscriberEmailAddresses` | `wangyunpeng@google.com`、`peishiuanwu@google.com` |

## 每次必须向用户询问的内容

**点选类——合并进开场那一个 `AskUserQuestion`（见〈交互方式 A〉），一次问完：**

- **受影响产品**：Cloud TPU / GKE / GCS（据此 + 故障症状自动推荐 `classification.id`，再用对话框让用户确认，见第 ③ 步）。
- **`priority`（P1–P4，团队约定不开 P0）—— 永远问用户，绝不自己判断**。四个选项正好贴合对话框上限。各选项 `description` 建议写明：
  - **P1**：关键功能受损，影响重大，最高紧急
  - **P2**：生产受影响但有 workaround（最常见）
  - **P3**：影响较小、非紧急
  - **P4**：咨询 / 最低优先级
- **`PROJECT_ID`**：默认项 `tpu-for-training`（标 `(默认)`），用户可改。

**自由文本——开场选完后集中一次问（见〈交互方式 B〉）：**

- **问题标题与描述**的具体内容，用于拼 `displayName`（纯英文）/ `description`（中英双语，见下）。

## displayName 只写英文；description 中英双语（英文在前、中文在后）

这是硬性格式要求：

- `displayName`：**只写英文**，单行简洁标题（不写中文、不加 `/ 中文`）。
- `description`：**保持中英双语**，先写完整英文，再写等价中文，用 `[EN]` / `[CN]` 分段：

```text
[EN] Objective: ...
Business impact: ...
Error details (commands, logs, errors): ...
Observed vs expected: ...
Environment (orchestration, framework, TPU type, repro steps): ...

[CN] 目标：……
业务影响：……
错误详情（命令、日志、报错）：……
实际 vs 预期：……
环境（编排、框架、TPU 型号、复现步骤）：……
```

## 输出格式：case 详情必须渲染后再展示

任何把 case 信息展示给用户的场合——② 列出已有 case、⑤ 提交后的结果反馈、后续查询单个 case 或评论——**禁止直接粘贴 API 返回的原始 JSON 或一整坨未分段文本**。先解析字段，再按下面的 Markdown 规范渲染：**关键信息加粗、段落之间留空行、链接可点击**。

**单个 case 详情**（提交结果、查询详情）按此模板渲染：

```markdown
### Case 123456 — Fix TPU node pool stuck in PROVISIONING

**状态**：OPEN　　**优先级**：P2　　**测试 case**：否

**分类**：Google Kubernetes Engine > Set up GKE clusters

**创建时间**：2026-07-14 10:32（Asia/Shanghai）　　**更新时间**：2026-07-14 11:05（Asia/Shanghai）

**控制台链接**：[在 Cloud Console 中打开](https://console.cloud.google.com/support/cases/detail/123456?project=tpu-for-training)

**问题描述**：

> [EN] Objective: ...
>
> [CN] 目标：……
```

硬性要求：

- **字段名一律加粗**（`**状态**：`、`**优先级**：`……），字段值紧跟其后；相关字段可并排一行，不相关字段之间**空一行**。
- `description` / 评论正文用引用块（`>`）包裹，**保留原文的段落换行**——多段之间在引用块里留 `>` 空行，绝不压成一行。
- 时间戳转成 `Asia/Shanghai` 的人类可读格式（`2026-07-14 10:32`），别直接给 `2026-07-14T02:32:11.123456Z` 这种 RFC3339 原串。
- Console 链接渲染成**可点击的 Markdown 链接**，不裸贴 URL 长串。
- **多个 case 的列表**（② 场景）用表格呈现：列固定为 **Case 编号 / 标题 / 优先级 / 状态 / 创建时间**，编号列直接做成指向 Console 的链接；表格后空一行再写你的结论（例如「其中 #123456 与本次故障同类，确认是否仍要新开？」）。
- 展示前先挑重点：用户没要求时不必罗列全部字段，但**编号、标题、状态、优先级、链接**五项永远要有。

## 效率：别让流程变慢

慢通常来自三件事，按下面做能显著提速：

- **token 只取一次**：`gcloud auth print-access-token` 较慢，**不要每条 curl 都现取**。开场存一次环境变量，后续复用：

  ```bash
  export TOKEN=$(gcloud auth print-access-token)
  export PROJECT_ID=tpu-for-training                     # 默认项；或用户在对话框改的值
  # 之后所有 curl 用：--header "Authorization: Bearer $TOKEN"
  ```

- **能并行就并行**：开场对话框弹给用户的同时，后台跑预检/列已有 case（互不依赖的命令放同一批一起发），别一条跑完再发下一条。
- **`gcloud services enable` 只在首次需要**：服务已启用时这步是多余的慢调用。可先跳过，仅当某次 API 报 `SERVICE_DISABLED` 时再执行。

## 流程

### ① 环境预检（可与开场对话框并行）

```bash
gcloud config list                                   # 确认 account / project
# 仅当后续 API 报 SERVICE_DISABLED 时才需要执行下面这行（已启用就别重复跑）：
# gcloud services enable cloudsupport.googleapis.com --project=$PROJECT_ID
```

> 调用 API 时**必须**带 `x-goog-user-project: $PROJECT_ID` 头，否则 ADC 用户凭证会因缺少 quota project 报 403。

### ② 先列出已有 case（避免重复开）

```bash
# 注意：过滤用 state=OPEN（不是 state!=CLOSED，后者会 400 INVALID_ARGUMENT）；
# --get + --data-urlencode 负责把 filter 正确 URL 编码。
curl -s \
  --header "Authorization: Bearer $TOKEN" \
  --header "x-goog-user-project: $PROJECT_ID" \
  --get --data-urlencode 'filter=state=OPEN' \
  "https://cloudsupport.googleapis.com/v2/projects/$PROJECT_ID/cases"
```

把未关闭的同类 case **按〈输出格式〉渲染成表格**告诉用户，确认是否真的要新开。

### ③ 自动推荐 classification.id，并用对话框让用户确认

**这一步必须主动推荐，但绝不替用户最终锁定。** 流程是「自动选一个 → 说清理由 → 弹对话框让用户确认」：

1. **自动推荐**：结合「受影响产品」+ 故障症状关键词，按下方〈症状 → 分类推荐〉表选一个最贴合的 `classification.id`（优先用快速参考里已解析好的 ID）。
2. **解释理由**：告诉用户推荐了哪个 `displayName`、为什么（命中了哪个症状）。
3. **独立确认门（对话框）**：用 `AskUserQuestion` 弹出，**不要**把它混进第 ④ 步整体请求体确认里。选项布局：
   - 第一个选项 = 推荐分类的 `displayName`，标注 `(推荐)`；
   - 其后 1–2 个 = 最可能的备选分类；
   - 最后 = 「都不对，重新查询」。
   - 用户选**推荐或备选** → 锁定该 `id`，进入 ④。
   - 用户选**重新查询**，或用「Other」描述了新方向 → 用下方命令实时查询候选，把结果作为新选项再弹一次对话框，直到锁定。

**绝不凭空编造 ID，API 会拒绝。** 快速参考都不贴合时实时查询：

```bash
curl -s \
  --header "Authorization: Bearer $TOKEN" \
  --header "x-goog-user-project: $PROJECT_ID" \
  "https://cloudsupport.googleapis.com/v2/caseClassifications:search?query=display_name:\"*Cloud TPU*\""
# 其它产品：display_name:"*Kubernetes Engine*" / display_name:"*Cloud Storage*"
```

**症状 → 分类推荐**（命中关键词即推荐对应子类，仍需用户在对话框确认）：

| 受影响产品 | 故障症状关键词 | 推荐子类 |
|---|---|---|
| GKE | node pool 创建/删除卡住、集群创建失败、节点起不来、cluster lifecycle | Set up GKE clusters |
| GKE | Pod/JobSet hang、workload 跑不起来、调度失败、容器 OOM/Crash | Deploy and Manage Workloads |
| GKE | 集群升级、扩缩容、节点配置、成本/性能调优 | Manage and Optimize Clusters |
| GKE | TPU/GPU 训练推理工作负载本身（GKE 上） | GKE AI/ML |
| GKE | 监控、指标、日志、告警相关 | Monitor |
| Cloud TPU | TPU VM/节点创建、删除、状态异常、维护事件 | Manage TPUs |
| Cloud TPU | 训练/推理任务 hang、掉卡、XID、性能异常 | Monitoring and Troubleshooting / Training and Inference |
| Cloud TPU | 部署 TPU 工作负载、排队/调度 | Deploy TPU workloads |
| GCS | 上传失败/慢、下载失败/慢、对象读写报错 | Manage objects :: Upload/Download objects |
| GCS | 请求重试、5xx、超时、限流 | Making requests :: Request retry strategy |
| GCS | bucket 配置、元数据、生命周期 | Manage storage buckets :: Bucket metadata and configuration |
| GCS | bucket 监控、用量、指标 | Monitoring :: Bucket monitoring |

> 表中症状有重叠或跨多个子类时（例如「node pool 卡住」既像 lifecycle 又像 workload），**推荐其一并把另一个作为备选明确列出**，由用户在确认对话框里定夺。

### ④ 复述请求体，用对话框确认「真实 case 还是测试 case」

先把最终请求体（标题、描述、priority、classification displayName）原样复述给用户，然后用 `AskUserQuestion` 弹出**最终确认门**——核心是让用户明确这是**真实 case 还是测试 case**。选项：

- **真实 case（`(推荐 / 默认)`，`testCase: false`）**：会正式路由给 **Google 工程师**处理。`testCase` 默认就是 `false`。
- **测试 case（`testCase: true`）**：**不会路由给工程师**，仅用于走通流程 / 演练。
- **取消**（不提交）。

（若用户还想改标题/描述/优先级/分类，可选「Other」说明，回到对应步骤改完再确认。）

**默认值是 `testCase: false`（真实 case）。** 未在此对话框得到用户明确选择之前，绝不发起 ⑤ 的提交请求。

### ⑤ 提交

按用户在第 ④ 步的选择设置 `testCase` 字段后提交（**默认 `false` = 真实 case**；用户选「测试」才用 `true`）：

```bash
curl -s --request POST \
  --header "Authorization: Bearer $TOKEN" \
  --header "x-goog-user-project: $PROJECT_ID" \
  --header 'Content-Type: application/json' \
  --data '{
    "displayName": "<English title only / 纯英文标题>",
    "description": "<[EN] ... \n\n[CN] ...>",
    "classification": { "id": "<已确认的分类 id>" },
    "priority": "<P1–P4，来自用户>",
    "timeZone": "Asia/Shanghai",
    "languageCode": "zh-CN",
    "subscriberEmailAddresses": ["wangyunpeng@google.com", "peishiuanwu@google.com"],
    "testCase": false
  }' \
  "https://cloudsupport.googleapis.com/v2/projects/$PROJECT_ID/cases"
```

提交后**必须**从返回里取出 `name`（形如 `projects/<PROJECT_ID>/cases/123456`），并**按〈输出格式〉的单 case 模板**把提交结果渲染给用户——case 编号、状态、优先级、分类、**可点击的 Cloud Console 链接**一项不少，方便后续查询。

可用一行命令直接从响应里提取编号并拼好链接（把上面的提交响应存到 `$RESP`）：

```bash
CASE_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin)["name"].split("/")[-1])')
echo "Case: $CASE_ID"
echo "Link: https://console.cloud.google.com/support/cases/detail/${CASE_ID}?project=${PROJECT_ID}"
```

### 后续操作

```bash
# 追加评论
curl -s --request POST \
  --header "Authorization: Bearer $TOKEN" \
  --header "x-goog-user-project: $PROJECT_ID" \
  --header 'Content-Type: application/json' \
  --data '{ "body": "..." }' \
  "https://cloudsupport.googleapis.com/v2/projects/$PROJECT_ID/cases/<CASE_ID>/comments"

# 关闭 case
curl -s --request POST \
  --header "Authorization: Bearer $TOKEN" \
  --header "x-goog-user-project: $PROJECT_ID" \
  "https://cloudsupport.googleapis.com/v2/projects/$PROJECT_ID/cases/<CASE_ID>:close"
```

查询/追评后向用户展示 case 状态或评论内容时，同样**按〈输出格式〉渲染**。

## 产品分类 ID 快速参考

> 解析时间 2026-06-18。若 API 返回 `INVALID_ARGUMENT` 或分类报错，按 ③ 重新查询并更新本表。

**Cloud TPU**（`Other Google Cloud Products > Cloud TPU > …`）

| 子类 | classification.id |
|---|---|
| Manage TPUs | `584L8PB3D1N6IOR1DGP10GRCDTQM8NQKA1ALUCPO70RJ0C9Q296M2RJ1CTILUL2GALPLUDPJ6SQ38DI2204024GC8TNMUPRCCKG46R3FELI0` |
| Monitoring and Troubleshooting | `584L8PB3D1N6IOR1DGP10GRCDTQM8NQKA1ALUCPO70RJ0C9Q5H374OBDCLRMUSJBEDFIGL35DPPMUSJ6DHNNEB2VA1SN8RRICDK2ONQA85C2INPL60R38D1H8880G08I1H3MURR7DHII0GRCDTQM8` |
| Training and Inference | `584L8PB3D1N6IOR1DGP10GRCDTQM8NQKA1ALUCPO70RJ0C9Q256NAR3KD5PMOQB3CLFJIDPI6GO3CGGG100H4327DTNMER35411MORRLCG` |
| Deploy TPU workloads | `584L8PB3D1N6IOR1DGP10GRCDTQM8NQKA1ALUCPO70RJ0C9Q4926AS3CDTSLUL2GALFNERRIDDM6UOB4EDFMURIV8T5KANPJ60R32CPJ8880G08I1H3MURR7DHII0GRCDTQM8` |

**Google Kubernetes Engine**（`Google Kubernetes Engine > …`）

| 子类 | classification.id |
|---|---|
| Deploy and Manage Workloads | `584L8PB3D1N6IOR1DGP1GIRLC9IN4RJ5EHIN6NQ5DPJMIRJ5BSR30C1J74SJK8I4CLO6ORRPBTGMSP2V9LGMSOB7CLFLERRIDDM6UOB4EDFJACHM6SSJ6GGG100H4327DTNMER35411MORRLCG` |
| Set up GKE clusters | `584L8PB3D1N6IOR1DGP1GIRLC9IN4RJ5EHIN6NQ5DPJMIRJ5BSR30C1J74SJK6IJCLQ5UTBGBT3KMHAVCDM7ASRKCLP76NPH6OQJ0CPL8880G08I1H3MURR7DHII0GRCDTQM8` |
| Manage and Optimize Clusters | `584L8PB3D1N6IOR1DGP1GIRLC9IN4RJ5EHIN6NQ5DPJMIRJ5BSR30C1J74SJK8QDC5N62PR5BTGMSP2V9TO78QBDD5T6ANQ3DHQN6T35E9PLUCHN74SJGE22204024GC8TNMUPRCCKG46R3FELI0` |
| GKE AI/ML | `584L8PB3D1N6IOR1DGP1GIRLC9IN4RJ5EHIN6NQ5DPJMIRJ5BSR30C1J74SJK7A4CLO6ORRPBT0KIBQD9HFLERRIDDM6UOB4EDFJAC9L60PJEGGG100H4327DTNMER35411MORRLCG` |
| Monitor | `584L8PB3D1N6IOR1DGP1GIRLC9IN4RJ5EHIN6NQ5DPJMIRJ5BSR30C1J74SJK3IDDTN6IT3FE9FJCD1J6CRJGGGG100H4327DTNMER35411MORRLCG` |

**Cloud Storage**（`Other Google Cloud Products > Cloud Storage > …`）

| 子类 | classification.id |
|---|---|
| Making requests :: Request retry strategy | `584L8PB3D1N6IOR1DGP18GRCDTQM8NQJEHNN4OB7CLFJ4C9O6CO32EHG9LGMMQBECTFN4PBHELIN6T3JBST3KNQICLONAPBJEHFN4PBKE9SLUSRKE9GN8PB7F5FJ6D9M64PJEGGG100H4327DTNMER35411MORRLCG` |
| Manage objects :: Upload objects | `584L8PB3D1N6IOR1DGP18GRCDTQM8NQJEHNN4OB7CLFJ4C9O6CO32EHE8PN48EIVADIN4TJ9DPJJKNQ5E9P6USJJBSK6SRRE5LGNAT38BTP6AR31EHIM8AAV70OJGC1J6D1102012864ERRFCTM6A823DHNNAP0` |
| Manage objects :: Download objects | `584L8PB3D1N6IOR1DGP18GRCDTQM8NQJEHNN4OB7CLFJ4C9O6CO32EHF9TH6KPB3EHFKQOBEC5JMARB5DPQ5UEHQBT26UTREDHNM2P39DPJLUJR2D9IM6T3JBSSJ8C1N6SR444080490OHRFDTJMOP908DM6UTB4` |
| Manage storage buckets :: Bucket metadata and configuration | `584L8PB3D1N6IOR1DGP18GRCDTQM8NQJEHNN4OB7CLFJ4C9O6CO32EI29LGMSOB7CLFN6T3FE9GMEPAVC9QM6QR5EHPLUEHQBT17AORBCLQ5URB5EHGM8OBKC5FM2RJ4BTHMURJ6D5JNASJ1EHKMURIV6KS32E9K651102012864ERRFCTM6A823DHNNAP0` |
| Monitoring :: Bucket monitoring | `584L8PB3D1N6IOR1DGP18GRCDTQM8NQJEHNN4OB7CLFJ4C9O6CO32EH59LIN8PBID5N6EEIV9LNMSQBKDTP6IRJ7BSJ5UJ3FCTJMIRJ7BSPJEE9I70P444080490OHRFDTJMOP908DM6UTB4` |

## Common Mistakes

| 错误 | 正确做法 |
|---|---|
| 自己判断 priority 并直接填进去 | **永远问用户** P1–P4，不设默认 |
| `displayName` 里掺了中文 | 标题**只写英文**；中文只放进 `description` |
| `description` 只写英文（或只写中文） | `description` 必须**中英双语，英文在前、中文在后** |
| 不向用户确认就直接提交 | 提交前必须复述请求体 + 弹**真实/测试**确认门 |
| 提交后只回个 case 编号、不给链接 | 反馈 case 编号 + **可点击的 Console 链接**（`…/support/cases/detail/<ID>?project=<PROJECT_ID>`） |
| 把 API 返回的原始 JSON / 一坨未分段文本直接甩给用户 | 按〈输出格式〉渲染：**字段名加粗、段落空行、链接可点击**，列表用表格 |
| case 描述压成一行、时间戳裸贴 RFC3339 原串 | 描述用引用块保留段落换行；时间转 `Asia/Shanghai` 人类可读格式 |
| 编造一个 `classification.id` | 用快速参考，或实时 `caseClassifications:search` |
| 自动选了分类却没单独让用户确认 | 推荐 + 说理由后，**单独弹对话框**确认分类 |
| 把分类确认混进整体请求体确认里一笔带过 | 分类要有**独立**的确认门（对话框），先于第 ④ 步 |
| 产品/优先级/分类/提交用纯文本提问、等用户打字 | 这四处**一律用 `AskUserQuestion` 对话框**让用户点选 |
| 产品/优先级/项目拆成多轮对话框逐个问 | **开场合并成一个 `AskUserQuestion`**（多问题）一次选完 |
| priority 列出 P0 或超过 4 项 → `too_big`/无效 | priority **只列 P1–P4 四项**（不开 P0，P0 也无法经 API 设置） |
| 某问题只列 1 个选项（如项目只放 `tpu-for-training`）→ `too_small` 报错 | 每问**至少 2 个选项**；单默认值也要补第 2 项（如「其它项目」） |
| 列 case 用 `filter=state!=CLOSED` → 400 INVALID_ARGUMENT | 用 `--get --data-urlencode 'filter=state=OPEN'` |
| 标题/描述分多轮挤牙膏地问 | 开场选完后**列一个清单一次性**收集自由文本 |
| 每条 curl 都 `$(gcloud auth print-access-token)` 现取 token | 开场 `export TOKEN=…` **取一次复用**；互不依赖的命令并行发 |
| 每次都跑 `gcloud services enable`（已启用仍重复） | 默认跳过，仅当 API 报 `SERVICE_DISABLED` 时才执行 |
| 把标题/描述硬塞进对话框选项 | 标题/描述用**自由文本对话**收集，不做成选项 |
| 忘记 `x-goog-user-project` 头 → 403 | 每个 curl 都带该头 |
| 把 `timeZone` 填成 `-07:00`、漏掉 `languageCode` | 固定 `Asia/Shanghai` + `zh-CN` |
| 用 `<teammate@...>` 占位抄送人 | 固定两位 subscriber 邮箱 |
| 假设存在 `gcloud beta support cases create` 子命令 | 用 Cloud Support API v2 + curl |

## Red Flags — 停下来重做

- 我自己定了 priority（没问用户）
- `displayName` 标题里掺了中文（标题应只写英文）
- `description` 不是中英双语，或中文在前（应英文在前、中文在后）
- 还没向用户复述请求体、没弹「真实/测试」确认门就发起提交
- 我没让用户明确选「真实 case / 测试 case」就替他定了 `testCase`
- 我编了一个看起来像那么回事的 `classification.id`
- 我自动选了分类，但没单独让用户确认就往下走了
- 产品 / 优先级 / 分类确认 / 最终提交，我用的是纯文本提问而不是对话框（`AskUserQuestion`）
- 产品/优先级/项目我拆成了多轮对话框，而不是开场一次性合并问完
- 我把 case 详情以原始 JSON 或一坨未分段文本丢给了用户，没按〈输出格式〉渲染（字段加粗、段落空行、链接可点击）
- 我在多条 curl 里反复 `$(gcloud auth print-access-token)` 现取 token，而不是开场存一次复用
- 我给某个对话框问题列的选项数不在 2–4 之间（>4 报 `too_big`、<2 报 `too_small`；项目只列 1 个会炸）
- priority 我列了 P0，或没按 P1–P4 来（P0 无法经 API 设置）
- 我用 `state!=CLOSED` 去过滤 case（会 400；正确是 `state=OPEN`）

以上任意一条出现 → 退回对应步骤，按流程重做。
