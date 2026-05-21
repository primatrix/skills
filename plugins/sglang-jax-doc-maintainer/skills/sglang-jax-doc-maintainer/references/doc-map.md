# sglang-jax 文档映射

本文件用于候选文档召回和文档漂移检查。路径映射只负责召回，不证明文档一定需要编辑；做出更新决策前必须阅读当前代码、测试和当前文档。

## 路径规范化

- 接受绝对路径、workspace 相对路径和 code repo 相对路径。
- code path 匹配前去掉 code repository root。
- wiki 文档验证阶段使用 wiki repo root 相对路径，例如 `docs/projects/sglang-jax/03-scheduler.md`；用户报告中可同时保留 workspace 相对路径 `wiki/docs/projects/sglang-jax/03-scheduler.md`。
- code path 去掉 package prefix `python/sglang/` 或 `python/sgl_jax/`。
- 使用规范化后的路径匹配，例如 `srt/managers/scheduler.py`、`srt/layers/attention/backend_selector.py` 或 `server_args.py`。

## 正向映射：代码路径到候选文档

| 代码区域 | 主要文档 | 次要文档 |
|---|---|---|
| `srt/entrypoints/**` | `02-entrypoints-and-tokenization.md` | `01-architecture-overview.md`, `13-configuration-reference.md` |
| `srt/managers/tokenizer_manager.py` | `02-entrypoints-and-tokenization.md` | `03-scheduler.md` |
| `srt/managers/detokenizer_manager.py` | `02-entrypoints-and-tokenization.md` | `01-architecture-overview.md` |
| `srt/managers/scheduler.py` | `03-scheduler.md` | `01-architecture-overview.md` |
| `srt/managers/schedule_batch.py` | `03-scheduler.md` | `04-model-executor.md` |
| `srt/managers/schedule_policy.py` | `03-scheduler.md` | `07-kv-cache.md` |
| `srt/model_executor/**` | `04-model-executor.md` | `01-architecture-overview.md`, `13-configuration-reference.md` |
| `srt/models/**` | `05-models.md` | `01-architecture-overview.md` |
| `srt/layers/**` | `06-layers-and-attention.md` | `08-pallas-kernels.md` |
| `srt/layers/attention/**` | `06-layers-and-attention.md` | `07-kv-cache.md`, `08-pallas-kernels.md` |
| `srt/mem_cache/**` | `07-kv-cache.md` | `03-scheduler.md`, `13-configuration-reference.md` |
| `srt/kernels/**` | `08-pallas-kernels.md` | `06-layers-and-attention.md`, `11-quantization.md`, `09-speculative-decoding.md` 按 symbol / caller 决定 |
| `benchmark/kernels/**` | `08-pallas-kernels.md` | 仅 benchmark 结论有证据时更新 |
| `srt/speculative/**` | `09-speculative-decoding.md` | `08-pallas-kernels.md` |
| `srt/lora/**` | `10-lora.md` | `07-kv-cache.md` |
| `srt/configs/quantization_config.py` | `11-quantization.md` | `13-configuration-reference.md` |
| `srt/utils/quantization/**` | `11-quantization.md` | `08-pallas-kernels.md` |
| `srt/kernels/quantized_matmul/**` | `11-quantization.md` | `08-pallas-kernels.md` |
| `srt/multimodal/**` | `12-multimodal.md` | `01-architecture-overview.md` |
| `srt/server_args.py` or `server_args.py` | `13-configuration-reference.md` | 搜索受影响 flag 的 feature docs、entrypoint docs 和调用点 |
| `sgl_jax/global_config.py` | `13-configuration-reference.md` | 搜索受影响配置的 feature docs 和调用点 |
| `srt/configs/**` | `13-configuration-reference.md` | 搜索受影响配置、默认值和使用点 |
| `.github/workflows/**` | 默认不更新架构文档 | 仅当开发者文档已描述该行为时更新 |
| `test/**` or `tests/**` | 默认不更新架构文档 | 仅当测试揭示已文档化行为变化时更新相关文档 |

## 当前主文档范围

默认维护 `wiki/docs/projects/sglang-jax/` 下：

- `01-architecture-overview.md`
- `02-entrypoints-and-tokenization.md`
- `03-scheduler.md`
- `04-model-executor.md`
- `05-models.md`
- `06-layers-and-attention.md`
- `07-kv-cache.md`
- `08-pallas-kernels.md`
- `09-speculative-decoding.md`
- `10-lora.md`
- `11-quantization.md`
- `12-multimodal.md`
- `13-configuration-reference.md`

01-13 之外文件默认不得新增或修改，除非用户明确授权。

## 反向索引：文档到权威代码来源

| 文档 | 主要权威来源 | 常见符号 / 配置 | 相关验证 |
|---|---|---|---|
| `01-architecture-overview.md` | managers、model_executor、entrypoints、mem_cache、layers 的系统边界 | subsystem 名称、跨模块数据流、feature family | 与 02-13 的术语、链接和抽象层级一致 |
| `02-entrypoints-and-tokenization.md` | `srt/entrypoints/**`, tokenizer/detokenizer managers | server entrypoint、tokenizer manager、detokenizer manager | 请求入口和 token 流程是否仍一致 |
| `03-scheduler.md` | `scheduler.py`, `schedule_batch.py`, `schedule_policy.py` | `Scheduler`, `ScheduleBatch`, scheduling policy, prefill/decode | 职责边界、队列流和 token budget 位置 |
| `04-model-executor.md` | `srt/model_executor/**`, model runner, worker | model executor、runner、parallelism | 执行边界、batch 输入、并行策略 |
| `05-models.md` | `srt/models/**`, model configs | model class、loader、architecture mapping | 模型加载路径和权重分片 |
| `06-layers-and-attention.md` | `srt/layers/**`, `srt/layers/attention/**` | `AttentionBackend`, backend selector, layer classes | backend 列表、接口和选择路径 |
| `07-kv-cache.md` | `srt/mem_cache/**`, scheduler cache use | KV cache、req-to-token pool、page / block | cache 生命周期和 scheduler 交互 |
| `08-pallas-kernels.md` | `srt/kernels/**`, kernel benchmarks | Pallas kernel、attention / matmul kernels | kernel entry、调用方和 benchmark 证据 |
| `09-speculative-decoding.md` | `srt/speculative/**` | draft model、verify、speculative path | 推测解码生命周期 |
| `10-lora.md` | `srt/lora/**` | LoRA manager、adapter、weight path | adapter 加载和缓存 |
| `11-quantization.md` | quantization configs、quantized kernels、utils | quantization config、quantized matmul | config 与 kernel 路径一致 |
| `12-multimodal.md` | `srt/multimodal/**`, model-specific multimodal code | image/video processor、multimodal input | 输入处理和模型路径 |
| `13-configuration-reference.md` | `server_args.py`, config classes, global config | ServerArgs fields、defaults、env vars | 默认值、CLI 暴露、调用点 |

反向索引用于验证文档事实是否仍有权威代码来源。若文档陈述找不到对应来源，输出 drift 或 unknown，不要编造解释。

## 候选扩展规则

正向映射后必须继续检查：

- Symbol search：搜索新增或修改的 class、function、config flag、backend name 是否出现在 01-13 文档中。
- Neighbor search：读取 caller/callee、配置入口、测试覆盖点和默认值使用位置。
- Doc graph search：读取候选文档的 cross-link、overview 和 sibling concept 文档。

配置变更必须额外检查：声明位置、默认值、CLI / ServerArgs 暴露、环境变量、schema / config class、downstream call sites、提到该配置的 feature docs。

删除、移动或 rename 必须检查旧文件名、旧 symbol 和旧链接是否仍出现在 01-13 文档中。

## 置信度

- high：代码事实、文档命中和 symbol / config 匹配同时存在。
- medium：路径命中且语义相关，但还需要读取邻近代码确认。
- low：只有 changed-file-list、目录名相似或 PR 描述线索。
- unknown：base/diff 缺失、证据冲突、代码无法访问或文档陈述无法回查。

low 或 unknown 不得直接编辑；先请求 diff、代码上下文，或让用户补充项目事实依据。用户确认编辑计划只授权执行范围，不会提高项目事实本身的证据等级。
