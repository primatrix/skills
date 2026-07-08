# 文档来源反向索引

本文件用于从文档陈述回查权威代码来源，发现 drift、重复和过期陈述。它不是完整架构说明，只是验证入口。

| 文档 | 责任边界 | 权威代码来源 | 常见 drift 信号 |
|---|---|---|---|
| `01-architecture-overview.md` | 系统分层、核心子系统、横向 taxonomy | managers、model_executor、mem_cache、layers、entrypoints | 子文档新增 feature family 但 overview 无入口；术语不一致 |
| `02-entrypoints-and-tokenization.md` | 请求入口、tokenization、detokenization | `srt/entrypoints/**`, `tokenizer_manager.py`, `detokenizer_manager.py` | entrypoint 或 manager 职责移动；token 流程过期 |
| `03-scheduler.md` | 调度、队列、batch、prefill/decode 决策 | `scheduler.py`, `schedule_batch.py`, `schedule_policy.py` | Scheduler 与 ScheduleBatch 职责描述错位 |
| `04-model-executor.md` | 执行边界、runner、worker、parallelism | `srt/model_executor/**` | executor 接收数据或并行策略过期 |
| `05-models.md` | 模型加载、模型结构、权重路径 | `srt/models/**`, model configs | model family 或 loader 路径过期 |
| `06-layers-and-attention.md` | layers、attention backend、selector | `srt/layers/**`, `srt/layers/attention/**` | backend 列表、接口或选择逻辑过期 |
| `07-kv-cache.md` | KV cache、memory pool、request-token mapping | `srt/mem_cache/**`, scheduler cache usage | cache 生命周期或 pool 类型过期 |
| `08-pallas-kernels.md` | Pallas kernel、kernel 调用与限制 | `srt/kernels/**`, kernel benchmarks | kernel 入口、调用方或 benchmark 结论过期 |
| `09-speculative-decoding.md` | speculative decode 生命周期 | `srt/speculative/**` | draft/verify 路径或配置过期 |
| `10-lora.md` | LoRA adapter 加载、缓存、执行路径 | `srt/lora/**` | adapter 生命周期或 cache 交互过期 |
| `11-quantization.md` | quantization config、kernel、模型路径 | quantization config、utils、quantized kernels | config 与 kernel 支持不一致 |
| `12-multimodal.md` | multimodal input 和模型处理路径 | `srt/multimodal/**`, model-specific multimodal code | processor 或输入路径过期 |
| `13-configuration-reference.md` | ServerArgs、global config、默认值、env | `server_args.py`, config classes, global config | flag 名称、默认值、CLI 暴露或调用点过期 |

## 使用方法

1. 从候选文档中提取受影响陈述。
2. 用本表找到权威代码入口。
3. 搜索文档中的文件名、类名、函数名、配置名和术语是否仍存在。
4. 若文档陈述无法回查到代码或用户确认，标记 unknown 或 drift。
5. 只有证据充足时才计划修改；证据不足时询问用户。
