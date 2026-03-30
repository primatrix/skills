---
name: xprof-profiling-analysis
description: Use when analyzing TPU/XLA profiling data (xprof, trace.json.gz, op_stats, xplane), understanding HLO op performance, backward pass structure (gmm/tgmm), MFU calculation for MoE models, communication bottleneck identification, or comparing GPU vs TPU training performance
---

# TPU/XLA Profiling 分析

TPU 训练性能分析的完整方法论：从 trace 数据解析到性能瓶颈定位和优化路径推导。

## 数据源

xprof profiling 数据位于 `tensorboard/plugins/profile/<timestamp>/` 下：

| 文件 | 用途 |
|------|------|
| `*.trace.json.gz` | Timeline 事件（算子耗时、通信、HLO 分类） |
| `*.op_stats_v2.pb` | 算子级统计（protobuf） |
| `*.xplane.pb` | 完整算子数据（protobuf，含硬件 peak、model_flops、tf_op） |
| `*.memory_viewer.json` | 显存分配时间线 |

**⚠️ trace.json.gz 有 1,000,000 event 硬上限**——大模型训练（GA>1、MoE）单 device 可产生 800K+ events，trace 会被严重截断（`dropped_traces` 字段标记丢弃数）。**必须先检查 trace 是否完整**：若 `Complete (X) events == 1,000,000`，则数据被截断，必须改用 xplane.pb。

**数据源优先级**：
1. **xplane.pb**（推荐）——完整数据，含硬件 peak spec、model_flops、tf_op、source_stack
2. **trace.json.gz**——仅当 event 数 < 1M 时可用，JSON 易解析但可能截断

## Trace 解析

### 事件结构

```python
# trace.json.gz 中的 traceEvents
{
  "ph": "X",           # Complete event
  "pid": 12,           # Process ID
  "ts": 1000000,       # Timestamp (us)
  "dur": 500,          # Duration (us)
  "name": "tgmm.42",   # Op name
  "args": {
    "hlo_category": "custom-call",
    "long_name": "...",           # HLO 指令全名，含 shape、replica_groups 等
    "bytes_accessed": 939524096,
    "model_flops": 137438953472,  # 算子 FLOPs（Pallas kernel 可能报 0）
    "tf_op": "jit(train_step)/...",  # JAX call stack，含层名、fwd/bwd 信息
    "source": "path/to/file.py:123"  # Python 源码位置
  }
}
```

### 定位 TPU 设备

```python
# 从 metadata events 找 TPU device PIDs（排除 SparseCore）
for e in events:
    if e.get('ph') == 'M' and e.get('name') == 'process_name':
        name = e['args']['name']
        if '/device:TPU:' in name and 'SparseCore' not in name:
            tpu_pids[e['pid']] = name
# 取 TPU:0 做分析
tpu0_pid = min(tpu_pids.keys())
```

### Step Time 提取

搜索 `train_step` 事件获取总 step time，再按算子分类累计计算/通信/空闲时间。

**跨 device 一致性检查**：对所有 TPU device（TPU:0 ~ TPU:N）提取 step time 并比较。方差大（>5%）说明存在 load imbalance（常见于 MoE expert routing 不均匀或 PP stage 划分不均）。方差小（<1%）说明负载均衡良好。

### 从 trace 提取并行配置

通信算子的 `long_name` 包含 `replica_groups` 和 tensor shape，可直接推断集群配置：

```python
# 示例：AllGather long_name
# "all-gather... bf16[157184,16]{...} → bf16[157184,2048]{...}, replica_groups=[1,128]<=[128]"
# → 128 devices in one group, FSDP=128 (16→2048 = 128x expansion)

# 示例：AllToAll long_name
# "all-to-all... bf16[128,2,4096,16]{...}, replica_groups=[1,128]<=[128], dimensions={0}"
# → EP 通过 AllToAll 在 128 devices 间路由

# 解析方法：
for e in step_ops:
    long_name = e.get('args', {}).get('long_name', '')
    if 'replica_groups=' in long_name:
        # 提取 replica_groups 推断集群大小和通信拓扑
        # [1,128] 表示 1 组 128 devices
        # [4,32] 表示 4 组各 32 devices
```

| 信息 | 提取来源 |
|------|---------|
| 集群总 device 数 | `replica_groups` 中的最大值 |
| FSDP degree | AllGather shape 扩展倍数（如 16→2048 = 128x） |
| EP degree | AllToAll `replica_groups` 中每组 device 数 |
| TP degree | AllReduce `replica_groups` 的分组方式 |
| model_dim | AllGather 全量维度（如 2048） |
| vocab_size | Embedding AllGather 第一维（如 157184） |

### 利用 tf_op 字段识别层和 fwd/bwd

`tf_op` 字段包含完整的 JAX call stack，比算子名更可靠地判断层归属和 fwd/bwd：

```python
# 示例 tf_op 值：
# Forward GMM:
#   "jit(train_step)/jvp(TransformerLinenPure.apply)/TransformerLinenPure/decoder/moe_layers_0/shard_map/jit(gmm)/pallas_call:"
# Backward GMM:
#   "jit(train_step)/transpose(jvp(TransformerLinenPure.apply))/TransformerLinenPure/decoder/moe_layers_0/..."

# 关键规则：
# - 含 "transpose(jvp(...))" → backward pass
# - 含 "jvp(...)" 但不含 "transpose" → forward pass
# - "moe_layers_N" → 第 N 个 MoE 层
# - "mtp_block" → Multi-Token Prediction 模块
```

## HLO 算子分类

XLA 编译后的算子名不保留 JAX 语义，需按 `hlo_category` + 算子名规则分类：

| 高级类别 | hlo_category / 算子名特征 | 含义 |
|---------|--------------------------|------|
| `matmul` | `convolution fusion` | 矩阵乘（XLA 用 convolution 表示 matmul） |
| `attention` | 名称含 `splash_mha`, `flash_attention` | Attention kernel |
| `communication` | 名称含 `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`（包括其 `async-start`/`async-done` 变体） | 集合通信 |
| `custom_kernel` | `custom-call`（但排除 GMM/TGMM/splash_mha） | Pallas kernel、offload 等 |
| `custom_fusion` | `custom fusion` | XLA 自定义融合（含 gather/scatter_custom_fusion） |
| `elementwise_fusion` | `loop fusion`, `non-fusion elementwise` | 逐元素运算融合 |
| `data_formatting` | `data formatting` | Tensor layout 转换（FSDP sharding 常引发大量 copy） |
| `routing` | `sort` | MoE TopK routing |

**通信分类注意事项**：名称含 `all-reduce`/`all-gather` 等关键字的 `async-done` 事件（如 `all-reduce.1725.cloned.1.call-done`）**必须归入 `communication`**，不要单独分为 `async_comm`。这些 async-done 事件的 duration 就是 compute engine 等待通信完成的 **stall 时间**，是衡量 exposed communication cost 的核心指标。

### 关键算子识别

| 算子名模式 | 含义 | 阶段 |
|-----------|------|------|
| `convolution_bitcast_fusion.*` | MatMul（forward 或 backward dlhs） | fwd/bwd |
| `tgmm.*` | MoE 权重梯度（Transposed Grouped Matrix Multiply） | **仅 backward** |
| `splash_mha_fwd_*` | SplashAttention forward | forward |
| `splash_mha_dkv_*` | SplashAttention dK/dV 梯度 | backward |
| `splash_mha_dq_*` | SplashAttention dQ 梯度 | backward |
| `ragged-all-to-all` | MoE expert routing 通信（EP 场景） | fwd/bwd |

## Backward Pass 结构（MoE 模型）

### GMM / TGMM 关系

MoE 层的 forward 用 `gmm()`（grouped matrix multiply），backward 通过 `jax.custom_vjp` 拆分为：

```
Forward:  Y = gmm(X, W)       # 分组矩阵乘

Backward:
  dlhs = gmm(dY, W^T)         # 激活梯度 → convolution_bitcast_fusion
  drhs = tgmm(X^T, dY)        # 权重梯度 → tgmm.*
```

- **dlhs（激活梯度）**：在 critical path 上，各层串行依赖（dlhs_N → dlhs_{N-1} → ... → dlhs_1）
- **drhs（权重梯度）**：各层独立，只依赖 incoming gradient dY_i（= dlhs_{i+1}）

### MoE 层数推断

从 GMM/TGMM 算子数量可直接推断 MoE 层数：

```
每个 MoE 层有 2 个 expert matmul（如 gate_proj/up_proj + down_proj），
Forward: 2 GMMs/layer, Backward dlhs: 2 GMMs/layer, Backward drhs: 2 TGMMs/layer

MoE_layers = GMM_count / 4    # 2 matmuls × (fwd + bwd_dlhs)
MoE_layers = TGMM_count / 2   # 2 matmuls × bwd_drhs

例：120 GMM + 60 TGMM → 30 MoE layers
```

类似地，`splash_mha_fwd` 的数量 = attention layer 数量。结合 `tf_op` 中的 `moe_layers_N` 可验证。

### XLA 调度特征

Timeline 中 backward 通常呈现**两次循环**：
1. 第一轮：所有层的 dlhs（激活梯度链，串行）
2. 第二轮：所有层的 drhs/tgmm（权重梯度，可并行）

XLA scheduler 延迟 tgmm 的原因：
- **优先 critical path**：dlhs 链是串行依赖，优先连续执行
- **通信友好**：drhs 需要 ReduceScatter，集中执行便于通信 overlap
- 注意：延迟 tgmm 不一定是显存最优——逐层计算 dlhs+drhs 可以更早释放 dY_i 和 X_i

### Forward/Backward 判别

XLA HLO 名称不含 "backward"/"gradient" 等语义标签。判别方法（按可靠性排序）：

**方法 1：`tf_op` 字段（最可靠）**
- 含 `transpose(jvp(...))` → backward pass
- 含 `jvp(...)` 但不含 `transpose` → forward pass
- 可同时提取层编号（如 `moe_layers_0`）

**方法 2：算子名模式（快速判断）**
- `tgmm.*` → **一定是 backward**（权重梯度）
- `splash_mha_dkv_*`, `splash_mha_dq_*` → backward attention
- `splash_mha_fwd_*` → forward attention
- 其余 matmul（`convolution_bitcast_fusion`）→ 仅靠名称无法区分，需用 `tf_op` 或 HLO graph

## MFU 计算

### 公式

```
MFU = F_useful / (Peak_cluster × T_step)

F_useful = GBS × seq_len × FLOPs_per_token(fwd+bwd)
Peak_cluster = num_devices × peak_flops_per_device
```

### 从 trace 提取 FLOPs

trace 中每个算子的 `args.model_flops` 字段记录了该算子的 FLOPs。可直接累加：

```python
total_flops = sum(int(e['args'].get('model_flops', 0)) for e in step_ops)
```

**注意**：`model_flops` 的值可能是 int 或 string 类型，需做类型转换。

**Pallas kernel 的 model_flops 为 0**：SplashAttention、GMM、TGMM 等 Pallas custom-call 的 `model_flops` 通常报 0，但实际上它们有大量计算。需要注意：
- GMM/TGMM 的 `model_flops` 在较新版本中已正确填充（可从 trace 验证）
- SplashAttention 的 `model_flops` 仍然报 0，需手动估算或从模型配置计算
- 因此从 trace 直接累加的 FLOPs 可能**低估**实际计算量，MFU 会偏低

### MoE 模型注意事项

MoE 的 per-token 有效 FLOPs 远低于同规模 dense 模型（因为每 token 只激活 top-k experts）：

```
例：256 experts top-8, 17.43B params
  → per-token FLOPs(fwd) ≈ 3.25 GFLOPs
  → per-token FLOPs(fwd+bwd) ≈ 9.75 GFLOPs
  → 同规模 dense 模型约为 6×N FLOPs/token ≈ 105 GFLOPs
```

MoE MFU 数值天然偏低（~10%），这是模型架构特性，不代表硬件利用率差。**跨架构对比必须用 MFU**，不能用 per-chip throughput（设备数不同时无法对比）。

### GPU 基线 MFU 计算

```python
gpu_tokens_per_sec = tokens_per_day / 86400
gpu_peak = num_gpus × peak_flops_bf16_dense  # 用 dense bf16，不含 sparsity
gpu_mfu = gpu_tokens_per_sec × flops_per_token_fwdbwd / gpu_peak
```

## 通信分析

### 通信原语对应关系

| 原语 | 并行策略 | 用途 |
|------|---------|------|
| AllGather | FSDP | 收集分片权重 |
| ReduceScatter | FSDP | 聚合并分发梯度 |
| AllReduce | DP（非 FSDP） | 梯度同步 |
| AllToAll | EP（Expert Parallelism） | Expert routing 数据交换 |

### 瓶颈特征

- **FSDP 瓶颈**：AllGather + ReduceScatter 耗时高，随 FSDP degree 增大恶化
- **EP 瓶颈**：AllToAll 耗时高（EP=16 场景可占 39% step time）

### 通信 overlap 判断

**关键概念**：单个 TPU device 上所有算子是**串行执行**的。通信 overlap 发生在通信引擎（ICI/DCN）与计算引擎（TensorCore）之间。在 trace timeline 上：

- `async-start` 事件：通信发起，duration 通常极短（~0μs）
- 通信在后台执行，compute engine 继续运行其他算子
- `async-done` 事件：compute engine 需要通信结果时触发
  - **duration ≈ 0**：通信已完成，无 stall → 通信被有效 overlap
  - **duration > 0**：compute engine 在等待通信完成 → 这就是 **exposed communication cost**

**正确的 overlap 度量**：

```python
# 统计 async-done 的总 stall 时间
for comm_type in ['all-reduce', 'all-gather', 'reduce-scatter']:
    done_ops = [e for e in step_ops
                if comm_type in e['name'].lower() and 'done' in e['name'].lower()]
    stall = sum(e['dur'] for e in done_ops)
    print(f"{comm_type} exposed stall: {stall/1000:.2f} ms")
```

**错误做法**：不要试图在单 device timeline 上做"区间重叠"分析来判断 overlap，因为单 device 上所有事件本就是串行的。

### Data Formatting 分析

`data formatting` 类算子是 tensor layout 转换（copy 操作），常见原因：
- FSDP sharding 导致权重 layout 与计算 layout 不一致，需反复 copy
- AllGather 后的 reshape/transpose

当 data formatting 占比 >5% 时值得关注：
- 检查 top copy ops 的 `bytes_accessed`，确认是否是大 tensor 的重复 copy
- 检查 sharding spec 是否合理，能否减少 layout 转换

## XLA Flags 配置检查

通信 overlap 效果取决于 XLA flags 配置。当发现通信未被有效 overlap 时，应检查 launch script 中的 `LIBTPU_INIT_ARGS`。

### 关键 flags 及作用

**Continuation Fusion（CF）**——将 async 通信与计算融合到一个 fusion 中执行：

| Flag | 作用 |
|------|------|
| `xla_tpu_enable_async_collective_fusion=true` | 启用 async collective fusion pass |
| `xla_tpu_enable_async_collective_fusion_fuse_all_gather=true` | AllGather 的 CF |
| `xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true` | AllReduce 的 CF |
| `xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` | ReduceScatter 的 CF |
| `xla_tpu_enable_async_collective_fusion_multiple_steps=true` | 跨多步 fusion |
| `xla_tpu_overlap_compute_collective_tc=true` | TensorCore 上 overlap compute/comm |
| `xla_enable_async_all_gather=true` | 启用 async AllGather |
| `xla_enable_async_all_reduce=true` | 启用 async AllReduce |

**SparseCore Offloading**——v7x 推荐方案，将通信 offload 到 SparseCore 执行：

| Flag | 作用 |
|------|------|
| `xla_tpu_enable_sparse_core_collective_offload_all_gather=true` | AllGather 走 SparseCore |
| `xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true` | ReduceScatter 走 SparseCore |
| `xla_tpu_enable_sparse_core_collective_offload_all_reduce=true` | AllReduce 走 SparseCore |

**注意：CF 和 SparseCore 对同一通信原语互斥**。例如用 SparseCore offload AllReduce 时，须设 `fuse_all_reduce=false`。

**Data Parallel Overlap**——优化 DP AllReduce pipeline：

| Flag | 作用 |
|------|------|
| `xla_tpu_enable_data_parallel_all_reduce_opt=true` | 优化 DCN AllReduce |
| `xla_tpu_data_parallel_opt_different_sized_ops=true` | 支持不同大小 op 的 pipeline |

### 常见配置问题

| 症状 | 可能的 flags 缺失 |
|------|------------------|
| AllReduce async-done stall 大 | 缺少 `fuse_all_reduce=true` 或 SparseCore offload |
| AllGather async-done stall 大 | 缺少 `fuse_all_gather=true` 或 SparseCore offload |
| 所有通信都未 overlap | 缺少 `async_collective_fusion=true` 基础 flag |
| MoE 模型通信未优化 | 缺少 `DATA_PARALLEL_OVERLAP` flags |
| VMEM 不足导致 overlap 失败 | `xla_tpu_scoped_vmem_limit_kib` 设置过低 |

### 检查路径

1. Launch script 中的 `export LIBTPU_INIT_ARGS="..."`
2. 项目中的 `xla_flags_library.py`（如 MaxText benchmarks/ 目录）
3. 对比同类模型 benchmark config 使用的 flags 组合

## Roofline 与 System Bound

### Roofline（单设备上限）

```
Ridge Point = Peak_FLOPS / HBM_BW (FLOPs/byte)
  TPU v7x: 2307T / 7380 GB/s ≈ 313 FLOPs/byte

算子 AI > Ridge Point → compute-bound (MFU≈100%)
算子 AI < Ridge Point → memory-bound (MFU 由带宽决定)
```

Roofline MFU 是混合算子的加权 MFU 上限（无通信、无并行开销）。

### System Bound（多设备理论值）

通过 parallelism analyzer 搜索最优并行配置：
- 遍历 TP × DP × PP × EP × FSDP × CP 组合
- 对每种组合计算：计算时间 + 通信时间（考虑 overlap）+ 显存约束
- 输出最优配置及其 MFU

关键规律：
- **PP 减少 FSDP 通信**：PP=8 将模型切到 8 段，每段独立做小规模 FSDP/DP
- **大 GBS 摊薄通信**：GBS↑ → micro_batch 更大 → 同样通信下计算更多
- **Remat 影响 micro_batch 大小**：save_all 占显存多 → micro_batch 小 → GA 轮次多

## 分析报告结构

推荐简洁实用的 5 段结构：

1. **背景与分析思路**：目标、方法（profiling + 理论 + GPU 对比）、环境
2. **当前优化效果**：配置对比表（step time / compute / comm / MFU）+ 关键结论 + 图表
3. **理论分析**：MFU 瀑布（Roofline → System Bound → Actual）+ Top-N 最优并行策略
4. **剩余工作**：按优先级列出待完成 profiling 任务
5. **优化路径**：分阶段措施 + 预期效果

图表类型：step time 堆叠柱状图、算子分布饼图、Top-N 算子条形图、通信分解图、MFU 瀑布图。

## 常见陷阱

| 陷阱 | 正确做法 |
|------|---------|
| XLA 的 `convolution` = CNN 卷积 | XLA 用 convolution 表示所有 matmul |
| hlo_category 可直接判断 fwd/bwd | 不能，需要结合算子名模式（tgmm, splash_mha_d*）或 `tf_op` 字段 |
| MFU 低 = 硬件利用率差 | MoE 模型 MFU 天然低，需理解 per-token FLOPs |
| 用 per-device throughput 跨集群对比 | 不同集群规模下必须用 MFU 对比 |
| FSDP degree 越大通信开销不变 | FSDP=128 的 AllGather/ReduceScatter 远大于 FSDP=16 |
| 延迟 tgmm 一定省显存 | 不一定，逐层做 dlhs+drhs 可以更早释放中间张量 |
| 在单 device timeline 上做区间重叠分析来判断通信 overlap | 单 device 上所有事件是串行的，应看 async-done 的 duration 来衡量 exposed comm |
| `model_flops` 累加 = 真实 FLOPs | Pallas kernel（SplashAttention）的 model_flops 报 0，累加值会低估 |
| 通信未 overlap = 需要优化算法 | 先检查 XLA flags 配置（CF/SparseCore/DATA_PARALLEL_OVERLAP），往往是 flags 缺失 |
| `async_comm` 是独立于通信的类别 | 通信的 async-done 事件（如 `all-reduce.*.call-done`）就是通信类别，其 duration = stall 时间 |
| trace.json.gz 包含完整数据 | **trace.json.gz 有 1M event 硬上限**，大模型（GA>1、MoE）单 device 可有 800K+ events，trace 会截断。检查 `Complete (X) events` 是否 = 1,000,000，若是则必须用 xplane.pb |

## xplane.pb 解析

当 trace.json.gz 被截断时，使用 xplane.pb 作为数据源：

```python
from tensorflow.tsl.profiler.protobuf import xplane_pb2

xspace = xplane_pb2.XSpace()
with open('*.xplane.pb', 'rb') as f:
    xspace.ParseFromString(f.read())

# 找 TPU:0 plane
tpu0 = [p for p in xspace.planes if p.name == '/device:TPU:0'][0]

# 元数据是 map 类型
event_meta = {k: tpu0.event_metadata[k] for k in tpu0.event_metadata}
stat_meta = {k: tpu0.stat_metadata[k].name for k in tpu0.stat_metadata}

# XLA Ops line 包含主要算子
xla_line = [l for l in tpu0.lines if l.name == 'XLA Ops'][0]

# 事件时间：offset_ps (picoseconds), duration_ps
for ev in xla_line.events:
    meta = event_meta[ev.metadata_id]
    name = meta.display_name or meta.name  # display_name 对应 trace 中的 name
    full_name = meta.name                   # 对应 trace 中的 long_name
    offset_us = ev.offset_ps / 1e6
    dur_us = ev.duration_ps / 1e6
    # hlo_category, model_flops, tf_op 等在 meta.stats 中
    for s in meta.stats:
        sname = stat_meta[s.metadata_id]
        # s.str_value, s.int64_value, s.double_value 等

# 硬件信息在 plane-level stats 中
# peak_teraflops_per_second, peak_hbm_bw_gigabytes_per_second, device_type_string
for s in tpu0.stats:
    sname = stat_meta[s.metadata_id]
    # 如 device_type_string="TPU v7x", peak_teraflops_per_second=1028.75

# 检查 dropped_traces
# dropped_traces > 0 说明 trace.json.gz 不完整
```

### xplane.pb vs trace.json.gz 字段映射

| xplane.pb | trace.json.gz |
|-----------|---------------|
| `meta.display_name` | `event.name` |
| `meta.name` | `event.args.long_name` |
| `meta.stats[hlo_category]` | `event.args.hlo_category` |
| `meta.stats[model_flops]` | `event.args.model_flops` |
| `meta.stats[tf_op]` | `event.args.tf_op` |
| `ev.offset_ps / 1e6` | `event.ts`（微秒） |
| `ev.duration_ps / 1e6` | `event.dur`（微秒） |
