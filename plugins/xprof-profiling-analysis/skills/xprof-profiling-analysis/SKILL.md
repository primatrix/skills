---
name: xprof-profiling-analysis
description: Use when analyzing TPU/XLA profiling data (xprof, trace.json.gz, op_stats, xplane), understanding HLO op performance, backward pass structure (gmm/tgmm), MFU calculation for MoE models, communication bottleneck identification, HBM memory analysis (peak composition, buffer categorization, optimization targets), or comparing GPU vs TPU training performance
---

# TPU/XLA Profiling 分析

TPU 训练性能分析的完整方法论：从 trace 数据解析到性能瓶颈定位和优化路径推导。

## 统一分析工具 `xprof`

`tpu-profiling/scripts/xprof.py` 是统一的分析 CLI，覆盖所有 profiling 需求：

```bash
# 查看 run 目录下有哪些数据文件
xprof discover --run-dir ci-prof-run115

# ── 显存分析 ──
xprof memory peak           --run-dir ci-prof-run115    # 峰值 HBM 组成
xprof memory diagnose        --run-dir ci-prof-run115    # 异常诊断（尖刺/长生命周期/overlap）
xprof memory runtime         --run-dir ci-prof-run115    # 运行时 HBM（xplane BFC）
xprof memory theory                                      # 理论显存估算（bottom-up）
xprof memory compare-theory  --run-dir ci-prof-run115    # 理论 vs 实际对比

# ── 计算分析 ──
xprof compute breakdown      --run-dir ci-prof-run115    # 算子耗时分类
xprof compute mfu            --run-dir ci-prof-run115 --gbs 5120 --num-chips 128  # MFU
xprof compute phases         --run-dir ci-prof-run115    # fwd/bwd/optimizer 阶段分解
xprof compute layers         --run-dir ci-prof-run115    # 逐层耗时（含 fwd/bwd 拆分）
xprof compute operators      --xplane *.xplane.pb         # 算子级 roofline 分析（优化目标识别）

# ── 通信分析 ──
xprof comm overlap           --run-dir ci-prof-run115    # 通信 stall 分析
xprof comm primitives        --run-dir ci-prof-run115    # 通信原语分解（AG/RS/AR/A2A）
xprof comm balance           --run-dir ci-prof-run115    # 跨 device 负载均衡

# ── 跨 run 对比 ──
xprof compare --runs 115,183                             # A/B 性能对比

# ── 配置审计 ──
xprof audit flags --flags-file launch.sh                 # XLA flags 优化检查
```

所有命令支持 `--json` 输出结构化数据，`--output` 指定报告路径，`--run-dir` 自动发现数据文件。

### xprof 命令与数据源对应

| 命令 | 输入数据 | 输出 |
|------|---------|------|
| `memory peak` | memory_viewer.json | 峰值 buffer 分类、top buffers、vocab-sized 优化目标 |
| `memory diagnose` | memory_viewer.json | 尖刺检测、buffer overlap、lifetime 异常、phase 曲线、优化建议 |
| `memory runtime` | xplane.pb | BFC allocator 真实 HBM（reserved - available） |
| `memory theory` | 模型配置（硬编码） | 单设备理论显存（无并行/无 remat） |
| `memory compare-theory` | memory_viewer.json + 模型配置 | 理论 vs 实际差异分析 |
| `compute breakdown` | xplane.pb | 算子类别耗时（matmul/gmm/attention/stall/...） |
| `compute mfu` | xplane.pb | MFU、tokens/sec、TFLOPS/chip |
| `compute phases` | xplane.pb (tf_op) | forward/backward/optimizer 阶段耗时 |
| `compute layers` | xplane.pb (tf_op) | 逐层 fwd+bwd 耗时（可定位慢层） |
| `compute operators` | xplane.pb | 算子级 roofline 分析：per-category 利用率、top-N 算子、优化目标排序 |
| `comm overlap` | xplane.pb | async-done stall 分析（exposed communication cost） |
| `comm primitives` | xplane.pb | AllGather/ReduceScatter/AllReduce/AllToAll 分解 |
| `comm balance` | 多个 xplane.pb | 跨 device step time 方差 |
| `compare` | 多个 xplane.pb | 跨 run 性能对比表 |
| `audit flags` | launch script / flags 文件 | XLA flags 缺失/冲突检查 |

## 数据源

xprof profiling 数据位于 `tensorboard/plugins/profile/<timestamp>/` 下：

| 文件 | 用途 |
|------|------|
| `*.trace.json.gz` | Timeline 事件（算子耗时、通信、HLO 分类） |
| `*.op_stats_v2.pb` | 算子级统计（protobuf） |
| `*.xplane.pb` | 完整算子数据（protobuf，含硬件 peak、model_flops、tf_op） |
| `*.memory_viewer.json` | 显存峰值组成分析（从 HLO heap simulator 生成） |
| `*.hlo_proto.pb` | HLO 编译产物（含 buffer_assignment、heap_simulator_trace） |

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

## HBM 显存分析

### 数据源与优先级

显存分析有三层数据源，从轻到重：

| 数据源 | 适用场景 | 优势 | 限制 |
|--------|---------|------|------|
| `memory_viewer.json` | **首选**：峰值组成、优化目标识别 | JSON 格式易解析，含完整 buffer 分类 | 静态分析，不含运行时 aliasing 效果 |
| xplane.pb `/host:CPU` | 运行时真实 HBM 用量 | 反映 XLA 运行时 buffer reuse | 仅有 BFC allocator 汇总数据 |
| `*.hlo_proto.pb` | 最详细的 buffer 级分析 | 完整 heap simulator replay + op_name 元数据 | 文件大（几百 MB），解析慢 |

**工具**：使用统一 CLI `xprof`（见上方命令表）。

```bash
# 峰值组成
xprof memory peak --run-dir ci-prof-run115

# 异常诊断（尖刺、长生命周期 buffer、overlap）
xprof memory diagnose --run-dir ci-prof-run115

# 运行时 BFC allocator
xprof memory runtime --run-dir ci-prof-run115

# 理论 vs 实际
xprof memory compare-theory --run-dir ci-prof-run115
```

### memory_viewer.json 解析

TensorBoard memory_viewer 插件从 HLO buffer_assignment 生成，是峰值分析的首选数据源。

**数据结构**：

```python
{
  "heapSizes": [float, ...],           # 每条 HLO 指令处的堆大小（MiB），timeline 数据
  "unpaddedHeapSizes": [float, ...],   # 不含 XLA padding 的堆大小
  "maxHeap": [                          # 峰值时刻所有存活 buffer（按 timeline 顺序）
    {
      "logicalBufferSizeMib": 32.0,    # buffer 大小
      "shapeString": "f32[8,2048,512]{2,1,0:T(8,128)}",  # 含 layout
      "tfOpName": "state.params['params']['decoder']...",  # JAX op 路径（分类关键）
      "instructionName": "param.21790", # HLO 指令名
      "groupName": "Parameter",         # "Parameter" | "Temporary" | ""
      "opCode": "parameter",            # HLO opcode
    }, ...
  ],
  "maxHeapBySize": [...],              # 同 maxHeap 但按大小排序
  "logicalBufferSpans": {              # buffer 生命周期 {id: {start, limit}}
    "505278": {"start": 54184, "limit": 57711}, ...
  },
  "peakHeapMib": 90167.9,             # 峰值堆大小（MiB）
  "peakUnpaddedHeapMib": 89890.3,     # 不含 padding 的峰值
  "entryComputationParametersMib": 6239.4,  # 入口参数（params + optimizer）
  "peakHeapSizePosition": 9335,        # 峰值在 heapSizes 中的位置
  "indefiniteLifetimes": [...],        # 无限生命周期 buffer（通常是参数）
}
```

**关键分析方法**：

```python
import json

with open('memory_viewer.json') as f:
    data = json.load(f)

peak_gib = data['peakHeapMib'] / 1024
params_gib = data['entryComputationParametersMib'] / 1024
padding_mib = data['peakHeapMib'] - data['peakUnpaddedHeapMib']

# 峰值 buffer 分类（按 tfOpName 路径判断）
for buf in data['maxHeap']:
    op = buf.get('tfOpName', '')
    group = buf.get('groupName', '')  # "Parameter" vs "Temporary"
    size = buf['logicalBufferSizeMib']
    # 用 op path 判断：params、optimizer、logits、attention、moe 等

# 峰值位置分析：peak 发生在 schedule 的哪个阶段
peak_pos = data['peakHeapSizePosition']
total = len(data['heapSizes'])
print(f"Peak at {peak_pos}/{total} = {peak_pos/total*100:.1f}% through schedule")
```

**Buffer 分类规则**（按 `tfOpName` / `groupName`）：

| 分类 | 判断规则 | 典型形状 |
|------|---------|---------|
| Parameters | `groupName=="Parameter"` 且 `state.params` | 各种权重 shape |
| Optimizer (mu/nu) | `groupName=="Parameter"` 且 `opt_state` | 同权重 shape，f32 |
| Logits/CE | `logits_dense` / `cross_entropy` / `softmax` | `[B,T,vocab]` |
| MoE experts | `moe` / `expert` / `gmm` / `router` | `[num_experts,dim,dim]` |
| Attention/MLA | `attention` / `mla` / `wq_` / `wkv_` | `[B,T,H,D]` 等 |
| MTP | `mtp` / `multi_token` | 含 vocab-sized logits |
| Gradient (bwd) | `transpose(jvp(...)` | 同 fwd 激活 shape |
| Forward act. | `jvp(...)` 无 `transpose` | 各种激活 shape |

### XPlane 运行时显存

xplane.pb 的 `/host:CPU` plane 记录 XLA BFC allocator 事件。**关键陷阱**：

```
bytes_reserved  = 82.51 GiB   ← 总 HBM 池大小
bytes_allocated = 6.97 GiB    ← 仅动态 I/O buffer（参数 + optimizer 入口）
bytes_available = 5.26 GiB    ← 池中空闲空间

⚠️ bytes_allocated ≠ 实际 HBM 用量！
实际 HBM = bytes_reserved - bytes_available = 77.25 GiB
```

`bytes_allocated` 只跟踪传入编译程序的**动态 I/O buffer**（参数和优化器状态的入口 buffer），**不包含**编译程序的内部工作空间（~70 GiB）——该工作空间由 XLA buffer assignment 静态分配，作为单个大 allocation 在池内管理。

```python
from tensorflow.tsl.profiler.protobuf import xplane_pb2

xspace = xplane_pb2.XSpace()
with open('*.xplane.pb', 'rb') as f:
    xspace.ParseFromString(f.read())

# 找 /host:CPU plane
host_plane = [p for p in xspace.planes if '/host:CPU' in p.name][0]
stat_names = {sid: sm.name for sid, sm in host_plane.stat_metadata.items()}

# 提取 BFC allocator 事件
peak_reserved = 0
min_available = float('inf')
for line in host_plane.lines:
    for event in line.events:
        for stat in event.stats:
            name = stat_names.get(stat.metadata_id, '')
            if name == 'bytes_reserved':
                peak_reserved = max(peak_reserved, stat.int64_value)
            elif name == 'bytes_available':
                min_available = min(min_available, stat.int64_value)

actual_hbm = peak_reserved - min_available  # 这才是真实 HBM 用量
```

### HLO Proto Heap Simulator Replay

最详细的分析方法，从 HLO buffer assignment 完整重放堆模拟器：

```python
from tensorflow.compiler.xla.service import hlo_pb2

hlo_proto = hlo_pb2.HloProto()
with open('*.hlo_proto.pb', 'rb') as f:
    hlo_proto.ParseFromString(f.read())

ba = hlo_proto.buffer_assignment
mod = hlo_proto.hlo_module

# 1. 参数分配（固定内存）
for alloc in ba.buffer_allocations:
    if alloc.is_entry_computation_parameter and not alloc.is_tuple:
        # alloc.size = 参数大小（已含 FSDP sharding）
        # 通过 assigned logical buffer → instruction → metadata.op_name 获取 JAX 路径

# 2. Heap simulator replay
main_trace = None
for trace in ba.heap_simulator_traces:
    if trace.whole_module_simulation:
        main_trace = trace  # 找最大的 whole-module trace

# 重放 ALLOC(0)/FREE(1)/SHARE_WITH(2) 事件
current_size = 0
peak_size = 0
for ev in main_trace.events:
    if ev.kind == 0:    # ALLOC
        buf_size = lb_by_id[ev.buffer_id].size
        current_size += buf_size
        if current_size > peak_size:
            peak_size = current_size
            # 记录峰值快照
    elif ev.kind == 1:  # FREE
        current_size -= alloc_buffers[ev.buffer_id]
    elif ev.kind == 2:  # SHARE_WITH
        pass  # 不增加 current_size
```

**Heap sim vs 运行时差异**：HLO heap sim 是**静态上限**，实际 XLA 运行时通过 buffer aliasing（input/output aliasing for AllGather）、pipeline scheduling（expert 权重顺序 fetch）和 remat 进一步降低内存。差异通常在 5-15%。

### 显存优化目标识别

**1. Vocab-sized buffers（Chunked CE 目标）**：

```python
# 在 maxHeap 中搜索含 vocab_size 维度的 buffer
# 例：bf16[10, 4096, 157184] = 11.99 GiB 的 logits tensor
# Chunked CE 可将其减少到 bf16[10, chunk_size, 157184]
# chunk_size=1024 → 2.99 GiB，节省 9 GiB/GA step

for buf in data['maxHeap']:
    dims = parse_shape_dims(buf['shapeString'])
    if any(d >= 100000 for d in dims):  # vocab-sized
        print(f"Optimization target: {buf['logicalBufferSizeMib']:.0f} MiB — {buf['tfOpName']}")
```

**2. 关键指标**：

| 指标 | 计算方法 | 优化含义 |
|------|---------|---------|
| Parameter vs Temporary 比例 | `groupName` 分组 | Temporary 大 → activation 优化空间大 |
| Padding overhead | `peakHeapMib - peakUnpaddedHeapMib` | > 500 MiB 说明 XLA tiling 浪费大 |
| 峰值位置 | `peakHeapSizePosition / len(heapSizes)` | < 50% → forward 峰值; > 70% → backward 峰值 |
| HBM 利用率 | `actual_hbm / hbm_capacity` | > 90% 需要优化; < 50% 有余量加大 batch |

**3. 常见优化路径**：

| 优化 | 预期节省 | 影响 |
|------|---------|------|
| Chunked cross-entropy | ~18 GiB（3× vocab logits） | 无精度影响，需 custom_vjp |
| save_out_proj remat | 大量激活不保存 | 增加约 33% 重计算 |
| FSDP 增大 → 减小 per-chip 参数 | 线性减少 | 增加通信 |
| Gradient accumulation 减少 | 减少 GA loop 中同时存活的激活 | 改变 GBS |
| bf16 optimizer states (mu_dtype) | ~2 GiB（mu 从 f32 → bf16） | 可能影响训练稳定性 |

### 多源交叉验证

当同时有多个数据源时，应交叉验证：

```
                 memory_viewer.json    xplane runtime    HLO proto replay
Peak HBM:       88.05 GiB (static)    77.25 GiB (actual) 79.23 GiB (upper bound)
Params:         6.09 GiB              6.97 GiB (dynamic) 6.09 GiB
```

- **memory_viewer 和 HLO proto** 给出静态上限（无运行时 aliasing）
- **xplane** 给出运行时实际值（含 buffer reuse，更低）
- 三者一致性说明分析可靠；差异大需要调查原因

### 显存异常诊断（`xprof memory diagnose`）

`memory diagnose` 从 `memory_viewer.json` 的 `heapSizes` 时间线和 `logicalBufferSpans` 生命周期数据提供四种异常检测：

**1. 尖刺检测**：找 heapSizes 中最大的 delta 跳变。例：

```
[9330]  77888 MiB  (Δ +12280)  fusion.60899
[9335]  90168 MiB  (Δ +12280)  select_reduce_fusion.112  ◀ PEAK
[9337]  77888 MiB  (Δ -12280)  fusion.60904
```

尖刺原因：多个大 buffer 短暂重叠。上例中 3 个 `[10,4096,157184]` vocab-sized buffer 在 7 条指令窗口内同时存活。

**2. Buffer Overlap 分析**：按 shape 分组统计峰值时刻同时存活的 buffer，量化各组贡献：

| Shape 分组 | 数量 | 总量 | 含义 |
|-----------|------|------|------|
| `bf16[10,4096,157184]` | 3 | 36 GiB | Logits/CE/MTP — chunked CE 目标 |
| `bf16[256,2048,512]` | 40 | 20 GiB | MoE expert AllGather — XLA 运行时会 pipeline |
| `f32[10,64,16,128,128]` | 1 | 640 MiB | GLA backward remat checkpoint |

**3. Lifetime 异常**：从 `logicalBufferSpans` 检测生命周期异常长的 Temporary buffer：

```python
for bid, span in logicalBufferSpans.items():
    start = span.get('start', 0)
    limit = span.get('limit', total_instr)
    lifetime_pct = (limit - start) / total_instr * 100
    # > 80% 生命周期 + > 100 MiB → critical
    # > 50% → warning
    # > 30% → info
```

注意：MoE expert AllGather buffer 在 heap sim 中表现为 >90% 生命周期（所有层的权重同时 "逻辑存活"），但 XLA 运行时实际 pipeline 加载，每次只有 1-2 层物理驻留。这是静态分析的已知 false positive。

**4. Phase 曲线**：按 schedule 百分比采样 heapSizes，显示显存随计算阶段的变化趋势。典型 MoE 模型曲线：
- 0-2%: 参数加载（6 GiB → 42 GiB）
- 2-8%: forward + MTP logits 计算（42 → 90 GiB，含峰值）
- 10-60%: backward 激活梯度链（~48 GiB 稳态）
- 60-85%: backward 权重梯度 + AllGather（48 → 75 GiB）
- 85-95%: optimizer update（下降到 8 GiB）
- 95-100%: 回到参数基线（6 GiB）

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
| `bytes_allocated` = 实际 HBM 用量 | **bytes_allocated 只跟踪动态 I/O buffer**（参数 + optimizer 入口），不含编译程序内部工作空间（~70 GiB）。实际 HBM = `bytes_reserved - bytes_available` |
| memory_viewer.json 峰值 = 运行时峰值 | memory_viewer 是**静态上限**（HLO heap sim），运行时通过 buffer aliasing 和 reuse 实际更低（差 5-15%）。xplane 运行时数据是权威来源 |
| 峰值 HBM 由 activations 主导 | 对 MoE 模型，expert 权重 AllGather 和 logits tensor 可能是最大的 buffer。先看 top buffers 再下结论 |
| Logits tensor 大小固定不可优化 | Chunked cross-entropy 可将 `[B,T,vocab]` 减到 `[B,chunk,vocab]`，节省 ~18 GiB |

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
