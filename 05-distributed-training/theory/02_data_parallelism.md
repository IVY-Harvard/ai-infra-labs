# 02 — 数据并行：DDP 原理、Ring AllReduce、FSDP 显存优化

## 1. 从朴素 DP 到 DDP

### 1.1 朴素数据并行 (nn.DataParallel)

PyTorch 最早的数据并行实现，已基本弃用：

```python
model = nn.DataParallel(model)  # 不推荐
```

**问题**：
- GPU 0 是 parameter server：收集所有梯度、更新参数、广播新参数
- GPU 0 成为瓶颈（通信 + 计算集中在一张卡）
- GIL 限制多线程效率
- 显存不均衡：GPU 0 需要存所有梯度

### 1.2 分布式数据并行 (DistributedDataParallel, DDP)

DDP 使用 AllReduce 替代 parameter server，所有 GPU 对等参与：

```python
# 每个进程
model = DDP(model, device_ids=[local_rank])
```

**DDP 训练流程**：
```
Step 1: 每个 rank 用不同 mini-batch 做前向计算
Step 2: 每个 rank 本地反向传播计算梯度
Step 3: AllReduce 同步梯度（所有 rank 得到相同的平均梯度）
Step 4: 每个 rank 用相同梯度更新参数（保证参数一致）
```

## 2. DDP 关键实现细节

### 2.1 Gradient Bucketing

DDP 不是等所有梯度计算完再通信，而是将梯度分桶，反向传播过程中**边计算边通信**：

```
时间 →
反向传播:  [Layer N 梯度] [Layer N-1 梯度] [Layer N-2 梯度] ...
通信:                    [Bucket 1 AllReduce]  [Bucket 2 AllReduce] ...
                         ↑ 与反向传播重叠
```

**Bucket 大小**（默认 25MB）影响：
- Bucket 太大 → 等待时间长，overlap 效果差
- Bucket 太小 → 通信次数多，startup latency 累积
- 最优值取决于网络带宽和计算速度

### 2.2 梯度同步时机

DDP 使用 `autograd hook`，在每个参数的梯度计算完成时触发：

```python
# 伪代码：DDP 内部注册的 hook
def grad_hook(param):
    bucket = find_bucket(param)
    bucket.mark_ready(param)
    if bucket.all_ready():
        # 异步启动 AllReduce
        bucket.allreduce_async()
```

### 2.3 Broadcast 初始化

DDP 构造时，rank 0 的参数会 broadcast 到所有 rank，确保初始参数一致：

```python
# DDP.__init__ 内部
for param in module.parameters():
    dist.broadcast(param.data, src=0)
```

## 3. Ring AllReduce 推导

### 3.1 什么是 AllReduce

AllReduce = Reduce + Broadcast：所有节点的数据执行 reduce 操作（如求和），结果广播到所有节点。

```
Before:  GPU 0: [a0, a1, a2]   GPU 1: [b0, b1, b2]   GPU 2: [c0, c1, c2]
After:   GPU 0: [a0+b0+c0, a1+b1+c1, a2+b2+c2]  (same on all GPUs)
```

### 3.2 朴素实现的通信量

如果用一个节点做 reduce 再 broadcast：
- Reduce: (N-1) × M 数据量发送到一个节点
- Broadcast: (N-1) × M 从一个节点发出
- 总通信量: 2(N-1) × M
- **瓶颈**：一个节点需要收发所有数据

### 3.3 Ring AllReduce 算法

Ring AllReduce 分两个阶段，每个阶段 N-1 步（N = GPU 数量）：

**Phase 1: ReduceScatter**

将数据分成 N 个 chunk，通过环形传递，每步将一个 chunk 的值累加：

```
N = 3, 数据分 3 个 chunk

Step 0: GPU 0 发 chunk[0] → GPU 1     GPU 1 发 chunk[1] → GPU 2     GPU 2 发 chunk[2] → GPU 0
Step 1: GPU 0 发 chunk[2] → GPU 1     GPU 1 发 chunk[0] → GPU 2     GPU 2 发 chunk[1] → GPU 0

经过 2 步 (N-1 步):
  GPU 0 持有 chunk[1] 的完整 reduce 结果
  GPU 1 持有 chunk[2] 的完整 reduce 结果
  GPU 2 持有 chunk[0] 的完整 reduce 结果
```

**Phase 2: AllGather**

将每个 GPU 持有的完整 chunk 通过环形传递给所有 GPU：

```
Step 0: GPU 0 发 chunk[1] → GPU 1     GPU 1 发 chunk[2] → GPU 2     GPU 2 发 chunk[0] → GPU 0
Step 1: GPU 0 发 chunk[0] → GPU 1     GPU 1 发 chunk[1] → GPU 2     GPU 2 发 chunk[2] → GPU 0

经过 2 步: 所有 GPU 持有完整的 reduce 结果
```

### 3.4 通信量分析

```
每步每个 GPU 发送: M/N 数据
ReduceScatter 步数: N-1
AllGather 步数: N-1
总步数: 2(N-1)

每个 GPU 总发送量 = 2(N-1) × M/N ≈ 2M (当 N 大时)
每个 GPU 总接收量 = 2(N-1) × M/N ≈ 2M

关键特性:
1. 总通信量与 N 无关！（近似 2M）
2. 每步通信量 = M/N，可以完全利用环上每条链路的带宽
3. 是带宽最优的 AllReduce 算法
```

### 3.5 时间复杂度

```
Ring AllReduce 时间 = 2(N-1) × [α + M/(N×β)]

α = latency (每次通信的启动开销)
β = bandwidth (每条链路的带宽)
M = 数据量
N = GPU 数量

当 M 很大时（如梯度同步）:
  时间 ≈ 2M/β  (与 N 几乎无关)

当 N 很大时:
  α 项 2(N-1)α 变大 → Ring AllReduce 的 latency 问题
  → 解决方案: Tree AllReduce (latency = O(log N))
```

## 4. FSDP 原理 (Fully Sharded Data Parallelism)

### 4.1 从 DDP 到 FSDP 的动机

DDP 的显存模型（BF16 混合精度，7B 模型）：

```
每张 GPU:
  模型参数 (BF16):    14 GB
  梯度 (BF16):        14 GB
  优化器状态 (FP32):   84 GB  (master_weight + momentum + variance)
  ─────────────────────────
  合计:               112 GB  × 8 卡 = 896 GB 总显存使用

  实际模型只有 14 GB 参数，但每卡要 112 GB
  冗余率: 112 × 8 / 112 = 8x（8 张卡存了 8 份完全相同的东西）
```

FSDP 思路：**把参数/梯度/优化器状态切分（shard）到所有 GPU，需要时再 AllGather 恢复**。

### 4.2 FSDP 训练流程

```
Step 1 (Forward):
  对于每个 FSDP unit（通常是一个 Transformer 层）:
    AllGather: 从所有 GPU 收集完整参数
    前向计算
    释放非本地的参数分片（可选，取决于 sharding strategy）

Step 2 (Backward):
  对于每个 FSDP unit（反向顺序）:
    AllGather: 再次收集完整参数
    反向计算梯度
    ReduceScatter: 将梯度 reduce 并 scatter 到各 GPU
    释放非本地的参数和完整梯度

Step 3 (Optimizer):
  每个 GPU 只更新自己负责的参数分片
  优化器状态也只存本地分片
```

### 4.3 显存节省分析

**FSDP FULL_SHARD (等效 ZeRO-3)**，8 GPU：

```
每张 GPU:
  模型参数分片 (BF16):  14 / 8 = 1.75 GB
  梯度分片 (BF16):      14 / 8 = 1.75 GB
  优化器状态分片 (FP32): 84 / 8 = 10.5 GB
  ───────────────────────────────
  合计:                  14 GB（相比 DDP 的 112 GB）

  峰值显存（Forward 期间需要 AllGather 恢复完整参数）:
  14 GB + 14 GB (当前层的完整参数) = 28 GB
  + 激活值（取决于 batch size 和是否使用 activation checkpointing）
```

### 4.4 通信量分析

```
DDP:
  AllReduce(梯度) = 2M (一步一次)

FSDP FULL_SHARD:
  Forward: AllGather(参数) × L 次 (L = 层数)
  Backward: AllGather(参数) × L 次 + ReduceScatter(梯度) × L 次

  总通信量:
  AllGather: 每次 M × (N-1)/N ≈ M (数据量)
  Forward AllGather: L × M_layer → 总计 ≈ M
  Backward AllGather: L × M_layer → 总计 ≈ M
  Backward ReduceScatter: L × M_layer → 总计 ≈ M
  ─────────────────
  总计: 3M

  对比 DDP 的 2M：FSDP 多了约 50% 通信量
  换来的是：显存从 112GB 降到 ~28GB（峰值）
```

### 4.5 Sharding Strategy 选择

```python
from torch.distributed.fsdp import ShardingStrategy

# FULL_SHARD: 参数+梯度+优化器全切分 (= ZeRO-3)
# 最省显存，通信量最大 (3M)
FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)

# SHARD_GRAD_OP: 只切分梯度+优化器 (= ZeRO-2)
# 中等显存，通信量 2M（Forward 不需要 AllGather）
FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)

# NO_SHARD: 不切分 (= DDP)
FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD)

# HYBRID_SHARD: 节点内 FULL_SHARD，节点间 DDP
# 适合多机训练，减少跨机通信
FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)
```

### 4.6 FSDP2 (PyTorch 2.x)

FSDP2 (fully_shard API) 是新一代实现：

```python
from torch.distributed._composable.fsdp import fully_shard

# FSDP2 更灵活：per-module 粒度
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)
```

改进点：
- **可组合**：与 TP、compile() 等可以 compose
- **更细粒度**：per-parameter sharding，不再绑定到 nn.Module
- **DTensor 原生**：底层使用 DTensor 而非 FlatParameter
- **通信优化**：更好的 prefetching 和 overlap

## 5. Activation Checkpointing

### 5.1 动机

激活值（中间结果）在反向传播中需要，但占大量显存：

```
Transformer 层的激活值 (per layer, BF16):
  Attention:  2 × B × S × H × 2 bytes       (Q*K^T 结果 + softmax 输出)
  MLP:        B × S × 4H × 2 bytes           (中间激活)
  LayerNorm:  2 × B × S × H × 2 bytes        (两个 LN 的输入)

  7B 模型 (H=4096, S=2048, B=4):
  每层 ≈ 4 × 2048 × 4096 × 2 × 3 ≈ 200 MB
  32 层 → ~6.4 GB
```

### 5.2 原理

选择性地丢弃中间激活值，反向传播时重新计算：

```
Normal:      Forward → 保存所有激活 → Backward（使用保存的激活）
Checkpoint:  Forward → 只保存输入 → Backward（从输入重新计算激活）

时间增加: ~33%（多一次部分前向）
显存减少: 激活值从 O(L) 降到 O(√L) 或 O(1)
```

### 5.3 与 FSDP 配合

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

# 对每个 Transformer 层做 activation checkpointing
for layer in model.transformer.layers:
    checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

# 然后 wrap FSDP
model = FSDP(model, ...)
```

## 6. DDP vs FSDP 选择指南

```
                    ┌──────────────────────┐
                    │ 模型能放入单卡？       │
                    └──────────┬───────────┘
                         Yes   │   No
                    ┌──────────┴───────────┐
                    ▼                       ▼
              ┌──────────┐          ┌──────────────┐
              │ DDP      │          │ FSDP/ZeRO-3  │
              │ 最简单    │          │ 切分所有状态  │
              └──────────┘          └──────────────┘
                    │
                    ▼
           显存还是紧张？
           (大 batch / 长 seq)
                    │ Yes
                    ▼
           ┌──────────────┐
           │ FSDP          │
           │ SHARD_GRAD_OP │
           │ (ZeRO-2)     │
           └──────────────┘
```

**8×H20 场景建议**：

| 模型规模 | 推荐方案 | 理由 |
|---------|---------|------|
| < 3B | DDP | 简单高效 |
| 3B - 7B | FSDP SHARD_GRAD_OP | 显存优化 + 低通信开销 |
| 7B - 13B | FSDP FULL_SHARD | 显存必须切分 |
| > 13B | FSDP + TP=4 或 TP+PP | 单靠数据并行不够 |

## 7. 下一步

- [03_tensor_parallelism.md](03_tensor_parallelism.md)：深入张量并行的矩阵切分和通信分析
- [Lab 01](../labs/01_ddp_basics/)：DDP 实战
- [Lab 02](../labs/02_fsdp_practice/)：FSDP 实战与显存对比
