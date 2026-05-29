# 06 — 集合通信原语：AllReduce / AllGather / ReduceScatter / P2P

## 1. 通信原语概览

### 1.1 分类

```
点对点通信 (Point-to-Point):
  Send / Recv — 一对一通信

集合通信 (Collective):
  Broadcast      — 一对多
  Reduce         — 多对一 (with operation)
  AllReduce      — 多对多 (Reduce + Broadcast)
  Gather         — 多对一 (concatenation)
  AllGather      — 多对多 (Gather + Broadcast)
  Scatter        — 一对多 (split + send)
  ReduceScatter  — 多对多 (Reduce + Scatter)
  All-to-All     — 多对多 (personalized)
```

### 1.2 通信量公式总结

设 N = 进程数，M = 每个进程的数据量

| 操作 | 每进程发送 | 每进程接收 | 总通信量 | 备注 |
|------|-----------|-----------|---------|------|
| Broadcast | M (root) | M | M | Root → All |
| Reduce | M | M (root) | M×(N-1) | All → Root |
| AllReduce | 2M(N-1)/N | 2M(N-1)/N | 2M(N-1) | Ring optimal |
| Gather | M | M×N (root) | M×(N-1) | 拼接 |
| AllGather | M | M×N | M×(N-1) | 每个都拼接 |
| Scatter | M×N (root) | M | M×(N-1) | 分割 |
| ReduceScatter | M | M/N | M×(N-1)/N | Reduce→每人一份 |
| All-to-All | M | M | M×(N-1) | 个性化交换 |

## 2. AllReduce 详解

### 2.1 语义

```
输入: rank i 有数据 xᵢ ∈ R^M
输出: 所有 rank 得到 y = op(x₀, x₁, ..., x_{N-1})

op 通常是 SUM (梯度同步)，也可以是 MAX, MIN, PROD
```

### 2.2 实现算法

**Ring AllReduce (带宽最优)**:
```
Phase 1: ReduceScatter (N-1 步)
  每步: 每个 rank 发送 M/N 数据给下一个 rank
  完成后: rank i 持有 chunk i 的 reduce 结果

Phase 2: AllGather (N-1 步)
  每步: 每个 rank 发送 M/N 给下一个 rank
  完成后: 所有 rank 持有完整结果

时间: 2(N-1) × [α + M/(N×β)]
     = 2(N-1)α + 2(N-1)M/(Nβ)
     ≈ 2M/β  (当 M >> Nα/β)

特点: 带宽最优 (每条链路充分利用)
      延迟非最优 (O(N) 步)
```

**Tree AllReduce (延迟最优)**:
```
Phase 1: Tree Reduce (log₂N 步)
  树形结构逐层 reduce，每步通信量 M

Phase 2: Tree Broadcast (log₂N 步)
  从 root 逐层广播

时间: 2log₂N × [α + M/β]

特点: 延迟最优 (O(logN) 步)
      带宽非最优 (每步传 M，但只用一条链路)
```

**Recursive Halving-Doubling**:
```
Phase 1: Recursive Halving (ReduceScatter, log₂N 步)
  Step k: 与距离 2^k 的 rank 交换一半数据并 reduce
  每步通信量: M/2, M/4, ..., M/2^(logN) → 总计 M(1-1/N)

Phase 2: Recursive Doubling (AllGather, log₂N 步)
  反向操作

时间: 2log₂N × α + 2(N-1)M/(Nβ)

特点: 延迟和带宽都不错的平衡
```

**NCCL 的选择策略**:
```
小消息 (< 256KB): Tree (减少延迟)
中消息 (256KB - 几 MB): Recursive Halving-Doubling
大消息 (> 几 MB): Ring (最大化带宽利用)

NCCL 还有 CollNet (InfiniBand switch reduction) 等优化
```

### 2.3 在分布式训练中的用途

```
1. DDP 梯度同步: AllReduce(gradients, op=SUM) → 然后除以 N
2. TP 行并行输出合并: AllReduce(partial_outputs, op=SUM)
3. Metrics 聚合: AllReduce(loss, op=SUM) → 平均 loss
```

## 3. AllGather 详解

### 3.1 语义

```
输入: rank i 有数据 xᵢ ∈ R^M
输出: 所有 rank 得到 y = [x₀, x₁, ..., x_{N-1}] ∈ R^{NM}

即: 收集所有 rank 的数据，拼接成完整向量
```

### 3.2 实现

```
Ring AllGather (N-1 步):
  Step k: 每个 rank 发送收到的前一块给下一个 rank
  每步通信量: M
  总时间: (N-1) × [α + M/β]
  带宽: (N-1)M / [(N-1)(M/β)] = β  → 带宽利用 100%
```

### 3.3 在分布式训练中的用途

```
1. FSDP Forward: AllGather(sharded_params) → 恢复完整参数
2. TP 输出拼接: 有时需要 AllGather 列并行的输出
3. Embedding: AllGather(partial_embedding) → 完整 embedding table
```

## 4. ReduceScatter 详解

### 4.1 语义

```
输入: rank i 有完整数据 xᵢ ∈ R^{NM}
输出: rank i 得到 yᵢ = op(x₀[i], x₁[i], ..., x_{N-1}[i]) ∈ R^M

即: 每个 rank 先 Reduce 所有数据，再 Scatter 结果的第 i 块给 rank i
```

### 4.2 实现

```
Ring ReduceScatter (N-1 步):
  等同于 Ring AllReduce 的 Phase 1
  每步通信量: M (实际是 NM/N = M)
  总时间: (N-1) × [α + M/β]
```

### 4.3 在分布式训练中的用途

```
1. FSDP Backward: ReduceScatter(gradients) → 每 rank 只存自己的梯度分片
2. Sequence Parallel: ReduceScatter 替代 AllReduce 的前半部分
3. ZeRO: ReduceScatter(gradients) → 分布式存储梯度
```

## 5. Broadcast 详解

### 5.1 语义

```
输入: root rank 有数据 x ∈ R^M
输出: 所有 rank 得到 x 的副本
```

### 5.2 实现

```
Tree Broadcast (log₂N 步):
  Step 1: root → 1 个 rank
  Step 2: 2 个 rank → 2 个 rank
  Step k: 2^(k-1) → 2^(k-1)
  
  总时间: log₂N × [α + M/β]

Pipeline Broadcast (N-1 步，适合大消息):
  将 M 分成块，流水线式传递
  时间: (N-1)α + M/β  → 带宽利用更好
```

### 5.3 在分布式训练中的用途

```
1. DDP 初始化: Broadcast(rank0_params) → 同步初始参数
2. PP Stage 0: Broadcast 输入 embedding
3. 超参数同步: Broadcast(learning_rate) (如果动态调整)
```

## 6. P2P (Send / Recv) 详解

### 6.1 语义

```
Send(tensor, dst): 将 tensor 发送到 dst rank
Recv(tensor, src): 从 src rank 接收 tensor 到 tensor buffer
```

### 6.2 在分布式训练中的用途

```
1. Pipeline Parallelism:
   Forward: stage_i.Send(activation, dst=stage_i+1)
   Backward: stage_i.Send(gradient, dst=stage_i-1)

2. Ring Attention:
   Send(KV_block, dst=next_rank)
   Recv(KV_block, src=prev_rank)
```

### 6.3 通信模式

```
同步 P2P:
  send 和 recv 必须配对，否则死锁
  dist.send(tensor, dst=1)  # rank 0
  dist.recv(tensor, src=0)  # rank 1

异步 P2P:
  handle = dist.isend(tensor, dst=1)
  handle = dist.irecv(tensor, src=0)
  handle.wait()
```

## 7. All-to-All 详解

### 7.1 语义

```
输入: rank i 有数据 xᵢ = [xᵢ₀, xᵢ₁, ..., xᵢ_{N-1}]  (N 块，每块给不同 rank)
输出: rank i 得到 yᵢ = [x₀ᵢ, x₁ᵢ, ..., x_{N-1}ᵢ]  (从所有 rank 收集给自己的块)

即: 个性化的多对多通信
```

### 7.2 在分布式训练中的用途

```
1. Expert Parallelism (MoE):
   Token dispatch: All-to-All(tokens → experts)
   Token combine: All-to-All(expert_outputs → original_ranks)

2. Sequence Parallel (某些实现):
   切换并行维度时使用 All-to-All
```

## 8. NCCL 实现细节

### 8.1 NCCL 架构

```
NCCL (NVIDIA Collective Communications Library):
  - 针对 NVIDIA GPU 优化的集合通信库
  - 支持 NVLink, PCIe, InfiniBand, RoCE, TCP
  - 自动检测拓扑，选择最优算法
  - 支持 multi-node, multi-GPU

拓扑检测:
  1. 检测 GPU 间的互联类型 (NVLink, PCIe, ...)
  2. 检测网络拓扑 (IB switch topology)
  3. 构建最优通信图 (ring, tree, 或混合)
```

### 8.2 NCCL Ring 构建

```
NCCL 如何构建 Ring:
  1. 检测所有 GPU 对之间的带宽
  2. 选择使 ring 总带宽最大的排列
  3. 可能构建多个 ring（multi-ring），并行传输

例: 8 GPU with NVLink mesh:
  Ring 0: 0→1→2→3→4→5→6→7→0
  Ring 1: 0→3→6→1→4→7→2→5→0
  → 两个 ring 可以并行传输，利用 NVLink 的多条链路
```

### 8.3 NCCL Tree 实现

```
Tree AllReduce 用于小消息:
  
  Binary tree:
       0
      / \
     1   2
    / \ / \
   3  4 5  6
   |
   7

  Reduce: 叶 → 根 (3 steps for 8 GPUs)
  Broadcast: 根 → 叶 (3 steps)
  总 latency: 2×log₂(N) × α
```

### 8.4 CollNet (SHARP)

```
InfiniBand SHARP (Scalable Hierarchical Aggregation and Reduction Protocol):
  - 在 IB switch 硬件上做 reduce
  - 不需要数据到达 GPU 就能完成聚合
  
好处: 减少网络流量和延迟
限制: 需要支持 SHARP 的 IB switch
```

## 9. 通信优化技术

### 9.1 通信计算重叠 (Overlap)

```
策略 1: Gradient Bucketing (DDP)
  反向传播过程中，已完成的梯度桶立即开始 AllReduce
  
  Timeline:
  Compute:  [Layer N grad] [Layer N-1 grad] [Layer N-2 grad] ...
  Comm:                    [Bucket 1 AR]    [Bucket 2 AR]    ...

策略 2: FSDP Prefetch
  在计算当前层时，提前 AllGather 下一层的参数
  
  Timeline:
  Compute:  [Layer i fwd]  [Layer i+1 fwd] ...
  Comm:     [AG layer i+1] [AG layer i+2] ...

策略 3: PP + DP overlap
  Pipeline 后期的反向传播与 DP AllReduce 重叠
```

### 9.2 通信压缩

```
1. FP16/BF16 通信:
   梯度用 FP16 传输 → 通信量减半
   
2. 梯度压缩 (Gradient Compression):
   - Top-K sparsification: 只传最大的 K 个梯度
   - 1-bit Adam: 用 1 bit 编码梯度方向
   - PowerSGD: 低秩近似梯度

3. 量化通信:
   FP8 AllReduce (实验性)
```

### 9.3 通信分组

```
Hierarchical AllReduce:
  Step 1: 节点内 AllReduce (NVLink, 快)
  Step 2: 节点间 AllReduce (IB, 慢，但数据量小了)
  Step 3: 节点内 Broadcast 结果

好处: 减少跨网络的通信量
     每个节点只有一个代表参与节点间通信
```

## 10. 通信性能分析

### 10.1 关键指标

```
1. 带宽 (Bandwidth):
   算法带宽 = 数据量 / 时间
   Bus 带宽 = 算法带宽 × 修正因子
   
   修正因子:
   AllReduce: 2(N-1)/N ≈ 2 (Ring)
   AllGather: (N-1)/N ≈ 1
   ReduceScatter: (N-1)/N ≈ 1

2. 延迟 (Latency):
   单次通信启动开销 α
   Ring: 2(N-1) × α
   Tree: 2×log₂(N) × α

3. 利用率:
   实际带宽 / 硬件峰值带宽
   NVLink H20: 理论 ~450 GB/s, 实际 AllReduce ~400 GB/s (88%)
```

### 10.2 性能公式

```
AllReduce 时间 (Ring, 大消息):
  T = 2(N-1)/N × M/BW ≈ 2M/BW

例: M=1GB, BW=400GB/s (NVLink):
  T = 2×1/400 = 5 ms

例: M=1GB, BW=25GB/s (IB):
  T = 2×1/25 = 80 ms
  → 这就是为什么大模型的 DP AllReduce 需要 overlap
```

### 10.3 NCCL 环境变量调优

```bash
# 显示 NCCL 调试信息
export NCCL_DEBUG=INFO

# 选择算法
export NCCL_ALGO=Ring    # 或 Tree, CollNet

# 协议选择
export NCCL_PROTO=Simple  # 或 LL, LL128

# 网络接口
export NCCL_SOCKET_IFNAME=eth0

# 多 NIC 绑定
export NCCL_NET_GDR_LEVEL=5

# NCCL buffer 大小
export NCCL_BUFFSIZE=4194304  # 4MB

# 多 ring
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8
```

## 11. PyTorch 分布式通信 API

### 11.1 基础 API

```python
import torch.distributed as dist

# 初始化
dist.init_process_group(backend="nccl")

# 集合通信
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.all_gather(output_tensors, input_tensor)
dist.reduce_scatter(output_tensor, input_tensors)
dist.broadcast(tensor, src=0)

# P2P
dist.send(tensor, dst=1)
dist.recv(tensor, src=0)

# 异步
handle = dist.all_reduce(tensor, async_op=True)
handle.wait()

# Process Group
tp_group = dist.new_group(ranks=[0,1,2,3])
dist.all_reduce(tensor, group=tp_group)
```

### 11.2 ProcessGroup 管理

```python
# 创建多个通信组
def create_parallel_groups(tp_size, pp_size, dp_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # TP groups: 相邻的 tp_size 个 rank
    for i in range(world_size // tp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    
    # PP groups: 间隔 tp_size 的 rank
    for i in range(tp_size):
        for j in range(dp_size):
            ranks = [i + j * tp_size * pp_size + k * tp_size 
                     for k in range(pp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                pp_group = group
    
    # DP groups: 间隔 tp_size * pp_size 的 rank
    for i in range(tp_size * pp_size):
        ranks = [i + j * tp_size * pp_size for j in range(dp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            dp_group = group
    
    return tp_group, pp_group, dp_group
```

## 12. 下一步

- [07_deepspeed_zero.md](07_deepspeed_zero.md)：ZeRO 如何用 ReduceScatter/AllGather 优化显存
- [Lab 08](../labs/08_communication_optimization/)：手写 Ring AllReduce + NCCL Benchmark
