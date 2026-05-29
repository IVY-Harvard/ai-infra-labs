# 05 - Continuous Batching：连续批处理

## 核心问题

> 为什么 Continuous Batching 能比 Static Batching 提升 2-8x 吞吐？
> Scheduler 的设计哲学是什么？

## Static Batching 的问题

### 工作方式

```
Static Batching (传统方案):

1. 收集一个 batch 的请求 (等待填满或超时)
2. 所有请求一起 Prefill
3. 所有请求一起 Decode, 直到 最慢的请求 完成
4. 一起返回结果
5. 收集下一个 batch

时间线:
┌──────────────────────────────────────────────────────────┐
│                                                           │
│  Batch 1:                                                 │
│  Req A: [Prefill][====Decode 200 tokens====][等 B 完成]   │
│  Req B: [Prefill][==========Decode 500 tokens==========]  │
│  Req C: [Prefill][==Decode 100 tokens==][====等 B 完成====]│
│                                                           │
│  ← Batch 1 结束, 开始 Batch 2 →                          │
│                                                           │
│  Req D: [Prefill][===Decode 300 tokens===][等 F]          │
│  Req E: [Prefill][=Decode 50 tokens=][=====等 F=========] │
│  Req F: [Prefill][========Decode 400 tokens========]      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 问题分析

```
问题 1: 短板效应
  Batch 中最慢的请求决定整个 batch 的时间
  快请求完成后只能"空等" → GPU 资源浪费

问题 2: 排队延迟
  新请求必须等当前 batch 完全结束才能开始
  即使有空闲 GPU 资源

问题 3: 填充浪费
  不同请求的 prompt 长度不同 → padding 到最长
  短 prompt 的 padding 是纯浪费

量化分析:
  假设生成长度: Req A=200, B=500, C=100
  Static Batch 时间: 500 步 (等 B 完成)
  有效计算: (200+500+100) / (500×3) = 53.3%
  浪费: 46.7% 的 GPU 计算!
```

## Continuous Batching 的设计

### 核心理念: Iteration-Level Scheduling

```
关键转变: 调度粒度从 "请求级" 变为 "迭代级 (step 级)"

Static:  以 batch 为单位 → 整个 batch 完成才调度
Continuous: 以 iteration 为单位 → 每一步都可以调度

原则:
  - 完成的请求 → 立即移出 batch, 释放资源
  - 新到的请求 → 立即加入 batch, 开始服务
  - 每一步的 batch 组成都可以不同
```

### 工作方式

```
Continuous Batching 时间线:

Step 1: [A-prefill, B-prefill, C-prefill]  ← 3 个请求同时 prefill
Step 2: [A-decode,  B-decode,  C-decode ]  ← 3 个一起 decode
...
Step 100: [A-decode, B-decode, C-done→out] ← C 完成! 释放 slot
Step 101: [A-decode, B-decode, D-prefill ] ← D 立即加入!
...
Step 200: [A-done,   B-decode, D-decode  ] ← A 完成! 释放 slot
Step 201: [E-prefill,B-decode, D-decode  ] ← E 立即加入!
...
Step 400: [E-decode, B-decode, D-done    ] ← D 完成! 
Step 401: [E-decode, B-decode, F-prefill ] ← F 加入
...

┌──────────────────────────────────────────────────────────┐
│  GPU 利用率对比:                                          │
│                                                           │
│  Static:     ████░░████░░████░░  (有大量空闲)             │
│  Continuous: ████████████████████  (持续满载)              │
│                                                           │
│  没有"等最慢请求"的浪费!                                   │
│  没有"batch 间隙"的空闲!                                   │
└──────────────────────────────────────────────────────────┘
```

## 为什么能提升吞吐

### 数学分析

```
假设:
  - 请求生成长度: 均匀分布 [100, 500] tokens
  - 平均生成长度: 300 tokens
  - 最大生成长度: 500 tokens
  - Batch size: 32
  - Decode 每步时间: 40ms (不论 batch 中有多少请求)

Static Batching:
  每个 batch 时间 = 500 步 × 40ms = 20s (等最慢的)
  每个 batch 完成: 32 个请求
  吞吐 = 32 / 20s = 1.6 req/s
  有效计算比 = 平均长度/最大长度 = 300/500 = 60%

Continuous Batching:
  持续保持 batch=32 (请求完成即补入新请求)
  每步处理 32 个 token (几乎满载)
  每秒: 1000ms/40ms = 25 步 × 32 tokens = 800 tokens/s
  平均每请求 300 tokens → 吞吐 = 800/300 = 2.67 req/s
  有效计算比 ≈ 100% (始终满 batch)

提升: 2.67 / 1.6 = 1.67x

如果生成长度分布更不均匀 (如 [50, 2000]):
  Static: 2000步 × 40ms = 80s, 吞吐 = 32/80 = 0.4 req/s
  Continuous: 仍然 ≈ 2.67 req/s
  提升: 6.7x !!!

结论: 生成长度差异越大, Continuous Batching 优势越明显
```

### 关键等式

```
Static Batching 吞吐:
  T_static = batch_size / (max_gen_len × step_time)

Continuous Batching 吞吐:
  T_continuous = batch_size / (avg_gen_len × step_time)

提升比:
  T_continuous / T_static = max_gen_len / avg_gen_len

当生成长度差异大时:
  max_gen_len >> avg_gen_len → 提升倍数大
  
实际场景中, max/avg 通常在 2-8x → 吞吐提升 2-8x
```

## Scheduler 设计

### vLLM Scheduler 架构

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM Scheduler                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  三个队列:                                               │
│                                                          │
│  ┌──────────────┐                                       │
│  │  waiting      │  新请求等待区                          │
│  │  (FIFO queue) │  → 等待被调度 Prefill                 │
│  └──────┬───────┘                                       │
│         │ schedule()                                     │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │  running      │  正在执行区                            │
│  │  (active set) │  → 每步 Decode                        │
│  └──────┬───────┘                                       │
│         │ finish/preempt                                 │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │  swapped      │  被抢占区                              │
│  │  (swap queue)  │  → KV Cache 在 CPU 内存              │
│  └──────────────┘                                       │
│                                                          │
│  每步调度逻辑:                                           │
│  1. 检查 swapped 队列, 尝试恢复被抢占的请求               │
│  2. 检查 running 中完成的请求, 释放资源                   │
│  3. 检查 waiting 队列, 尝试加入新请求                     │
│  4. 如果显存不足, 抢占 running 中优先级最低的请求          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 调度算法: FCFS + 抢占

```python
# 简化的调度逻辑
def schedule(self):
    scheduled = []
    
    # Phase 1: 恢复 swapped 请求
    while self.swapped and has_blocks_for(self.swapped[0]):
        seq = self.swapped.popleft()
        swap_in(seq)  # CPU → GPU
        scheduled.append(seq)
    
    # Phase 2: 处理 running 请求
    for seq in self.running:
        if seq.is_finished():
            free_blocks(seq)
            continue
        if not can_append_slot(seq):
            # 显存不够给这个 seq 新 token → 抢占
            preempt(seq)  # GPU → CPU (swap) 或丢弃 (recompute)
        else:
            append_slot(seq)  # 分配新 Block (如果需要)
            scheduled.append(seq)
    
    # Phase 3: 调度新请求
    while self.waiting and has_blocks_for(self.waiting[0]):
        seq = self.waiting.popleft()
        allocate_blocks(seq)  # 分配初始 Block
        scheduled.append(seq)
    
    return scheduled
```

### 抢占策略 (Preemption)

```
当显存不足时, 需要驱逐某些请求:

策略 1: Swap (交换到 CPU)
  - KV Cache: GPU → CPU memory
  - 恢复时: CPU → GPU (swap in)
  - 优点: 不丢失计算结果
  - 缺点: swap 带宽有限 (PCIe ~32 GB/s)

策略 2: Recomputation (重新计算)
  - 丢弃 KV Cache
  - 恢复时: 重新 Prefill
  - 优点: 不需要 CPU 内存
  - 缺点: 浪费已完成的计算

vLLM 默认: Swap (优先保留计算结果)
选择依据: 如果 swap 时间 < recompute 时间 → Swap
          否则 → Recompute
          
一般经验: 短序列 recompute, 长序列 swap
```

## Chunked Prefill 与 Continuous Batching 的结合

### 问题

```
Continuous Batching 中, 新请求加入时需要 Prefill:

Step N:   [A-decode, B-decode, C-decode]
Step N+1: [A-decode, B-decode, D-PREFILL(2000 tokens)]  ← 长 Prefill!

D 的 Prefill 会:
1. 长时间霸占 GPU (2000 tokens 的大 GEMM)
2. A, B 的 Decode 被阻塞
3. TPOT 产生尖峰 (latency spike)
```

### Chunked Prefill 解决

```
把长 Prefill 分成小块, 和 Decode 穿插执行:

Step N:   [A-decode, B-decode, D-prefill-chunk1(512)]
Step N+1: [A-decode, B-decode, D-prefill-chunk2(512)]
Step N+2: [A-decode, B-decode, D-prefill-chunk3(512)]
Step N+3: [A-decode, B-decode, D-prefill-chunk4(464)]
Step N+4: [A-decode, B-decode, D-decode]  ← D 开始正常 decode

优势:
  - A, B 的 TPOT 稳定 (不被长 Prefill 阻塞)
  - GPU 利用率更均匀
  - TTFT 稍微增加 (4 步 vs 1 步), 但 TPOT 稳定性大幅提升

vLLM 参数: --enable-chunked-prefill --max-num-batched-tokens 2048
```

## 吞吐对比实测

```
模型: LLaMA-2-70B on 8×H20
数据: ShareGPT 分布 (avg prompt=200, avg output=300)
并发: 持续 100 req/s 请求压力

┌─────────────────────────────────────────────────────┐
│  方案                 │ 吞吐 (req/s) │ Avg Latency  │
├─────────────────────────────────────────────────────┤
│  Static Batch (bs=32) │    4.2       │   7.6s       │
│  Continuous Batch     │   14.8       │   2.1s       │
│  + Chunked Prefill    │   15.2       │   1.9s       │
│  + Prefix Caching     │   18.6       │   1.5s       │
├─────────────────────────────────────────────────────┤
│  提升                 │   4.4x       │   5.1x       │
└─────────────────────────────────────────────────────┘

注: 数字为典型值参考, 实际取决于负载模式
```

## 与 PagedAttention 的协同

```
Continuous Batching + PagedAttention = 完美组合

Continuous Batching 需要:
  ✓ 动态加入/移出请求 → PagedAttention 支持动态分配/释放 Block
  ✓ 不同请求不同长度 → PagedAttention 按需分配, 无碎片
  ✓ 最大化 batch size → PagedAttention 提高显存利用率
  ✓ 抢占和恢复 → PagedAttention 支持 Block 级 swap

没有 PagedAttention 的 Continuous Batching:
  - 仍然需要预分配 → 显存浪费
  - batch size 受限 → 吞吐提升有限
  - 碎片化严重 → 新请求可能无法加入

没有 Continuous Batching 的 PagedAttention:
  - 显存利用率提高了
  - 但 batch 内的短板效应仍在
  - 吞吐提升有限

两者结合:
  PagedAttention 解决"空间效率" (显存利用率)
  Continuous Batching 解决"时间效率" (GPU 利用率)
  → 共同实现 2-4x 吞吐提升
```

## 知识要点框架

### "Continuous Batching 为什么能提升吞吐？"

```
回答框架:

"Static Batching 有两个核心问题:
1. 短板效应: 整个 batch 等最慢的请求 → GPU 空等
2. 批次间隙: batch 之间有空闲期

Continuous Batching 的核心改变:
调度粒度从 batch 级变为 iteration 级 (每步都可调度)

- 请求完成 → 立即释放 slot, 新请求立即加入
- GPU 始终保持满载 (不再空等)
- 吞吐提升 = max_gen_len / avg_gen_len, 通常 2-8x

配合 PagedAttention:
- PA 提供高效的 KV Cache 动态管理
- 使得请求的加入/退出不产生碎片
- 两者结合实现空间+时间的双重效率"
```

### 追问: "Continuous Batching 有什么缺点？"

```
"主要挑战:
1. Prefill 干扰: 新请求 Prefill 可能阻塞其他请求的 Decode
   → 解决: Chunked Prefill

2. 调度复杂度: 每步都要做调度决策
   → 但 scheduler overhead << decode 时间, 可忽略

3. 显存管理复杂: 频繁分配/释放 Block
   → PagedAttention 的 Block 池化管理解决

4. 抢占策略: 显存不足时需要决定驱逐谁
   → FCFS 是简单策略, 更优策略还在研究"
```

## 小结

| 要点 | 记住 |
|------|------|
| Static 的问题 | 短板效应 + 批次间隙 |
| Continuous 的核心 | Iteration-Level Scheduling |
| 吞吐提升来源 | 消除 GPU 空等时间 |
| 提升公式 | max_gen_len / avg_gen_len (2-8x) |
| 与 PA 协同 | PA 解决空间, CB 解决时间 |
| Chunked Prefill | 解决 Prefill 干扰 Decode 的问题 |
| Scheduler 核心 | 三个队列: waiting → running → swapped |
