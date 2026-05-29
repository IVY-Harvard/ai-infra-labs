# 04 - PagedAttention 深度解析

## 核心问题

> PagedAttention 借鉴了操作系统虚拟内存的什么思想？它是怎样减少显存浪费的？

这是实践中最高频的推理系统问题，必须能从原理到实现完整讲清楚。

## 传统方案的问题回顾

### 连续内存分配

```
传统方案: 为每个序列预分配一块 连续 的显存空间

┌──────────── GPU Memory ──────────────────────────┐
│                                                   │
│  Seq A: [██████████████████████████████████████]  │  预分配 max_len
│         [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │  实际使用 20%
│                                                   │
│  Seq B: [██████████████████████████████████████]  │  预分配 max_len
│         [██████████████░░░░░░░░░░░░░░░░░░░░░░]  │  实际使用 35%
│                                                   │
│  ──── 空闲 ───────────────────────────────────── │  想放 Seq C？
│  [                                              ] │  空间够但不连续
│  [        ][    ][             ][     ]           │  外部碎片!
│                                                   │
└──────────────────────────────────────────────────┘

三种浪费:
  1. Internal Fragmentation (内部碎片): 预分配但未使用 → 60-90%
  2. External Fragmentation (外部碎片): 空闲块不连续 → 无法分配
  3. Reservation Waste: 为未来生成预留空间 → 不确定需要多少
```

### 量化浪费

```
典型场景 (LLaMA-70B, max_seq=4096):

预分配: 4096 × 0.3MB = 1.25 GB / 请求
实际用: ~500 × 0.3MB = 0.15 GB / 请求 (平均)

每个请求浪费: 1.1 GB
50 个请求: 55 GB 浪费

在 8×H20 (96 GB/GPU) 上:
  可用 KV Cache: ~75 GB/GPU
  预分配 50 请求: 62.5 GB → 只能服务 50 请求
  实际需要: 7.5 GB → 理论可服务 500 请求!

PagedAttention 的显存利用率接近 100% → 吞吐提升 2-4x
```

## 操作系统虚拟内存回顾

### 核心思想

```
┌─────────────────────────────────────────────────────────┐
│  操作系统虚拟内存                                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  问题: 进程需要连续地址空间, 但物理内存是碎片化的          │
│                                                          │
│  解决: 虚拟地址 → 页表映射 → 物理页 (不要求连续!)        │
│                                                          │
│  Virtual Address Space        Physical Memory            │
│  ┌───┐                       ┌───┐                      │
│  │ 0 │ ────────────────────▶ │ 5 │                      │
│  ├───┤                       ├───┤                      │
│  │ 1 │ ──────────┐          │   │ (空闲)                │
│  ├───┤           │          ├───┤                      │
│  │ 2 │ ────┐     │          │ 2 │ ◀── Page 2           │
│  ├───┤     │     │          ├───┤                      │
│  │ 3 │     │     └────────▶ │ 8 │                      │
│  └───┘     │                ├───┤                      │
│  (连续)    └──────────────▶ │ 1 │                      │
│                              └───┘                      │
│                              (不连续, 但没关系!)         │
│                                                          │
│  关键特性:                                               │
│  1. 按需分配 (Demand Paging): 用到才分配物理页            │
│  2. 页大小固定 (如 4KB): 消除外部碎片                     │
│  3. 页表映射: 虚拟连续 → 物理不连续                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## PagedAttention 的设计

### 核心类比

```
┌─────────────────────────────────────────────────────────┐
│  OS Virtual Memory     →    PagedAttention               │
├─────────────────────────────────────────────────────────┤
│  进程                  →    序列 (Sequence)               │
│  虚拟页                →    逻辑 Block                    │
│  物理页                →    物理 Block (GPU 显存块)       │
│  页表                  →    Block Table                   │
│  页大小 (4KB)          →    Block Size (如 16 tokens)    │
│  按需分配              →    按需分配 Block                │
│  Page Fault            →    新 Block 分配                │
│  Copy-on-Write         →    Beam Search 共享              │
│  Swap (内存↔磁盘)      →    Swap (GPU↔CPU)              │
└─────────────────────────────────────────────────────────┘
```

### Block 的结构

```
一个 Physical Block 存储 block_size 个 token 的 KV:

┌────────────────────────────────────────────┐
│  Physical Block #7                          │
│                                             │
│  Block Size = 16 tokens                     │
│                                             │
│  K tensor: [16, num_kv_heads, head_dim]     │
│  V tensor: [16, num_kv_heads, head_dim]     │
│                                             │
│  大小 = 16 × 2 × n_kv × d_h × dtype bytes │
│                                             │
│  Slots: [tok₀│tok₁│tok₂│...│tok₁₅]        │
│         [████│████│████│░░░│░░░░░]          │
│          filled         empty (可继续填)     │
│                                             │
│  fill_count = 3 (当前已填充 3 个 token)     │
│                                             │
└────────────────────────────────────────────┘
```

### Block Table 映射

```
序列 A: "The quick brown fox jumps over the lazy dog generates more tokens"
Token 数: 12
Block Size: 4

Block Table for Seq A:
  Logical Block 0 → Physical Block 3  [The |quick|brown|fox  ]
  Logical Block 1 → Physical Block 7  [jumps|over|the |lazy ]  
  Logical Block 2 → Physical Block 1  [dog  |gen. |more|tok. ]

物理 Block 不需要连续!

┌──────────── GPU KV Cache Memory ────────────────────────┐
│                                                          │
│  Block 0: [Seq B, blk 0 ]  ← 被 Seq B 使用             │
│  Block 1: [Seq A, blk 2 ]  ← Seq A 的第 3 块           │
│  Block 2: [   空闲       ]                               │
│  Block 3: [Seq A, blk 0 ]  ← Seq A 的第 1 块           │
│  Block 4: [Seq C, blk 0 ]  ← 被 Seq C 使用             │
│  Block 5: [   空闲       ]                               │
│  Block 6: [Seq B, blk 1 ]  ← 被 Seq B 使用             │
│  Block 7: [Seq A, blk 1 ]  ← Seq A 的第 2 块           │
│                                                          │
│  物理上不连续, 但通过 Block Table 逻辑上连续!            │
│  没有外部碎片! 空闲块可立即分配给任何序列!               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 为什么 PagedAttention 能减少浪费

### 消除内部碎片

```
传统方案:
  预分配 max_seq_len = 4096 tokens
  实际用 500 tokens
  浪费: 3596 tokens 的空间 (87.8%)

PagedAttention:
  只分配 ceil(500/16) = 32 个 Block
  最后一个 Block 可能有内部碎片: 500 % 16 = 4 → 浪费 12 slots
  浪费: 12/500 = 2.4% !!!

碎片率从 87.8% 降到 2.4% → 几乎为零
```

### 消除外部碎片

```
传统方案:
  需要连续空间 → 即使总空闲够, 也可能分配失败
  
  Memory: [AAA][  ][BBB][  ][CCC][    ][DDD]
  想分配 6 个连续单元 → 失败! (最大连续空闲 = 4)

PagedAttention:
  不需要连续空间 → 有空闲 Block 就能分配
  
  Memory: [AAA][  ][BBB][  ][CCC][    ][DDD]
  想分配 6 个 Block → 成功! 使用 [ ][ ][ ][ ][ ][ ] 
  (散落在各处, 通过 Block Table 映射)
```

### 按需分配

```
传统方案:
  请求到达 → 立即分配 max_seq_len 空间
  (还不知道会生成多少 token)

PagedAttention:
  请求到达 → 分配 prompt 长度的 Block
  每生成 token → 当前 Block 满了才分配新 Block
  请求完成 → 立即释放所有 Block

时间线对比:
  传统:  [████████████████████████████░░░░░░░░░░░░░░]
         ↑ 立即分配 max                   释放 ↑
         
  Paged: [████]→[████████]→[████████████]→释放
         ↑按需  ↑需要时再分配  ↑继续分配
```

## Copy-on-Write (CoW)

### 场景: Parallel Sampling

```
请求: "Write a poem about AI" (temperature=0.8, n=4)
→ 需要生成 4 个不同的回答, 共享同一个 prompt

朴素方案: 复制 4 份 prompt 的 KV Cache
  显存: 4 × prompt_KV_size

Copy-on-Write 方案:
  
  Seq 1 ──┐
  Seq 2 ──┤── 共享 prompt 的 Block (引用计数 = 4)
  Seq 3 ──┤
  Seq 4 ──┘

  Block Table:
    Seq 1: [Block 3, Block 7, Block 1, ...]  ← 前 N 个 Block 共享
    Seq 2: [Block 3, Block 7, Block 1, ...]  ← 相同物理 Block!
    Seq 3: [Block 3, Block 7, Block 1, ...]
    Seq 4: [Block 3, Block 7, Block 1, ...]

  Physical Block 3: ref_count = 4

  当 Seq 1 需要修改最后一个共享 Block 时:
    if ref_count > 1:
      new_block = allocate()           # 分配新 Block
      copy(shared_block, new_block)    # 复制内容
      shared_block.ref_count -= 1      # 减少引用
      seq1.block_table[-1] = new_block # 更新映射
    
  显存节省: prompt 部分只存 1 份, 节省 (n-1)/n = 75%
```

### 场景: Beam Search

```
Beam Search with beam_width=4:

Step 1: 4 个 beam 共享 prompt KV Cache
  Beam 0,1,2,3 → 都指向相同的 prompt Blocks

Step 2: 每个 beam 扩展不同的 token
  Beam 0 → "The"  (新 Block)
  Beam 1 → "A"    (新 Block)
  Beam 2 → "In"   (新 Block)
  Beam 3 → "The"  (可能和 Beam 0 共享!)

Step 3: 剪枝, 保留 top-4 beams
  Beam 0 被丢弃 → 释放其独有 Block
  Beam 1 被选中 → 可能 fork 出新 beam (CoW)
  
  CoW 避免了大量 KV Cache 复制!
```

## PagedAttention Kernel

### 计算逻辑

```
传统 Attention:
  Output = softmax(Q × K^T / √d) × V
  K, V 是连续的 tensor

PagedAttention:
  K, V 分散在不同的物理 Block 中
  需要通过 Block Table 找到实际位置

伪代码:
  for each query token q:
    scores = []
    for block_idx in block_table[seq_id]:
      block = physical_blocks[block_idx]
      k_block = block.key   # [block_size, head_dim]
      s = q @ k_block.T     # [1, block_size]
      scores.append(s)
    
    scores = concat(scores)   # [1, total_seq_len]
    weights = softmax(scores / sqrt(d))
    
    output = 0
    for i, block_idx in enumerate(block_table[seq_id]):
      block = physical_blocks[block_idx]
      v_block = block.value
      w = weights[i*block_size : (i+1)*block_size]
      output += w @ v_block  # 加权求和
    
    return output
```

### 性能考虑

```
PagedAttention 比传统方案慢吗?

理论:
  - 额外开销: Block Table 查找 + 非连续内存访问
  - 非连续访问 → 可能降低 GPU Cache 命中率

实际:
  - vLLM 的 CUDA kernel 做了深度优化
  - Block 内部是连续的 (block_size=16 的 mini-batch)
  - 性能损失 < 5% (可忽略)
  
  但显存利用率提升 60-90% → batch size 提升 2-4x → 吞吐提升 2-4x!
  
  5% 的 kernel 开销换来 200-400% 的吞吐提升 → 非常值得!
```

## Block Size 的选择

```
Block Size 的权衡:

小 Block (如 block_size=1):
  ✓ 零内部碎片
  ✗ Block Table 太大
  ✗ 太多小块 → kernel 效率低 (内存不连续)
  ✗ 管理开销大

大 Block (如 block_size=256):
  ✓ 减少 Block 数量, 管理简单
  ✓ 块内连续, kernel 高效
  ✗ 内部碎片增大 (平均浪费 block_size/2)
  
推荐: block_size=16 (vLLM 默认)
  - 内部碎片: 平均浪费 8 tokens ≈ 8×0.3KB = 2.4KB (微不足道)
  - Block Table 大小合理
  - Kernel 效率好 (16 tokens 的 mini-batch)

vLLM 参数: --block-size 16 (默认值)
```

## 与传统方案的完整对比

```
┌──────────────────────────────────────────────────────────────┐
│                  传统方案 vs PagedAttention                    │
├────────────────┬────────────────────┬────────────────────────┤
│    维度         │   传统 (连续分配)   │   PagedAttention       │
├────────────────┼────────────────────┼────────────────────────┤
│ 分配策略        │  预分配 max_len    │  按需分配 Block        │
│ 内部碎片        │  60-90%           │  < 4% (最后一个 Block)  │
│ 外部碎片        │  严重 (连续要求)   │  无 (任意 Block 可用)  │
│ 显存利用率      │  20-40%           │  > 95%                 │
│ 可服务 batch    │  小               │  大 (2-4x)             │
│ CoW 支持        │  需要复制整个 KV   │  Block 级 CoW          │
│ Swap 支持       │  整个序列 swap     │  Block 级 swap         │
│ 共享 (prefix)   │  不支持           │  Block 级共享           │
│ Kernel 性能     │  最优 (连续访存)   │  微小开销 (< 5%)      │
│ 实现复杂度      │  简单             │  复杂 (Block Table等)  │
│ 总吞吐         │  基准             │  2-4x 提升             │
└────────────────┴────────────────────┴────────────────────────┘
```

## Prefix Caching（前缀缓存）

### 动机

```
大量请求共享相同前缀 (System Prompt):

请求 1: [System Prompt (2000 tokens)] + [User: "Hello"]
请求 2: [System Prompt (2000 tokens)] + [User: "Help me"]
请求 3: [System Prompt (2000 tokens)] + [User: "Write code"]

每个请求都要 Prefill System Prompt → 重复计算 + 重复存储!

Prefix Caching:
  System Prompt 的 KV Cache Block → 只计算一次, 所有请求共享
  新请求: 直接复用已有的 Block (ref_count++)
  
  节省:
  - Prefill 计算: 2000 tokens × N 个请求 → 只算 1 次
  - KV Cache 显存: 2000 × 0.3MB × N → 只存 1 份
  
  vLLM 参数: --enable-prefix-caching
```

### 实现

```
Prefix Caching 的 Block 管理:

Hash-based Block Matching:
  1. 对每个 Block 的 token content 计算 hash
  2. 新请求的 token 序列 → 按 block_size 分块 → 查 hash
  3. 命中 → 复用物理 Block (跳过计算)
  4. 未命中 → 新分配 Block, Prefill 计算

Block Hash:
  hash(block) = hash(token_ids[i*block_size : (i+1)*block_size])
  
  注意: 第 i 个 Block 的 hash 需要依赖前面所有 Block
  (因为 attention 有因果性, 相同 tokens 在不同位置的 KV 不同)
  
  实际: hash(block_i) = hash(prefix_hash || token_ids_i)
```

## 知识要点框架

### "请解释 PagedAttention 的原理"

```
"PagedAttention 借鉴了操作系统虚拟内存的分页机制:

1. 核心思想:
   把 KV Cache 显存划分为固定大小的物理 Block (类比物理页)
   每个序列通过 Block Table (类比页表) 映射到物理 Block
   物理 Block 不需要连续

2. 解决了三个问题:
   - 内部碎片: 按需分配, 从 60-90% 降到 < 4%
   - 外部碎片: 不要求连续, 完全消除
   - 共享浪费: Copy-on-Write, Block 级复用

3. 效果:
   显存利用率从 20-40% 提升到 95%+
   同样的显存可以服务 2-4x 的 batch size
   吞吐提升 2-4x, kernel 开销 < 5%

4. 额外能力:
   - Prefix Caching: 共享 System Prompt 的 KV
   - Beam Search 优化: CoW 避免 KV 复制
   - GPU-CPU Swap: Block 级别的交换, 更灵活"
```

### 追问: "Block Size 怎么选？"

```
"Block Size 是碎片和效率的权衡:
- 太小: 管理开销大, kernel 效率低
- 太大: 内部碎片增加
- 默认 16 tokens 是经验值
- 内部碎片只有最后一个 Block, 平均浪费 8 tokens ≈ 2.4KB
- 这个浪费相比传统方案的 GB 级浪费可忽略不计"
```

## 小结

| 概念 | 类比 | 解决的问题 |
|------|------|-----------|
| Physical Block | 物理页 | 固定大小, 消除外部碎片 |
| Block Table | 页表 | 虚拟连续 → 物理不连续 |
| 按需分配 | Demand Paging | 消除内部碎片 |
| Copy-on-Write | CoW | 共享复用, 减少复制 |
| Swap | Page Swap | GPU↔CPU 显存管理 |
| Prefix Cache | Shared Memory | 前缀复用, 减少重复 |

**一句话总结**: PagedAttention 把 KV Cache 管理从"为每个序列预分配连续大块"变成"按需分配固定大小的块并通过页表映射"，将显存利用率从 20-40% 提升到 95%+，是 vLLM 最核心的创新。
