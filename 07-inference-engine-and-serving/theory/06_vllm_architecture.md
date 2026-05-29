# 06 - vLLM 整体架构

## 概述

vLLM 是目前最流行的 LLM 推理引擎。理解它的架构，是从"会用"进阶到"理解"的关键。

## 架构全景

```
┌──────────────────────────────────────────────────────────────┐
│                        vLLM Architecture                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────┐             │
│  │              API Layer                       │             │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │             │
│  │  │ OpenAI   │  │  gRPC    │  │ Offline   │  │             │
│  │  │ API      │  │ Service  │  │ LLM()     │  │             │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  │             │
│  └───────┼──────────────┼─────────────┼────────┘             │
│          └──────────────┼─────────────┘                      │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │              LLMEngine                        │            │
│  │  (核心引擎, 协调所有组件)                       │            │
│  │                                               │            │
│  │  ┌──────────┐  ┌──────────────┐              │            │
│  │  │Tokenizer │  │ Scheduler    │              │            │
│  │  │          │  │ ┌──────────┐ │              │            │
│  │  │          │  │ │ waiting  │ │              │            │
│  │  │          │  │ │ running  │ │              │            │
│  │  │          │  │ │ swapped  │ │              │            │
│  │  │          │  │ └──────────┘ │              │            │
│  │  └──────────┘  │ ┌──────────┐ │              │            │
│  │                │ │BlockMgr  │ │              │            │
│  │                │ │(GPU+CPU) │ │              │            │
│  │                │ └──────────┘ │              │            │
│  │                └──────────────┘              │            │
│  └────────────────────┬─────────────────────────┘            │
│                       │ execute_model()                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────┐            │
│  │              Worker(s)                        │            │
│  │                                               │            │
│  │  Worker 0 (GPU 0)  Worker 1 (GPU 1)  ...     │            │
│  │  ┌──────────────┐  ┌──────────────┐          │            │
│  │  │ ModelRunner  │  │ ModelRunner  │          │            │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │          │            │
│  │  │ │  Model   │ │  │ │  Model   │ │          │            │
│  │  │ │(Attention │ │  │ │(Attention │ │          │            │
│  │  │ │+ FFN)    │ │  │ │+ FFN)    │ │          │            │
│  │  │ └──────────┘ │  │ └──────────┘ │          │            │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │          │            │
│  │  │ │CacheEngine│ │  │ │CacheEngine│ │          │            │
│  │  │ └──────────┘ │  │ └──────────┘ │          │            │
│  │  └──────────────┘  └──────────────┘          │            │
│  └──────────────────────────────────────────────┘            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. LLMEngine — 大脑

```
职责:
  - 接收请求 (add_request)
  - 协调 Scheduler 和 Worker
  - 驱动主循环 (step)
  - 返回结果

核心方法:
  add_request(request_id, prompt, params)  → 加入 waiting 队列
  step()                                   → 执行一步推理
  abort_request(request_id)                → 取消请求

主循环 (简化):
  while has_unfinished_requests():
    scheduler_output = scheduler.schedule()  # 决定这步执行谁
    output = worker.execute_model(scheduler_output)  # 执行推理
    results = process_output(output)  # 处理输出 (采样等)
    update_sequences(results)  # 更新序列状态
    yield results  # 返回给调用者
```

### 2. Scheduler — 调度中心

```
职责:
  - 管理三个队列 (waiting/running/swapped)
  - 每步决定哪些序列参与计算
  - 管理 BlockManager (分配/释放 Block)
  - 处理抢占逻辑

关键数据结构:
  SequenceGroup: 一个请求 (可能包含多个 Sequence, 如 beam search)
  Sequence: 一个具体的 token 序列
  
  waiting:  deque[SequenceGroup]  # FIFO 等待队列
  running:  deque[SequenceGroup]  # 正在执行的请求
  swapped:  deque[SequenceGroup]  # 被抢占到 CPU 的请求

调度优先级:
  1. swapped (恢复被抢占的, 避免饥饿)
  2. running (继续执行中的请求)
  3. waiting (调度新请求)
```

### 3. BlockManager — 显存管家

```
职责:
  - 管理 GPU 和 CPU 的物理 Block 池
  - 为序列分配/释放 Block
  - 处理 Copy-on-Write
  - 管理 Swap (GPU ↔ CPU)

核心数据结构:
  BlockTable: Dict[seq_id, List[PhysicalBlock]]
  FreeBlocks: List[PhysicalBlock]  # GPU 空闲 Block 池
  CpuFreeBlocks: List[PhysicalBlock]  # CPU 空闲 Block 池
  
关键方法:
  can_allocate(seq_group) → bool     # 能否分配足够 Block
  allocate(seq_group)                 # 分配初始 Block (Prefill)
  can_append_slot(seq_group) → bool   # 能否分配新 slot (Decode)
  append_slot(seq)                    # 为新 token 分配 slot
  free(seq)                           # 释放序列的所有 Block
  swap_out(seq_group) → mapping       # GPU → CPU
  swap_in(seq_group) → mapping        # CPU → GPU
```

### 4. Worker — 执行者

```
职责:
  - 管理一个 GPU
  - 加载和运行模型
  - 管理 CacheEngine (KV Cache 物理内存)

架构 (Tensor Parallelism):
  TP=8 时: 8 个 Worker, 每个管理 1 个 GPU
  Worker 0 (rank 0, GPU 0) ← Driver (接收调度指令)
  Worker 1 (rank 1, GPU 1) ← 跟随者
  ...
  Worker 7 (rank 7, GPU 7) ← 跟随者

通信:
  Driver Worker 广播指令 → 所有 Worker 同步执行
  NCCL All-Reduce 同步中间结果
```

### 5. ModelRunner — 模型执行器

```
职责:
  - 准备模型输入 (token ids, positions, attention metadata)
  - 执行前向传播
  - 采样输出 token

两种执行模式:
  Prefill: 输入 N 个 token → 输出 KV Cache + 1st token
  Decode:  输入 1 个 token × B 个序列 → 输出 B 个新 token

关键方法:
  prepare_input(scheduler_output) → ModelInput
  execute_model(model_input) → SamplerOutput
```

### 6. CacheEngine — KV Cache 物理层

```
职责:
  - 管理 GPU/CPU 上 KV Cache 的物理内存
  - 执行 Block 间的 copy/swap 操作

物理结构:
  gpu_cache: List[Tuple[Tensor, Tensor]]  # per layer: (K, V)
  cpu_cache: List[Tuple[Tensor, Tensor]]  # per layer: (K, V)
  
  每个 tensor shape: [num_blocks, block_size, num_kv_heads, head_dim]
  
  gpu_cache[layer_idx][0]  → K cache for layer
  gpu_cache[layer_idx][1]  → V cache for layer
  
  访问 Block 7, Layer 3 的 K:
    gpu_cache[3][0][7]  → [block_size, num_kv_heads, head_dim]
```

## 请求生命周期

### 完整流程

```
┌──────────────────────────────────────────────────────────┐
│  一个请求的完整生命周期                                    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. 请求到达                                              │
│     Client → API Server → LLMEngine.add_request()        │
│     → Tokenize prompt                                     │
│     → 创建 SequenceGroup + Sequence                      │
│     → 加入 Scheduler.waiting 队列                         │
│                                                           │
│  2. 调度 Prefill                                          │
│     Scheduler.schedule():                                 │
│       waiting 中取出 → 检查 BlockManager.can_allocate()   │
│       → allocate() 分配初始 Block                         │
│       → 加入 scheduled_prefills                           │
│     状态: WAITING → RUNNING                               │
│                                                           │
│  3. 执行 Prefill                                          │
│     Worker.execute_model():                               │
│       ModelRunner.prepare_input() → 准备输入               │
│       Model.forward() → 前向传播 (计算所有层的 KV)        │
│       → KV 写入 CacheEngine 的 Block 中                  │
│       Sampler → 采样第一个输出 token                      │
│     返回: first token + 更新 Sequence                     │
│                                                           │
│  4. 迭代 Decode                                           │
│     每步:                                                 │
│       Scheduler.schedule():                               │
│         检查 can_append_slot() → append_slot() 分配新 slot│
│       Worker.execute_model():                             │
│         新 token 的 Q 与 KV Cache 做 PagedAttention       │
│         采样下一个 token                                  │
│       检查是否完成:                                       │
│         hit max_tokens? → 完成                            │
│         generated EOS? → 完成                             │
│         被抢占? → swap out → SWAPPED 状态                 │
│                                                           │
│  5. 完成                                                  │
│     Scheduler: 从 running 移除                            │
│     BlockManager: free() 释放所有 Block                   │
│     Detokenize: token ids → text                         │
│     API Server: 返回结果给 Client                         │
│     状态: RUNNING → FINISHED                              │
│                                                           │
│  [异常路径]                                               │
│  5a. 被抢占 (Preempted)                                   │
│      BlockManager.swap_out(): GPU Block → CPU Block      │
│      加入 swapped 队列                                    │
│      状态: RUNNING → SWAPPED                              │
│      稍后: swap_in() 恢复 → 继续 Decode                  │
│                                                           │
│  5b. 被取消 (Aborted)                                     │
│      LLMEngine.abort_request()                            │
│      释放资源, 从队列移除                                  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Tensor Parallelism 在 vLLM 中的实现

```
8×H20 运行 LLaMA-70B (TP=8):

模型分片:
  每个 Attention head → 分配到不同 GPU
  FFN 的权重 → 列/行切分到 8 个 GPU

Worker 通信:
  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │GPU 0 │──│GPU 1 │──│GPU 2 │──│GPU 3 │
  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
     │         │         │         │     NCCL
  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐
  │GPU 4 │──│GPU 5 │──│GPU 6 │──│GPU 7 │
  └──────┘  └──────┘  └──────┘  └──────┘

每步推理:
  1. Driver Worker (GPU 0) 广播: scheduler_output
  2. 所有 Worker: 各自计算自己负责的 head/FFN 分片
  3. All-Reduce: 同步 Attention 输出 + FFN 输出
  4. Driver Worker: 收集 logits, 执行采样
  5. Driver Worker: 广播采样结果
```

## vLLM 配置参数详解

```
# 核心配置 (你的 8×H20 环境)

vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 8        # TP=8, 使用 8 个 GPU
  --gpu-memory-utilization 0.9    # KV Cache 使用 90% 可用显存
  --max-model-len 4096            # 最大序列长度
  --block-size 16                 # Block 大小 (tokens)
  --swap-space 4                  # CPU swap 空间 (GB)
  --max-num-seqs 256              # 最大并发序列数
  --enable-chunked-prefill        # 启用分块 Prefill
  --max-num-batched-tokens 2048   # 每步最大 token 数
  --enable-prefix-caching         # 启用前缀缓存
  --kv-cache-dtype fp8            # KV Cache 使用 FP8
  
# 性能相关
  --enforce-eager                 # 禁用 CUDA Graph (调试用)
  --disable-log-stats             # 关闭统计日志 (生产环境)
  
# 采样参数 (请求级)
  temperature=0.7
  top_p=0.9
  max_tokens=512
  presence_penalty=0.0
  frequency_penalty=0.0
```

## 源码目录结构

```
vllm/
├── engine/
│   ├── llm_engine.py          # LLMEngine 核心引擎
│   ├── async_llm_engine.py    # 异步引擎 (API Server 用)
│   └── arg_utils.py           # 参数解析
├── core/
│   ├── scheduler.py           # Scheduler 调度器 ← 最核心
│   ├── block_manager.py       # BlockManager ← 最核心
│   └── block/                 # Block 相关实现
│       ├── block_table.py
│       ├── cpu_gpu_block_allocator.py
│       └── prefix_caching_block.py
├── worker/
│   ├── worker.py              # GPU Worker
│   ├── model_runner.py        # ModelRunner ← 核心
│   └── cache_engine.py        # CacheEngine
├── model_executor/
│   ├── models/                # 各模型实现
│   │   ├── llama.py
│   │   ├── qwen2.py
│   │   └── ...
│   └── layers/
│       ├── attention/         # Attention 层
│       │   ├── attention.py
│       │   └── backends/
│       │       ├── flash_attn.py
│       │       └── paged_attn.py  # PagedAttention kernel
│       ├── linear.py
│       └── sampler.py         # 采样器
├── attention/
│   ├── ops/
│   │   └── paged_attn.py      # PagedAttention CUDA kernel 接口
│   └── backends/
└── entrypoints/
    ├── openai/                # OpenAI 兼容 API
    │   ├── api_server.py
    │   └── serving_chat.py
    └── llm.py                 # 离线推理入口
```

## 核心知识点

### "描述 vLLM 的请求处理流程"

```
回答框架:

"vLLM 的请求处理经过以下阶段:

1. 请求入队: API Server → LLMEngine.add_request() → 
   Tokenize → 创建 SequenceGroup → 进入 waiting 队列

2. 调度: Scheduler 每步执行 schedule():
   - 从 waiting 取请求, 检查 BlockManager 是否有足够 Block
   - 有 → allocate Block, 加入 running
   - 没有 → 尝试抢占 running 中优先级低的请求

3. 执行: Worker.execute_model()
   - Prefill: 一次处理所有 prompt token, 写入 KV Cache Block
   - Decode: 每步一个 token, PagedAttention 读取 KV Cache

4. 完成: 释放 Block, Detokenize, 返回结果

核心组件配合:
- Scheduler 决定"谁参与计算"
- BlockManager 管理"KV Cache 放哪里"
- Worker/ModelRunner 执行"具体的推理计算"
- CacheEngine 管理"KV Cache 的物理存储"

关键创新:
- PagedAttention: 按需分页分配 KV Cache
- Continuous Batching: 每步都可以调整 batch 组成
- Chunked Prefill: 避免长 Prefill 阻塞 Decode"
```

## 小结

| 组件 | 职责 | 关键方法 |
|------|------|----------|
| LLMEngine | 协调所有组件 | add_request(), step() |
| Scheduler | 决定每步执行谁 | schedule() |
| BlockManager | 管理 KV Cache Block | allocate(), append_slot(), free() |
| Worker | 管理 GPU, 执行模型 | execute_model() |
| ModelRunner | 准备输入, 前向传播 | prepare_input(), execute_model() |
| CacheEngine | KV Cache 物理存储 | copy(), swap_in(), swap_out() |
