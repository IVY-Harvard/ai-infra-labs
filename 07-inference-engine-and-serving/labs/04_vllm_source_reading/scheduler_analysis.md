# Scheduler 核心逻辑分析

## 源码位置

`vllm/core/scheduler.py`

## 核心数据结构

```python
class Scheduler:
    def __init__(self, ...):
        self.waiting: Deque[SequenceGroup] = deque()   # 等待 Prefill
        self.running: Deque[SequenceGroup] = deque()   # 正在 Decode
        self.swapped: Deque[SequenceGroup] = deque()   # 被抢占到 CPU
        
        self.block_manager = BlockSpaceManager(...)
```

## schedule() 主逻辑

```python
def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
    """每步调用一次，决定这步执行谁"""
    
    # 核心调度逻辑
    scheduler_outputs = self._schedule()
    
    # 准备 SequenceGroupMetadata (给 ModelRunner)
    seq_group_metadata_list = []
    for scheduled in scheduler_outputs.scheduled_seq_groups:
        seq_group = scheduled.seq_group
        # 包含: token_ids, block_tables, is_prompt, etc.
        metadata = SequenceGroupMetadata(...)
        seq_group_metadata_list.append(metadata)
    
    return seq_group_metadata_list, scheduler_outputs
```

## _schedule() 的三阶段

```python
def _schedule(self) -> SchedulerOutputs:
    """调度的核心: 三阶段处理"""
    
    budget = SchedulingBudget(
        token_budget=self.scheduler_config.max_num_batched_tokens,
        max_num_seqs=self.scheduler_config.max_num_seqs,
    )
    
    # Phase 1: 处理 running 中的请求
    # → 完成的释放, 需要更多 Block 的分配
    running_scheduled = self._schedule_running(budget)
    
    # Phase 2: 恢复 swapped 请求
    # → 如果有足够 GPU Block, swap in
    swapped_in = self._schedule_swapped(budget)
    
    # Phase 3: 调度新请求 (从 waiting → running)
    # → 如果有足够 Block, allocate 并开始 Prefill
    prefills = self._schedule_prefills(budget)
    
    return SchedulerOutputs(
        scheduled_seq_groups=running_scheduled + swapped_in + prefills,
        ...
    )
```

## _schedule_running() 详解

```python
def _schedule_running(self, budget) -> List:
    """处理正在运行的请求"""
    
    scheduled = []
    
    while self.running:
        seq_group = self.running[0]
        
        # 检查是否完成
        if seq_group.is_finished():
            self.running.popleft()
            self._free_seq_group(seq_group)
            continue
        
        # 检查能否继续 (有没有 Block 给新 token)
        while not self._can_append_slots(seq_group):
            # Block 不够! 需要抢占
            # 从 running 末尾取优先级最低的请求
            victim = self.running.pop()
            self._preempt(victim)  # swap out 或 recompute
        
        # 分配新 slot
        self._append_slots(seq_group)
        scheduled.append(seq_group)
        self.running.popleft()
    
    # 重新放回 running
    for sg in scheduled:
        self.running.append(sg)
    
    return scheduled
```

## 抢占策略

```python
def _preempt(self, seq_group):
    """抢占: 当 GPU 显存不足时"""
    
    if self.scheduler_config.preemption_mode == "swap":
        # Swap: GPU Block → CPU Block
        self._preempt_by_swap(seq_group)
    else:
        # Recompute: 丢弃 KV Cache, 重新 Prefill
        self._preempt_by_recompute(seq_group)

def _preempt_by_swap(self, seq_group):
    """Swap Out: KV Cache 从 GPU 移到 CPU"""
    self.block_manager.swap_out(seq_group)
    seq_group.status = SWAPPED
    self.swapped.appendleft(seq_group)

def _preempt_by_recompute(self, seq_group):
    """Recompute: 丢弃 KV Cache, 回到 WAITING 重新 Prefill"""
    self.block_manager.free(seq_group)
    seq_group.status = WAITING
    seq_group.reset()  # 重置生成状态
    self.waiting.appendleft(seq_group)  # 放到队首 (优先级高)
```

## 关键设计决策

### 1. FCFS (First Come First Served)

```
为什么用 FCFS 而不是更复杂的调度算法?

- 简单可预测: 用户请求按到达顺序处理
- 公平性: 没有请求被饿死
- 低开销: O(1) 调度决策
- 实际够用: 配合 Continuous Batching, FCFS 已经很高效

更复杂的策略 (SJF, 优先级队列) 在特定场景有优势,
但增加了复杂度和不可预测性。
```

### 2. 预算控制 (Budget)

```python
class SchedulingBudget:
    token_budget: int   # 这步最多处理多少 token
    max_num_seqs: int   # 这步最多多少序列
    
# 为什么需要预算?
# 1. 控制 GPU 内存使用 (不能一次 Prefill 太长)
# 2. 控制延迟 (一步不能做太多事)
# 3. Chunked Prefill: token_budget 限制每步 Prefill 的量
```

### 3. Swapped vs Recompute

```
什么时候 Swap? 什么时候 Recompute?

Swap 更好:
  - 长序列 (KV Cache 大, 重算代价高)
  - CPU 内存充足
  - PCIe 带宽够

Recompute 更好:
  - 短序列 (Prefill 很快)
  - CPU 内存紧张
  - 序列刚开始 (KV Cache 小)

vLLM 默认: 先尝试 Swap, 失败则 Recompute
```

## 阅读建议

1. 从 `Scheduler.schedule()` 入口开始
2. 重点关注 `_schedule_running()` (最核心)
3. 理解 `SchedulingBudget` 的作用
4. 看 `_preempt()` 的两种策略
5. 最后看 `_schedule_prefills()` 中的 Chunked Prefill 逻辑
