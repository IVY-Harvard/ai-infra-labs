# 一个请求的完整生命周期追踪

## 场景

用户发送: `POST /v1/chat/completions` with `{"messages": [{"role": "user", "content": "Hello"}]}`

## 阶段 1: 请求接收

```
entrypoints/openai/api_server.py
  → app.post("/v1/chat/completions")
    → create_chat_completion()
      → 解析请求参数 (temperature, max_tokens, etc.)
      → engine.generate(prompt, sampling_params, request_id)
```

源码位置: `vllm/entrypoints/openai/serving_chat.py`

## 阶段 2: 进入引擎

```
engine/async_llm_engine.py: AsyncLLMEngine.generate()
  → engine/llm_engine.py: LLMEngine.add_request()
    → tokenizer.encode("Hello") → [15496]
    → 创建 SequenceGroup:
        seq = Sequence(seq_id=0, prompt_token_ids=[15496])
        seq_group = SequenceGroup(request_id="req-xxx", seqs=[seq], ...)
    → scheduler.add_seq_group(seq_group)
        → 放入 self.waiting 队列
```

源码位置: `vllm/engine/llm_engine.py` 的 `add_request()`

## 阶段 3: 调度

```
engine/llm_engine.py: LLMEngine.step()
  → core/scheduler.py: Scheduler.schedule()
    
    _schedule_default():
      # Phase 1: running 中完成的 → 释放 Block
      for seq_group in self.running:
        if seq_group.is_finished():
          block_manager.free(seq)
          
      # Phase 2: 尝试恢复 swapped
      for seq_group in self.swapped:
        if block_manager.can_swap_in(seq_group):
          block_manager.swap_in(seq_group)
          self.running.append(seq_group)
          
      # Phase 3: 调度新请求 (我们的请求在这里!)
      for seq_group in self.waiting:
        if block_manager.can_allocate(seq_group):
          block_manager.allocate(seq_group)  # 分配初始 Block!
          self.running.append(seq_group)
          # 标记为 PREFILL
    
    返回 SchedulerOutput:
      scheduled_seq_groups = [...]
      blocks_to_swap_in = {}
      blocks_to_swap_out = {}
```

关键: `block_manager.allocate()` 只分配 prompt 长度的 Block (不是 max_seq_len!)

## 阶段 4: Prefill 执行

```
worker/model_runner.py: ModelRunner.execute_model(scheduler_output)
  → prepare_model_input():
      # 收集所有需要 Prefill 的请求
      token_ids = [15496]  # "Hello" 的 token
      positions = [0]
      # Block Table: seq_id → [physical_block_ids]
      block_tables = {0: [7]}  # 例如分配到 Block 7
      
  → model.forward(token_ids, positions, kv_caches, attn_metadata)
      # Layer 0: 
      #   QKV projection → q, k, v
      #   写入 KV Cache: kv_caches[0][block_7] = k, v
      #   Self-Attention (FlashAttention)
      #   FFN
      # Layer 1: ... 同上
      # ...
      # Layer L-1: 最后一层
      #   → hidden_states
      
  → sampler.forward(hidden_states)
      # logits = hidden_states @ lm_head_weight
      # 应用 temperature, top_p, etc.
      # token = sample(logits)
      # 假设采样到 token_id = 306 ("I")
      
  返回: SamplerOutput(token_ids=[306], ...)
```

## 阶段 5: Decode 迭代

```
每次 LLMEngine.step():

1. Scheduler.schedule():
   → block_manager.can_append_slot(seq_group) 
   → block_manager.append_slot(seq)  # 为新 token 分配 slot
   (如果当前 Block 满了 → 分配新 Block)

2. ModelRunner.execute_model():
   → 输入只有 1 个 token: [306] ("I")
   → position: [1]
   → 与 Block 7 中已有的 KV 做 PagedAttention
   → 采样下一个 token, 假设得到 289 ("'m")

3. 重复...
   Step 3: input=[289], 采样得 "a"
   Step 4: input=[64], 采样得 " helpful"
   ...
   直到: 生成 EOS 或达到 max_tokens
```

## 阶段 6: 完成与返回

```
LLMEngine.step():
  → process_outputs():
      if stop_condition_met (EOS or max_tokens):
        seq_group.status = FINISHED
        
  → Scheduler: 从 running 移除
  → BlockManager: free(seq) → 释放所有 Block
  → Detokenize: [15496, 306, 289, 64, ...] → "Hello, I'm a helpful..."
  → 通过 SSE 流式返回给客户端
```

## 数据结构变化追踪

```
Step 0 (请求到达):
  Sequence: tokens=[15496], status=WAITING
  BlockTable: (未分配)

Step 1 (Prefill):
  Sequence: tokens=[15496, 306], status=RUNNING
  BlockTable: [Block 7] (Block 7 存了 1 token 的 KV)

Step 2 (Decode):
  Sequence: tokens=[15496, 306, 289], status=RUNNING
  BlockTable: [Block 7] (Block 7 存了 2 tokens 的 KV)

...

Step N (Block 满了):
  Sequence: tokens=[15496, ..., token_16, token_17], status=RUNNING
  BlockTable: [Block 7, Block 12] (分配了新 Block 12)

Final (完成):
  Sequence: status=FINISHED
  BlockTable: (释放, Block 7 和 12 回到 free pool)
```
