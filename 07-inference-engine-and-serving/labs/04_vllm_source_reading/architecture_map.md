# vLLM 核心模块关系图

## 整体架构

```
                           ┌────────────────────┐
                           │   User Request      │
                           │  (OpenAI API)       │
                           └────────┬───────────┘
                                    │
                           ┌────────▼───────────┐
                           │  AsyncLLMEngine     │
                           │  (异步封装)          │
                           └────────┬───────────┘
                                    │
                           ┌────────▼───────────┐
                           │    LLMEngine        │
                           │  ┌──────────────┐  │
                           │  │  Tokenizer   │  │
                           │  ├──────────────┤  │
                           │  │  Scheduler ──┼──┼──→ 决定"谁"参与计算
                           │  │  │           │  │
                           │  │  └─BlockMgr  │  │──→ 管理"KV 放哪"
                           │  ├──────────────┤  │
                           │  │  InputProc   │  │──→ 准备输入
                           │  └──────────────┘  │
                           └────────┬───────────┘
                                    │ execute_model()
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │ Worker 0 │   │ Worker 1 │   │ Worker N │
              │ (GPU 0)  │   │ (GPU 1)  │   │ (GPU N)  │
              │┌────────┐│   │┌────────┐│   │┌────────┐│
              ││ModelRun││   ││ModelRun││   ││ModelRun││
              ││  ┌────┐││   ││  ┌────┐││   ││  ┌────┐││
              ││  │Model│││   ││  │Model│││   ││  │Model│││
              ││  └────┘││   ││  └────┘││   ││  └────┘││
              ││CacheEng││   ││CacheEng││   ││CacheEng││
              │└────────┘│   │└────────┘│   │└────────┘│
              └──────────┘   └──────────┘   └──────────┘
                    │               │               │
                    └───── NCCL All-Reduce ─────────┘
```

## 数据流

```
请求到达 → Tokenize → add_request()
                           │
                    ┌──────▼──────┐
                    │  WAITING    │  (等待调度)
                    └──────┬──────┘
                           │ schedule()
                    ┌──────▼──────┐
                    │  RUNNING    │  (正在执行)
                    └──┬──────┬───┘
                       │      │
              完成 ←───┘      └───→ 抢占
              (FINISHED)          (SWAPPED)
```

## 关键调用链

```
LLMEngine.step()
  │
  ├── scheduler.schedule()
  │     ├── _schedule_running()    # 处理运行中的请求
  │     ├── _schedule_swapped()    # 恢复被抢占的
  │     ├── _schedule_prefills()   # 调度新请求
  │     └── block_manager.allocate/append_slot/free
  │
  ├── model_executor.execute_model(scheduler_output)
  │     ├── worker.execute_model()
  │     │     ├── model_runner.prepare_input()
  │     │     ├── model.forward()
  │     │     │     ├── attention (PagedAttention kernel)
  │     │     │     └── ffn
  │     │     └── sampler.forward()
  │     └── return SamplerOutput
  │
  └── process_outputs()
        ├── update sequences
        ├── check stop conditions
        └── yield results
```

## 核心源码文件索引

| 文件 | 行数(约) | 核心类/函数 | 读优先级 |
|------|---------|------------|---------|
| `core/scheduler.py` | ~1500 | `Scheduler._schedule()` | ⭐⭐⭐⭐⭐ |
| `core/block_manager.py` | ~800 | `BlockSpaceManager` | ⭐⭐⭐⭐⭐ |
| `engine/llm_engine.py` | ~1200 | `LLMEngine.step()` | ⭐⭐⭐⭐ |
| `worker/model_runner.py` | ~1000 | `ModelRunner.execute_model()` | ⭐⭐⭐⭐ |
| `worker/worker.py` | ~500 | `Worker` | ⭐⭐⭐ |
| `attention/backends/paged_attn.py` | ~300 | PagedAttention kernel | ⭐⭐⭐ |
| `sequence.py` | ~400 | `Sequence`, `SequenceGroup` | ⭐⭐⭐ |
