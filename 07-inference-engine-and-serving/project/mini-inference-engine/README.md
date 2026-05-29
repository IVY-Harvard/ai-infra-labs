# Mini Inference Engine

一个简化版 LLM 推理引擎，实现核心功能：

- **PagedAttention** — 分页 KV Cache 管理
- **Continuous Batching** — 连续批处理调度
- **OpenAI 兼容 API** — `/v1/chat/completions`
- **流式输出** — SSE Streaming
- **Prometheus 监控** — TTFT/TPOT/吞吐指标

## 架构

```
API Request → FastAPI Server → LLMEngine → Scheduler → Worker → GPU
                  │                │           │          │
                  │                │       BlockManager  ModelRunner
                  │                │           │          │
              Streaming ←     Results ←    KV Cache ← PagedAttention
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务 (使用小模型测试)
python -m src.serving.api_server --model gpt2 --port 8000

# 发送请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# 运行测试
pytest tests/

# 压测
python benchmarks/throughput_benchmark.py
```

## 项目结构

```
src/
├── core/
│   ├── engine.py          # LLMEngine 主入口
│   ├── scheduler.py       # 请求调度器
│   ├── sequence.py        # Sequence 数据结构
│   └── block_manager.py   # Block 分配器
├── attention/
│   ├── paged_attention.py # PagedAttention 计算
│   └── cache_engine.py    # KV Cache 引擎
├── model/
│   ├── model_loader.py    # 模型加载器
│   └── model_runner.py    # 模型执行器
├── serving/
│   ├── api_server.py      # FastAPI 服务
│   └── streaming.py       # SSE 流式输出
├── worker/
│   ├── worker.py          # Worker 进程
│   └── tp_worker.py       # Tensor Parallel Worker
└── metrics/
    ├── prometheus_metrics.py  # Prometheus 指标
    └── latency_tracker.py     # 延迟追踪
```

## 设计理念

这不是生产级引擎，而是学习工具。核心目标：

1. **可读性** > 性能：代码清晰，注释充分
2. **核心完整**：Scheduler + BlockManager + PagedAttention 逻辑完整
3. **可运行**：能实际加载小模型 (GPT-2) 跑通全流程
4. **对照 vLLM**：命名和结构对齐 vLLM，方便源码对照
