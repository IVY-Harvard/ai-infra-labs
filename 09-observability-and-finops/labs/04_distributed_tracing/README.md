# Lab 04: GPU 推理服务分布式追踪

## 概述

在 GPU 推理场景中，一个看似简单的 API 请求实际经过多个阶段：
负载均衡 → API Server → Tokenizer → Scheduler → Prefill → Decode → Detokenize → Response。

传统的 Metrics + Logs 只能告诉你 **"慢了"**，分布式追踪告诉你 **"哪里慢了"**。

本 Lab 基于 OpenTelemetry + Jaeger 构建端到端推理追踪系统，
实现从请求入口到 GPU 计算的全链路可观测。

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Inference Request Flow                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Client ─────→ Gateway ─────→ vLLM API Server ─────→ Response          │
│    │              │                  │                    ▲              │
│    │              │                  ▼                    │              │
│    │              │           ┌─────────────┐            │              │
│    │              │           │  Tokenizer  │            │              │
│    │              │           └──────┬──────┘            │              │
│    │              │                  ▼                    │              │
│    │              │           ┌─────────────┐            │              │
│    │              │           │  Scheduler  │            │              │
│    │              │           └──────┬──────┘            │              │
│    │              │                  ▼                    │              │
│    │              │    ┌─────────────────────────┐       │              │
│    │              │    │   Model Execution       │       │              │
│    │              │    │  ┌───────┐ ┌─────────┐  │       │              │
│    │              │    │  │Prefill│ │  Decode  │  │       │              │
│    │              │    │  │(TTFT) │ │  (TPOT)  │  │       │              │
│    │              │    │  └───────┘ └─────────┘  │       │              │
│    │              │    └─────────────────────────┘       │              │
│    │              │                  ▼                    │              │
│    │              │           ┌─────────────┐            │              │
│    │              │           │ Detokenizer │────────────┘              │
│    │              │           └─────────────┘                           │
│    │              │                                                     │
│    ▼              ▼                  ▼                                  │
│  ┌────────────────────────────────────────────────────────┐            │
│  │              OpenTelemetry SDK (Auto + Manual)          │            │
│  │  trace_id: abc123  span_id: xxx  parent_span: yyy     │            │
│  └────────────────────────────────────────┬───────────────┘            │
│                                           │ OTLP gRPC                  │
│                                           ▼                            │
│  ┌────────────────────────────────────────────────────────┐            │
│  │              OTel Collector (Pipeline)                   │            │
│  │  receivers → processors → exporters                     │            │
│  │  ┌─────────┐  ┌──────────────┐  ┌──────────────────┐  │            │
│  │  │OTLP gRPC│  │tail_sampling │  │ jaeger_exporter  │  │            │
│  │  │         │  │batch         │  │ prometheus_exp   │  │            │
│  │  │         │  │attributes    │  │ file_exporter    │  │            │
│  │  └─────────┘  └──────────────┘  └──────────────────┘  │            │
│  └────────────────────────────────────────┬───────────────┘            │
│                                           │                            │
│                          ┌────────────────┼────────────────┐           │
│                          ▼                ▼                ▼           │
│                    ┌──────────┐   ┌────────────┐   ┌───────────┐      │
│                    │  Jaeger  │   │ Prometheus │   │   Loki    │      │
│                    │  (Trace) │   │  (Metrics) │   │  (Logs)   │      │
│                    └──────────┘   └────────────┘   └───────────┘      │
│                          │                │                │           │
│                          └────────────────┼────────────────┘           │
│                                           ▼                            │
│                                    ┌─────────────┐                     │
│                                    │   Grafana   │                     │
│                                    │ Trace→Metric│                     │
│                                    │ Trace→Log   │                     │
│                                    └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 核心概念

### 为什么 GPU 推理需要分布式追踪？

| 传统 Web 服务追踪 | GPU 推理追踪特殊性 |
|-------------------|-------------------|
| 毫秒级 span | Prefill 可达数秒, Decode 可达数十秒 |
| 确定性延迟 | 受 KV Cache / Batch / Preemption 影响极大 |
| 单次执行 | Continuous Batching 下多请求共享 GPU 时间片 |
| CPU 密集型 | GPU 异步执行, CUDA Event 计时 |
| 简单的服务调用链 | 包含调度、抢占、swap、prefix cache 等复杂状态 |

### Trace 结构设计 (GPU 推理特化)

```
Trace: inference_request (trace_id: abc123)
│
├── Span: gateway.route (2ms)
│   └── attributes: route=/v1/chat/completions, method=POST
│
├── Span: api_server.process_request (5ms)
│   └── attributes: model=qwen2.5-72b, stream=true
│
├── Span: tokenizer.encode (3ms)
│   └── attributes: prompt_tokens=1024, truncated=false
│
├── Span: scheduler.schedule (1ms)
│   └── attributes: action=prefill, queue_position=0, waiting_time_ms=0
│
├── Span: model.prefill (450ms)  ← TTFT 核心
│   ├── attributes: num_tokens=1024, batch_size=16
│   ├── attributes: kv_cache_usage=0.72, prefix_cache_hit=0.85
│   ├── attributes: gpu_id=[0,1,2,3,4,5,6,7], tp_size=8
│   └── events:
│       ├── prefix_cache_hit (matched 870/1024 tokens)
│       └── kv_blocks_allocated (blocks=10)
│
├── Span: model.decode (3200ms)  ← 生成阶段
│   ├── attributes: output_tokens=128, avg_tpot_ms=25
│   ├── attributes: batch_size_range=[12,24], preempted=false
│   └── events:
│       ├── decode_step[0]: batch=16, tpot=22ms
│       ├── decode_step[63]: batch=20, tpot=27ms (batch 增大)
│       └── decode_step[127]: batch=12, tpot=20ms (eos)
│
├── Span: tokenizer.decode (1ms)
│   └── attributes: output_text_length=512
│
└── Span: response.stream (3250ms)
    └── attributes: chunks_sent=128, connection=keep-alive
```

---

## 文件结构

```
04_distributed_tracing/
├── README.md                    ← 本文件
├── otel_setup.py               ← OpenTelemetry 初始化 & 配置
├── inference_tracing.py         ← vLLM 推理链路追踪实现
└── jaeger_setup.yaml           ← Jaeger + OTel Collector 部署
```

---

## 关键追踪指标

| Span 名称 | 对应指标 | 诊断价值 |
|-----------|---------|---------|
| scheduler.wait | TTFT 中的排队部分 | 容量不足的直接证据 |
| model.prefill | TTFT 中的计算部分 | Prompt 长度 / Prefix Cache 效果 |
| model.decode | output_tokens × TPOT | Decode 效率 / Batch 影响 |
| scheduler.preempt | Preemption 事件 | KV Cache 瓶颈直接观测 |
| kv_cache.swap_out | CPU Offload 延迟 | 性能退化根因 |
| prefix_cache.lookup | Cache 查询耗时 | Cache 策略效果 |

---

## 与 Lab 03 (vLLM Metrics) 的关联

- **Metrics 回答**: 系统整体 TTFT P99 是多少？过去 1 小时 preemption 了多少次？
- **Traces 回答**: 这个具体请求为什么慢？它在哪个阶段卡了多久？被谁抢占了？

两者结合实现 **Metrics → Traces 下钻**：
1. Grafana 仪表盘看到 TTFT P99 突然升高
2. 点击异常时间段，跳转 Jaeger 查看 slow traces
3. 发现某些请求被 preempt → KV Cache 满
4. 关联 Prometheus 确认 gpu_cache_usage_perc > 0.95

---

## 运行前提

- Python 3.10+
- OpenTelemetry SDK: `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`
- Jaeger: `docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest`
- Kubernetes 环境 (用于完整部署)
