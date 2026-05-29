# Module 07: Inference Engine & Serving — 推理引擎与模型服务

> **本模块是整个 AI Infra 学习路径中最核心的模块之一。**

## 为什么推理引擎如此重要？

对于 AI Infra 工程师而言，推理引擎是日常工作的核心战场：
- **训练是一次性的，推理是持续性的** — 模型上线后 99% 的 GPU 时间花在推理
- **推理优化直接影响成本** — 吞吐提升 2x = GPU 成本降低 50%
- **核心技术点** — PagedAttention、Continuous Batching、KV Cache 管理是必须掌握的关键知识

## 学习目标

完成本模块后，学习者将能够：
- 理解 vLLM 每一层架构设计
- 从源码级别解释 PagedAttention
- 手写简化版推理引擎
- 进行架构选型和性能调优

## 模块结构

### 📖 Theory（理论深度）

| # | 文件 | 核心问题 | 重要程度 |
|---|------|----------|----------|
| 01 | inference_pipeline | LLM 推理全流程是什么？ | ⭐⭐⭐ |
| 02 | prefill_vs_decode | 为什么 Prefill 和 Decode 性质完全不同？ | ⭐⭐⭐⭐⭐ |
| 03 | kv_cache_management | KV Cache 为什么是推理瓶颈？ | ⭐⭐⭐⭐⭐ |
| 04 | paged_attention | PagedAttention 怎么解决显存浪费？ | ⭐⭐⭐⭐⭐ |
| 05 | continuous_batching | 连续批处理为什么能提升吞吐？ | ⭐⭐⭐⭐⭐ |
| 06 | vllm_architecture | vLLM 内部怎么运转？ | ⭐⭐⭐⭐ |
| 07 | speculative_decoding | 投机解码为什么不影响质量？ | ⭐⭐⭐ |
| 08 | quantization_serving | 量化对推理有什么影响？ | ⭐⭐⭐⭐ |
| 09 | tensorrt_llm | TRT-LLM vs vLLM 怎么选？ | ⭐⭐⭐ |
| 10 | serving_architecture | 生产级推理服务怎么设计？ | ⭐⭐⭐⭐ |

### 🔬 Labs（动手实验）

| # | 实验 | 学习收获 |
|---|------|----------|
| 01 | KV Cache Fundamentals | 计算 KV Cache 显存，理解为什么 128K 上下文吃掉大量显存 |
| 02 | Paged Attention | 实现 Block 分配器，理解虚拟内存思想 |
| 03 | Continuous Batching | 对比静态/连续批处理，观察吞吐差距 |
| 04 | vLLM Source Reading | 跟着指南读 vLLM 源码，深入理解内部机制 |
| 05 | TensorRT-LLM | 构建 TRT-LLM engine，与 vLLM 对比 |
| 06 | SGLang Practice | 体验 SGLang 的 Radix Attention |
| 07 | Quantization | 实测各种量化方案的精度/速度权衡 |
| 08 | Speculative Decoding | 实现投机解码，观察加速效果 |
| 09 | Multimodal Serving | 部署视觉/语音模型 |
| 10 | Model Routing | 构建多模型路由和负载均衡 |

### 🏗️ Project: Mini Inference Engine

一个简化版推理引擎，实现核心功能：
- **PagedAttention** — 分页显存管理
- **Continuous Batching** — 连续批处理调度
- **OpenAI 兼容 API** — `/v1/chat/completions`
- **流式输出** — SSE streaming
- **Prometheus 监控** — TTFT/TPOT/吞吐指标

## 学习路线

```
Week 1: 理论基础
  Day 1-2: theory/01-03 (推理流程 + KV Cache)
  Day 3-4: theory/04-05 (PagedAttention + Continuous Batching) ← 最核心
  Day 5:   theory/06 (vLLM 架构)

Week 2: 动手实验
  Day 1: labs/01-02 (KV Cache + PagedAttention 实现)
  Day 2: labs/03 (Continuous Batching 对比)
  Day 3: labs/04 (vLLM 源码阅读)
  Day 4: labs/07-08 (量化 + 投机解码)
  Day 5: labs/05-06 (TRT-LLM + SGLang)

Week 3: 项目实战
  Day 1-2: project/ 核心引擎 (Engine + Scheduler + BlockManager)
  Day 3:   project/ API 层 (FastAPI + Streaming)
  Day 4:   project/ 监控与压测
  Day 5:   整体串联 + 知识巩固

Week 4: 补充模块
  Day 1-2: theory/07-10 + labs/09-10
  Day 3-5: 综合练习 + 查漏补缺
```

## 核心知识点 Top 5

1. **PagedAttention 的原理** → theory/04 + labs/02
2. **Prefill 和 Decode 阶段的区别** → theory/02
3. **Continuous Batching 相比 Static Batching 的优势** → theory/05 + labs/03
4. **KV Cache 的显存占用计算** → theory/03 + labs/01
5. **vLLM 的请求处理流程** → theory/06 + labs/04

## 环境要求

- **GPU**: 8×H20 (96GB HBM3 per GPU)
- **Python**: 3.10+
- **核心依赖**: vLLM, TensorRT-LLM, SGLang, transformers, torch
- **服务依赖**: FastAPI, Prometheus, Grafana
