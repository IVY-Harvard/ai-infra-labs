# 09 - TensorRT-LLM 架构与 vLLM 对比

## 概述

TensorRT-LLM (TRT-LLM) 是 NVIDIA 官方的 LLM 推理优化框架。理解它与 vLLM 的差异，是做企业级部署选型的基础。

## TensorRT-LLM 架构

```
┌──────────────────────────────────────────────────────────┐
│                TensorRT-LLM Architecture                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐                                     │
│  │  Model Definition │  Python API 定义模型结构           │
│  │  (Python)         │  类似 PyTorch 但用 TRT 算子       │
│  └────────┬──────────┘                                   │
│           │ build                                         │
│           ▼                                               │
│  ┌─────────────────┐                                     │
│  │  TRT Engine      │  编译优化后的推理引擎               │
│  │  (Serialized)    │  包含所有优化 (融合/量化/etc)       │
│  └────────┬──────────┘                                   │
│           │ load                                          │
│           ▼                                               │
│  ┌─────────────────┐                                     │
│  │  Runtime         │                                     │
│  │  ┌────────────┐ │                                     │
│  │  │ GptSession │ │  管理推理会话                        │
│  │  ├────────────┤ │                                     │
│  │  │ Executor   │ │  批处理执行器 (Inflight Batching)   │
│  │  ├────────────┤ │                                     │
│  │  │ KV Cache   │ │  Paged KV Cache (类似 PagedAttn)   │
│  │  │ Manager    │ │                                     │
│  │  └────────────┘ │                                     │
│  └─────────────────┘                                     │
│                                                           │
│  ┌─────────────────┐                                     │
│  │  Triton Server   │  生产部署 (gRPC/HTTP)               │
│  │  (Serving Layer)  │                                    │
│  └─────────────────┘                                     │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## TRT-LLM 的核心优化

### 1. 编译时优化 (Build Phase)

```
TRT-LLM 在 build 阶段做的优化:

1. 算子融合 (Kernel Fusion):
   LayerNorm + QKV Linear → 一个 kernel
   Attention + Softmax → 一个 kernel
   → 减少 kernel launch overhead 和中间数据读写

2. 精度优化:
   自动选择最快的数据类型 (FP16/BF16/FP8/INT8)
   Mixed Precision: 关键层保持高精度

3. 内存优化:
   静态内存规划 (编译时确定所有 buffer 大小)
   最小化显存碎片

4. 硬件适配:
   为特定 GPU 生成最优 kernel
   利用 GPU 特有指令 (如 H20 的 FP8)
```

### 2. Inflight Batching

```
TRT-LLM 的 Continuous Batching 实现:

与 vLLM 类似:
  - 请求可以动态加入/退出
  - Iteration-Level Scheduling
  - Paged KV Cache

独特之处:
  - Executor API 封装更完整
  - 与 Triton Server 深度集成
  - 支持更细粒度的调度策略
```

## vLLM vs TRT-LLM 详细对比

```
┌──────────────────────────────────────────────────────────────────┐
│              vLLM vs TensorRT-LLM 对比                            │
├─────────────┬──────────────────────┬────────────────────────────┤
│   维度       │      vLLM            │      TRT-LLM               │
├─────────────┼──────────────────────┼────────────────────────────┤
│ 开发者       │ UC Berkeley          │ NVIDIA                     │
│ 开源        │ 完全开源 (Apache 2.0) │ 开源 (Apache 2.0)          │
│ 语言        │ Python + CUDA        │ C++ + Python + CUDA        │
│ 易用性       │ ⭐⭐⭐⭐⭐ (pip install)│ ⭐⭐⭐ (需要 build engine)  │
│ 模型支持     │ 极广 (HF 直接加载)   │ 需要转换/支持列表         │
│ 新模型适配   │ 快 (社区活跃)        │ 较慢 (NVIDIA 维护)        │
│             │                      │                            │
│ Prefill 性能│ 好 (FlashAttention)  │ 更好 (深度融合)            │
│ Decode 性能 │ 好                   │ 更好 (编译优化)            │
│ 首 token 延迟│ 较好               │ 更好 (kernel 融合)         │
│ 量化支持     │ FP8/GPTQ/AWQ       │ FP8/INT8/INT4/更全面       │
│             │                      │                            │
│ 部署复杂度   │ 低 (直接启动)       │ 高 (build→deploy)          │
│ 迭代速度     │ 快 (Python 为主)    │ 慢 (需要重新 build)        │
│ 调试难度     │ 低 (Python 可调试)  │ 高 (C++ 核心)              │
│ 生产稳定性   │ 好                  │ 很好 (NVIDIA 支持)         │
│             │                      │                            │
│ 适合场景     │ 快速迭代/多模型     │ 单模型极致性能             │
│             │ 研究探索/中小规模    │ 大规模生产部署             │
│             │ 灵活配置             │ 性能第一                   │
├─────────────┼──────────────────────┼────────────────────────────┤
│ 性能差距     │ 通常慢 10-30%       │ 基准 (编译优化加成)        │
│ 开发效率     │ 基准 (极快)         │ 慢 2-5x                   │
└─────────────┴──────────────────────┴────────────────────────────┘
```

## 选型决策树

```
┌─────────────────────────────────────────────────┐
│                                                  │
│  你的场景是什么？                                 │
│       │                                          │
│       ├─ 快速原型/研究 → vLLM                    │
│       │                                          │
│       ├─ 需要支持很多模型 → vLLM                 │
│       │   (vLLM 模型适配更快更全)                 │
│       │                                          │
│       ├─ 单模型追求极致性能 → TRT-LLM            │
│       │   (编译优化能带来 10-30% 性能提升)        │
│       │                                          │
│       ├─ 团队没有 C++ 能力 → vLLM                │
│       │                                          │
│       ├─ NVIDIA 企业支持很重要 → TRT-LLM          │
│       │                                          │
│       ├─ 需要灵活的量化策略 → vLLM               │
│       │   (vLLM 支持 runtime 切换)               │
│       │                                          │
│       └─ 大规模生产 + 稳定性优先 → TRT-LLM        │
│           + Triton Server                         │
│                                                  │
└─────────────────────────────────────────────────┘
```

## SGLang — 新兴竞争者

```
SGLang (SG Language):
  - 来自 LMSYS (LMSys Chatbot Arena 团队)
  - Radix Attention: 基于 radix tree 的自动 prefix caching
  - 高效的多轮对话和复杂 prompt 管理

独特优势:
  - Radix Attention 自动前缀共享 (无需手动配置)
  - 对多轮对话场景特别友好
  - 结构化输出 (JSON mode) 性能很好
  - 某些场景吞吐超过 vLLM

三者定位:
  vLLM:    通用性最好, 社区最大, 快速迭代
  TRT-LLM: 极致性能, NVIDIA 生态, 企业级
  SGLang:  创新最快, 多轮对话优, 上升势头
```

## 知识要点框架

### "vLLM 和 TensorRT-LLM 怎么选？"

```
"选择取决于场景:

vLLM 适合:
- 需要快速迭代、支持多种模型
- 团队以 Python 为主
- 中小规模部署, 灵活性优先

TRT-LLM 适合:
- 单模型大规模部署, 性能优先
- 有 NVIDIA 企业支持需求
- 愿意投入 build/优化的工程时间

性能差距: TRT-LLM 通常快 10-30% (编译优化)
开发效率: vLLM 快 2-5x (直接 pip install + serve)

我们的实践: 
开发测试用 vLLM (快速迭代),
生产环境用 vLLM 或 TRT-LLM (取决于性能要求)。
对于 8×H20 环境, vLLM 的 FP8 支持已经能满足大多数场景。"
```

## 小结

| 框架 | 核心优势 | 适用场景 | 性能 |
|------|----------|----------|------|
| vLLM | 易用, 灵活, 社区强 | 通用推理服务 | 基准 |
| TRT-LLM | 极致优化, NVIDIA 支持 | 大规模生产 | +10-30% |
| SGLang | Radix Attention, 创新 | 多轮对话, 结构化输出 | 接近或超 vLLM |
