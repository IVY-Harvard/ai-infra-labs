# Module 01: GPU Computing and Kernel Optimization

## 模块定位

本模块是 AI Infra 工程师学习路径的第一个模块，聚焦于 GPU 计算的底层原理和内核优化技术。完成本模块后，读者将能够：

- 深入理解 GPU 硬件架构（特别是 H20）与软件执行模型的映射关系
- 判断一个算子是计算密集还是访存密集，并选择对应的优化策略
- 使用 CUDA/Triton 编写和优化 GPU 内核
- 理解 FlashAttention、算子融合等关键优化技术的原理
- 具备 GPU 性能分析和调优的完整工具链能力

## 前置要求

- 1 年 HPC 经验（熟悉并行计算基本概念）
- Python 熟练，C/C++ 基础
- 了解深度学习基本概念（矩阵乘、Attention 等）
- 硬件环境：8 张 NVIDIA H20 GPU

## 学习路线

```
Week 1: 理论基础 + 环境搭建
  ├── theory/01 GPU 架构
  ├── theory/02 CUDA 编程模型
  ├── lab/01 查询设备参数
  └── lab/02 CUDA 入门

Week 2: 内存层次 + Tensor Core
  ├── theory/03 计算vs访存分析
  ├── theory/04 Tensor Core与混合精度
  ├── lab/03 内存层次实测
  └── lab/04 Tensor Core GEMM

Week 3: 高级内核优化
  ├── theory/05 Triton与编译器
  ├── theory/06 算子融合
  ├── lab/05 Attention 内核
  ├── lab/06 Triton 编程
  └── lab/07 torch.compile

Week 4: 工程实践 + 项目
  ├── theory/07 多后端适配
  ├── lab/08 算子融合实战
  ├── lab/09 多后端适配
  ├── lab/10 Profiling 实战
  └── project: GPU Kernel Benchmark Suite
```

## Theory 理论文件

| # | 文件 | 核心内容 |
|---|------|----------|
| 01 | gpu_architecture.md | SM/Warp/CUDA Core/Tensor Core/Memory Hierarchy/NVLink，H20 参数 |
| 02 | cuda_programming_model.md | Grid/Block/Thread/Warp，执行模型，调度与同步 |
| 03 | compute_vs_memory_bound.md | Roofline 模型，Arithmetic Intensity，Prefill vs Decode |
| 04 | tensor_core_and_mixed_precision.md | Tensor Core 原理，FP16/BF16/FP8/INT8/TF32 |
| 05 | triton_and_compiler.md | Triton 编程模型，torch.compile/Dynamo/Inductor，编译器生态 |
| 06 | operator_fusion.md | 融合原理，FlashAttention，Kernel Fusion 策略 |
| 07 | multi_backend_adaptation.md | CUDA vs ROCm vs 昇腾/寒武纪，适配层设计 |

## Labs 实验

| # | 实验 | 关键收获 |
|---|------|----------|
| 01 | GPU 架构查询 | 理解 H20 硬件参数的工程含义 |
| 02 | CUDA 基础 | 掌握 CUDA 编程的基本 pattern |
| 03 | 内存层次 | 实测各级存储带宽，体会共享内存的加速效果 |
| 04 | Tensor Core GEMM | 朴素/WMMA/cuBLAS 三种实现的性能差异 |
| 05 | Attention 内核 | 理解 FlashAttention 为什么快 |
| 06 | Triton 编程 | 用 Python 写 GPU 内核 |
| 07 | torch.compile | 理解 PyTorch 2.0 编译流程 |
| 08 | 算子融合 | 手动 vs 自动融合的效果 |
| 09 | 多后端适配 | 设计可移植的算子适配层 |
| 10 | Profiling 实战 | NSight/Roofline/NVML 全套工具 |

## Project 企业级项目

**GPU Kernel Benchmark Suite** — 一个完整的 GPU 基准测试工具套件：

- 测试 FP16/FP32/INT8 算力（TFLOPS）
- 测试 HBM 带宽
- 测试 GPU 间 P2P/NVLink 带宽
- 生成 HTML 对比报告
- 支持多卡并行测试
- 容器化部署

## 如何使用本模块

1. **先读理论**：每个 lab 开始前，先阅读对应的 theory 文件
2. **动手实验**：按 lab 编号顺序完成，每个 lab 的 README 有详细步骤
3. **做项目**：理论和实验完成后，用 project 检验综合能力
4. **记笔记**：在每个 lab 目录下创建自己的 `notes.md`
