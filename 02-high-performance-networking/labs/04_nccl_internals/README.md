# Lab 04: NCCL Internals

## 概述

本实验深入探索 NVIDIA Collective Communications Library (NCCL) 的内部机制，包括环境变量调优、性能基准测试和拓扑分析。

## 实验内容

### 1. NCCL 环境变量参考 (`nccl_env_reference.md`)
- 30+ 常用 NCCL 环境变量
- 每个变量的含义、默认值和推荐配置
- 不同场景下的最佳实践

### 2. NCCL 基准测试 (`nccl_benchmark.py`)
- 封装 nccl-tests 工具
- 支持 all_reduce, all_gather, broadcast 等操作
- 自动化多组参数测试与结果收集

### 3. 拓扑分析器 (`topology_analyzer.py`)
- 解析 `nvidia-smi topo -m` 输出
- 识别 GPU 间通信路径 (NVLink, PCIe, SYS)
- 生成拓扑建议报告

## 前置条件

- NVIDIA GPU (多卡环境)
- CUDA Toolkit >= 11.0
- NCCL >= 2.10
- nccl-tests (已编译)

## 使用方法

```bash
# 运行 NCCL 基准测试
python nccl_benchmark.py --test all_reduce --min-size 1M --max-size 1G

# 分析 GPU 拓扑
python topology_analyzer.py --output report.json
```
