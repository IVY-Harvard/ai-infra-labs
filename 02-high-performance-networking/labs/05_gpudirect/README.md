# Lab 05: GPUDirect

## 概述

本实验探索 GPUDirect 技术，包括 GPU 间 P2P 通信和 GPUDirect RDMA，用于最大化 GPU 间及 GPU 与网络设备间的数据传输效率。

## 实验内容

### 1. P2P 连接检查 (`p2p_check.py`)
- 检测所有 GPU 对之间的 P2P 可达性
- 测量 P2P 带宽和延迟
- 生成连接矩阵报告

### 2. GPUDirect RDMA 带宽测试 (`gdrdma_benchmark.py`)
- 测试 GPU 与 RDMA 网卡之间的直接数据传输
- 比较 GDR 启用/禁用时的带宽差异
- 评估不同数据大小下的传输效率

## 技术背景

### GPUDirect P2P
允许同一 PCIe 总线或 NVLink 连接的 GPU 之间直接传输数据，绕过 CPU 内存。

### GPUDirect RDMA
允许第三方 PCIe 设备（如 InfiniBand HCA）直接访问 GPU 显存，实现零拷贝网络传输。

## 前置条件

- NVIDIA GPU (Kepler 架构以上)
- nvidia-peermem 模块已加载
- Mellanox OFED (GPUDirect RDMA)
- CUDA Toolkit >= 11.0

## 使用方法

```bash
# 检查 P2P 连接
python p2p_check.py --bandwidth

# GPUDirect RDMA 带宽测试
python gdrdma_benchmark.py --ib-dev mlx5_0 --gpu 0
```
