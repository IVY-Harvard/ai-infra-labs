# Lab 06: Network Topology

## 概述

本实验专注于 GPU 集群网络拓扑的可视化与性能分析，包括节点内 GPU 拓扑可视化和节点间带宽矩阵测量。

## 实验内容

### 1. 拓扑可视化 (`topology_visualizer.py`)
- 收集 GPU、NIC、PCIe Switch 拓扑信息
- 生成可视化拓扑图 (ASCII 或 Graphviz DOT)
- 标注连接类型和带宽

### 2. 带宽矩阵 (`bandwidth_matrix.py`)
- 测量集群节点间的网络带宽
- 支持多种传输模式 (TCP, RDMA)
- 生成热力图数据用于性能分析

## 技术背景

了解集群网络拓扑对分布式训练至关重要:
- **胖树 (Fat-tree)**: 常见的交换机拓扑，提供等分带宽
- **Dragonfly**: 高效的低直径网络拓扑
- **Rail-Optimized**: 针对 GPU 服务器优化的网络拓扑

## 前置条件

- 多节点 GPU 集群
- SSH 免密登录配置
- iperf3 或 perftest 工具
- Python 3.8+ (graphviz 可选)

## 使用方法

```bash
# GPU 拓扑可视化
python topology_visualizer.py --format dot --output topo.dot

# 节点间带宽矩阵
python bandwidth_matrix.py --hosts nodes.txt --mode rdma
```
