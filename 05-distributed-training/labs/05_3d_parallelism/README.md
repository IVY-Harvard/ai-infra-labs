# Lab 05: 3D 并行 — TP + PP + DP 组合

## 目标

1. 理解并行组（Process Group）的创建逻辑
2. 实现拓扑映射器：根据 GPU 拓扑自动分配 TP/PP/DP 组
3. 在 8×H20 上配置 TP=4, PP=2, DP=1 的 3D 并行

## 前置知识

- 完成 Lab 01-04
- 对应理论：[theory/05_3d_parallelism.md](../../theory/05_3d_parallelism.md)

## 文件说明

| 文件 | 说明 |
|------|------|
| `hybrid_parallel_config.py` | 3D 并行配置生成器：创建 TP/PP/DP 组 |
| `topology_mapper.py` | GPU 拓扑感知映射 |

## 实验

```bash
torchrun --nproc_per_node=8 hybrid_parallel_config.py --tp 4 --pp 2
torchrun --nproc_per_node=8 topology_mapper.py
```
