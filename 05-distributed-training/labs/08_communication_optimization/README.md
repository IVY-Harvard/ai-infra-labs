# Lab 08: 通信优化

## 目标

1. 手写 Ring AllReduce，深入理解带宽最优算法
2. NCCL 性能 benchmark
3. 实现通信-计算重叠（overlap）

## 文件说明

| 文件 | 说明 |
|------|------|
| `ring_allreduce.py` | 手写 Ring AllReduce (ReduceScatter + AllGather) |
| `nccl_benchmark.py` | NCCL 各种集合通信操作的带宽测试 |
| `overlap_comm_compute.py` | 通信与计算重叠技术演示 |

## 运行

```bash
torchrun --nproc_per_node=8 ring_allreduce.py
torchrun --nproc_per_node=8 nccl_benchmark.py
torchrun --nproc_per_node=8 overlap_comm_compute.py
```
