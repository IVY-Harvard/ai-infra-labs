# Mini Training Framework — 微型分布式训练框架

## 概述

一个简化但可运行的分布式训练框架，支持 TP + PP + DP 混合并行。
面向理解原理而设计，代码量精简，核心逻辑清晰可读。

**目标环境**: 8 × H20 GPU (NVLink 互联)

## 架构

```
mini-training-framework/
├── src/
│   ├── parallel/                # 并行策略
│   │   ├── data_parallel.py     # 数据并行封装 (DDP + FSDP)
│   │   ├── tensor_parallel.py   # 张量并行 (Column/Row Parallel)
│   │   ├── pipeline_parallel.py # 流水线并行 (1F1B 调度)
│   │   └── hybrid_parallel.py   # 混合并行编排器
│   ├── communication/           # 通信后端
│   │   ├── backend.py           # 通信后端抽象
│   │   ├── collective_ops.py    # 集合通信操作
│   │   └── topology.py          # 拓扑感知通信
│   ├── checkpoint/              # Checkpoint
│   │   ├── saver.py             # 同步保存
│   │   ├── loader.py            # 加载
│   │   └── async_saver.py       # 异步保存
│   ├── scheduler/               # 调度
│   │   ├── task_scheduler.py    # 训练任务调度
│   │   └── resource_manager.py  # GPU 资源管理
│   └── monitor/                 # 监控
│       ├── throughput_tracker.py # 吞吐量追踪
│       └── communication_profiler.py  # 通信性能分析
├── configs/
│   └── 8gpu_tp4_pp2.yaml       # 8 GPU 配置示例
├── tests/
│   ├── test_parallel.py         # 并行模块测试
│   └── test_communication.py    # 通信模块测试
├── Dockerfile                   # 容器化部署
├── requirements.txt             # 依赖
└── README.md                    # 本文件
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 8 GPU 训练

```bash
torchrun --nproc_per_node=8 -m src.parallel.hybrid_parallel \
    --config configs/8gpu_tp4_pp2.yaml
```

### 测试

```bash
torchrun --nproc_per_node=4 -m pytest tests/test_parallel.py -v
torchrun --nproc_per_node=4 -m pytest tests/test_communication.py -v
```

## 核心设计

### 1. 混合并行编排器 (HybridParallelEngine)

编排 TP + PP + DP 的核心类：
- 根据配置创建并行组 (TP group, PP group, DP group)
- 将模型切分到不同 stage (PP) 和不同 GPU (TP)
- 管理训练循环中的通信调度

### 2. 通信后端抽象

统一的通信接口，底层调用 NCCL：
- `all_reduce`, `all_gather`, `reduce_scatter`
- `send`, `recv` (P2P)
- 拓扑感知：自动将 TP 通信放在 NVLink 组内

### 3. Checkpoint 管理

支持同步和异步保存：
- 分布式保存：每 rank 保存自己的分片
- 异步保存：后台线程写入磁盘
- 自动清理旧 checkpoint

## 配置说明

```yaml
# configs/8gpu_tp4_pp2.yaml
parallel:
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
  data_parallel_size: 1     # 自动计算: 8 / (4 * 2) = 1
  
model:
  hidden_size: 4096
  num_layers: 32
  num_heads: 32
  vocab_size: 32000
  
training:
  micro_batch_size: 2
  num_micro_batches: 8      # global batch = 2 * 8 * 1 = 16
  max_steps: 1000
  learning_rate: 3e-4
  precision: bf16
```

## 注意事项

- 这是教学用的简化框架，不是生产级工具
- 生产环境推荐使用 Megatron-Core 或 DeepSpeed
- 代码注释详细，适合阅读理解原理
