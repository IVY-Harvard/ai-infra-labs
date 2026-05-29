# Module 05: Distributed Training

## 模块定位

本模块是 AI Infra 工程师学习路径的核心模块，聚焦大规模分布式训练的理论与实践。
目标环境：**8 × NVIDIA H20 GPU**（单机多卡，NVLink 互联）。

## 前置知识

- 熟悉 PyTorch 基础（nn.Module、DataLoader、Optimizer）
- 了解 GPU 显存模型（HBM、L2 Cache）
- 了解 TP/PP 的基本概念（本模块将深入展开）
- 使用过 Slurm 进行作业调度

## 学习目标

完成本模块后，读者将能够：

1. **理论层面**：清晰理解 DP/TP/PP/EP/CP/SP 六大并行策略的原理、通信量公式和适用场景
2. **工程层面**：独立配置和调优 3D 并行训练（TP=4, PP=2, DP=1 on 8×H20）
3. **框架层面**：熟练使用 DeepSpeed ZeRO / Megatron-Core / PyTorch FSDP
4. **系统层面**：手写关键原语（Ring AllReduce、列并行 Linear），深入理解通信瓶颈
5. **生产层面**：实现分布式 Checkpoint、容错恢复、混合精度训练

## 模块结构

```
05-distributed-training/
├── README.md                          # 本文件
├── theory/                            # 理论文档（10 篇）
│   ├── 01_parallelism_overview.md     # 并行策略全景
│   ├── 02_data_parallelism.md         # 数据并行：DDP/FSDP
│   ├── 03_tensor_parallelism.md       # 张量并行：Megatron-style
│   ├── 04_pipeline_parallelism.md     # 流水线并行：GPipe/1F1B
│   ├── 05_3d_parallelism.md           # 3D 并行组合策略
│   ├── 06_communication_primitives.md # 集合通信原语
│   ├── 07_deepspeed_zero.md           # DeepSpeed ZeRO 系列
│   ├── 08_megatron_core.md            # Megatron-LM/Core 架构
│   ├── 09_mixed_precision_training.md # 混合精度训练
│   └── 10_fault_tolerance.md          # 容错与弹性训练
├── labs/                              # 动手实验（10 个）
│   ├── 01_ddp_basics/                 # DDP 基础
│   ├── 02_fsdp_practice/              # FSDP 实践
│   ├── 03_tensor_parallelism/         # 张量并行手写
│   ├── 04_pipeline_parallelism/       # 流水线并行调度
│   ├── 05_3d_parallelism/             # 3D 并行配置
│   ├── 06_deepspeed_zero/             # DeepSpeed ZeRO
│   ├── 07_megatron_core/              # Megatron-Core
│   ├── 08_communication_optimization/ # 通信优化
│   ├── 09_mixed_precision/            # 混合精度
│   └── 10_checkpoint_and_recovery/    # Checkpoint 与恢复
└── project/
    └── mini-training-framework/       # 企业级项目：微型分布式训练框架
```

## 学习路线

### Week 1: 基础并行（Theory 01-03, Labs 01-03）
- 理解所有并行策略的全景
- 从单卡到 DDP 到 FSDP 的渐进式实验
- 手写张量并行的列并行和行并行

### Week 2: 高级并行（Theory 04-06, Labs 04-06）
- 流水线并行调度算法（GPipe → 1F1B → Interleaved）
- 3D 并行的配置逻辑（TP 放 NVLink 内，PP 跨机，DP 最外层）
- DeepSpeed ZeRO Stage 1/2/3 显存拆解

### Week 3: 框架与优化（Theory 07-09, Labs 07-09）
- Megatron-Core 的 Sequence Parallelism 和 Context Parallelism
- 通信原语手写 + NCCL 性能分析
- 混合精度训练的 Loss Scaling 机制

### Week 4: 生产化 + 项目（Theory 10, Lab 10, Project）
- Checkpoint、容错、弹性训练
- 完成 mini-training-framework 项目

## 关键公式速查

| 并行策略 | 通信量（每步） | 通信模式 | 适用带宽 |
|---------|--------------|---------|---------|
| DDP     | 2 × model_size | AllReduce | 任意 |
| FSDP    | 3 × model_size | AllGather + ReduceScatter | 任意 |
| TP      | 每层 2 × hidden_size × seq_len × batch | AllReduce | NVLink |
| PP      | micro_batch_size × hidden_size × seq_len | P2P Send/Recv | 任意 |

## 环境配置

```bash
# 基础环境
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed
pip install megatron-core
pip install flash-attn

# 监控工具
pip install nvidia-ml-py3 wandb tensorboard

# 验证 GPU 拓扑
nvidia-smi topo -m
```

## 参考资料

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
