# 模块 03：存储与数据管道

## 模块定位

当训练规模从单卡扩展到 8 卡甚至多节点时，存储和数据管道会成为真正的瓶颈。本模块系统讲解分布式存储架构、数据缓存策略、Checkpoint 工程化、模型分发，以及从原始数据到训练样本的完整数据管道。

## 目标读者

- 有多卡 GPU 的实际使用经验
- 用过 NFS/共享存储挂载模型文件
- 对分布式存储和数据管道尚未深入

## 学习目标

1. **存储选型能力** — 能根据 AI 工作负载特征（大文件顺序读、小文件随机读、Checkpoint 突发写）选择合适的存储方案
2. **JuiceFS 实战** — 能部署和调优 JuiceFS，理解其元数据引擎+对象存储+本地缓存的三层架构
3. **缓存策略设计** — 能设计多级缓存（内存→SSD→HDD→远端），掌握 Alluxio 和 JuiceFS 缓存调优
4. **Checkpoint 工程化** — 理解同步/异步/增量 Checkpoint，掌握 PyTorch DCP 和 GC 策略
5. **模型分发** — 能设计 P2P 分发、多级缓存加载、模型预热等企业级模型分发方案
6. **数据管道构建** — 能构建流式数据加载管道（WebDataset/MosaicML StreamingDataset），处理多模态数据
7. **数据质量飞轮** — 能搭建数据去重、质量打分、合成数据生成的完整数据飞轮

## 模块结构

```
03-storage-and-data-pipeline/
├── README.md                          # 本文件：模块总览
├── theory/                            # 理论知识（7 篇）
│   ├── 01_distributed_storage_landscape.md   # AI 存储全景
│   ├── 02_juicefs_architecture.md            # JuiceFS 架构详解
│   ├── 03_data_caching.md                    # AI 数据缓存策略
│   ├── 04_checkpoint_engineering.md          # Checkpoint 工程化
│   ├── 05_model_distribution.md              # 模型分发策略
│   ├── 06_training_data_pipeline.md          # 训练数据管道
│   └── 07_data_quality_and_flywheel.md       # 数据质量与飞轮
├── labs/                              # 动手实验（10 个）
│   ├── 01_distributed_fs_comparison/         # 分布式文件系统对比测试
│   ├── 02_juicefs_practice/                  # JuiceFS 部署与调优
│   ├── 03_alluxio_caching/                   # Alluxio 缓存实战
│   ├── 04_checkpoint_management/             # Checkpoint 管理
│   ├── 05_model_distribution/                # 模型分发实战
│   ├── 06_streaming_dataloader/              # 流式数据加载
│   ├── 07_tokenization_pipeline/             # Tokenization 流水线
│   ├── 08_data_quality/                      # 数据质量工具
│   ├── 09_synthetic_data/                    # 合成数据生成
│   └── 10_data_flywheel/                     # 数据飞轮
└── project/                           # 企业级项目
    └── model-asset-platform/                 # 模型资产管理平台
```

## 学习路径

### 第一周：存储基础（theory 01-03 + labs 01-03）

| 天数 | 内容 | 时间 |
|------|------|------|
| Day 1 | 理论 01：AI 存储全景 + Lab 01：存储基准测试 | 3h |
| Day 2 | 理论 02：JuiceFS 架构 + Lab 02：JuiceFS 部署 | 3h |
| Day 3 | 理论 03：缓存策略 + Lab 03：Alluxio 缓存实战 | 3h |

### 第二周：Checkpoint 与模型分发（theory 04-05 + labs 04-05）

| 天数 | 内容 | 时间 |
|------|------|------|
| Day 4 | 理论 04：Checkpoint 工程化 + Lab 04：Checkpoint 管理 | 3h |
| Day 5 | 理论 05：模型分发 + Lab 05：P2P 分发与多级缓存 | 3h |

### 第三周：数据管道（theory 06-07 + labs 06-10）

| 天数 | 内容 | 时间 |
|------|------|------|
| Day 6 | 理论 06：训练数据管道 + Lab 06-07：流式加载与 Tokenization | 4h |
| Day 7 | 理论 07：数据质量飞轮 + Lab 08-10：数据质量、合成与飞轮 | 4h |

### 第四周：综合项目

| 天数 | 内容 | 时间 |
|------|------|------|
| Day 8-10 | 模型资产管理平台完整开发 | 12h |

## 前置知识

- Linux 文件系统基础（挂载、权限、IO 操作）
- Python 基础（异步编程、多进程）
- PyTorch 基础（DataLoader、模型保存/加载）
- Docker 和 Kubernetes 基础概念
- 了解 S3 等对象存储的基本使用

## 核心工具版本

| 工具 | 版本 | 用途 |
|------|------|------|
| JuiceFS | 1.2+ | 分布式文件系统 |
| Alluxio | 2.9+ | 数据编排与缓存 |
| PyTorch | 2.3+ | DCP / DataLoader |
| WebDataset | 0.2+ | 流式数据加载 |
| MosaicML StreamingDataset | 0.7+ | 流式数据集 |
| MinIO | latest | 本地对象存储 |
| Redis | 7.0+ | JuiceFS 元数据引擎 |
| FastAPI | 0.110+ | API 服务 |

## 关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| POSIX 语义 | 文件系统提供的标准接口（open/read/write/close），兼容传统应用但限制并发性能 |
| 对象存储 | 以对象（Key-Value）方式存储数据，天然分布式，但不支持 POSIX 原地修改 |
| 元数据引擎 | 管理文件名、目录结构、权限等信息的组件，其性能决定了 ls/stat 等操作的速度 |
| Checkpoint | 训练中间状态的快照，用于故障恢复和训练续跑 |
| 数据飞轮 | 模型上线→收集反馈→标注→训练→更新模型的持续改进闭环 |
