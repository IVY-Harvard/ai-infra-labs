# Module 02: High-Performance Networking for AI Infrastructure

## Target Audience

有多卡 GPU 实操经验、做过 InfiniBand 网卡在 K8s 中纳管的工程师。  
已了解基本网络概念，但需要深入理解 RDMA、NCCL、GPUDirect 等底层原理并具备网络调优能力。

## Learning Objectives

完成本模块后，读者将能够：

1. **理论层面**：深入理解 RDMA/IB/RoCE 底层原理，掌握 NCCL 通信算法与调优方法论
2. **实操层面**：独立完成 IB/RoCE 网络诊断、NCCL 性能调优、GPUDirect 验证
3. **架构层面**：根据训练规模与并行策略选择合适的网络拓扑
4. **工程层面**：在 K8s 环境中配置高性能网络（SR-IOV/Multus/RDMA Device Plugin）
5. **项目层面**：构建集群网络诊断与健康检查平台

## Module Structure

```
02-high-performance-networking/
├── README.md                          # 本文件
├── theory/                            # 理论篇（7 个专题）
│   ├── 01_rdma_fundamentals.md        # RDMA 技术全解
│   ├── 02_infiniband_deep_dive.md     # InfiniBand 深度解析
│   ├── 03_roce_and_ethernet.md        # RoCE 与以太网
│   ├── 04_nccl_internals.md           # NCCL 通信库深度
│   ├── 05_gpudirect_technology.md     # GPUDirect 技术族
│   ├── 06_network_topology_design.md  # AI 集群网络拓扑设计
│   └── 07_container_networking.md     # K8s 容器高性能网络
├── labs/                              # 实验篇（10 个实验）
│   ├── 01_rdma_fundamentals/          # RDMA 基础实验
│   ├── 02_infiniband_practice/        # InfiniBand 实战
│   ├── 03_roce_practice/              # RoCE 配置实战
│   ├── 04_nccl_internals/             # NCCL 深度实验
│   ├── 05_gpudirect/                  # GPUDirect 实验
│   ├── 06_network_topology/           # 拓扑分析实验
│   ├── 07_container_networking/       # K8s 网络配置
│   ├── 08_uec_ethernet/              # UEC 超级以太网
│   ├── 09_network_diagnosis/          # 网络诊断工具
│   └── 10_bandwidth_optimization/     # 带宽优化
└── project/                           # 企业级项目
    └── cluster-network-diagnostics/   # 集群网络诊断平台
```

## Learning Path

### Phase 1: Foundation (Week 1)

| Day | Theory | Lab | Focus |
|-----|--------|-----|-------|
| 1 | `theory/01` RDMA 基础 | `labs/01` RDMA 信息采集 | 理解为什么 AI 训练需要 RDMA |
| 2 | `theory/02` InfiniBand | `labs/02` IB 诊断 | 掌握 IB 架构与诊断工具 |
| 3 | `theory/03` RoCE | `labs/03` RoCE 配置 | 理解 RoCE vs IB 的选型 |

### Phase 2: Core Technologies (Week 2)

| Day | Theory | Lab | Focus |
|-----|--------|-----|-------|
| 4 | `theory/04` NCCL | `labs/04` NCCL 调优 | 掌握 NCCL 环境变量与算法 |
| 5 | `theory/05` GPUDirect | `labs/05` GPUDirect 验证 | 理解 GPU 直通数据路径 |
| 6 | `theory/06` 拓扑设计 | `labs/06` 拓扑分析 | 学会根据场景选拓扑 |

### Phase 3: Production Practice (Week 3)

| Day | Theory | Lab | Focus |
|-----|--------|-----|-------|
| 7 | `theory/07` K8s 网络 | `labs/07` K8s 配置 | 在容器环境配置 RDMA |
| 8 | — | `labs/08` UEC 标准 | 了解以太网新趋势 |
| 9 | — | `labs/09` 网络诊断 | 掌握全套诊断方法 |
| 10 | — | `labs/10` 带宽优化 | 生产环境性能调优 |

### Phase 4: Project (Week 4)

集群网络诊断与健康检查平台 — 整合前三周所学，构建可部署的企业级工具。

## Prerequisites

### Hardware

- 8x NVIDIA H20 GPU（PCIe 连接）
- InfiniBand HDR/NDR 或 RoCE 网卡（推荐 ConnectX-7）
- 支持 PFC 的以太网交换机（如果使用 RoCE）

### Software

```bash
# 基础环境
NVIDIA Driver >= 535
CUDA >= 12.2
MLNX_OFED >= 23.10

# 工具
ibverbs-utils, perftest, rdma-core
nccl >= 2.19, nccl-tests
Python >= 3.10, pip

# K8s 环境
Kubernetes >= 1.28
Multus CNI, SR-IOV CNI
RDMA Device Plugin
```

### Knowledge

- Linux 系统管理基础（systemd/网络配置/内核参数）
- K8s 基本操作（已做过 IB 网卡在 K8s 纳管）
- 对分布式训练的基本理解（Data Parallel / Tensor Parallel）

## Quick Validation

在开始本模块前，运行以下命令验证环境：

```bash
# 1. 验证 RDMA 设备
ibv_devices
rdma link show

# 2. 验证 GPU
nvidia-smi --query-gpu=name,pci.bus_id --format=csv

# 3. 验证 NCCL
python3 -c "import torch; print(torch.cuda.nccl.version())"

# 4. 验证 K8s RDMA 资源
kubectl get nodes -o json | jq '.items[].status.allocatable | with_entries(select(.key | contains("rdma")))'
```

## Key References

- [RDMA Aware Networks Programming User Manual (NVIDIA)](https://docs.nvidia.com/networking/display/rdmacore60)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Ultra Ethernet Consortium](https://ultraethernet.org/)
- [K8s Network Plumbing Working Group](https://github.com/k8snetworkplumbingwg)
