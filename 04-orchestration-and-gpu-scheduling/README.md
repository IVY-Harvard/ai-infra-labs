# Module 04: Orchestration and GPU Scheduling

## 模块概述

本模块深入探讨 Kubernetes 环境下的 GPU 调度与编排机制。面向拥有多卡 GPU 集群、
熟悉 K8s 和 KubeRay 基本操作、配置过 Slurm 但对 K8s 调度内部机制了解不深的工程师。

## 学习目标

完成本模块后，读者将能够：

1. **理解 K8s 调度器内部机制** — 调度周期、扩展点、自定义调度器开发
2. **掌握 GPU 资源管理** — Device Plugin、GPU Operator 架构与运维
3. **实施 GPU 共享策略** — MIG/MPS/Time-Slicing 的选型与配置
4. **部署批处理调度器** — Volcano/Kueue 满足 AI 训练场景需求
5. **运用 Ray/KubeRay** — 分布式训练和推理的编排
6. **设计多集群方案** — 混合云 GPU 调度与弹性伸缩
7. **实现容错机制** — 训练中断恢复、推理高可用

## 前置知识

| 技能 | 要求级别 | 说明 |
|------|----------|------|
| Kubernetes | 中级 | 理解 Pod/Deployment/Service/DaemonSet |
| GPU 基础 | 中级 | 了解 CUDA、nvidia-smi |
| KubeRay | 初级 | 部署过 RayCluster |
| Slurm | 初级 | 配置过基本调度 |
| Go/Python | 初级 | 能读懂代码 |

## 实验环境

```
硬件：8x NVIDIA H20 (96GB HBM3 each)
系统：Ubuntu 22.04 + CUDA 12.x
K8s：v1.28+ (kubeadm 或 kind 均可)
```

## 模块结构

```
04-orchestration-and-gpu-scheduling/
├── README.md                          # 本文件
├── theory/                            # 理论知识（7 篇）
│   ├── 01_k8s_scheduling_deep_dive.md
│   ├── 02_gpu_resource_management.md
│   ├── 03_gpu_sharing.md
│   ├── 04_batch_scheduling.md
│   ├── 05_ray_and_kuberay.md
│   ├── 06_multi_cluster_hybrid_cloud.md
│   └── 07_fault_tolerance_ha.md
├── labs/                              # 动手实验（10 个）
│   ├── 01_k8s_scheduling_deep_dive/
│   ├── 02_device_plugin/
│   ├── 03_gpu_operator/
│   ├── 04_custom_scheduler/
│   ├── 05_gpu_sharing/
│   ├── 06_volcano_kueue/
│   ├── 07_multi_cluster/
│   ├── 08_spot_elasticity/
│   ├── 09_fault_tolerance/
│   └── 10_hybrid_cloud/
└── project/gpu-cluster-scheduler/     # 综合项目
    ├── src/
    ├── tests/
    ├── deploy/
    └── Dockerfile
```

## 学习路径

```
Week 1: 理论 01-03 + Lab 01-03（调度基础 + GPU 管理）
Week 2: 理论 04-05 + Lab 04-06（批调度 + Ray）
Week 3: 理论 06-07 + Lab 07-10（多集群 + 容错）
Week 4: 综合项目开发
```

## 与 Slurm 的对比视角

对于有 Slurm 经验的读者，K8s GPU 调度有以下关键差异：

| 维度 | Slurm | Kubernetes |
|------|-------|------------|
| 调度粒度 | Job/Step | Pod/Container |
| GPU 分配 | GRES 插件 | Device Plugin + Extended Resource |
| 队列管理 | Partition/QOS | Namespace/ResourceQuota/PriorityClass |
| Gang Scheduling | 原生支持 | 需要 Volcano/Coscheduling |
| 弹性 | 手动扩缩 | HPA/VPA/Cluster Autoscaler |
| 容错 | Job Requeue | Pod Restart Policy + Checkpoint |

## 参考资源

- [Kubernetes Scheduler 源码](https://github.com/kubernetes/kubernetes/tree/master/pkg/scheduler)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Volcano 文档](https://volcano.sh/docs/)
- [Kueue 文档](https://kueue.sigs.k8s.io/)
- [KubeRay 文档](https://ray-project.github.io/kuberay/)
