# Lab 06 - Volcano 与 Kueue 批调度

## 目标

掌握 K8s 生态中两个主流批调度框架——Volcano 和 Kueue。
理解 Gang Scheduling、队列管理、公平调度等批处理调度核心概念。

## 背景

```
为什么需要批调度框架？

K8s 默认调度器的局限（对 GPU 训练）：
  1. 逐 Pod 调度 — 分布式训练需要所有 worker 同时就绪
  2. 没有队列概念 — 无法排队等待资源
  3. 没有公平共享 — 先到先得，大租户会饿死小租户
  4. 没有 Gang Scheduling — 可能只调度了部分 worker

解决方案：
  ┌────────────────────────────────────────┐
  │        批调度框架                        │
  │                                         │
  │  Volcano (CNCF Sandbox)                │
  │  - Gang Scheduling                      │
  │  - Job 生命周期管理                      │
  │  - 队列 + 优先级                         │
  │  - 适合 HPC/批处理                      │
  │                                         │
  │  Kueue (Kubernetes SIG)                │
  │  - 资源配额管理                          │
  │  - 公平共享 + 借用                       │
  │  - 与原生 Job/PyTorchJob 集成          │
  │  - 更轻量，K8s 原生                     │
  └────────────────────────────────────────┘
```

## 实验内容

### 实验 1：Volcano 安装与使用

详见 [volcano_setup.md](./volcano_setup.md) — Volcano 安装和配置。

### 实验 2：Gang Scheduling 演示

详见 [gang_scheduling_demo.yaml](./gang_scheduling_demo.yaml) — Volcano Gang Scheduling 示例。

### 实验 3：Kueue 安装与使用

详见 [kueue_setup.md](./kueue_setup.md) — Kueue 安装和配置。

### 实验 4：Kueue 工作负载管理

详见 [workload_demo.yaml](./workload_demo.yaml) — Kueue Workload 提交示例。

## Volcano vs Kueue 决策

```
选择 Volcano 的场景：
  ✅ 需要 Gang Scheduling（分布式训练必须）
  ✅ 复杂的 Job DAG 编排
  ✅ 已有 Volcano vcjob 生态
  ✅ 需要自定义调度算法（插件机制）

选择 Kueue 的场景：
  ✅ 多租户资源配额管理
  ✅ 希望用原生 K8s Job/PyTorchJob
  ✅ 需要资源借用（队列间弹性）
  ✅ 更轻量、维护简单

两者结合：
  → 用 Kueue 管理配额和队列
  → 用 Volcano 的 Gang Scheduling 保证训练任务原子性
  → K8s 1.29+ 支持两者协同工作
```

## 思考题

1. 多卡 GPU 集群如果有 3 个团队共享，如何用 Kueue 分配配额？
2. 一个需要 8 GPU 的训练任务，Gang Scheduling 在什么情况下会帮大忙？
3. 如何实现"非工作时间 A 队列可以借用 B 队列的配额"？
