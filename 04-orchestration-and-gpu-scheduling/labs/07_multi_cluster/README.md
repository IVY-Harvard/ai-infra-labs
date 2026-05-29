# Lab 07 - 多集群 GPU 调度

## 目标

理解在多个 K8s 集群之间调度 GPU 工作负载的方案。
掌握 KubeFed、Admiralty、Liqo 等多集群方案的特点。

## 背景

```
多集群场景（可能的扩展路径）：

当前：单集群 8x H20
未来：
  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │  私有集群 (IDC)  │    │  云集群 A (GCP)  │    │  云集群 B (AWS)  │
  │  8x H20         │    │  按需 A100      │    │  Spot H100      │
  │  日常训练+推理   │    │  burst 训练      │    │  大规模训练      │
  └─────────────────┘    └─────────────────┘    └─────────────────┘
           │                       │                       │
           └───────────── 统一调度层 ──────────────────────┘

何时需要多集群 GPU 调度：
  - 私有集群 GPU 不够 → burst 到云上
  - 不同区域需要推理服务（低延迟）
  - 利用 Spot/Preemptible 实例降成本
  - 合规要求（数据不出境）
```

## 实验内容

### 实验 1：多集群方案对比

详见 [kubefed_overview.md](./kubefed_overview.md) — 各多集群方案的对比分析。

### 实验 2：跨集群 GPU Job 提交

详见 [cross_cluster_job.yaml](./cross_cluster_job.yaml) — 跨集群调度配置示例。

### 实验 3：模拟多集群调度

```bash
# 使用 kind 创建两个模拟集群
kind create cluster --name cluster-onprem --config kind-config-onprem.yaml
kind create cluster --name cluster-cloud --config kind-config-cloud.yaml

# 验证
kubectl --context kind-cluster-onprem get nodes
kubectl --context kind-cluster-cloud get nodes

# 安装 Fake GPU Plugin 模拟 GPU
kubectl --context kind-cluster-onprem apply -f ../02_device_plugin/daemonset.yaml
kubectl --context kind-cluster-cloud apply -f ../02_device_plugin/daemonset.yaml
```

## 思考题

1. 跨集群训练时，数据传输和模型同步的带宽瓶颈怎么处理？
2. 在混合云方案中，如何决定一个任务应该在私有集群还是云上运行？
3. 多集群故障域隔离如何与 GPU 任务的 Gang Scheduling 兼容？
