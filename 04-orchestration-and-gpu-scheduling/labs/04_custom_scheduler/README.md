# Lab 04 - 自定义 GPU 调度器

## 目标

实现一个自定义的 GPU 调度器，包含拓扑感知打分和 Bin Packing 策略。
理解如何通过 Scheduler Extender 或 Scheduling Framework 扩展 K8s 调度逻辑。

## 前置条件

- Python 3.10+
- K8s 集群
- kubectl 已配置

## 架构设计

```
自定义 GPU 调度器架构：

┌──────────────────────────────────────────────┐
│              kube-scheduler                   │
│                                               │
│  内置 Filter/Score                            │
│       │                                       │
│       ▼                                       │
│  ┌──────────────────────────────────┐        │
│  │    Scheduler Extender (webhook)   │        │
│  │    POST /filter                   │        │
│  │    POST /prioritize               │        │
│  └──────────────┬───────────────────┘        │
└─────────────────┼────────────────────────────┘
                  │ HTTP
                  ▼
┌──────────────────────────────────────────────┐
│        自定义调度器 (Python)                   │
│                                               │
│  ┌────────────────┐  ┌─────────────────┐     │
│  │ topology_scorer │  │  bin_packing    │     │
│  │ NVLink 亲和性   │  │  GPU 集中分配   │     │
│  └────────────────┘  └─────────────────┘     │
│                                               │
│  数据来源：                                    │
│  - K8s API (节点/Pod 信息)                     │
│  - DCGM (GPU 拓扑和利用率)                     │
│  - 自定义 CRD (GPU 分配状态)                   │
└──────────────────────────────────────────────┘
```

## 实验内容

### 实验 1：GPU 拓扑感知调度

详见 [topology_scorer.py](./topology_scorer.py) — 基于 NVLink 拓扑的节点打分算法。

### 实验 2：Bin Packing 策略

详见 [bin_packing.py](./bin_packing.py) — GPU Bin Packing 调度策略实现。

### 实验 3：部署自定义调度器

详见 [scheduler_config.yaml](./scheduler_config.yaml) — K8s 调度器配置文件。

```bash
# 运行自定义调度器
pip install flask kubernetes
python topology_scorer.py  # 启动 extender server

# 在另一个终端测试
curl -X POST http://localhost:8888/prioritize \
  -H "Content-Type: application/json" \
  -d '{"pod": {"metadata": {"name": "test"}}, "nodes": [...]}'
```

### 实验 4：对比调度策略

```bash
# 场景：8x H20，提交多个不同 GPU 需求的 Job
# 对比默认调度 vs Bin Packing vs 拓扑感知

# 默认调度（LeastAllocated — 倾向分散）
# Job A (2 GPU) → Node 0
# Job B (2 GPU) → Node 1  ← 分散到不同节点
# Job C (4 GPU) → 无法调度！（每个节点只剩 6 GPU，但都被占了 2）

# Bin Packing（集中分配）
# Job A (2 GPU) → Node 0
# Job B (2 GPU) → Node 0  ← 集中到同一节点
# Job C (4 GPU) → Node 1  ← 可以调度
```

## 清理

```bash
kubectl delete -f scheduler_config.yaml
```

## 思考题

1. Scheduler Extender 和 Scheduling Framework Plugin 各有什么优劣？
2. 在多卡 GPU 集群上，什么场景用 Bin Packing，什么场景用 Spread？
3. 如何结合 GPU 利用率实时数据来做更智能的调度决策？
