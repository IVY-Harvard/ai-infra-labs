# Lab 08 - Spot 实例与弹性训练

## 目标

掌握如何利用云上 Spot/Preemptible GPU 实例降低训练成本，
同时通过弹性训练框架保证训练任务不因 Spot 中断而失败。

## 背景

```
Spot GPU 实例的经济账：

                On-Demand      Spot/Preemptible    节省
H100 (80GB)    $3.50/hr       $1.05/hr           70%
A100 (80GB)    $2.50/hr       $0.75/hr           70%
H20 (96GB)     $2.00/hr       $0.60/hr           70%  (估算)

8x H20 训练一个 70B 模型：
  On-Demand: 8 × $2.00 × 720hr = $11,520/月
  Spot:      8 × $0.60 × 720hr = $3,456/月
  节省：$8,064/月

代价：Spot 实例随时可能被回收（通常有 30s-2min 预警）
解决：弹性训练 + Checkpoint + 自动恢复
```

## 实验内容

### 实验 1：Spot 实例中断处理

详见 [spot_handler.py](./spot_handler.py) — Spot 中断信号处理器。

### 实验 2：弹性训练配置

详见 [elastic_training.yaml](./elastic_training.yaml) — PyTorch Elastic Training on Spot。

### 实验 3：混合 On-Demand + Spot 策略

```bash
# 节点标签标识 Spot vs On-Demand
kubectl label node spot-node-0 cloud.provider/instance-lifecycle=spot
kubectl label node ondemand-node-0 cloud.provider/instance-lifecycle=normal

# Spot 节点加 Taint（只有容忍 Spot 中断的 Pod 才能调度）
kubectl taint nodes spot-node-0 cloud.provider/spot=true:NoSchedule
```

## 架构设计

```
弹性 GPU 训练架构（Spot + On-Demand 混合）：

┌───────────────────────────────────────────────────────┐
│                    训练控制器                           │
│  (PyTorch Elastic / Kubeflow Training Operator)       │
└───────────────────┬───────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│Worker 0 │   │Worker 1 │   │Worker 2 │    ← 弹性 worker 数量
│On-Demand│   │  Spot   │   │  Spot   │
│ (保底)  │   │(可被回收)│   │(可被回收)│
│ 2 GPU   │   │ 2 GPU   │   │ 2 GPU   │
└─────────┘   └────┬────┘   └─────────┘
                    │
              Spot 中断信号
                    │
                    ▼
         ┌──────────────────┐
         │ 1. 保存 checkpoint │
         │ 2. 通知其他 worker │
         │ 3. 缩小训练规模   │
         │ 4. 等待新 Spot    │
         │ 5. 扩大恢复       │
         └──────────────────┘
```

## 成本优化计算器

```python
# 简易计算
on_demand_cost_per_hour = 8 * 2.00  # 8x H20 On-Demand
spot_cost_per_hour = 8 * 0.60       # 8x H20 Spot
spot_interruption_rate = 0.05       # 假设 5% 时间在恢复

# 有效训练时间
effective_spot_hours = 720 * (1 - spot_interruption_rate)  # 684 小时

# 月度总成本
on_demand_monthly = on_demand_cost_per_hour * 720
spot_monthly = spot_cost_per_hour * 720  # Spot 即使被中断也按使用时间计费
savings = on_demand_monthly - spot_monthly

print(f"On-Demand: ${on_demand_monthly:.0f}/月")
print(f"Spot:      ${spot_monthly:.0f}/月")
print(f"节省:      ${savings:.0f}/月 ({savings/on_demand_monthly*100:.0f}%)")
```

## 思考题

1. 如果 Spot 中断率高达 20%，训练效率会下降多少？什么时候 Spot 不再划算？
2. Checkpoint 保存频率如何根据 Spot 中断概率动态调整？
3. 混合 On-Demand+Spot 时，如何确保至少 1 个 On-Demand worker 存活？
