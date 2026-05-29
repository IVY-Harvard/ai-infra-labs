# Lab 05 - GPU 共享技术

## 目标

掌握 GPU 共享的三种主要方式：MIG、MPS、Time-Slicing。
理解每种方式的适用场景、性能特征和配置方法。

## 背景

```
为什么需要 GPU 共享？

多卡 GPU 场景：
  - H20 有 96GB HBM3 显存
  - 一个小模型推理只需 10GB 显存
  - 如果每个推理服务独占一张卡 → 86GB 浪费
  - GPU 利用率可能只有 10-20%

GPU 共享方式对比：

方式        | 隔离性 | 性能损失 | 显存隔离 | 适用 GPU     | 最小粒度
-----------+-------+--------+---------+------------+---------
MIG        | 强    | ~0%    | 硬件隔离 | A100/H100  | 1/7 GPU
MPS        | 中    | 5-15%  | 无隔离   | 所有 NVIDIA | 灵活
Time-Slice | 弱    | 10-30% | 无隔离   | 所有 NVIDIA | 时间片

注意：H20 不支持 MIG！但了解 MIG 对于混合集群（A100+H20）很重要。
H20 可以使用 MPS 或 Time-Slicing。
```

## 实验内容

### 实验 1：MIG 配置（A100/H100 环境）

详见 [mig_setup_guide.md](./mig_setup_guide.md) — MIG 分区配置完整指南。

### 实验 2：MPS 配置

详见 [mps_config.md](./mps_config.md) — MPS 在 K8s 中的配置方法。

### 实验 3：Time-Slicing 配置

详见 [time_slicing.yaml](./time_slicing.yaml) — NVIDIA Device Plugin 的 Time-Slicing 配置。

### 实验 4：性能基准测试

详见 [comparison_benchmark.py](./comparison_benchmark.py) — 对比不同共享方式的性能。

## 快速决策指南

```
多卡 GPU 环境推荐：

场景                    | 推荐方式      | 原因
----------------------+-------------+------------------
多个小模型推理 (< 20GB) | MPS         | 低延迟，适合推理
开发/调试多人共享       | Time-Slicing | 简单，不需要改代码
大模型训练             | 独占         | 训练需要全部显存和带宽
混合工作负载           | MPS + 独占   | 推理用 MPS，训练独占
```

## 清理

```bash
# 恢复 Device Plugin 默认配置（移除 Time-Slicing）
kubectl delete configmap device-plugin-config -n gpu-operator
# 重启 Device Plugin DaemonSet
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n gpu-operator
```

## 思考题

1. 在 H20 上用 MPS 共享给 4 个推理服务，每个服务的显存上限怎么控制？
2. Time-Slicing 下如果一个 Pod 跑满了 GPU 计算，其他 Pod 会怎样？
3. 有没有办法在 K8s 层面限制每个 Pod 的 GPU 显存使用量？
