# Lab 09 - GPU 容错实践

## 目标

实现完整的 GPU 故障检测、节点隔离、Pod 自动迁移和训练恢复流程。
结合 theory/07 的理论知识进行动手实践。

## 背景

```
多卡 GPU 集群的故障场景：

场景 1：GPU ECC 错误累积
  → 某张卡 ECC uncorrectable 错误超过阈值
  → 需要隔离该卡，迁移上面的 Pod

场景 2：训练进程卡死
  → NCCL 通信超时，但 Pod 仍显示 Running
  → 需要检测并重启

场景 3：节点宕机
  → 整个节点 NotReady
  → 需要快速将 Pod 迁移到其他节点

场景 4：GPU 驱动崩溃
  → nvidia-smi 无响应
  → 所有 GPU Pod 受影响
```

## 实验内容

### 实验 1：GPU 故障检测器

详见 [node_failure_handler.py](./node_failure_handler.py) — 自定义 GPU 故障检测和处理控制器。

### 实验 2：自动恢复配置

详见 [auto_recovery.yaml](./auto_recovery.yaml) — K8s 自动恢复相关配置。

### 实验 3：健康检查探针

详见 [health_check_probe.py](./health_check_probe.py) — GPU 训练/推理健康检查。

### 实验 4：端到端故障模拟

```bash
# 模拟 GPU 故障（在有 Fake GPU Plugin 的环境中）
# 1. 给节点添加 GPU 故障 taint
kubectl taint nodes gpu-node-0 nvidia.com/gpu-unhealthy=ecc-error:NoExecute

# 2. 观察 Pod 被驱逐
kubectl get pods -o wide -w

# 3. 观察 Pod 被重调度到健康节点
kubectl get events --sort-by=.lastTimestamp | head -20

# 4. 清理 — 模拟 GPU 修复
kubectl taint nodes gpu-node-0 nvidia.com/gpu-unhealthy-
```

## 验证清单

```
□ GPU 健康检查脚本可以检测模拟的 ECC 错误
□ 故障节点被自动添加 Taint
□ Pod 被正确驱逐并重调度
□ 训练任务从 checkpoint 自动恢复
□ 推理服务保持可用（PDB 生效）
□ 告警通知正确发送
```

## 思考题

1. 如何区分 GPU 硬件永久故障和临时性错误（如温度过高自动降频）？
2. 在什么条件下应该自动重启节点而不是仅隔离 GPU？
3. 如何避免故障检测的"误报"导致不必要的 Pod 迁移？
