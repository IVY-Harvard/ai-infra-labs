# 调度器 Trace 抓取与分析指南

## 1. 启用调度器 Verbose 日志

```bash
# 修改 kube-scheduler 启动参数，提高日志级别
# 在 /etc/kubernetes/manifests/kube-scheduler.yaml 中添加：
spec:
  containers:
  - command:
    - kube-scheduler
    - --v=6                    # 级别 6 可以看到完整的调度决策过程
    - --config=/etc/kubernetes/scheduler-config.yaml
```

各日志级别包含的信息：

| 级别 | 内容 |
|------|------|
| v=2 | 调度结果（Pod 绑定到哪个节点） |
| v=4 | Filter/Score 的详细流程 |
| v=6 | 每个插件的执行结果和耗时 |
| v=8 | 完整的 API 调用（非常冗长） |

## 2. 一次完整的 GPU Pod 调度 Trace

```
# 提交一个需要 2 GPU 的 Pod，以下是完整的调度日志分析：

[Scheduling Queue]
  I0601 10:00:01.001  "Pod added to activeQ"  pod="default/train-job-0"
  # Pod 被加入活跃队列，等待调度

[Pre-Filter Phase]
  I0601 10:00:01.002  "Running PreFilter"  plugin="NodeResourcesFit"
  # 检查 Pod 的资源请求是否合理（不超过集群最大单节点容量）

[Filter Phase]
  I0601 10:00:01.003  "Running Filter"  plugin="NodeResourcesFit"
  # 逐个检查节点的可用资源

  Node: gpu-node-0  (8 GPU total, 4 GPU allocated)
    → NodeResourcesFit: 可用 GPU 4 >= 请求 2 → PASS
  Node: gpu-node-1  (8 GPU total, 7 GPU allocated)
    → NodeResourcesFit: 可用 GPU 1 < 请求 2 → FILTERED OUT
  Node: cpu-node-0  (0 GPU total)
    → NodeResourcesFit: 可用 GPU 0 < 请求 2 → FILTERED OUT

  I0601 10:00:01.004  "Filter result"  feasibleNodes=1  totalNodes=3

[Score Phase]
  I0601 10:00:01.005  "Running Score"  plugin="NodeResourcesBalancedAllocation"
  Node: gpu-node-0  score=60
  
  I0601 10:00:01.005  "Running Score"  plugin="NodeResourcesFit" (LeastAllocated)
  Node: gpu-node-0  score=50  # (4 free / 8 total) * 100 = 50

  I0601 10:00:01.006  "Final scores"
  Node: gpu-node-0  finalScore=110

[Reserve Phase]
  I0601 10:00:01.007  "Reserving resources"  node="gpu-node-0"

[Bind Phase]
  I0601 10:00:01.010  "Binding Pod to Node"  pod="train-job-0"  node="gpu-node-0"
  # 总调度耗时: 9ms
```

## 3. 使用 Prometheus 监控调度延迟

```bash
# 调度器暴露的关键指标
scheduler_scheduling_attempt_duration_seconds_bucket  # 调度尝试延迟分布
scheduler_pending_pods                                # 各队列中等待的 Pod 数
scheduler_pod_scheduling_sli_duration_seconds         # SLI 调度延迟
scheduler_plugin_execution_duration_seconds           # 各插件执行耗时

# PromQL 查询示例：
# P99 调度延迟
histogram_quantile(0.99, sum(rate(
  scheduler_scheduling_attempt_duration_seconds_bucket[5m]
)) by (le, result))

# GPU Pod 的平均调度延迟（按标签过滤）
histogram_quantile(0.5, sum(rate(
  scheduler_pod_scheduling_sli_duration_seconds_bucket{
    attempts="1"
  }[5m]
)) by (le))
```

## 4. 常见调度问题诊断

```
问题 1：GPU Pod 卡在 Pending
  → kubectl describe pod <name> 查看 Events
  → 常见原因：
    a) 集群 GPU 资源不足 → 等待其他 Pod 释放
    b) Taint/Toleration 不匹配 → 检查 GPU 节点是否有 Taint
    c) NodeAffinity 过于严格 → 放宽亲和性规则

问题 2：调度到了错误的节点
  → 检查 Score 插件的权重配置
  → 确认 GPU 拓扑亲和性规则是否正确

问题 3：调度延迟过高
  → 检查 Informer Cache 同步状态
  → 检查是否有大量 Unschedulable Pod 阻塞队列
  → 考虑增加 parallelism 参数
```
