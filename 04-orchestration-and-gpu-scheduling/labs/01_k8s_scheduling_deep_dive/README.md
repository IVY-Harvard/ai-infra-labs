# Lab 01 - K8s 调度器深入探索

## 目标

通过实际操作理解 kube-scheduler 的内部工作机制，包括调度队列、Filter/Score 流程、
调度器扩展点，以及如何编写一个简单的调度插件。

## 前置条件

- 可用的 K8s 集群（本地 kind/minikube 或多卡 GPU 集群）
- kubectl 已配置
- Go 1.21+ （调度插件开发需要）

## 实验内容

### 实验 1：观察调度器行为

```bash
# 1. 查看调度器配置
kubectl get pods -n kube-system -l component=kube-scheduler -o yaml

# 2. 查看调度器日志（提高日志级别）
kubectl logs -n kube-system kube-scheduler-<node-name> -f

# 3. 创建一个 GPU Pod，观察调度事件
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-scheduling-test
spec:
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# 4. 观察调度事件
kubectl describe pod gpu-scheduling-test | grep -A 20 Events
kubectl get events --field-selector involvedObject.name=gpu-scheduling-test
```

### 实验 2：调度器 Trace 分析

详见 [scheduler_trace.md](./scheduler_trace.md) — 完整的调度 trace 抓取与分析指南。

### 实验 3：调度器插件开发

详见 [scheduling_plugin_demo.go](./scheduling_plugin_demo.go) — 实现一个简单的 GPU
拓扑感知 Score 插件。

编译运行：

```bash
# 构建自定义调度器
cd scheduling-plugin
go build -o gpu-topo-scheduler .

# 运行（连接到 K8s 集群）
./gpu-topo-scheduler \
  --kubeconfig=$HOME/.kube/config \
  --config=scheduler-config.yaml \
  --v=5
```

### 实验 4：调度性能分析

```bash
# 部署 100 个 GPU Pod 测试调度吞吐
for i in $(seq 1 100); do
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: sched-perf-test-$i
  labels:
    test: scheduling-perf
spec:
  containers:
  - name: sleep
    image: busybox
    command: ["sleep", "10"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF
done

# 观察调度延迟
kubectl get pods -l test=scheduling-perf -o json | \
  jq '.items[] | {name: .metadata.name,
      created: .metadata.creationTimestamp,
      scheduled: (.status.conditions[] | select(.type=="PodScheduled") | .lastTransitionTime)}'
```

## 清理

```bash
kubectl delete pod gpu-scheduling-test
kubectl delete pods -l test=scheduling-perf
```

## 思考题

1. 在多卡 GPU 集群中，调度一个需要 4 GPU 的 Pod 时，Filter 阶段会排除哪些节点？
2. 如果两个节点都有 4 个空闲 GPU，Score 阶段默认会怎么排序？
3. 如何让调度器优先把 GPU Pod 集中到同一节点（Bin Packing）？
