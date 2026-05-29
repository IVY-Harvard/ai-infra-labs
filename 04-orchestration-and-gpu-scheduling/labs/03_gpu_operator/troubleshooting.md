# GPU Operator 故障排查指南

## 诊断流程

```
GPU 工作负载失败
    │
    ├─ Pod Pending → kubectl describe pod → 检查事件
    │   └─ "Insufficient nvidia.com/gpu"
    │       → 检查 Device Plugin 是否运行正常
    │
    ├─ Pod CrashLoop → kubectl logs → 检查容器日志
    │   └─ "CUDA error: no CUDA-capable device"
    │       → 检查 Container Toolkit 和驱动
    │
    └─ GPU Operator Pod 不正常 → 按组件逐一排查
```

## 常见问题

### 问题 1：Driver DaemonSet 卡在 Init

```bash
# 现象
kubectl get pods -n gpu-operator
# nvidia-driver-daemonset-xxx  0/1  Init:0/1

# 诊断
kubectl logs -n gpu-operator nvidia-driver-daemonset-xxx -c nvidia-driver-ctr
# 可能看到：内核头文件缺失

# 解决
# 方案 A：安装内核头文件
apt-get install linux-headers-$(uname -r)

# 方案 B：使用预编译驱动
helm upgrade gpu-operator nvidia/gpu-operator \
  --set driver.usePrecompiled=true

# 方案 C：使用 host 驱动（跳过容器化驱动）
helm upgrade gpu-operator nvidia/gpu-operator \
  --set driver.enabled=false
```

### 问题 2：Device Plugin 运行但节点没有 GPU 资源

```bash
# 现象
kubectl describe node <gpu-node> | grep nvidia
# 没有 nvidia.com/gpu

# 诊断步骤
# 1. 检查 Device Plugin 日志
kubectl logs -n gpu-operator nvidia-device-plugin-daemonset-xxx

# 2. 检查 kubelet 日志
journalctl -u kubelet | grep device-plugin

# 3. 检查 socket 文件
ls -la /var/lib/kubelet/device-plugins/nvidia*.sock

# 4. 检查驱动是否正常
nvidia-smi

# 常见原因
# a) Container Toolkit 没有正确配置 containerd
kubectl logs -n gpu-operator nvidia-container-toolkit-daemonset-xxx
# 检查 /etc/containerd/config.toml 是否包含 nvidia runtime

# b) containerd 需要重启
systemctl restart containerd
```

### 问题 3：Pod 启动后看不到 GPU

```bash
# 现象
kubectl exec -it <pod> -- nvidia-smi
# Failed to initialize NVML: Unknown Error

# 诊断
# 1. 检查 Pod 是否确实请求了 GPU
kubectl get pod <pod> -o jsonpath='{.spec.containers[*].resources}'

# 2. 检查环境变量
kubectl exec -it <pod> -- env | grep NVIDIA
# 应该有 NVIDIA_VISIBLE_DEVICES=0,1,...

# 3. 检查设备挂载
kubectl exec -it <pod> -- ls /dev/nvidia*

# 解决
# 通常是 Container Toolkit 版本与 containerd 不兼容
# 升级 nvidia-container-toolkit
```

### 问题 4：DCGM Exporter 无指标

```bash
# 现象
curl http://<node-ip>:9400/metrics
# 空或报错

# 诊断
kubectl logs -n gpu-operator nvidia-dcgm-exporter-xxx
# 常见错误：
# "Unable to connect to DCGM" → DCGM daemon 未启动
# "No GPUs found" → 驱动问题

# 解决
# 确认 DCGM 与驱动版本兼容
kubectl exec -it nvidia-dcgm-exporter-xxx -n gpu-operator -- dcgmi discovery -l
```

### 问题 5：Validator 失败

```bash
# 现象
kubectl get pods -n gpu-operator
# nvidia-operator-validator-xxx  0/1  Init:2/4

# 诊断 — 查看哪一步失败
kubectl logs -n gpu-operator nvidia-operator-validator-xxx -c driver-validation
kubectl logs -n gpu-operator nvidia-operator-validator-xxx -c toolkit-validation
kubectl logs -n gpu-operator nvidia-operator-validator-xxx -c plugin-validation
kubectl logs -n gpu-operator nvidia-operator-validator-xxx -c cuda-validation

# 对应的验证步骤失败说明上游组件有问题
```

## 排查清单（你的 8x H20 环境）

```
□ nvidia-smi 在 host 上可以正常执行
□ GPU Operator 所有 Pod 状态为 Running
□ kubectl describe node 显示 nvidia.com/gpu: 8
□ 节点有 nvidia.com/gpu.product=NVIDIA-H20 标签
□ DCGM Exporter 的 /metrics 端点有数据
□ 测试 Pod 可以运行 nvidia-smi
□ 多 GPU Pod（请求 4 或 8 GPU）可以正常调度和运行
```

## 日志收集

```bash
# 收集所有 GPU Operator 组件日志
for pod in $(kubectl get pods -n gpu-operator -o name); do
  echo "=== $pod ==="
  kubectl logs -n gpu-operator $pod --all-containers 2>/dev/null
done > gpu-operator-logs.txt

# 收集节点信息
kubectl describe nodes > nodes-info.txt

# 收集 ClusterPolicy 状态
kubectl get clusterpolicy -o yaml > clusterpolicy.yaml
```
