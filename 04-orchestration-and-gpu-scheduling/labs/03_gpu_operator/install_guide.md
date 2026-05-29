# GPU Operator 安装指南

## 1. 前置条件检查

```bash
# 检查 GPU 硬件
lspci | grep -i nvidia
# 预期输出：NVIDIA H20 (或你的 GPU 型号)

# 检查内核版本（驱动兼容性）
uname -r
# 推荐：5.15+ (Ubuntu 22.04) 或 5.14+ (RHEL 9)

# 检查容器运行时
kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.containerRuntimeVersion}'
# 支持：containerd 1.6+, CRI-O 1.24+

# 检查 K8s 版本
kubectl version --short
# GPU Operator v24.x 支持 K8s 1.27-1.30
```

## 2. 使用 Helm 安装

```bash
# 添加 NVIDIA Helm 仓库
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# 安装 GPU Operator（标准配置）
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set gfd.enabled=true \
  --set migManager.enabled=false \
  --set operator.defaultRuntime=containerd
```

## 3. 针对 8x H20 环境的配置

```bash
# H20 优化配置
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --values - <<EOF
driver:
  enabled: true
  version: "550.90.07"           # H20 推荐驱动版本
  
toolkit:
  enabled: true

devicePlugin:
  enabled: true
  config:
    name: device-plugin-config
    default: h20-config
    data:
      h20-config: |
        version: v1
        sharing:
          timeSlicing:
            renameByDefault: false
            resources:
            - name: nvidia.com/gpu
              replicas: 1          # 不开启 time-slicing（训练场景）

dcgmExporter:
  enabled: true
  serviceMonitor:
    enabled: true                  # 如果有 Prometheus Operator

gfd:
  enabled: true                    # GPU Feature Discovery

migManager:
  enabled: false                   # H20 不支持 MIG，关闭

operator:
  defaultRuntime: containerd
EOF
```

## 4. 已有驱动的环境

```bash
# 如果节点已经安装了 NVIDIA 驱动（如你的 H20 已有驱动）
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true
```

## 5. 安装验证

```bash
# 检查所有 Pod 状态
kubectl get pods -n gpu-operator -w

# 检查 ClusterPolicy（GPU Operator 的 CRD）
kubectl get clusterpolicy cluster-policy -o yaml

# 查看 GPU 节点标签
kubectl get nodes --show-labels | grep nvidia
# 预期标签：
#   nvidia.com/cuda.driver.major=550
#   nvidia.com/gpu.count=8
#   nvidia.com/gpu.product=NVIDIA-H20
#   nvidia.com/gpu.memory=98304

# 运行验证 Pod
kubectl run gpu-verify --rm -it \
  --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

## 6. 升级与卸载

```bash
# 升级 GPU Operator
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --reuse-values \
  --set driver.version="550.100.01"

# 卸载（注意：会中断所有 GPU 工作负载）
helm uninstall gpu-operator -n gpu-operator
```
