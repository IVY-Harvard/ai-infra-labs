# Lab 03 - NVIDIA GPU Operator

## 目标

理解 GPU Operator 如何自动化 GPU 节点的全栈管理，包括驱动安装、容器运行时配置、
设备插件部署和监控。掌握 GPU Operator 的故障排查方法。

## 背景

手动管理 GPU 节点涉及很多组件：

```
传统方式（手动）：
  1. 安装 NVIDIA 驱动       → 内核版本敏感，容易出问题
  2. 安装 CUDA Toolkit      → 版本兼容性
  3. 安装 nvidia-container-toolkit  → containerd/CRI-O 配置
  4. 部署 nvidia-device-plugin      → DaemonSet
  5. 部署 DCGM Exporter     → 监控
  6. 配置 GPU Feature Discovery → 标签

GPU Operator（自动化）：
  → 一个 Operator 管理以上所有组件
  → 自动检测 GPU 硬件，选择正确的驱动版本
  → 通过 CRD 声明式管理配置
```

## 实验内容

### 实验 1：安装 GPU Operator

详见 [install_guide.md](./install_guide.md) — 完整安装步骤和配置选项。

### 实验 2：理解各组件

详见 [components_analysis.md](./components_analysis.md) — 每个组件的作用和交互关系。

### 实验 3：故障排查

详见 [troubleshooting.md](./troubleshooting.md) — 常见问题和诊断步骤。

## 快速验证

```bash
# 安装后验证所有组件就绪
kubectl get pods -n gpu-operator

# 预期输出：
# NAME                                       READY   STATUS
# gpu-operator-xxx                           1/1     Running
# nvidia-driver-daemonset-xxx                1/1     Running
# nvidia-container-toolkit-daemonset-xxx     1/1     Running
# nvidia-device-plugin-daemonset-xxx         1/1     Running
# nvidia-dcgm-exporter-xxx                   1/1     Running
# gpu-feature-discovery-xxx                  1/1     Running
# nvidia-operator-validator-xxx              1/1     Running

# 验证 GPU 可用
kubectl run gpu-test --rm -it --image=nvidia/cuda:12.2.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 -- nvidia-smi
```

## 思考题

1. GPU Operator 与手动安装 Device Plugin 相比，在多卡 GPU 环境中有什么优势？
2. 如果 GPU 驱动需要升级，GPU Operator 如何做到不中断运行中的 GPU 工作负载？
3. 为什么 GPU Operator 默认用容器化驱动而不是 host 驱动？
