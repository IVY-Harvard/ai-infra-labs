# GPU Operator 组件分析

## 架构总览

```
┌────────────────────────────────────────────────────────┐
│                   GPU Operator                          │
│                                                         │
│  ┌──────────────┐  监控 ClusterPolicy CRD              │
│  │  Controller  │──── 根据节点 GPU 硬件决定部署哪些组件  │
│  └──────┬───────┘                                       │
│         │ 部署/管理                                      │
│         ▼                                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  各组件（均为 DaemonSet，部署在 GPU 节点上）      │   │
│  │                                                   │   │
│  │  ① NVIDIA Driver    → 容器化 GPU 驱动            │   │
│  │  ② Container Toolkit → 配置容器运行时             │   │
│  │  ③ Device Plugin     → 向 K8s 注册 GPU 资源      │   │
│  │  ④ DCGM Exporter    → GPU 指标采集               │   │
│  │  ⑤ GPU Feature Disc. → 节点 GPU 标签             │   │
│  │  ⑥ MIG Manager      → MIG 分区管理（可选）       │   │
│  │  ⑦ Validator        → 验证全栈正常               │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

## 各组件详解

### 1. NVIDIA Driver DaemonSet

```
职责：在每个 GPU 节点上安装/管理 NVIDIA 内核驱动

工作方式：
  - 容器内编译并加载内核模块（nvidia.ko, nvidia-uvm.ko 等）
  - 自动匹配节点内核版本
  - 支持预编译驱动（加速部署）

你的 H20 环境：
  - 推荐驱动版本：550.x (Data Center Driver)
  - 如果已手动安装驱动，可以 driver.enabled=false 跳过
```

### 2. NVIDIA Container Toolkit DaemonSet

```
职责：配置容器运行时，使容器能访问 GPU

工作方式：
  - 修改 containerd/CRI-O 配置
  - 注入 nvidia-container-runtime hook
  - 使容器启动时自动挂载 GPU 设备和驱动库

关键配置文件：
  - /etc/containerd/config.toml (containerd)
  - /etc/nvidia-container-runtime/config.toml
```

### 3. NVIDIA Device Plugin DaemonSet

```
职责：向 kubelet 注册 nvidia.com/gpu 资源

工作方式（见 Lab 02 详细分析）：
  - 通过 gRPC 与 kubelet 通信
  - 上报 GPU 设备列表和健康状态
  - 处理 GPU 分配请求

与你的 H20：
  - 会注册 8 个 nvidia.com/gpu 资源
  - 支持 time-slicing 共享配置
```

### 4. DCGM Exporter DaemonSet

```
职责：采集 GPU 指标，暴露 Prometheus 格式 metrics

关键指标（H20）：
  - DCGM_FI_DEV_GPU_UTIL          GPU 利用率
  - DCGM_FI_DEV_FB_USED           显存使用量
  - DCGM_FI_DEV_FB_FREE           显存空闲量
  - DCGM_FI_DEV_GPU_TEMP          GPU 温度
  - DCGM_FI_DEV_POWER_USAGE       功耗
  - DCGM_FI_DEV_ECC_DBE_VOL_TOTAL ECC 双位错误（关键！）
  - DCGM_FI_DEV_NVLINK_BANDWIDTH  NVLink 带宽

端口：默认 9400
路径：/metrics
```

### 5. GPU Feature Discovery DaemonSet

```
职责：自动给 GPU 节点打标签

你的 H20 节点标签示例：
  nvidia.com/gpu.count=8
  nvidia.com/gpu.product=NVIDIA-H20
  nvidia.com/gpu.memory=98304         # 96GB HBM3
  nvidia.com/gpu.family=hopper
  nvidia.com/cuda.driver.major=550
  nvidia.com/cuda.runtime.major=12
  nvidia.com/mig.capable=false        # H20 不支持 MIG
  nvidia.com/gpu.compute.major=9
  nvidia.com/gpu.compute.minor=0

用途：nodeSelector / nodeAffinity 中引用这些标签
```

### 6. MIG Manager DaemonSet

```
职责：管理 MIG（Multi-Instance GPU）分区

H20 注意：H20 不支持 MIG，只有 H100/A100/A30 支持。
在你的环境中应该禁用此组件（migManager.enabled=false）。
```

### 7. Operator Validator DaemonSet

```
职责：验证整个 GPU 栈是否正常工作

验证步骤：
  1. 驱动加载 → 检查 nvidia-smi
  2. 容器运行时 → 运行一个测试容器
  3. Device Plugin → 检查 nvidia.com/gpu 资源已注册
  4. CUDA → 运行 vectorAdd 测试
  
如果验证失败 → 标记节点不可用
```

## 组件间依赖关系

```
Driver → Toolkit → Device Plugin → Validator
                 ↘                ↗
                  DCGM Exporter
                 ↗
GPU Feature Discovery

启动顺序：Driver 必须先就绪，然后 Toolkit 才能配置运行时，
         最后 Device Plugin 才能注册设备。
```
