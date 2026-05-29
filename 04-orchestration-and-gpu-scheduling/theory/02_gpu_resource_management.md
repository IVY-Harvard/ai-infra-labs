# 02 - K8s GPU 资源管理

## 引言：GPU 在 K8s 中为什么特殊

CPU 和 Memory 是 K8s 的 **原生资源**，调度器天然理解它们。但 GPU 是 **扩展资源**
(Extended Resource)，K8s 核心不知道 GPU 是什么、怎么分配、怎么隔离。

这就需要两层抽象：
1. **Device Plugin** — 告诉 K8s "我这个节点有哪些 GPU 可用"
2. **GPU Operator** — 自动化安装 driver、toolkit、device plugin 等所有 GPU 相关组件

## 1. Device Plugin 机制

### 1.1 设计哲学

K8s 设计者的决策：**不把硬件特定逻辑放入 kubelet 核心**。

```
对比两种方案：

方案 A（未采用）：kubelet 内置 GPU 管理
  kubelet → 直接调用 nvidia-smi → 报告 GPU 数量
  问题：每加一种硬件（GPU/FPGA/TPU/SmartNIC）都要改 kubelet

方案 B（采用）：Device Plugin 接口
  kubelet ←gRPC→ Device Plugin（厂商实现）
  kubelet 只需要知道：有多少个 "nvidia.com/gpu" 资源可用
  不关心底层是 NVIDIA/AMD/Intel 的 GPU
```

### 1.2 Device Plugin 注册流程

```
┌──────────────────────────────────────────────────────────────┐
│                         Node                                  │
│                                                              │
│  ┌─────────┐       Unix Socket         ┌─────────────────┐ │
│  │ kubelet │◄──────────────────────────►│ Device Plugin   │ │
│  │         │  /var/lib/kubelet/         │ (nvidia-device- │ │
│  │         │  device-plugins/           │  plugin)        │ │
│  │         │  nvidia.sock              │                 │ │
│  └────┬────┘                           └────────┬────────┘ │
│       │                                         │           │
│       │ Registration RPC                         │           │
│       │ ◄────────────────────────────────────────┘           │
│       │   "我是 nvidia.com/gpu 的管理者"                       │
│       │                                                      │
│       │ ListAndWatch RPC                                     │
│       │ ◄──────────────── 持续上报设备列表和健康状态              │
│       │   [{id:"GPU-uuid-0", healthy:true},                  │
│       │    {id:"GPU-uuid-1", healthy:true}, ...]             │
│       │                                                      │
│       │ Allocate RPC (Pod 被调度到此节点时)                     │
│       │ ────────────────► "请分配 2 个 GPU 给这个容器"          │
│       │ ◄──────────────── 返回：环境变量 + 设备挂载             │
│       │   {envs: {NVIDIA_VISIBLE_DEVICES: "0,1"},            │
│       │    devices: [{/dev/nvidia0}, {/dev/nvidia1}]}        │
│       │                                                      │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 三个核心 gRPC 接口

```protobuf
service DevicePlugin {
  // 持续流式上报设备列表
  rpc ListAndWatch(Empty) returns (stream ListAndWatchResponse) {}

  // 分配设备给容器
  rpc Allocate(AllocateRequest) returns (AllocateResponse) {}

  // 获取设备的首选分配方案（可选）
  rpc GetPreferredAllocation(PreferredAllocationRequest)
      returns (PreferredAllocationResponse) {}
}
```

**ListAndWatch** — 设备发现与健康监控

```go
func (p *NvidiaPlugin) ListAndWatch(e *pluginapi.Empty,
    s pluginapi.DevicePlugin_ListAndWatchServer) error {
    // 初始上报
    s.Send(&pluginapi.ListAndWatchResponse{
        Devices: p.getDevices(), // 所有 GPU 列表
    })
    // 持续监控
    for {
        select {
        case <-p.healthCheck:
            // GPU 状态变化时重新上报
            s.Send(&pluginapi.ListAndWatchResponse{
                Devices: p.getDevices(),
            })
        case <-p.stop:
            return nil
        }
    }
}
```

**Allocate** — 设备分配

```go
func (p *NvidiaPlugin) Allocate(ctx context.Context,
    req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {

    responses := &pluginapi.AllocateResponse{}
    for _, containerReq := range req.ContainerRequests {
        // containerReq.DevicesIDs = ["GPU-uuid-0", "GPU-uuid-3"]
        response := &pluginapi.ContainerAllocateResponse{
            Envs: map[string]string{
                "NVIDIA_VISIBLE_DEVICES": strings.Join(containerReq.DevicesIDs, ","),
            },
            // 不需要显式挂载 /dev/nvidia* — nvidia-container-runtime 处理
        }
        responses.ContainerResponses = append(responses.ContainerResponses, response)
    }
    return responses, nil
}
```

**GetPreferredAllocation** — 拓扑感知分配（关键！）

```go
// 当 Pod 请求 4 个 GPU 时，节点有 8 个 GPU
// 应该分配哪 4 个？选择 NVLink 互联的 4 个！
func (p *NvidiaPlugin) GetPreferredAllocation(ctx context.Context,
    req *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {

    // 基于 GPU 拓扑（NVLink/NVSwitch 连接关系）选择最优组合
    // H20: 8 GPU 通过 NVLink 全连接
    // 如果是 A100 DGX: 2 组 4GPU 通过 NVSwitch 连接
    preferred := selectByTopology(req.AvailableDeviceIDs, req.MustIncludeDeviceIDs, req.AllocationSize)
    return &pluginapi.PreferredAllocationResponse{
        ContainerResponses: []*pluginapi.ContainerPreferredAllocationResponse{
            {DeviceIDs: preferred},
        },
    }, nil
}
```

### 1.4 Device Plugin 的局限性

| 局限 | 说明 | 影响 |
|------|------|------|
| 整数分配 | 只能分配整数个设备 | 无法原生做 GPU 共享 |
| 无跨节点视图 | 每个节点独立运行 | 无法做多节点 GPU 拓扑调度 |
| 无动态资源 | 设备数量重启后才能变 | MIG 重分区需要重启 Plugin |
| 分配不可逆 | 分配后只能删 Pod 释放 | 无法在线迁移 GPU |
| 无细粒度信息 | 只报告数量和健康状态 | 无法表达 GPU 型号、显存等 |

## 2. GPU Operator 架构

### 2.1 为什么需要 GPU Operator

只有 Device Plugin 够吗？看看裸机部署 GPU 需要多少步骤：

```bash
# 手动部署流程（每个节点！）
1. 安装 NVIDIA Driver         （内核模块编译，版本匹配）
2. 安装 nvidia-container-toolkit（容器运行时 hook）
3. 配置 containerd/docker      （添加 nvidia runtime）
4. 安装 nvidia-device-plugin   （DaemonSet）
5. 安装 DCGM                  （GPU 监控）
6. 配置 MIG（如需要）          （分区管理）
7. 安装 Node Feature Discovery  （GPU 特性标签）
8. 配置 GDS（如需要）          （GPU Direct Storage）
```

**GPU Operator 的价值：把上述所有步骤自动化为声明式管理。**

### 2.2 架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Operator Controller                        │
│                                                                  │
│  ClusterPolicy CR ──► Reconcile Loop                            │
│                        │                                         │
│         ┌──────────────┼──────────────────────────────┐         │
│         ▼              ▼              ▼               ▼          │
│  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌──────────────┐ │
│  │  Driver  │  │  Container   │  │  DCGM  │  │   Device     │ │
│  │ DaemonSet│  │  Toolkit     │  │Exporter│  │   Plugin     │ │
│  │          │  │  DaemonSet   │  │        │  │  DaemonSet   │ │
│  └──────────┘  └──────────────┘  └────────┘  └──────────────┘ │
│         │              │              │               │          │
│         ▼              ▼              ▼               ▼          │
│  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌──────────────┐ │
│  │   MIG    │  │     Node     │  │  GPU   │  │   Validator  │ │
│  │ Manager  │  │   Feature    │  │Feature │  │  DaemonSet   │ │
│  │DaemonSet │  │  Discovery   │  │Discovery│ │              │ │
│  └──────────┘  └──────────────┘  └────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 各组件职责详解

#### Driver DaemonSet
```
职责：在每个 GPU 节点上安装 NVIDIA 内核驱动
实现：使用预编译的驱动容器镜像，通过 init container 加载内核模块
特殊点：
  - 使用 hostPID + privileged 模式
  - 编译内核模块时需要匹配节点内核版本
  - 支持预编译驱动（推荐）和运行时编译两种模式
与你的环境：H20 需要 driver >= 535.xx
```

#### Container Toolkit DaemonSet
```
职责：安装和配置 nvidia-container-toolkit
功能：
  - 注册 nvidia container runtime hook
  - 使容器能够访问 GPU 设备
  - 配置 containerd 的 runtime class
结果：Pod 中可以通过 NVIDIA_VISIBLE_DEVICES 环境变量访问 GPU
```

#### DCGM Exporter
```
职责：暴露 GPU Metrics 供 Prometheus 采集
关键指标：
  - DCGM_FI_DEV_GPU_UTIL         GPU 利用率
  - DCGM_FI_DEV_MEM_COPY_UTIL    显存带宽利用率
  - DCGM_FI_DEV_FB_USED          已用显存
  - DCGM_FI_DEV_POWER_USAGE      功耗
  - DCGM_FI_DEV_GPU_TEMP         温度
  - DCGM_FI_DEV_NVLINK_BANDWIDTH NVLink 带宽
端口：9400 (默认)
```

#### MIG Manager DaemonSet
```
职责：管理 A100/H100/H20 的 MIG 分区
工作流：
  1. 读取 ConfigMap 中的 MIG 配置
  2. 在 GPU 上执行 MIG 分区操作
  3. 重启 Device Plugin 使新分区生效
  4. 更新节点标签
注意：MIG 重分区需要无 Pod 使用该 GPU（需要 drain）
```

#### Node Feature Discovery (NFD)
```
职责：自动发现节点硬件特性，添加标签
GPU 相关标签：
  nvidia.com/gpu.product=NVIDIA-H20
  nvidia.com/gpu.memory=98304
  nvidia.com/gpu.count=8
  nvidia.com/gpu.family=hopper
  nvidia.com/mig.capable=true
  nvidia.com/gpu.compute.major=9
用途：NodeSelector/NodeAffinity 可以基于这些标签调度
```

#### GPU Feature Discovery
```
职责：更细粒度的 GPU 特性发现
额外标签：
  nvidia.com/cuda.driver.major=535
  nvidia.com/cuda.runtime.major=12
  nvidia.com/gpu.deploy.dcgm-exporter=true
  nvidia.com/mig.config=all-balanced
```

#### Validator DaemonSet
```
职责：验证所有组件正确安装
检查项：
  - Driver 加载成功
  - Container Toolkit 配置正确
  - Device Plugin 运行正常
  - 能成功创建使用 GPU 的测试 Pod
结果：设置节点标签 nvidia.com/gpu.deploy.operatorValidator=true
```

### 2.4 ClusterPolicy CR

```yaml
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: cluster-policy
spec:
  operator:
    defaultRuntime: containerd

  driver:
    enabled: true
    repository: nvcr.io/nvidia
    image: driver
    version: "535.129.03"
    # 对于已预装驱动的节点
    # enabled: false

  toolkit:
    enabled: true
    repository: nvcr.io/nvidia/k8s
    image: container-toolkit
    version: v1.14.3

  devicePlugin:
    enabled: true
    repository: nvcr.io/nvidia
    image: k8s-device-plugin
    version: v0.14.3
    config:
      name: device-plugin-config
      default: any
    # MIG 策略
    # "none" = 不使用 MIG
    # "single" = 每个 MIG 实例作为单独设备
    # "mixed" = 同一 GPU 上混合 MIG 和非 MIG
    mig:
      strategy: single

  dcgmExporter:
    enabled: true
    repository: nvcr.io/nvidia/k8s
    image: dcgm-exporter
    version: 3.3.0-3.2.0

  migManager:
    enabled: true
    config:
      name: mig-config

  nodeFeatureDiscovery:
    enabled: true

  gfd:
    enabled: true
```

## 3. 资源请求与限制

### 3.1 GPU 资源声明

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-train
spec:
  containers:
  - name: trainer
    image: nvcr.io/nvidia/pytorch:24.01-py3
    resources:
      limits:
        nvidia.com/gpu: 4           # 请求 4 个 GPU
        # 注意：GPU 不支持 requests != limits
        # 设置 limits 时 requests 自动等于 limits
      requests:
        cpu: "32"
        memory: "128Gi"
        # nvidia.com/gpu 的 requests 不需要显式设置
```

### 3.2 为什么 GPU requests 必须等于 limits

```
CPU/Memory：requests < limits 允许超卖（overcommit）
  - 容器可以 burst 使用更多 CPU
  - 基于 cgroups 做软隔离

GPU：requests 必须等于 limits
  - GPU 是离散设备，不像 CPU 可以分时复用
  - 分配 2 个 GPU 就是物理的 2 块卡，无法 burst 到 3 块
  - 除非使用 GPU 共享方案（MIG/MPS/Time-Slicing）
```

### 3.3 Topology Manager 与 GPU

```yaml
# kubelet 配置
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
topologyManagerPolicy: "best-effort"  # 或 "restricted" / "single-numa-node"
topologyManagerScope: "container"      # 或 "pod"
```

Topology Manager 确保 GPU 和 CPU/Memory 的 NUMA 亲和性：

```
NUMA Node 0: CPU 0-31, Memory 256GB, GPU 0-3 (PCIe)
NUMA Node 1: CPU 32-63, Memory 256GB, GPU 4-7 (PCIe)

当 Pod 请求 2 GPU + 16 CPU 时：
  best-effort: 尽量分配同一 NUMA 的 GPU 和 CPU
  restricted:  必须在同一 NUMA，否则拒绝调度
  single-numa-node: 所有资源必须来自单个 NUMA
```

## 4. 运维最佳实践

### 4.1 健康检查与故障检测

```yaml
# Device Plugin 检测到 GPU 故障时
# 1. 将设备标记为 Unhealthy
# 2. kubelet 更新节点 Allocatable
# 3. 使用该 GPU 的 Pod 不会被自动驱逐！（需要额外机制）

# 推荐：配合 GPU Health Check DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-health-monitor
spec:
  template:
    spec:
      containers:
      - name: monitor
        image: gpu-health-checker:v1
        env:
        - name: CHECK_INTERVAL
          value: "30"    # 每 30 秒检查
        - name: ECC_ERROR_THRESHOLD
          value: "10"    # ECC 错误超过阈值标记不健康
        - name: TEMP_THRESHOLD
          value: "90"    # 温度超过 90°C 告警
```

### 4.2 驱动升级策略

```yaml
# 方式一：Rolling Update（推荐）
# GPU Operator 支持滚动更新驱动
spec:
  driver:
    upgradePolicy:
      autoUpgrade: true
      maxParallelUpgrades: 1     # 一次只升级一个节点
      maxUnavailable: "25%"
      waitForCompletion:
        timeoutSeconds: 0
      podDeletion:
        force: false
        timeoutSeconds: 300
        deleteEmptyDir: false
      drain:
        enable: true
        force: true
        timeoutSeconds: 300
```

### 4.3 监控告警

```yaml
# Prometheus AlertRule 示例
groups:
- name: gpu-alerts
  rules:
  - alert: GPUMemoryAlmostFull
    expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE > 0.95
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU {{ $labels.gpu }} memory > 95%"

  - alert: GPUTemperatureCritical
    expr: DCGM_FI_DEV_GPU_TEMP > 85
    for: 2m
    labels:
      severity: critical

  - alert: GPUUnhealthy
    expr: DCGM_FI_DEV_XID_ERRORS > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GPU XID error detected on {{ $labels.node }}"
```

## 5. DRA (Dynamic Resource Allocation) — 未来方向

K8s 1.26+ 引入的 DRA 是 Device Plugin 的下一代替代：

```
Device Plugin 的根本限制：
  - 只能表达 "数量"（我要 2 个 GPU）
  - 无法表达 "属性"（我要 2 个有 NVLink 连接的 A100-80GB GPU）
  - 无法表达 "共享"（这个 GPU 可以被 3 个 Pod 共享）

DRA 的解决方案：
  - ResourceClaim：声明式的资源需求（类似 PVC）
  - ResourceClass：资源类型定义（类似 StorageClass）
  - 支持结构化参数（structured parameters）表达复杂需求
```

```yaml
# DRA 示例（alpha feature）
apiVersion: resource.k8s.io/v1alpha2
kind: ResourceClaim
metadata:
  name: gpu-claim
spec:
  resourceClassName: nvidia-gpu
  parametersRef:
    apiGroup: gpu.nvidia.com
    kind: GpuClaimParameters
    name: my-gpu-params
---
apiVersion: gpu.nvidia.com/v1alpha1
kind: GpuClaimParameters
metadata:
  name: my-gpu-params
spec:
  count: 4
  memory: "80Gi"
  interconnect: "nvlink"
  migProfile: ""
```

## 小结

```
Device Plugin：最小化接口，解决 "让 K8s 知道 GPU 存在" 的问题
GPU Operator：自动化运维，解决 "在 K8s 上完整管理 GPU 栈" 的问题
DRA：下一代方案，解决 "表达复杂 GPU 需求" 的问题

对于你的 8xH20 集群：
  ✅ 使用 GPU Operator 管理（省心）
  ✅ 配置 Topology Manager = restricted（保证 NUMA 亲和）
  ✅ 启用 DCGM Exporter（监控）
  ✅ 根据需求决定是否启用 MIG Manager
  ⏳ 关注 DRA 进展（K8s 1.30+ 可能 GA）
```
