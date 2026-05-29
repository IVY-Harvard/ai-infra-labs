# Kubernetes 容器高性能网络

## 概述

在 Kubernetes 上运行 GPU 训练任务时，网络是最容易成为瓶颈的环节。默认的容器网络栈 (veth + bridge/overlay) 引入了显著的额外延迟和 CPU 开销，完全无法满足 RDMA 通信需求。本章介绍在 K8s 中实现高性能网络的三种主流方案，以及 RDMA 设备暴露和多网卡管理的完整实践。

---

## 三种高性能网络方案对比

### 总览

| 方案 | 原理 | 性能 | 隔离性 | 复杂度 |
|------|------|------|--------|--------|
| Host Network | Pod 直接使用宿主机网络命名空间 | 原生性能 | 无隔离 | 最低 |
| Macvlan | 在物理网卡上创建虚拟 MAC 子接口 | 接近原生 | L2 隔离 | 低 |
| SR-IOV | 硬件虚拟化，物理网卡拆分为多个 VF | 原生性能 | 硬件隔离 | 高 |

### Host Network

```yaml
# Pod 使用 hostNetwork
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  hostNetwork: true        # 直接使用宿主机网络
  dnsPolicy: ClusterFirstWithHostNet
  containers:
  - name: trainer
    image: nvcr.io/nvidia/pytorch:24.01-py3
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]  # RDMA 需要锁定内存
    resources:
      limits:
        nvidia.com/gpu: 8
```

优点：
- 零额外开销，Pod 内网络性能 = 宿主机性能
- 配置最简单，RDMA 设备天然可见
- 无需额外 CNI 插件

缺点：
- **零网络隔离**：Pod 共享宿主机所有网络接口和端口
- **端口冲突**：多 Pod 不能绑定相同端口
- **安全风险**：Pod 可访问宿主机所有网络资源
- **实际场景**：适合单租户、每节点单训练任务的场景

### Macvlan

```yaml
# Macvlan NetworkAttachmentDefinition
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: macvlan-rdma-net
spec:
  config: '{
    "cniVersion": "0.3.1",
    "type": "macvlan",
    "master": "ens1f0",
    "mode": "bridge",
    "ipam": {
      "type": "host-local",
      "subnet": "10.0.1.0/24",
      "rangeStart": "10.0.1.100",
      "rangeEnd": "10.0.1.200"
    }
  }'
```

工作原理：
```
物理网卡 (ens1f0)
├── macvlan0 → Pod A (MAC: aa:bb:cc:00:01, IP: 10.0.1.100)
├── macvlan1 → Pod B (MAC: aa:bb:cc:00:02, IP: 10.0.1.101)
└── 宿主机本身 (原始 MAC/IP)

每个 macvlan 子接口有独立 MAC 地址
交换机看到多个 MAC，分别转发
```

优点：
- 接近原生性能 (仅多一层 MAC 地址映射)
- L2 级别隔离，Pod 间不能直接通信 (bridge 模式除外)
- 配置相对简单

缺点：
- **RDMA 支持有限**：Macvlan 不天然传递 RDMA 设备
- **同主机通信受限**：同一宿主机上的 macvlan Pod 间通信受限
- **交换机 MAC 表压力**：大量 macvlan 接口增加 MAC 表条目

### SR-IOV (Single Root I/O Virtualization)

SR-IOV 在硬件层面将一个物理网卡 (PF) 拆分为多个虚拟功能 (VF)：

```
物理网卡 (PF: mlx5_0)
├── VF0 → Pod A (独立 PCIe 设备，独立 MAC/VLAN)
├── VF1 → Pod B (独立 PCIe 设备，独立 MAC/VLAN)
├── VF2 → Pod C
└── ...最多 128 个 VF (取决于硬件)

每个 VF 是独立的 PCIe 设备:
- 独立出现在 /sys/bus/pci/devices/
- 独立的中断和 DMA 引擎
- 硬件级别隔离 (不共享队列/缓冲区)
```

配置步骤：

```bash
# 1. 在宿主机上创建 VF
echo 4 > /sys/class/net/ens1f0/device/sriov_numvfs

# 2. 验证 VF 创建成功
lspci | grep -i mellanox
# 应看到 PF 和多个 VF

# 3. 检查 VF 的 RDMA 设备
ls /sys/class/infiniband/
# 每个 VF 应有独立的 IB 设备节点
```

K8s 中使用 SR-IOV：

```yaml
# SR-IOV NetworkAttachmentDefinition
apiVersion: k8s.cni.cncf.io/v1
kind: NetworkAttachmentDefinition
metadata:
  name: sriov-rdma-net
  annotations:
    k8s.v1.cni.cncf.io/resourceName: nvidia.com/sriov_rdma_vf
spec:
  config: '{
    "cniVersion": "0.3.1",
    "type": "sriov",
    "vlan": 100,
    "ipam": {
      "type": "host-local",
      "subnet": "10.0.2.0/24"
    }
  }'
```

优点：
- **原生 PCIe 性能**：VF 直通到 Pod，无虚拟化开销
- **完整 RDMA 支持**：每个 VF 有独立 RDMA 设备
- **硬件隔离**：VF 间在硬件层面隔离
- **适合多租户**：不同训练任务安全共享物理网卡

缺点：
- **配置复杂**：需要 SR-IOV Device Plugin + SR-IOV CNI + Network Operator
- **VF 数量有限**：单个 PF 最多 128 VF
- **固件/驱动依赖**：需要特定固件版本支持

---

## Multus CNI：多网卡管理

### 为什么需要多网卡

AI 训练 Pod 通常需要多个网络：
- **管理网络**：K8s Service/API 通信 (默认 CNI，如 Calico/Flannel)
- **数据网络**：GPU 间 RDMA 通信 (高性能 CNI)
- **存储网络**：访问分布式存储 (可能独立网络)

### Multus 架构

```
Multus CNI (元 CNI，管理多个 CNI 插件)
├── 默认网络: Calico/Flannel (eth0) → K8s Service 网络
├── 附加网络 1: SR-IOV (net1) → RDMA 数据平面
└── 附加网络 2: Macvlan (net2) → 存储网络

Pod 网络命名空间:
├── eth0    → 10.244.1.5   (Calico，K8s 管理网络)
├── net1    → 10.0.2.10    (SR-IOV VF，RDMA 数据网络)
└── net2    → 10.0.3.10    (Macvlan，存储网络)
```

### 部署 Multus

```bash
# 安装 Multus (以 thick plugin 模式)
kubectl apply -f https://raw.githubusercontent.com/k8snetworkplumbingwg/multus-cni/master/deployments/multus-daemonset-thick.yml

# 验证安装
kubectl get pods -n kube-system | grep multus
```

### 使用 Multus 的 Pod 示例

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
  annotations:
    k8s.v1.cni.cncf.io/networks: sriov-rdma-net  # 附加 RDMA 网络
spec:
  containers:
  - name: trainer
    image: nvcr.io/nvidia/pytorch:24.01-py3
    securityContext:
      capabilities:
        add: ["IPC_LOCK", "NET_RAW"]
    resources:
      requests:
        nvidia.com/gpu: 8
        nvidia.com/sriov_rdma_vf: 1    # 请求 1 个 SR-IOV VF
      limits:
        nvidia.com/gpu: 8
        nvidia.com/sriov_rdma_vf: 1
    env:
    - name: NCCL_SOCKET_IFNAME
      value: "eth0"              # NCCL bootstrap 用管理网络
    - name: NCCL_IB_HCA
      value: "mlx5_2"            # NCCL 数据通信用 SR-IOV VF 的 RDMA 设备
```

---

## RDMA Device Plugin

### K8s 中暴露 RDMA 设备

默认情况下，K8s Pod 无法看到宿主机的 RDMA 设备。RDMA Device Plugin 解决此问题：

```
NVIDIA RDMA Shared Device Plugin 架构:

DaemonSet (每个节点运行)
├── 扫描 /sys/class/infiniband/ 发现 RDMA 设备
├── 注册为 K8s Extended Resource
│   ├── rdma/hca_shared_devices_a: 64   (共享模式，多 Pod 共用)
│   └── 或 nvidia.com/sriov_rdma_vf: 4  (SR-IOV VF 模式，独占)
└── 设备分配时注入 /dev/infiniband/* 到 Pod
```

### 两种 RDMA 暴露模式

**共享模式 (Shared)**：
```yaml
# 多个 Pod 共享同一 RDMA 设备
# ConfigMap 配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: rdma-devices
  namespace: kube-system
data:
  config.json: |
    {
      "periodicUpdateInterval": 300,
      "configList": [
        {
          "resourceName": "hca_shared_devices_a",
          "rdmaHcaMax": 64,
          "devices": ["ens1f0"]
        }
      ]
    }

# Pod 使用
resources:
  limits:
    rdma/hca_shared_devices_a: 1
```

**SR-IOV 独占模式**：
```yaml
# 每个 Pod 获得独立 VF 和 RDMA 设备
# 通过 SR-IOV Network Operator 自动管理
resources:
  limits:
    nvidia.com/sriov_rdma_vf: 1
```

### NVIDIA Network Operator

NVIDIA Network Operator 是一站式解决方案，自动管理：
- MLNX_OFED 驱动安装 (通过 DaemonSet)
- SR-IOV Device Plugin 部署
- RDMA Shared Device Plugin 部署
- Multus CNI 配置
- Secondary 网络 (Macvlan/SR-IOV) 创建

```bash
# 使用 Helm 安装 Network Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install network-operator nvidia/network-operator \
  --namespace nvidia-network-operator \
  --create-namespace \
  --set deployCR=true \
  --set ofedDriver.deploy=true \
  --set rdmaSharedDevicePlugin.deploy=true \
  --set sriovDevicePlugin.deploy=true \
  --set secondaryNetwork.deploy=true
```

---

## 完整部署实践：IB/RoCE in K8s

### 从驱动到 Pod 调度的完整流程

```
第 1 步: 硬件准备
├── BIOS 启用 SR-IOV (如需)
├── BIOS 启用 Above 4G Decoding
└── 确认 IOMMU 设置 (P2P 可能需要关闭)

第 2 步: 驱动安装 (通过 Network Operator 或手动)
├── MLNX_OFED 驱动
├── nvidia-peermem 模块 (GPUDirect RDMA)
└── SR-IOV VF 创建 (如需)

第 3 步: K8s 网络组件
├── Multus CNI (管理多网络)
├── RDMA Device Plugin (暴露 RDMA 设备)
├── SR-IOV CNI + Device Plugin (如使用 SR-IOV)
└── NetworkAttachmentDefinition (定义附加网络)

第 4 步: Pod 调度与资源请求
├── GPU 资源: nvidia.com/gpu
├── RDMA 资源: rdma/hca_shared_devices_a
├── 亲和性: GPU 和 RDMA 设备在同一 NUMA 域
└── SecurityContext: IPC_LOCK capability

第 5 步: 应用层配置
├── NCCL 环境变量
├── torch.distributed 初始化
└── 通信性能验证
```

### 节点亲和性与 NUMA 感知调度

```yaml
# 确保 Pod 调度到有合适 RDMA 设备的节点
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training
spec:
  nodeSelector:
    nvidia.com/gpu.product: "H20"
    network.nvidia.com/ib-capable: "true"
  
  # Topology Manager 配合 (kubelet 配置)
  # --topology-manager-policy=single-numa-node
  # 确保 GPU + RDMA 设备分配在同一 NUMA 域
  
  containers:
  - name: trainer
    resources:
      limits:
        nvidia.com/gpu: 8
        rdma/hca_shared_devices_a: 1
```

Kubelet Topology Manager 配置：
```bash
# /var/lib/kubelet/config.yaml
topologyManagerPolicy: "single-numa-node"  # 或 "best-effort"
topologyManagerScope: "pod"                 # pod 级别 NUMA 对齐
```

### 常见问题排查

| 问题 | 症状 | 排查方法 |
|------|------|---------|
| RDMA 设备不可见 | Pod 内 `ibv_devinfo` 无输出 | 检查 Device Plugin 日志、securityContext |
| GPUDirect RDMA 不工作 | NCCL 回退到 Socket 传输 | 检查 nvidia-peermem、NCCL_DEBUG 日志 |
| SR-IOV VF 分配失败 | Pod Pending, Insufficient resources | 检查 sriov_numvfs、Device Plugin 资源计数 |
| NCCL 超时 | 训练卡住不动 | 检查 NCCL_SOCKET_IFNAME、防火墙、IB 端口状态 |
| 性能低于宿主机 | 带宽/延迟明显差于 bare metal | 检查 NUMA 亲和性、Topology Manager 配置 |

### Pod 内验证步骤

```bash
# 在 Pod 内执行
# 1. 检查网络接口
ip addr show
# 确认有 RDMA 网络接口 (net1 等)

# 2. 检查 RDMA 设备
ibv_devinfo
# 确认设备状态 Active，速率正确

# 3. RDMA 连通性测试 (两个 Pod 间)
# Pod A (服务端):
ib_write_bw -d mlx5_0 --report_gbits
# Pod B (客户端):
ib_write_bw -d mlx5_0 <pod_a_ip> --report_gbits

# 4. GPUDirect RDMA 测试
ib_write_bw -d mlx5_0 --use_cuda=0 --report_gbits  # 服务端
ib_write_bw -d mlx5_0 --use_cuda=0 <pod_a_ip> --report_gbits  # 客户端

# 5. NCCL 测试
# 使用 nccl-tests 的 all_reduce_perf
```

---

## 小结

在 K8s 上实现高性能 GPU 训练网络，核心是绕过默认容器网络栈的开销。三种方案各有取舍：Host Network 最简单但无隔离；Macvlan 提供基本隔离但 RDMA 支持有限；SR-IOV 提供硬件级隔离和完整 RDMA 支持但配置复杂。生产环境推荐 Multus CNI + SR-IOV (或 RDMA Shared Device Plugin) + NVIDIA Network Operator 的组合，配合 Topology Manager 实现 NUMA 感知调度。关键是确保 GPU、RDMA 设备、CPU 三者的 NUMA 亲和性——这往往是容器化部署中最容易忽视却影响最大的因素。
