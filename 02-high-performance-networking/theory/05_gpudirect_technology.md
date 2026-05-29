# GPUDirect 技术全解

## 概述

GPUDirect 是 NVIDIA 的一组技术，核心目标是**消除 GPU 数据传输中的多余拷贝**。在 AI 训练中，GPU 间的通信效率直接决定了扩展效率——每一次不必要的 CPU 中转和内存拷贝都意味着延迟增加和带宽浪费。对于 8×H20 集群，GPUDirect 技术是实现高效多节点训练的基础。

---

## GPUDirect Peer-to-Peer (P2P)

### 原理

GPUDirect P2P 允许同一 PCIe 总线域内的 GPU 直接通过 PCIe 交换机互传数据，完全绕过 CPU 和系统内存：

```
=== 无 GPUDirect P2P ===

GPU0 VRAM → PCIe → CPU → 系统内存 → CPU → PCIe → GPU1 VRAM
           拷贝1        拷贝2         拷贝3        拷贝4
总计: 4 次拷贝，CPU 全程参与

=== 有 GPUDirect P2P ===

GPU0 VRAM → PCIe Switch → GPU1 VRAM
            直接 DMA
总计: 1 次 DMA 传输，CPU 零参与
```

### 适用条件

- 两个 GPU 必须在同一 PCIe Root Complex 或通过 PCIe Switch 连接
- 需要 CUDA 驱动支持 (CUDA 4.0+)
- 跨 NUMA 节点的 GPU 对可能因跨 CPU 桥接而性能下降

### 在 H20 上的表现

H20 没有 NVLink，节点内 GPU 互连完全依赖 PCIe：

```
H20 节点典型 PCIe 拓扑:

PCIe Gen5 Root Complex 0 (CPU 0)
├── PCIe Switch A
│   ├── GPU0     ←→ GPU1 之间 P2P: ~24 GB/s (双向)
│   └── GPU1
├── PCIe Switch B
│   ├── GPU2
│   └── GPU3
└── NIC0 (mlx5_0)

PCIe Gen5 Root Complex 1 (CPU 1)
├── PCIe Switch C
│   ├── GPU4
│   └── GPU5
├── PCIe Switch D
│   ├── GPU6
│   └── GPU7
└── NIC1 (mlx5_1)

P2P 性能层级:
同 Switch: ~24 GB/s (最优)
同 Root Complex 不同 Switch: ~20 GB/s (经 Root Complex 转发)
跨 CPU (跨 NUMA): ~14-18 GB/s (经 UPI/QPI 桥接，显著下降)
```

### 验证与调试

```bash
# 检查 P2P 可达性
nvidia-smi topo -p2p r
# 输出矩阵: OK = 可达, NS = 不支持

# 检查连接类型
nvidia-smi topo -m
# 关注: PIX (同 Switch), PXB (同 Root), PHB (跨 NUMA), SYS (跨 Socket)

# CUDA 编程验证
# cudaDeviceCanAccessPeer(&canAccess, gpu0, gpu1)
# cudaDeviceEnablePeerAccess(targetGPU, 0)
```

---

## GPUDirect RDMA (GDR)

### 原理

GPUDirect RDMA 允许网卡 (HCA/NIC) 直接读写 GPU 显存，无需经过 CPU 和系统内存：

```
=== 无 GPUDirect RDMA ===

本机 GPU VRAM → PCIe → CPU → 系统内存 → PCIe → NIC → 网络
               拷贝1        拷贝2         拷贝3
→ 网络 → NIC → PCIe → CPU → 系统内存 → PCIe → 远程 GPU VRAM
         拷贝4        拷贝5         拷贝6
总计: 6 次拷贝，双侧 CPU 全程参与

=== 有 GPUDirect RDMA ===

本机 GPU VRAM → PCIe → NIC → 网络 → NIC → PCIe → 远程 GPU VRAM
               DMA读       RDMA传输      DMA写
总计: 1 次 DMA 读 + 网络传输 + 1 次 DMA 写，CPU 零参与
```

### 技术要求

| 组件 | 要求 |
|------|------|
| GPU | Kepler 架构+ (计算能力 3.5+)，H20 完全支持 |
| 网卡 | ConnectX-3 Pro+ (推荐 ConnectX-6/7) |
| 内核模块 | `nvidia-peermem` (替代旧的 `nv_peer_mem`) |
| 驱动版本 | NVIDIA Driver 470+, MLNX_OFED 5.0+ |
| PCIe 拓扑 | GPU 和 NIC 在同一 PCIe Root 下性能最佳 |

### 启用 GPUDirect RDMA

```bash
# 1. 确认 nvidia-peermem 模块已加载
lsmod | grep nvidia_peermem
# 如未加载:
modprobe nvidia-peermem

# 2. 验证 GDR 功能
# 使用 perftest 工具 (MLNX_OFED 自带)
# 服务端:
ib_write_bw -d mlx5_0 --use_cuda=0 --report_gbits
# 客户端:
ib_write_bw -d mlx5_0 --use_cuda=0 <server_ip> --report_gbits

# 3. 在 NCCL 中确认 GDR 已启用
export NCCL_DEBUG=INFO
# 日志中应出现: "NET/IB: Using GPUDirect RDMA"

# 4. 关键 NCCL 变量
export NCCL_NET_GDR_LEVEL=PHB    # PHB = 同 NUMA 域
export NCCL_NET_GDR_READ=1        # 启用 GDR Read (发送方向)
```

### GDR Read vs GDR Write

```
GDR Write (默认):
接收方 NIC 收到数据 → NIC 直接 DMA Write → 接收方 GPU VRAM
适用: 接收方向，NIC 和 GPU 在同一 PCIe 域

GDR Read:
发送方 NIC 需要数据 → NIC 直接 DMA Read → 发送方 GPU VRAM → NIC 发送
适用: 发送方向，减少发送延迟
需要: NCCL_NET_GDR_READ=1 显式启用
```

### PCIe 拓扑对 GDR 的影响

```
最佳情况 (同一 PCIe Switch):
GPU ←→ PCIe Switch ←→ NIC
      直接 DMA，延迟最低

次优情况 (同一 PCIe Root Complex):
GPU ← Root Complex → NIC
     经 Root Complex 转发，性能略降

较差情况 (跨 NUMA):
GPU ← CPU0 ← UPI → CPU1 → NIC
     跨 CPU 桥接，带宽/延迟显著恶化

8×H20 关键原则: 确保 GPU0-3 使用 NIC0 (同 NUMA 0)
                GPU4-7 使用 NIC1 (同 NUMA 1)
```

---

## GPUDirect Storage (GDS)

### 原理

GPUDirect Storage 允许 GPU 直接从 NVMe/NVMe-oF 存储读写数据，绕过 CPU 和系统内存的 Bounce Buffer：

```
=== 传统 I/O 路径 ===
NVMe SSD → PCIe → CPU → 系统内存 (Page Cache) → CPU → PCIe → GPU VRAM
           拷贝1        拷贝2                     拷贝3

=== GPUDirect Storage ===
NVMe SSD → PCIe → GPU VRAM
           直接 DMA (由 GPU 发起)
```

### 适用场景

- **Checkpoint 加载**：训练恢复时从 NVMe 直接加载模型权重到 GPU
- **数据加载**：大规模数据集直接读入 GPU (配合 DALI 等数据管线)
- **远程存储**：通过 NVMe-oF 直接读写远程 NVMe (需网络支持)

### 对 AI 训练的影响

| 操作 | 传统 I/O | GPUDirect Storage | 提升 |
|------|---------|-------------------|------|
| Checkpoint 加载 (10GB) | ~4.5s | ~1.5s | 3× |
| 数据预取 (批次) | CPU 内存受限 | GPU 直接消费 | 减少 CPU 瓶颈 |
| 训练恢复时间 | 分钟级 | 秒级 | 关键差异 |

```bash
# 启用 GPUDirect Storage
# 1. 安装 GDS 驱动
apt install nvidia-gds

# 2. 验证
/usr/local/cuda/gds/tools/gdscheck -p

# 3. 使用 cuFile API 或兼容库 (如 KvikIO)
```

---

## 三种 GPUDirect 技术总览

```
                    GPU0 VRAM      GPU1 VRAM (本机)    GPU2 VRAM (远程)
                       |               |                    |
GPUDirect P2P:    GPU0 ←──PCIe──→ GPU1                     |
                  节点内 GPU 直连                            |
                                                            |
GPUDirect RDMA:   GPU0 ←──PCIe──→ NIC ←═══网络═══→ NIC ──→ GPU2
                  GPU 显存直达网络                           |
                                                            |
GPUDirect Storage: GPU0 ←──PCIe──→ NVMe                    |
                  GPU 显存直达存储                           |
```

### 性能量化对比

以 8×H20 集群典型操作为例：

| 操作 | 无 GPUDirect | 有 GPUDirect | 加速比 | 瓶颈消除 |
|------|-------------|-------------|--------|---------|
| 节点内 AllReduce (8 GPU) | ~85 GB/s | ~180 GB/s | 2.1× | CPU 拷贝 |
| 节点间 AllReduce (2 节点) | ~15 GB/s | ~42 GB/s | 2.8× | CPU + 内存拷贝 |
| Checkpoint 保存 (20GB) | ~8s | ~3s | 2.7× | CPU Bounce Buffer |
| 训练 Step 通信开销 | ~35% | ~12% | — | 整体效率提升 |

注：数值为典型参考值，实际性能取决于具体硬件配置和工作负载。

---

## 部署检查清单

```bash
# === GPUDirect P2P ===
# 1. 检查 P2P 支持
nvidia-smi topo -p2p r
# 2. 检查 IOMMU 设置 (IOMMU 可能阻止 P2P)
dmesg | grep -i iommu
# 如果 IOMMU 启用，需在 BIOS 或内核参数中禁用:
# intel_iommu=off 或 amd_iommu=off
# 或在 BIOS 中设置 ACS (Access Control Services) 为 disabled

# === GPUDirect RDMA ===
# 1. 确认 nvidia-peermem 模块
lsmod | grep nvidia_peermem
modprobe nvidia-peermem  # 如未加载

# 2. 确认 GPU/NIC NUMA 亲和性
nvidia-smi topo -m  # 检查 NIC 与 GPU 的 PCIe 关系

# 3. 带宽测试
ib_write_bw --use_cuda=0 -d mlx5_0 -s 65536 --report_gbits  # 服务端
ib_write_bw --use_cuda=0 -d mlx5_0 -s 65536 <server_ip> --report_gbits  # 客户端

# === GPUDirect Storage ===
# 1. 检查 GDS 兼容性
/usr/local/cuda/gds/tools/gdscheck -p
# 2. 确认 NVMe 在支持列表中
# 3. 确认文件系统支持 (ext4/xfs)
```

---

## 小结

GPUDirect 三项技术分别解决了 GPU 通信中三个层面的数据搬运效率问题：P2P 解决节点内 GPU 互连、RDMA 解决跨节点 GPU 通信、Storage 解决 GPU 与存储交互。对 8×H20 集群而言，GPUDirect RDMA 是影响最大的技术——它直接决定了多节点训练的扩展效率。部署时务必确认 nvidia-peermem 加载、GPU-NIC NUMA 亲和性正确、IOMMU 配置合理。
