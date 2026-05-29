# 01 — RDMA 技术全解

## 1. 为什么 AI 训练需要 RDMA

### 1.1 传统 TCP/IP 网络的瓶颈

在分布式训练中，8 张 H20 GPU 的单节点扩展到多节点时，网络成为首要瓶颈：

```
传统 TCP/IP 数据路径：
App Buffer → Socket Buffer → TCP/IP Stack → NIC Driver → NIC TX
   ↓            ↓              ↓              ↓
 用户态       内核拷贝      协议处理(CPU)    中断处理
              (copy 1)      (copy 2)        (context switch)

关键瓶颈：
1. 多次内存拷贝：用户空间 → 内核空间 → 网卡 DMA 缓冲区
2. CPU 开销：协议栈处理、中断响应、上下文切换
3. 延迟：每个 packet 都经过完整内核路径 (~10-50μs)
4. 带宽：CPU 处理能力限制了可用带宽
```

**量化对比**：在 200Gbps 网络上，TCP/IP 协议栈可能消耗 4-8 个 CPU 核心，且实际带宽只能达到 60-70%。

### 1.2 分布式训练对网络的要求

```
场景分析 — 8xH20 多节点训练：
┌─────────────────────────────────────────────────────┐
│ Data Parallel (AllReduce)                           │
│ • 每步梯度同步：模型参数大小 × 2（reduce + broadcast）│
│ • LLaMA-70B: ~140GB 梯度 / step                    │
│ • 要求：高带宽 + 低延迟                              │
├─────────────────────────────────────────────────────┤
│ Tensor Parallel (AllReduce per layer)               │
│ • 每层前向/反向都需同步：极高频率                     │
│ • 延迟敏感：5μs vs 50μs 直接影响吞吐               │
│ • 要求：超低延迟                                     │
├─────────────────────────────────────────────────────┤
│ Pipeline Parallel (P2P)                             │
│ • microbatch 间点对点传输                            │
│ • 气泡时间与通信延迟直接相关                          │
│ • 要求：稳定低延迟                                   │
└─────────────────────────────────────────────────────┘
```

## 2. RDMA 核心原理

### 2.1 三大关键特性

#### Kernel Bypass（内核旁路）

```
RDMA 数据路径：
App Buffer ──────→ NIC (RNIC/HCA)  ──→ 网络
   ↑                    ↑
 用户态直接操作       硬件处理协议
 无需系统调用         零 CPU 参与

实现机制：
1. 应用通过 mmap 获得 NIC 的 doorbell 寄存器映射
2. 提交 Work Request (WR) 到 Send Queue / Receive Queue
3. NIC 硬件直接 DMA 读取 WR 和数据
4. 完成后写入 Completion Queue (CQ)
5. 全程不经过内核 → 延迟降到 1-2μs
```

#### Zero-Copy（零拷贝）

```
数据在内存中的路径对比：

TCP/IP (3 次拷贝):
  App Buffer → [copy1] → Socket Buffer → [copy2] → SKB → [copy3] → NIC DMA

RDMA (0 次拷贝):
  App Buffer ─── [NIC DMA 直接读取] ──→ 网络
  ↑
  内存已注册(pinned)，NIC 知道物理地址
  
远端接收 (RDMA Write):
  网络 ──→ NIC DMA ──→ 远端 App Buffer (无需远端 CPU 参与!)
```

#### Memory Registration（内存注册）

```
内存注册流程：
┌──────────────────────────────────────────────────────────┐
│ 1. ibv_reg_mr(pd, addr, length, access_flags)            │
│    ↓                                                     │
│ 2. 内核 pin 住物理页面（不允许 swap out）                  │
│    ↓                                                     │
│ 3. 创建虚拟地址→物理地址的转换表                           │
│    ↓                                                     │
│ 4. 将转换表下发到 NIC 硬件                                │
│    ↓                                                     │
│ 5. 返回 Memory Region (MR)，包含：                       │
│    - lkey: 本地访问密钥                                   │
│    - rkey: 远端访问密钥（用于 RDMA Read/Write）           │
└──────────────────────────────────────────────────────────┘

注意事项：
- 注册大块内存（GB 级别）时有显著延迟
- 注册的内存不能被 OS 回收 → 需要合理规划
- NCCL 使用 cuMemAlloc → 天然 pinned → 注册开销较小
```

### 2.2 RDMA 操作模型

```
Queue Pair (QP) 通信模型：

    Node A                              Node B
┌──────────────┐                 ┌──────────────┐
│  Application │                 │  Application │
│      ↕       │                 │      ↕       │
│  ┌────────┐  │                 │  ┌────────┐  │
│  │   QP   │  │                 │  │   QP   │  │
│  │ ┌────┐ │  │    Network      │  │ ┌────┐ │  │
│  │ │ SQ │─┼──┼─────────────────┼──┼→│ RQ │ │  │
│  │ └────┘ │  │                 │  │ └────┘ │  │
│  │ ┌────┐ │  │                 │  │ ┌────┐ │  │
│  │ │ RQ │←┼──┼─────────────────┼──┼─│ SQ │ │  │
│  │ └────┘ │  │                 │  │ └────┘ │  │
│  └────────┘  │                 │  └────────┘  │
│  ┌────────┐  │                 │  ┌────────┐  │
│  │   CQ   │  │                 │  │   CQ   │  │
│  └────────┘  │                 │  └────────┘  │
└──────────────┘                 └──────────────┘

四种 RDMA 操作：
┌──────────────┬───────────────────┬──────────────────────┐
│ 操作         │ 描述               │ AI 训练中的用途        │
├──────────────┼───────────────────┼──────────────────────┤
│ Send/Recv    │ 双端参与            │ 控制消息、小数据交换   │
│ RDMA Write   │ 写入远端内存        │ AllReduce 梯度写入    │
│ RDMA Read    │ 读取远端内存        │ 参数拉取              │
│ Atomic       │ 远端原子操作        │ 分布式锁/计数器       │
└──────────────┴───────────────────┴──────────────────────┘

RDMA Write 是 AI 训练的核心操作：
- 单端操作：远端 CPU 无需任何参与
- 最低延迟：无远端软件开销
- NCCL 的 AllReduce 底层就是 RDMA Write
```

### 2.3 传输模式

```
RC (Reliable Connected) — AI 训练首选
├── 可靠传输：硬件保证有序、不丢失
├── 面向连接：QP 一对一绑定
├── 支持所有 RDMA 操作
└── 缺点：N 节点需要 N×N 个 QP

UC (Unreliable Connected)
├── 不保证可靠：无 ACK/重传
├── 面向连接
└── 用途：对丢包不敏感的场景

UD (Unreliable Datagram)
├── 不可靠 + 无连接
├── 一个 QP 可与多个远端通信
├── MTU 限制：单个消息 ≤ 4KB
└── 用途：服务发现、小消息广播

DC (Dynamic Connected) — Mellanox 扩展
├── 可靠 + 动态连接
├── 解决 RC 的 QP 爆炸问题
├── 按需建立连接
└── 用途：大规模集群（1000+ 节点）
```

## 3. 三种 RDMA 实现

### 3.1 InfiniBand (IB)

```
特点：
- 专用网络架构：HCA + IB Switch + Subnet Manager
- 原生无损：基于 credit-based 流控
- 最低延迟：端到端 < 1μs (NDR)
- 最高带宽：NDR 400Gbps, XDR 800Gbps

AI 集群定位：
- 旗舰选择：NVIDIA DGX SuperPOD 标配
- 大规模训练：1000+ GPU 首选
- 代价：专用设备成本高，运维需要 IB 专业知识
```

### 3.2 RoCE (RDMA over Converged Ethernet)

```
特点：
- 在以太网上实现 RDMA
- RoCE v1: L2 only (同子网)
- RoCE v2: UDP/IP 封装 → 可路由 ← 主流选择
- 需要 PFC + ECN 配置实现"伪无损"

AI 集群定位：
- 性价比选择：复用现有以太网基础设施
- 中等规模：64-512 GPU 常见
- 代价：需要精细的交换机流控配置
- H20 GPU 集群的常见选择（中国市场）
```

### 3.3 iWARP (Internet Wide Area RDMA Protocol)

```
特点：
- 基于 TCP 的 RDMA
- 天然可路由，兼容现有网络
- 无需 PFC/ECN 配置
- 延迟较高：TCP 协议栈开销

AI 集群定位：
- 极少用于 GPU 训练
- 适合存储网络（NVMe-oF over iWARP）
- 代表厂商：Chelsio
```

### 3.4 三种实现对比

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 特性         │ InfiniBand   │ RoCE v2      │ iWARP        │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 传输层       │ IB 原生      │ UDP/IP       │ TCP/IP       │
│ 延迟         │ < 1μs        │ 2-5μs        │ 10-30μs      │
│ 带宽         │ 400/800Gbps  │ 100/400Gbps  │ 25/100Gbps   │
│ 无损保证     │ 原生(credit) │ PFC+ECN      │ TCP 重传     │
│ 路由能力     │ IB 路由      │ IP 路由      │ IP 路由      │
│ 交换机       │ IB 专用      │ 以太网       │ 以太网       │
│ 拥塞控制     │ 硬件内置     │ DCQCN        │ TCP CC       │
│ GPU训练适用性│ ★★★★★       │ ★★★★         │ ★★           │
│ 成本         │ 高           │ 中           │ 低           │
│ 运维难度     │ 需IB专家     │ 需网络调优   │ 简单         │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## 4. RDMA 软件栈

### 4.1 Verbs API 层次

```
应用层视角（以 NCCL 为例）：

┌─────────────────────────────────────────────────────────────┐
│                    NCCL / MPI / GLOO                        │
├─────────────────────────────────────────────────────────────┤
│                    libibverbs (Verbs API)                    │
│  ibv_open_device()  ibv_alloc_pd()  ibv_reg_mr()           │
│  ibv_create_qp()   ibv_post_send()  ibv_poll_cq()          │
├─────────────────────────────────────────────────────────────┤
│                    rdma-core / libibverbs providers          │
│  mlx5 provider  │  hfi1 provider  │  rxe provider (soft)   │
├─────────────────────────────────────────────────────────────┤
│                    Kernel RDMA Subsystem                     │
│  ib_core  │  ib_uverbs  │  rdma_cm  │  ib_umad             │
├─────────────────────────────────────────────────────────────┤
│                    Hardware Driver                           │
│  mlx5_ib (ConnectX)  │  hfi1 (OPA)  │  rdma_rxe (soft)    │
├─────────────────────────────────────────────────────────────┤
│                    Hardware (HCA/RNIC)                       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键数据结构

```
RDMA 编程核心对象：

Context (ibv_context)
  └── 表示一个 RDMA 设备的打开句柄

Protection Domain (ibv_pd)
  └── 安全域：限制 MR/QP/AH 的可见范围

Memory Region (ibv_mr)
  ├── addr, length: 注册的内存范围
  ├── lkey: 本地访问密钥（post_send 时使用）
  └── rkey: 远端访问密钥（RDMA Write/Read 时告知对端）

Queue Pair (ibv_qp)
  ├── Send Queue (SQ): 提交发送请求
  ├── Receive Queue (RQ): 提交接收缓冲区
  ├── qp_num: QP 编号（连接标识）
  └── state: RESET → INIT → RTR → RTS

Completion Queue (ibv_cq)
  ├── 发送/接收完成通知
  ├── ibv_poll_cq(): 轮询模式（低延迟，NCCL 使用）
  └── ibv_req_notify_cq(): 事件模式（省 CPU）

Address Handle (ibv_ah)
  └── UD 模式下的目标地址封装
```

## 5. RDMA 性能指标

### 5.1 关键指标定义

```
┌──────────────┬─────────────────────────────────────────────────────┐
│ 指标         │ 含义与基准                                          │
├──────────────┼─────────────────────────────────────────────────────┤
│ Latency      │ 单次操作延迟。IB NDR: 0.6μs, RoCE: 2-3μs          │
│ Bandwidth    │ 持续大消息吞吐。NDR: ~48GB/s, HDR: ~24GB/s         │
│ Message Rate │ 小消息/秒。ConnectX-7: 200M msg/s                  │
│ IOPS         │ 每秒 RDMA 操作数（类似 Message Rate）               │
│ 99th %ile    │ 尾延迟：P99 应 < 2x 平均延迟                       │
│ Jitter       │ 延迟抖动：训练稳定性的关键指标                      │
└──────────────┴─────────────────────────────────────────────────────┘
```

### 5.2 性能测试基准工具

```bash
# ib_write_bw — 带宽测试
# Server:
ib_write_bw -d mlx5_0 -F --report_gbits
# Client:
ib_write_bw -d mlx5_0 -F --report_gbits <server_ip>

# ib_write_lat — 延迟测试
ib_write_lat -d mlx5_0 -F
ib_write_lat -d mlx5_0 -F <server_ip>

# 预期结果 (NDR 400Gbps):
#   带宽: ~49 GB/s (393 Gbps)
#   延迟: ~0.6 μs (RDMA Write)
```

## 6. RDMA 与 AI 训练的结合

### 6.1 NCCL 如何使用 RDMA

```
NCCL AllReduce 的 RDMA 路径：

1. 初始化阶段：
   ncclCommInitRank()
   → 探测 IB/RoCE 设备 (ibv_get_device_list)
   → 每对 GPU 间创建 QP (ibv_create_qp, RC 模式)
   → 注册 GPU 显存为 MR (需要 GPUDirect RDMA 或 staging buffer)

2. 通信阶段 (以 Ring AllReduce 为例)：
   GPU0 ──RDMA Write──→ GPU1 ──RDMA Write──→ GPU2 → ...
   
   每步操作：
   a. GPU 计算出梯度 → 存在显存 buffer
   b. NCCL 提交 ibv_post_send (RDMA Write)
   c. NIC DMA 读取 GPU 显存 → 发送到网络
   d. 远端 NIC 收到 → DMA 写入远端 GPU 显存
   e. 远端 GPU 执行 reduce 操作
   f. 完成后进入下一环

3. 关键优化：
   - 流水线：通信与计算重叠 (overlap)
   - 分块传输：大 tensor 切分为 chunk，提高并行度
   - 多 QP：利用多条 IB 通道
```

### 6.2 通信量计算

```
AllReduce 通信量估算：

假设: 模型参数 P, 节点数 N, 每节点 GPU 数 G

Ring AllReduce (节点间):
  每节点发送: 2 × P × sizeof(dtype) × (N-1)/N
  
  示例 — LLaMA-70B, FP16, 8 节点:
  通信量 = 2 × 70B × 2 bytes × 7/8 = 245 GB / step
  
  如果 step 时间目标 = 1s:
  需要带宽 ≥ 245 GB/s → 约 5 条 NDR 400Gbps
  
  实际: 8 卡 H20 节点通常配 8 条 IB/RoCE → 足够
```

## 7. 常见问题与排查

```
Q: ibv_devices 看不到设备？
A: 检查: lsmod | grep mlx5_ib, modprobe mlx5_ib
   检查: dmesg | grep mlx5

Q: 注册大内存（>1GB）非常慢？
A: 使用 ODP (On-Demand Paging): ibv_reg_mr 时指定 IBV_ACCESS_ON_DEMAND
   或使用 huge pages: echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

Q: RDMA 带宽只有理论值的一半？
A: 检查 MTU: ibv_devinfo | grep active_mtu (应为 4096)
   检查 PCIe: lspci -vvv | grep LnkSta (应为 x16 Gen4/Gen5)
   检查对端: 两端 NIC 速率必须一致

Q: QP 数量过多导致内存不足？
A: 大规模集群使用 DC (Dynamic Connected) 传输模式
   或使用 Shared Receive Queue (SRQ) 减少 RQ 内存
```

## 8. 本章小结

```
核心要点：
1. RDMA 通过 kernel bypass + zero-copy 实现 μs 级延迟
2. IB 是 AI 训练的旗舰选择，RoCE 是性价比选择
3. Memory Registration 是 RDMA 编程的基础，理解 MR/QP/CQ 的关系
4. NCCL 封装了 RDMA 操作，AllReduce 底层使用 RDMA Write (RC 模式)
5. 8xH20 节点通常配 8 条 IB/RoCE，满足大模型训练带宽需求

下一步：深入 InfiniBand 架构细节 → theory/02
```
