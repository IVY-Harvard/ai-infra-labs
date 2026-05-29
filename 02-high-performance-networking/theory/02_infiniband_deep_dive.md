# InfiniBand 深度解析

## 概述

InfiniBand (IB) 是专为高性能计算和数据中心设计的网络互连技术。与以太网的"尽力而为"设计哲学不同，IB 从第一天起就为零丢包、低延迟、高吞吐而生。对于 8×H20 GPU 集群，理解 IB 架构是优化训练通信的基础。

---

## 核心架构组件

### Host Channel Adapter (HCA)

HCA 是 InfiniBand 的网络适配器，等价于以太网中的 NIC，但功能远超传统网卡：

| 特性 | HCA | 传统 NIC |
|------|-----|----------|
| 传输卸载 | 完整协议栈硬件实现 | 仅校验和/分段卸载 |
| 内存管理 | 硬件 MR (Memory Region) 注册 | 依赖内核页表 |
| 多租户 | 硬件级 QP 隔离 | 软件队列划分 |
| 延迟 | ~1μs (端到端) | ~10-50μs |

HCA 关键能力：
- **硬件协议卸载**：RDMA 操作完全在 HCA 硬件中完成，CPU 零参与
- **Memory Region 管理**：通过 `ibv_reg_mr()` 注册的内存区域由 HCA 直接 DMA 访问
- **多 Port 支持**：ConnectX-7 支持双端口，每端口独立 NDR 400Gbps

```bash
# 查看 HCA 信息
ibstat
# 输出示例：
# CA 'mlx5_0'
#   CA type: MT4129 (ConnectX-7)
#   Number of ports: 2
#   Port 1:
#     State: Active
#     Physical state: LinkUp
#     Rate: 400 (NDR)
```

### InfiniBand Switch

IB 交换机与以太网交换机的根本区别：

- **Cut-Through 转发**：不等完整帧到达即开始转发，延迟 ~90ns
- **Credit-Based Flow Control**：逐跳流控，物理上保证零丢包
- **VL (Virtual Lane) 仲裁**：硬件级 QoS，最多 16 个虚拟通道
- **子网管理接口**：内置 SMA (Subnet Management Agent)

典型交换机规格 (Quantum-2 NDR)：
- 64 端口 × 400Gbps = 25.6Tbps 总带宽
- 端口到端口延迟：~90ns
- 支持自适应路由 (Adaptive Routing)

### Subnet Manager (SM)

SM 是 IB 网络的"大脑"，负责：

1. **拓扑发现**：通过 SMP (Subnet Management Packet) 遍历所有节点和交换机
2. **LID 分配**：为每个端口分配本地标识符 (Local Identifier, 16-bit)
3. **路由计算**：生成线性转发表 (LFT) 或自适应路由表
4. **路径建立**：响应 PathRecord 查询，返回完整路径信息
5. **故障处理**：检测链路故障并重新计算路由

```bash
# 查看 SM 状态
sminfo
# 查看子网拓扑
ibnetdiscover
# 查看路由表
ibroute <switch_lid>
```

SM 运行模式：
- **主备模式**：一个 Master SM + 多个 Standby SM，Master 故障时自动切换
- OpenSM (开源) vs UFM (NVIDIA 商业版，支持 REST API 和遥测)

---

## Queue Pair / Completion Queue 工作模型

### Queue Pair (QP) 架构

QP 是 IB 通信的基本单元，每个 QP 包含：
- **Send Queue (SQ)**：发送方向的 Work Request 队列
- **Receive Queue (RQ)**：接收方向的 Work Request 队列

QP 类型：
| 类型 | 连接模式 | 适用场景 |
|------|----------|----------|
| RC (Reliable Connected) | 点对点 | NCCL 默认，可靠传输 |
| UC (Unreliable Connected) | 点对点 | 大块数据，允许偶尔重传 |
| UD (Unreliable Datagram) | 一对多 | 管理消息，地址解析 |
| DC (Dynamically Connected) | 动态连接 | 大规模集群，减少 QP 数量 |

### 工作流程

```
应用层                    HCA 硬件
  |                         |
  |-- ibv_post_send() -->   |
  |   (WQE 写入 SQ)        |-- DMA 读取源数据
  |                         |-- 组装 IB 包
  |                         |-- 发送到网络
  |                         |
  |                         |-- 收到 ACK
  |   <-- CQE 产生 --      |-- 写入 CQ
  |-- ibv_poll_cq()         |
```

关键概念：
- **WQE (Work Queue Element)**：描述一次 RDMA 操作的指令
- **CQE (Completion Queue Element)**：操作完成的通知
- **Doorbell**：应用通过 MMIO 写通知 HCA 有新 WQE

### Completion Queue (CQ)

CQ 接收操作完成通知，支持两种模式：
- **Polling 模式**：应用主动调用 `ibv_poll_cq()` 检查，延迟最低
- **Event 模式**：通过 Completion Channel 异步通知，节省 CPU

NCCL 默认使用 Polling 模式以获得最低延迟。

---

## 速率演进：NDR 到 XDR

### 当前代际

| 代际 | 单通道速率 | 端口速率 (4x) | 发布年份 |
|------|-----------|--------------|----------|
| HDR | 50 Gbps | 200 Gbps | 2018 |
| NDR | 100 Gbps | 400 Gbps | 2022 |
| XDR | 200 Gbps | 800 Gbps | 2024 |
| GDR | 400 Gbps | 1600 Gbps | 规划中 |

### NDR 400Gbps 技术细节

- 编码：PAM4 (脉冲幅度调制 4 级)
- FEC：RS-FEC (544, 514)，纠错能力更强
- 线缆：支持铜缆 (≤2m)、AOC (≤100m)、光模块 (≤10km)
- 实测带宽：单端口 ~48 GB/s (有效载荷，扣除协议开销)

### XDR 800Gbps 关键提升

- 单通道 200Gbps，4x 聚合 800Gbps
- ConnectX-8 HCA 支持
- Quantum-3 交换机：144 端口 × 800Gbps
- 对 H20 集群影响：AllReduce 通信时间理论减半

---

## 无损特性：Credit-Based Flow Control

### 为什么 IB 能保证零丢包

IB 使用**逐跳信用流控 (Link-Level Credit-Based Flow Control)**：

```
发送端                   接收端
  |                        |
  |  初始信用额度 = N 个包  |
  |<--- Credit Grant ---   |  (接收端告知可接收 N 个包)
  |                        |
  |--- 发送数据包 1 --->   |  credit -= 1
  |--- 发送数据包 2 --->   |  credit -= 1
  |  ...                   |
  |--- 发送数据包 N --->   |  credit = 0，停止发送
  |                        |
  |<--- Credit Return ---  |  (接收端处理完毕，归还信用)
  |--- 继续发送 ------->   |
```

核心机制：
- 每条物理链路独立维护信用计数
- 发送方在信用耗尽时**硬停止**，绝不溢出接收缓冲区
- 信用粒度为 MTU 大小 (IB 默认 4KB)
- 整个机制在硬件中实现，无软件参与

### 与以太网流控对比

| 特性 | IB Credit Flow Control | 以太网 PFC |
|------|----------------------|------------|
| 触发时机 | 预防性（信用耗尽前停止） | 反应性（缓冲区快满时暂停） |
| 丢包可能 | 物理上不可能 | PFC 风暴/死锁仍可能 |
| Head-of-Line Blocking | VL 机制缓解 | PFC 按优先级暂停整个类 |
| 复杂度 | 协议原生，零配置 | 需精细调参 (PFC threshold) |

---

## InfiniBand vs Ethernet：本质区别

### 设计哲学

**InfiniBand — 专用高性能互连**
- 从芯片到协议栈完全自研
- 封闭生态，NVIDIA (Mellanox) 主导
- 为 HPC/AI 场景极致优化

**Ethernet — 通用网络基础设施**
- 开放标准，多厂商竞争
- 功能通过叠加层层协议实现 (TCP/IP/RoCE)
- 兼顾万千场景，单场景非最优

### 实际影响对比

| 维度 | InfiniBand | 以太网 (RoCE v2) |
|------|-----------|-----------------|
| 延迟 | ~1μs | ~2-5μs |
| 丢包率 | 0 (by design) | 极低但非零 |
| 配置复杂度 | SM 自动管理 | PFC/ECN/DCQCN 需手动调参 |
| 故障排查 | ibdiagnet 一键诊断 | 多层协议栈分别排查 |
| 成本 | 高 (专用设备) | 中 (可复用现有基础设施) |
| 扩展性 | 数万节点 (已验证) | 数千节点 (大规模挑战多) |
| 生态锁定 | NVIDIA 单一供应商 | 多厂商可选 |

### 对 8×H20 集群的建议

- **首选 IB**：如果集群规模 ≥32 节点且预算充足
- **考虑 RoCE**：如果已有成熟以太网基础设施且网络团队经验丰富
- **混合部署**：计算网络用 IB，存储/管理网络用以太网 (最常见方案)

---

## 实用诊断命令

```bash
# 链路状态检查
ibstatus          # 快速查看端口状态和速率
ibstat            # 详细 HCA 信息

# 性能测试
ib_write_bw       # RDMA Write 带宽测试
ib_read_lat       # RDMA Read 延迟测试

# 故障排查
ibdiagnet         # 全网诊断 (需要 SM 权限)
perfquery         # 端口计数器查询 (错误/丢弃统计)
ibclearerrors     # 清除错误计数器

# 拓扑可视化
ibnetdiscover > topology.txt
iblinkinfo        # 链路信息汇总
```

---

## 小结

InfiniBand 为 AI 训练提供了确定性的、无损的高性能网络。其架构从硬件到协议的垂直整合设计，使得 8×H20 集群的 GPU 间通信可以达到接近线速的效率。理解 HCA、SM、QP/CQ 的工作原理，是后续优化 NCCL 通信和排查网络问题的基础。
