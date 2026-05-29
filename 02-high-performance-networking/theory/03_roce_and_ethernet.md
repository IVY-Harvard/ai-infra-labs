# RoCE v2 与以太网 RDMA

## 概述

RoCE (RDMA over Converged Ethernet) 将 RDMA 的高性能带入以太网世界。对于已有以太网基础设施的团队，RoCE v2 提供了一条无需全面更换设备即可获得 RDMA 能力的路径。但这条路并不轻松——需要精细的网络配置来模拟 InfiniBand 的无损特性。

---

## RoCE v2 协议原理

### 协议栈对比

```
InfiniBand 原生:        RoCE v1:           RoCE v2:
+------------+         +------------+      +------------+
| IB 传输层   |         | IB 传输层   |      | IB 传输层   |
+------------+         +------------+      +------------+
| IB 网络层   |         | IB 网络层   |      | UDP (4791) |
+------------+         +------------+      +------------+
| IB 链路层   |         | Ethernet   |      | IP (v4/v6) |
+------------+         +------------+      +------------+
| IB 物理层   |         | Eth PHY    |      | Ethernet   |
+------------+         +------------+      +------------+
                                           | Eth PHY    |
                                           +------------+
```

### RoCE v2 关键设计

- **UDP 封装**：使用目标端口 4791，源端口基于流哈希 (实现 ECMP 负载均衡)
- **IP 可路由**：支持跨 L3 网络传输 (RoCE v1 仅限 L2)
- **GRH (Global Route Header)**：映射为 IP 头，保持 IB 传输语义
- **ICRC**：不变的 CRC 校验，端到端数据完整性保护

### 与 TCP 的本质区别

| 特性 | RoCE v2 | TCP |
|------|---------|-----|
| 连接管理 | QP 建立 (类 IB) | 三次握手 |
| 数据传输 | 零拷贝 RDMA | 内核缓冲区拷贝 |
| 可靠性 | Go-Back-N 重传 | 滑动窗口 + SACK |
| 丢包容忍 | 极差 (性能断崖) | 优雅降级 |
| CPU 开销 | 近零 | 显著 (中断/拷贝) |

RoCE v2 的核心矛盾：**使用了不保证可靠的以太网，却需要无损传输来保证性能。**

---

## 构建无损以太网：PFC + ECN + DCQCN

### Priority Flow Control (PFC)

PFC (IEEE 802.1Qbb) 是以太网模拟无损传输的基石：

```
发送方                    交换机                    接收方
  |                        |                        |
  |--- 数据流量 --------> |--- 转发 ------------> |
  |--- 数据流量 --------> |  缓冲区使用率上升      |
  |--- 数据流量 --------> |  达到 XOFF 阈值       |
  |                        |                        |
  |<-- PFC PAUSE (pri=3) --|  (暂停优先级 3)       |
  |  [停止发送 pri=3]      |                        |
  |                        |  缓冲区降到 XON 阈值   |
  |<-- PFC RESUME ---------|                        |
  |--- 恢复发送 --------> |                        |
```

PFC 配置要点：
```bash
# 在交换机上启用 PFC (以 Mellanox Onyx 为例)
interface ethernet 1/1
  dcb priority-flow-control mode on
  dcb priority-flow-control priority 3 no-drop

# 设置缓冲区阈值
  dcb priority-flow-control xoff-threshold 80000
  dcb priority-flow-control xon-threshold 20000
```

PFC 的问题：
- **Head-of-Line Blocking**：一个优先级暂停可能影响共享缓冲区
- **PFC Storm**：故障端口持续发送 PAUSE 帧，瘫痪上游
- **PFC Deadlock**：环形依赖导致所有端口互相等待

### Explicit Congestion Notification (ECN)

ECN (RFC 3168) 提供端到端拥塞信号，减少对 PFC 的依赖：

```
发送方                 交换机                    接收方
  |                     |                        |
  |-- IP.ECN=10 -----> |                        |
  |  (ECT capable)     |  队列深度 > 阈值       |
  |                     |-- IP.ECN=11 --------> |  (CE marked)
  |                     |                        |
  |<-------- CNP (Congestion Notification Packet) --|
  |  [降低发送速率]     |                        |
```

ECN 配置：
```bash
# 交换机 ECN 标记阈值配置
interface ethernet 1/1
  traffic-class 3 congestion-control ecn
  traffic-class 3 ecn minimum-absolute 150KB maximum-absolute 1500KB
```

### DCQCN (Data Center QCN) 拥塞控制

DCQCN 是 RoCE v2 的标准拥塞控制算法，结合 ECN + 速率调节：

**算法核心流程：**

1. **发送方**：初始以线速发送
2. **交换机**：队列深度超阈值时标记 ECN (CE bit)
3. **接收方**：检测到 CE 标记，发送 CNP (Congestion Notification Packet)
4. **发送方收到 CNP**：
   - 立即将速率降低到 `Rate × (1 - α/2)`
   - α 值基于 ECN 标记频率动态更新
5. **恢复阶段**：
   - Timer-based：定时器触发速率恢复
   - Byte-counter-based：发送一定字节后恢复
   - 恢复公式：`Rate += (TargetRate - Rate) / HAI_factor`

**关键参数调优：**
```bash
# HCA 端 DCQCN 参数 (通过 mlxreg 或 mlnx_qos)
# CNP 优先级
mlnx_qos -i mlx5_0 --pfc 0,0,0,1,0,0,0,0

# 速率降低因子
echo 1 > /sys/class/net/eth0/ecn/roce_np/cnp_dscp
# Alpha 更新频率 (越快响应越灵敏)
echo 1024 > /sys/class/infiniband/mlx5_0/cc_params/rp_ai_rate
```

### 三层协同工作

```
性能影响程度：
                    
DCQCN ─────── 精细速率调节 (常态工作) ──── 延迟影响：微秒级
    |
    v (DCQCN 来不及反应)
ECN ─────── 标记拥塞信号 ──── 延迟影响：十微秒级
    |
    v (ECN 来不及缓解)
PFC ─────── 最后防线，暂停流量 ──── 延迟影响：毫秒级 (应极少触发)
```

理想状态：DCQCN 处理 99% 的拥塞，PFC 几乎不触发。

---

## UEC：Ultra Ethernet Consortium

### 背景

2023 年成立的 UEC 联盟，目标是为 AI/HPC 重新设计以太网传输层：

**核心动机**：
- RoCE v2 依赖的 PFC 机制在大规模部署时问题频发
- TCP 延迟太高无法满足 AI 训练需求
- 需要一个"从 AI 需求出发设计"的以太网传输协议

### UEC 技术方向

| 特性 | RoCE v2 | UEC 目标 |
|------|---------|---------|
| 流控机制 | PFC (逐跳) | 端到端拥塞控制，消除 PFC |
| 多路径 | 依赖 ECMP 哈希 | 原生 Packet Spraying |
| 乱序处理 | 不支持 (需按序到达) | 硬件乱序重组 |
| 丢包恢复 | Go-Back-N (整窗重传) | 选择性重传 |
| 拥塞信号 | ECN/CNP | 精细化拥塞反馈 |

### 对实际部署的影响

- **短期 (2024-2025)**：UEC 仍在标准化阶段，生产环境继续使用 RoCE v2/IB
- **中期 (2026-2027)**：首批 UEC 兼容硬件可能上市
- **长期影响**：可能改变 IB vs Ethernet 的竞争格局

---

## RoCE vs InfiniBand 选型决策

### 决策矩阵

| 维度 | InfiniBand | RoCE v2 | 权重 (AI 训练) |
|------|-----------|---------|---------------|
| 原始性能 | 极致 (~1μs) | 优秀 (~2-3μs) | 高 |
| 部署复杂度 | 低 (SM 自管理) | 高 (PFC/ECN/DCQCN 调参) | 中 |
| 设备成本 | 高 | 中 | 中 |
| 运维成本 | 低 (工具链完善) | 中高 (排障复杂) | 高 |
| 现有基础设施复用 | 不可能 | 可能 | 低 |
| 供应商多样性 | 仅 NVIDIA | 多厂商 | 低 |
| 规模上限 | 数万节点 | 数千节点 (经验值) | 中 |
| 与存储网络融合 | 需独立网络 | 可融合 | 低 |

### 场景化建议

**选择 InfiniBand 当：**
- 集群规模 ≥ 64 GPU (8+ 节点)
- 训练任务对通信延迟极度敏感 (大模型 TP 并行)
- 团队没有深厚的 DCB 网络调优经验
- 预算允许专用网络投资

**选择 RoCE v2 当：**
- 已有成熟的 25/100/400G 以太网基础设施
- 有经验丰富的数据中心网络团队
- 需要计算/存储网络融合降本
- 集群规模适中且通信模式相对简单 (纯 DP 训练)

**8×H20 集群具体建议：**
- 节点内：PCIe P2P (无需网络)
- 节点间首选：NDR 400G InfiniBand (单端口即可饱和 H20 通信需求)
- 节点间备选：400G RoCE v2 (需配合专业网络团队)
- 管理网络：独立 25G 以太网

---

## RoCE v2 部署检查清单

```bash
# 1. 验证 RDMA 功能
ibv_devinfo  # 确认 HCA 识别
show_gids    # 确认 GID 表正确

# 2. 验证 PFC 配置
mlnx_qos -i ens1f0  # 查看 PFC 状态
# 确认 RoCE 流量的优先级已启用 no-drop

# 3. 验证 ECN 配置
# 交换机侧确认 ECN marking threshold 合理
# HCA 侧确认 DCQCN 已启用

# 4. 性能基线测试
ib_write_bw -d mlx5_0 --report_gbits  # 带宽
ib_write_lat -d mlx5_0                  # 延迟

# 5. 监控 PFC 触发频率
ethtool -S ens1f0 | grep pause
# rx_pfc3_pause / tx_pfc3_pause 应接近零
```

---

## 小结

RoCE v2 通过 UDP 封装将 RDMA 带入以太网，但需要 PFC+ECN+DCQCN 三层机制协同才能接近无损传输。配置复杂度远高于 InfiniBand，但为已有以太网基础设施的团队提供了可行路径。UEC 标准有望在未来简化这一局面。对于 8×H20 集群，如果追求最简运维和最优性能，InfiniBand 仍是首选；如果成本和基础设施复用是关键考量，RoCE v2 是可行的替代方案。
