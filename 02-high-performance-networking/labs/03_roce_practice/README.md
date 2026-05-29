# Lab 03: RoCE Practice

## 概述

本实验聚焦 RoCE v2（RDMA over Converged Ethernet v2）的配置与实践，包括 RoCE 网卡配置、PFC（Priority Flow Control）和 ECN（Explicit Congestion Notification）等关键网络功能的设置。

## 学习目标

1. 理解 RoCE v1 与 RoCE v2 的区别
2. 掌握 RoCE v2 网卡与网络配置
3. 理解并配置 PFC（无损以太网的基础）
4. 理解并配置 ECN（拥塞通知机制）
5. 验证 RoCE 网络的连通性与性能

## 背景知识

### RoCE 协议栈对比

```
RoCE v1                          RoCE v2
┌──────────────┐                 ┌──────────────┐
│   IB 传输层   │                 │   IB 传输层   │
├──────────────┤                 ├──────────────┤
│   IB 网络层   │                 │   UDP (4791)  │
├──────────────┤                 ├──────────────┤
│              │                 │   IP (v4/v6)  │
│  Ethernet L2 │                 ├──────────────┤
│              │                 │  Ethernet L2  │
└──────────────┘                 └──────────────┘

  不可路由                         可路由 ✓
  仅限同一 L2                      支持 L3 路由
```

### 无损以太网机制

RoCE 需要无损网络，通过以下机制实现：

```
                     拥塞检测与控制
┌─────────────────────────────────────────────┐
│                                             │
│    PFC (链路级)        ECN (端到端)          │
│    ┌──────────┐       ┌──────────────┐      │
│    │ 缓冲区满  │       │ 交换机标记    │      │
│    │ → PAUSE  │       │ ECN CE 位    │      │
│    │ 特定优先级│       │ → 接收端通知  │      │
│    │          │       │ → 发送端降速  │      │
│    └──────────┘       └──────────────┘      │
│                                             │
│    DCQCN = ECN + PFC (联合使用)              │
│    CNP: Congestion Notification Packet       │
└─────────────────────────────────────────────┘
```

### 关键概念

| 概念 | 说明 |
|------|------|
| **DSCP** | Differentiated Services Code Point，IP 层 QoS 标记 |
| **PCP** | Priority Code Point，802.1Q VLAN 优先级 |
| **TC** | Traffic Class，硬件流量类别 |
| **PFC** | Priority Flow Control (802.1Qbb)，按优先级暂停 |
| **ECN** | Explicit Congestion Notification，显式拥塞通知 |
| **DCQCN** | Data Center QCN，RoCE 专用拥塞控制算法 |
| **CNP** | Congestion Notification Packet，拥塞通知报文 |

## 实验文件

| 文件 | 说明 |
|------|------|
| `roce_setup.sh` | RoCE v2 网卡配置脚本 |
| `pfc_ecn_config.sh` | PFC 和 ECN 综合配置脚本 |

## 实验内容

### 实验 1: RoCE v2 基本配置

```bash
chmod +x roce_setup.sh
sudo ./roce_setup.sh --interface ens1f0 --gid-index 3
```

### 实验 2: PFC 配置

```bash
chmod +x pfc_ecn_config.sh
sudo ./pfc_ecn_config.sh --interface ens1f0 --mode pfc --priority 3
```

### 实验 3: ECN 配置

```bash
sudo ./pfc_ecn_config.sh --interface ens1f0 --mode ecn --priority 3
```

### 实验 4: 完整 DCQCN 配置

```bash
sudo ./pfc_ecn_config.sh --interface ens1f0 --mode full --priority 3
```

### 实验 5: 连通性与性能验证

```bash
# 服务端
ib_write_bw -d mlx5_0 --report_gbits -R

# 客户端
ib_write_bw -d mlx5_0 --report_gbits -R <server_ip>

# 使用 RoCE 特定 GID
ib_write_bw -d mlx5_0 -x 3 --report_gbits <server_ip>
```

## 验证检查点

- [ ] 理解 RoCE v1 与 v2 的协议栈差异
- [ ] 能配置网卡使用 RoCE v2 默认模式
- [ ] 能配置 PFC 并验证其工作
- [ ] 能配置 ECN 并验证标记功能
- [ ] 理解 DSCP-to-priority 映射
- [ ] 能在 RoCE 网络上运行 perftest

## 网络交换机配置参考

RoCE 不仅需要主机端配置，交换机也需要相应配置。以下是常见交换机的参考配置：

### NVIDIA Spectrum 交换机 (Cumulus/NVOS)

```bash
# 配置 PFC on priority 3
nv set qos pfc priority 3
nv set interface swp1-48 qos pfc-enable on

# 配置 ECN
nv set qos ecn default min-threshold 150000
nv set qos ecn default max-threshold 1500000
```

### Cisco Nexus

```
# PFC
interface Ethernet1/1
  priority-flow-control mode on
  priority-flow-control priority 3 no-drop

# ECN (WRED)
policy-map type network-qos roce-policy
  class type network-qos c-roce
    congestion-control ecn
```

## 参考资料

- [RoCE v2 配置指南 (NVIDIA)](https://docs.nvidia.com/networking/display/rdmacore50/RoCE+Configuration)
- [Lossless RoCE 配置 (NVIDIA)](https://docs.nvidia.com/networking/display/mlnxofedv580230/lossless+roce+configuration)
- [DCQCN 论文](https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf)
