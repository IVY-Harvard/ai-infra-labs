# Lab 01: RDMA Fundamentals

## 概述

本实验帮助你理解 RDMA（Remote Direct Memory Access）的核心概念，掌握 RDMA 设备信息收集与验证的基本技能。

## 学习目标

1. 理解 RDMA 的基本架构和工作原理
2. 掌握 RDMA 设备信息查询工具的使用
3. 了解 RDMA 传输类型（RC、UC、UD、RD）
4. 理解 Queue Pair（QP）、Completion Queue（CQ）、Memory Region（MR）等核心概念

## 背景知识

### RDMA 架构

```
┌─────────────────────────────────────────────────┐
│                  Application                     │
├─────────────────────────────────────────────────┤
│              Verbs API (libibverbs)              │
├─────────────────────────────────────────────────┤
│           RDMA Core (内核模块)                    │
├──────────┬──────────┬──────────┬────────────────┤
│  mlx5    │  mlx4    │  rxe     │  其他驱动       │
├──────────┴──────────┴──────────┴────────────────┤
│           Hardware / Software RDMA               │
└─────────────────────────────────────────────────┘
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Queue Pair (QP)** | RDMA 通信的基本端点，包含 Send Queue 和 Receive Queue |
| **Completion Queue (CQ)** | 用于通知操作完成的队列 |
| **Memory Region (MR)** | 注册给 RDMA 硬件使用的内存区域 |
| **Protection Domain (PD)** | 安全隔离域，关联 QP、MR 等资源 |
| **Address Handle (AH)** | 用于 UD 类型 QP 的远端地址描述 |

### RDMA 操作类型

| 操作 | 说明 | 是否需要远端参与 |
|------|------|-----------------|
| **Send/Recv** | 传统的消息传递 | 是 |
| **RDMA Write** | 直接写入远端内存 | 否 |
| **RDMA Read** | 直接读取远端内存 | 否 |
| **Atomic** | 远端原子操作（CAS、Fetch-Add） | 否 |

### RDMA 传输协议

| 协议 | 全称 | 链路层 | 特点 |
|------|------|--------|------|
| **InfiniBand** | InfiniBand | InfiniBand | 原生 RDMA，最低延迟 |
| **RoCE v1** | RDMA over Converged Ethernet v1 | Ethernet L2 | 以太网二层封装 |
| **RoCE v2** | RDMA over Converged Ethernet v2 | Ethernet L3 (UDP) | 可路由，当前主流 |
| **iWARP** | Internet Wide Area RDMA Protocol | TCP | 基于 TCP，兼容性好 |

## 实验内容

### 实验 1: 环境检查

```bash
# 检查 RDMA 内核模块
lsmod | grep -E "rdma|ib_|mlx"

# 检查 rdma-core 包是否安装
rpm -qa | grep rdma-core 2>/dev/null || dpkg -l | grep rdma-core 2>/dev/null

# 检查 libibverbs 用户空间库
ibv_devices 2>/dev/null && echo "libibverbs OK" || echo "libibverbs NOT found"
```

### 实验 2: 设备信息收集

使用提供的 `rdma_info_collector.sh` 脚本收集完整的 RDMA 设备信息：

```bash
chmod +x rdma_info_collector.sh
sudo ./rdma_info_collector.sh
```

### 实验 3: 手动探索

逐步执行以下命令，观察输出并理解含义：

```bash
# 列出所有 RDMA 设备
ibv_devices

# 查看设备详细信息
ibv_devinfo

# 查看特定设备的特定端口
ibv_devinfo -d mlx5_0 -i 1

# 使用 rdma 工具查看链路状态
rdma link show

# 查看 RDMA 统计计数器
rdma statistic show
```

### 实验 4: 简单连通性测试

在两台机器上进行 RDMA 连通性验证：

```bash
# 服务端
ib_write_bw -d mlx5_0

# 客户端
ib_write_bw -d mlx5_0 <server_ip>
```

## 验证检查点

- [ ] 能列出系统中所有 RDMA 设备
- [ ] 能查看每个端口的状态（Active/Down）
- [ ] 理解 link_layer 字段（InfiniBand vs Ethernet）
- [ ] 能识别设备支持的传输类型
- [ ] 理解 max_qp、max_cq 等设备能力参数

## 常见问题

### Q: ibv_devices 无输出？
确保 RDMA 驱动已加载：
```bash
modprobe mlx5_ib   # Mellanox ConnectX-5/6/7
modprobe mlx4_ib   # Mellanox ConnectX-3
modprobe rdma_rxe  # 软件模拟 RXE
```

### Q: 端口状态为 Down？
检查物理连接和对端设备，确认链路层协议匹配。

## 参考资料

- [RDMA Aware Networks Programming User Manual](https://docs.nvidia.com/networking/display/rdmacore50)
- [libibverbs API 文档](https://man7.org/linux/man-pages/man7/ibv_devices.7.html)
- [rdma-core GitHub](https://github.com/linux-rdma/rdma-core)
