# Subnet Manager 配置与管理指南

## 概述

Subnet Manager（SM）是 InfiniBand 网络的核心控制平面组件。它负责拓扑发现、LID 分配、路由计算和网络状态监控。没有运行中的 SM，InfiniBand 网络无法正常工作。

## 目录

1. [SM 基础概念](#1-sm-基础概念)
2. [OpenSM 安装与启动](#2-opensm-安装与启动)
3. [OpenSM 配置详解](#3-opensm-配置详解)
4. [路由算法选择](#4-路由算法选择)
5. [高可用配置](#5-高可用配置)
6. [监控与故障排查](#6-监控与故障排查)
7. [生产环境最佳实践](#7-生产环境最佳实践)

---

## 1. SM 基础概念

### SM 的职责

| 职责 | 说明 |
|------|------|
| **拓扑发现** | 通过 SMP（Subnet Management Packets）探测网络中所有节点 |
| **LID 分配** | 为每个端口分配 Local Identifier（LID） |
| **路由计算** | 计算节点间路由并下发 LFT（Linear Forwarding Table）到交换机 |
| **状态监控** | 周期性轮询（Sweep）网络状态，处理拓扑变更 |
| **QoS 管理** | 配置服务等级（SL）到虚拟通道（VL）映射 |
| **分区管理** | 配置 PKey 分区，实现网络隔离 |

### SM 状态机

```
                ┌───────────┐
                │ Discovering│
                └─────┬─────┘
                      │
                      ▼
                ┌───────────┐
          ┌─────│  Standby   │◄─────────┐
          │     └─────┬─────┘           │
          │           │ (更高优先级      │
          │           │  SM 离线)       │
          │           ▼                 │
          │     ┌───────────┐           │
          │     │  Master    │──────────┘
          │     └───────────┘  (检测到更高
          │                     优先级 SM)
          │
          └──► Not Active
```

### SM 运行位置

| 位置 | 优点 | 缺点 |
|------|------|------|
| **交换机内置** | 独立于计算节点，不占用主机资源 | 功能可能受限 |
| **主机运行 (OpenSM)** | 功能完整，配置灵活 | 占用主机资源 |
| **专用管理节点** | 不影响计算节点 | 需要额外硬件 |

---

## 2. OpenSM 安装与启动

### 安装

```bash
# RHEL/CentOS
sudo yum install opensm opensm-libs

# Ubuntu/Debian
sudo apt-get install opensm

# MLNX_OFED (推荐)
# OpenSM 通常包含在 MLNX_OFED 安装包中
```

### 基本启动

```bash
# 使用 systemd 启动
sudo systemctl start opensm
sudo systemctl enable opensm

# 查看状态
sudo systemctl status opensm

# 手动启动 (调试用)
sudo opensm -g <port_guid> -f /var/log/opensm.log -D 0x02
```

### 启动参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-g <guid>` | 指定绑定的端口 GUID | `-g 0x0002c903000fe080` |
| `-p <priority>` | SM 优先级 (0-15, 越高越优先) | `-p 14` |
| `-R <engine>` | 路由引擎 | `-R updn` |
| `-f <file>` | 日志文件路径 | `-f /var/log/opensm.log` |
| `-D <flags>` | 调试标志 | `-D 0x02` |
| `-c <file>` | 配置文件路径 | `-c /etc/opensm/opensm.conf` |
| `-s <seconds>` | Sweep 间隔 | `-s 10` |

---

## 3. OpenSM 配置详解

### 生成默认配置文件

```bash
# 生成带注释的配置文件
opensm -c /etc/opensm/opensm.conf --create-config /etc/opensm/opensm.conf
```

### 核心配置项

```ini
# /etc/opensm/opensm.conf

# ─── SM 身份与优先级 ─────────────────────────────
# SM 优先级 (0-15)，用于 Master 选举
sm_priority 14

# SM 绑定的端口 GUID (0 表示自动选择)
guid 0

# ─── Sweep 配置 ──────────────────────────────────
# 轻量级 Sweep 间隔 (秒)
sweep_interval 10

# 重量级 Sweep 最大间隔 (秒)
max_wire_smps 4

# ─── 路由配置 ────────────────────────────────────
# 路由引擎: minhop / updn / dnup / ftree / lash / dor / torus-2QoS / nue
routing_engine updn

# 是否使用单播缓存 (加速重路由)
use_ucast_cache TRUE

# ─── LID 管理 ────────────────────────────────────
# LID 范围
lmc 0
subnet_prefix 0xfe80000000000000

# 是否重新分配 LID
reassign_lids FALSE

# ─── QoS 配置 ────────────────────────────────────
# 启用 QoS
qos FALSE

# SL 到 VL 映射
# qos_sl2vl 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,7

# ─── 分区配置 ────────────────────────────────────
# 分区配置文件
partition_config_file /etc/opensm/partitions.conf

# 默认分区 PKey
# 所有端口默认属于 0x7fff (Default) 分区

# ─── 日志配置 ────────────────────────────────────
log_file /var/log/opensm.log
log_max_size 100

# 日志级别标志 (位掩码)
# 0x01 - ERROR, 0x02 - INFO, 0x04 - VERBOSE,
# 0x08 - DEBUG, 0x10 - FUNCS, 0x20 - FRAMES
sminfo_polling_timeout 10000
```

### 分区配置

```ini
# /etc/opensm/partitions.conf

# 默认分区 - 所有节点全成员
Default=0x7fff : ALL=full ;

# 存储网络分区
StorageNet=0x0002 : 
    0x0002c903000fe080=full,
    0x0002c903000fe081=full,
    0x0002c903000fe082=limited ;

# GPU 计算分区
GPUCompute=0x0003 :
    ALL_SWITCHES=full,
    0x0002c903000fe090=full,
    0x0002c903000fe091=full ;
```

---

## 4. 路由算法选择

### 可用路由引擎对比

| 引擎 | 全称 | 适用拓扑 | 特点 |
|------|------|---------|------|
| **minhop** | Minimum Hop | 通用 | 默认算法，最少跳数 |
| **updn** | Up/Down | Fat-tree | 避免死锁，适合层级拓扑 |
| **ftree** | Fat Tree | Fat-tree | 专为 Fat-tree 优化，负载均衡 |
| **lash** | LASH | 通用 | 基于虚拟通道避免死锁 |
| **dor** | Dimension Order | Torus/Mesh | 维度序路由，适合规则拓扑 |
| **torus-2QoS** | Torus 2-level QoS | Torus | Torus 专用，支持 QoS |
| **nue** | NUE | 通用 | 新一代通用引擎 |
| **dfsssp** | DFSSSP | 通用 | 基于最短路径，支持 QoS |

### 选择建议

```
拓扑识别:
├── Fat-tree (叶-脊) → ftree 或 updn
├── Torus/Mesh       → dor 或 torus-2QoS
├── 不规则拓扑       → lash 或 nue
└── 不确定           → updn (安全默认)
```

### 配置示例

```ini
# Fat-tree 拓扑推荐
routing_engine ftree

# 如果 ftree 失败自动回退到 updn
routing_engine ftree,updn,minhop

# Torus 拓扑
routing_engine torus-2QoS
# 需要额外的 torus 拓扑配置文件
# torus_config /etc/opensm/torus.conf
```

---

## 5. 高可用配置

### 双 SM 主备模式

InfiniBand 规范支持多个 SM 同时运行，通过优先级选举 Master。

```
节点 A (Master, priority=14)          节点 B (Standby, priority=8)
┌──────────────────────┐              ┌──────────────────────┐
│  OpenSM              │              │  OpenSM              │
│  sm_priority 14      │◄── Polling ──│  sm_priority 8       │
│  状态: MASTER        │              │  状态: STANDBY       │
└──────────────────────┘              └──────────────────────┘
        │                                     │
        │     当 Master 故障时：               │
        │                                     ▼
        │  (故障)                    ┌──────────────────────┐
        ✗                            │  OpenSM              │
                                     │  状态: MASTER        │
                                     │  (接管网络管理)       │
                                     └──────────────────────┘
```

#### 主节点配置

```ini
# /etc/opensm/opensm.conf (主节点)
sm_priority 14
polling_retry_number 4
sminfo_polling_timeout 10000
```

#### 备节点配置

```ini
# /etc/opensm/opensm.conf (备节点)
sm_priority 8
polling_retry_number 4
sminfo_polling_timeout 10000
```

### 验证 HA 配置

```bash
# 查看当前 SM Master 信息
sminfo

# 查看所有 SM 实例
saquery -s

# 模拟故障转移 (在主节点)
sudo systemctl stop opensm

# 在备节点验证是否切换为 Master
sminfo
# 应该显示备节点的 GUID 和 Master 状态
```

---

## 6. 监控与故障排查

### 常用监控命令

```bash
# 查看 SM 基本信息
sminfo

# 查看所有 SA (Subnet Administrator) 中的节点
saquery -N   # 节点记录
saquery -P   # 路径记录
saquery -L   # 链路记录

# 查看交换机列表
ibswitches

# 查看网络拓扑
ibnetdiscover

# 查看链路状态
iblinkinfo

# 查看特定节点信息
smpquery nodeinfo -G <guid>
smpquery portinfo -G <guid> <port>

# 查看转发表
dump_lfts.sh     # 转储所有交换机的 LFT
ibroute <lid>    # 查看特定交换机的路由表
```

### 日志分析

```bash
# 实时查看 OpenSM 日志
tail -f /var/log/opensm.log

# 搜索错误信息
grep -E "(ERROR|WARN)" /var/log/opensm.log | tail -20

# 搜索拓扑变更事件
grep "TOPOLOGY" /var/log/opensm.log

# 搜索 SM 状态转换
grep "SM state" /var/log/opensm.log
```

### 常见故障排查

#### 故障 1: 端口无法 Active

```bash
# 检查物理状态
ibstat
# 确认 Physical state 为 LinkUp

# 如果 Physical state 不是 LinkUp
# → 检查线缆连接
# → 检查对端设备

# 如果 Physical state 是 LinkUp 但 State 不是 Active
# → SM 未运行或不可达
sminfo
sudo systemctl status opensm
```

#### 故障 2: 路由不通

```bash
# 检查路径
ibtracert <src_lid> <dst_lid>

# 检查转发表
ibroute <switch_lid>

# 强制 SM 重新 Sweep
kill -HUP $(pgrep opensm)
```

#### 故障 3: 性能下降

```bash
# 检查错误计数器
perfquery -x

# 检查链路速率是否降级
ibstat | grep Rate

# 检查拥塞
perfquery | grep -E "Xmit|Rcv"

# 清除计数器后重新观察
perfquery -R
sleep 10
perfquery
```

#### 故障 4: SM 选举震荡

```bash
# 检查是否有多个 SM 在竞争
saquery -s

# 确认优先级配置正确 (避免相同优先级)
grep sm_priority /etc/opensm/opensm.conf

# 检查 SM 日志中的状态转换
grep "MASTER\|STANDBY" /var/log/opensm.log | tail -20
```

---

## 7. 生产环境最佳实践

### 部署建议

1. **SM 部署位置**
   - 优先使用交换机内置 SM（如 NVIDIA Quantum 交换机）
   - 如果使用主机 SM，部署在专用管理节点
   - 至少部署 2 个 SM 实现高可用

2. **优先级设计**
   ```
   交换机 SM:   priority 15 (最高)
   管理节点 SM: priority 14
   备用 SM:     priority 8
   ```

3. **路由引擎选择**
   - Fat-tree 拓扑: `ftree`（最佳负载均衡）
   - 通用拓扑: `updn`（安全可靠）
   - 回退链: `ftree,updn,minhop`

4. **Sweep 间隔**
   - 生产环境: 10-30 秒
   - 稳定网络可设更长间隔以减少 SM 负载

5. **日志管理**
   - 日志级别: 生产用 0x01 (ERROR only)，排障用 0x03 (ERROR+INFO)
   - 定期轮转日志，避免磁盘占满

### 性能调优清单

- [ ] 确认路由引擎匹配物理拓扑
- [ ] 确认 LMC 设置合理（通常为 0）
- [ ] 启用单播缓存 (`use_ucast_cache TRUE`)
- [ ] 配置合适的 Sweep 间隔
- [ ] 确认分区配置正确
- [ ] 如需 QoS，正确配置 SL-to-VL 映射
- [ ] 监控 SM 日志无异常

### 安全建议

- 使用 PKey 分区隔离不同租户/业务流量
- 限制 SM 管理访问
- 定期审计分区配置
- 启用 SM 认证（如果交换机支持）
