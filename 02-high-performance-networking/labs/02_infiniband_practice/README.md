# Lab 02: InfiniBand Practice

## 概述

本实验深入 InfiniBand 网络的诊断、调优与 Subnet Manager 管理，掌握生产环境中 IB 网络运维的核心技能。

## 学习目标

1. 掌握 InfiniBand 网络诊断工具链
2. 理解 Subnet Manager（SM）的角色与配置
3. 学会使用 perftest 工具进行性能基准测试
4. 能够排查常见的 IB 网络故障

## 背景知识

### InfiniBand 网络架构

```
┌──────────────┐     ┌──────────────┐
│   HCA (Host  │     │   HCA (Host  │
│   Channel    │     │   Channel    │
│   Adapter)   │     │   Adapter)   │
└──────┬───────┘     └──────┬───────┘
       │                     │
       │   InfiniBand Link   │
       │                     │
┌──────┴─────────────────────┴──────┐
│          IB Switch                 │
│     (含 Subnet Manager)           │
└───────────────────────────────────┘
```

### InfiniBand 速率演进

| 代际 | 信号速率 | 1x 带宽 | 4x (HDR) 带宽 |
|------|---------|---------|---------------|
| SDR | 2.5 Gbps | 2 Gbps | 8 Gbps |
| DDR | 5.0 Gbps | 4 Gbps | 16 Gbps |
| QDR | 10 Gbps | 8 Gbps | 32 Gbps |
| FDR | 14.0625 Gbps | 13.64 Gbps | 54.54 Gbps |
| EDR | 25 Gbps | 25 Gbps | 100 Gbps |
| HDR | 50 Gbps | 50 Gbps | 200 Gbps |
| NDR | 100 Gbps | 100 Gbps | 400 Gbps |
| XDR | 200 Gbps | 200 Gbps | 800 Gbps |

### Subnet Manager 角色

Subnet Manager 是 InfiniBand 网络的核心管理组件，负责：
- **拓扑发现**: 识别网络中所有节点和交换机
- **LID 分配**: 为每个端口分配本地标识符
- **路由计算**: 计算并下发转发表
- **状态监控**: 持续监控网络健康状态

## 实验文件

| 文件 | 说明 |
|------|------|
| `ib_diagnostic.sh` | IB 网络综合诊断脚本 |
| `subnet_manager_guide.md` | Subnet Manager 配置与管理指南 |

## 实验内容

### 实验 1: IB 网络基础诊断

```bash
# 运行诊断脚本
chmod +x ib_diagnostic.sh
sudo ./ib_diagnostic.sh
```

### 实验 2: 性能测试

使用 perftest 套件进行带宽和延迟测试：

```bash
# 带宽测试 — 服务端
ib_write_bw -d mlx5_0 --report_gbits

# 带宽测试 — 客户端
ib_write_bw -d mlx5_0 --report_gbits <server_ip>

# 延迟测试 — 服务端
ib_write_lat -d mlx5_0

# 延迟测试 — 客户端
ib_write_lat -d mlx5_0 <server_ip>
```

### 实验 3: Subnet Manager 管理

参考 `subnet_manager_guide.md` 完成以下任务：
1. 安装并启动 OpenSM
2. 查看 SM 状态和网络拓扑
3. 调整路由算法
4. 配置 SM 高可用

### 实验 4: 故障排查练习

常见故障场景：

| 故障现象 | 可能原因 | 诊断命令 |
|---------|---------|---------|
| 端口 Down | 物理连接/驱动 | `ibstat`, `ibdiagnet` |
| 高延迟 | 拥塞/路由次优 | `perfquery`, `ibdiagnet` |
| 丢包 | 链路错误 | `perfquery -x` |
| SM 不可达 | SM 未运行/配置错误 | `sminfo`, `smpquery` |

## 验证检查点

- [ ] 能使用 ibstat/ibstatus 查看端口状态
- [ ] 能运行 ibdiagnet 进行网络诊断
- [ ] 理解 perftest 各项指标含义
- [ ] 能配置和管理 OpenSM
- [ ] 能排查常见 IB 网络故障

## 参考资料

- [NVIDIA InfiniBand 文档](https://docs.nvidia.com/networking/)
- [OpenSM 用户指南](https://docs.nvidia.com/networking/display/opensm)
- [perftest GitHub](https://github.com/linux-rdma/perftest)
