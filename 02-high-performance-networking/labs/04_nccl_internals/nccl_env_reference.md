# NCCL 环境变量参考手册

## 通信协议与传输层

| 变量名 | 含义 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| `NCCL_P2P_DISABLE` | 禁用 GPU 点对点通信 | 0 (启用) | 0 (保持启用) |
| `NCCL_P2P_LEVEL` | P2P 通信级别 (LOC/NVL/PIX/PXB/PHB/SYS) | 自动检测 | 根据拓扑设置 |
| `NCCL_SHM_DISABLE` | 禁用共享内存传输 | 0 (启用) | 0 |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 使用级别 | 自动 | PIX (同 PCIe switch) |
| `NCCL_NET_GDR_READ` | 启用 GPUDirect RDMA 读操作 | 0 | 1 (支持时启用) |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | 0 (启用) | 0 |
| `NCCL_SOCKET_IFNAME` | 指定网络接口名称 | 自动选择 | eth0 或 ib0 |
| `NCCL_IB_HCA` | 指定 InfiniBand HCA 设备 | 所有可用 | mlx5_0,mlx5_1 |

## 性能调优

| 变量名 | 含义 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| `NCCL_BUFFSIZE` | 通信缓冲区大小 (bytes) | 4194304 (4MB) | 8388608 (8MB) |
| `NCCL_NTHREADS` | NCCL 内核线程数 | 512 | 256-512 |
| `NCCL_MAX_NCHANNELS` | 最大通信通道数 | 自动 | 根据 GPU 数调整 |
| `NCCL_MIN_NCHANNELS` | 最小通信通道数 | 自动 | 2 |
| `NCCL_CHECKS_DISABLE` | 禁用参数检查以提升性能 | 0 | 1 (生产环境) |
| `NCCL_CHECK_POINTERS` | 检查指针有效性 | 0 | 0 (生产环境) |
| `NCCL_LAUNCH_MODE` | 内核启动模式 | PARALLEL | PARALLEL |
| `NCCL_IB_TIMEOUT` | InfiniBand 超时 (指数值) | 18 | 22 (大规模集群) |
| `NCCL_IB_RETRY_CNT` | InfiniBand 重试次数 | 7 | 7 |

## 算法选择

| 变量名 | 含义 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| `NCCL_ALGO` | 集合通信算法 | 自动 | Tree/Ring/CollnetDirect |
| `NCCL_PROTO` | 通信协议 | 自动 | Simple/LL/LL128 |
| `NCCL_GRAPH_DUMP_FILE` | 导出拓扑图文件路径 | 空 | /tmp/nccl_graph.xml |
| `NCCL_TOPO_FILE` | 自定义拓扑文件路径 | 空 | 按需设置 |
| `NCCL_TOPO_DUMP_FILE` | 导出检测到的拓扑 | 空 | /tmp/nccl_topo.xml |
| `NCCL_COLLNET_ENABLE` | 启用 CollNet (Sharp) | 0 | 1 (支持时) |

## 调试与日志

| 变量名 | 含义 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| `NCCL_DEBUG` | 调试日志级别 | WARN | INFO (调试) / WARN (生产) |
| `NCCL_DEBUG_SUBSYS` | 调试子系统过滤 | ALL | INIT,NET,GRAPH |
| `NCCL_DEBUG_FILE` | 日志输出文件 | stderr | /tmp/nccl_%h_%p.log |

## 网络与连接

| 变量名 | 含义 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| `NCCL_SOCKET_NTHREADS` | Socket 传输线程数 | 1 | 4-8 (高带宽网络) |
| `NCCL_NSOCKS_PERTHREAD` | 每线程 Socket 数 | 1 | 4-8 |
| `NCCL_IB_QPS_PER_CONNECTION` | 每连接 QP 数 | 1 | 1 |
| `NCCL_IB_GID_INDEX` | InfiniBand GID 索引 | 自动 | 3 (RoCEv2) |
| `NCCL_IB_TC` | InfiniBand Traffic Class | 0 | 106 (RoCE DSCP) |
| `NCCL_IB_SL` | InfiniBand Service Level | 0 | 0 |
| `NCCL_CROSS_NIC` | 允许跨 NIC 通信 | 2 (自动) | 1 (多 NIC) |

## 使用建议

### DGX A100 (8 GPU NVLink)
```bash
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_NET_GDR_READ=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

### PCIe 多卡服务器
```bash
export NCCL_P2P_LEVEL=PXB
export NCCL_SHM_DISABLE=0
export NCCL_BUFFSIZE=8388608
```

### 多节点训练
```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```
