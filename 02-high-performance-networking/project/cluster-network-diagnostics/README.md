# 集群网络诊断平台 (Cluster Network Diagnostics)

> Module 02: 高性能网络 — 综合实战项目

企业级AI/GPU集群网络诊断工具，整合RDMA设备发现、链路健康诊断、性能基准测试和可视化报告功能。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    诊断平台 (Diagnostics Platform)        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Discovery   │  │  Diagnosis   │  │  Benchmark   │   │
│  │  设备发现模块  │  │  健康诊断模块  │  │  基准测试模块  │   │
│  │              │  │              │  │              │   │
│  │ • RDMA扫描   │  │ • 链路健康    │  │ • 带宽测试   │   │
│  │ • 拓扑映射   │  │ • 错误分析    │  │ • 延迟分析   │   │
│  │              │  │ • PFC检测    │  │ • NCCL基准   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │            │
│         └────────────┬────┴────┬────────────┘            │
│                      │         │                         │
│              ┌───────▼─────────▼───────┐                 │
│              │    Visualization         │                 │
│              │    可视化模块             │                 │
│              │                          │                 │
│              │ • HTML/JSON报告          │                 │
│              │ • 拓扑图 (DOT/ASCII)     │                 │
│              └──────────────────────────┘                 │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  监控层: Prometheus + Grafana + Alertmanager             │
└──────────────────────────────────────────────────────────┘
```

## 功能模块

### 1. 设备发现 (Discovery)
- **RDMA设备扫描器**: 扫描集群所有节点的RDMA设备(IB/RoCE)，收集HCA信息、端口状态、固件版本
- **拓扑映射器**: 发现网络拓扑结构，识别Fat-Tree/Spine-Leaf架构，计算节点间距离

### 2. 健康诊断 (Diagnosis)
- **链路健康检查**: 检测端口错误计数器、FEC状态、线缆/光模块健康
- **错误趋势分析**: 分析错误计数器时间序列，检测增长趋势，推断故障根因
- **PFC死锁检测**: 监控RoCE部署中的PFC帧行为，检测风暴和死锁模式

### 3. 性能基准测试 (Benchmark)
- **带宽测试**: 使用ib_write_bw/ib_read_bw进行全对带宽测试
- **延迟分析**: 使用ib_write_lat/ib_read_lat测量P50/P99/P999延迟
- **NCCL基准**: 封装nccl-tests测试AllReduce/AllGather等集合操作性能

### 4. 可视化 (Visualization)
- **报告生成器**: 生成HTML/JSON格式的综合诊断报告
- **拓扑可视化**: 生成DOT/ASCII格式的网络拓扑图

## 项目结构

```
cluster-network-diagnostics/
├── config.yaml                          # 默认配置
├── requirements.txt                     # Python依赖
├── README.md
├── deploy/
│   ├── Dockerfile                       # 诊断Agent容器
│   └── docker-compose.yaml              # 完整部署（含Prometheus/Grafana）
├── src/
│   ├── discovery/
│   │   ├── rdma_device_scanner.py       # RDMA设备扫描
│   │   └── topology_mapper.py           # 拓扑映射
│   ├── diagnosis/
│   │   ├── link_health_checker.py       # 链路健康检查
│   │   ├── error_analyzer.py            # 错误趋势分析
│   │   └── pfc_deadlock_detector.py     # PFC死锁检测
│   ├── benchmark/
│   │   ├── bandwidth_tester.py          # RDMA带宽测试
│   │   ├── latency_profiler.py          # RDMA延迟分析
│   │   └── nccl_benchmark.py            # NCCL集合通信基准
│   └── visualization/
│       ├── report_generator.py          # HTML/JSON报告
│       └── topology_visualizer.py       # 拓扑可视化
└── tests/
    ├── test_discovery.py
    ├── test_diagnosis.py
    └── test_benchmark.py
```

## 快速开始

### 环境要求
- Python 3.8+
- Mellanox OFED 用户空间工具 (ibverbs-utils, infiniband-diags, perftest)
- SSH免密访问集群所有节点
- (可选) NCCL和nccl-tests
- (可选) Docker + Docker Compose

### 安装

```bash
# 安装Python依赖
pip install -r requirements.txt

# 编辑配置文件
cp config.yaml my_cluster.yaml
vim my_cluster.yaml  # 填入实际的节点信息
```

### 使用示例

```python
import yaml
from src.discovery.rdma_device_scanner import RDMADeviceScanner
from src.discovery.topology_mapper import TopologyMapper
from src.diagnosis.link_health_checker import LinkHealthChecker
from src.benchmark.bandwidth_tester import BandwidthTester
from src.visualization.report_generator import ReportGenerator

# 加载配置
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 1. 扫描RDMA设备
scanner = RDMADeviceScanner(config)
scan_results = scanner.scan_cluster()
scanner.export_inventory("inventory.json")

# 2. 发现网络拓扑
mapper = TopologyMapper(config)
topology = mapper.discover_topology()
print(mapper.export_dot())

# 3. 检查链路健康
checker = LinkHealthChecker(config)
health_report = checker.check_cluster()

# 4. 运行带宽基准测试
bw_tester = BandwidthTester(config)
bw_results = bw_tester.test_all_pairs()

# 5. 生成综合报告
report_gen = ReportGenerator(config)
report_gen.set_discovery_results(scanner.get_cluster_summary())
report = report_gen.build_report()
report_gen.save_report(report)
```

### Docker部署

```bash
cd deploy
docker-compose up -d

# 访问Grafana面板
# http://localhost:3000 (admin / cluster-diag-2024)
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_discovery.py -v
python -m pytest tests/test_diagnosis.py -v
python -m pytest tests/test_benchmark.py -v
```

## 覆盖的Lab知识点

| Lab | 主题 | 在本项目中的体现 |
|-----|------|-----------------|
| Lab 01 | RDMA基础 | 设备扫描、ibstat解析 |
| Lab 02 | InfiniBand架构 | 拓扑映射、交换机层级识别 |
| Lab 03 | RoCE配置 | PFC检测、GID Index配置 |
| Lab 04 | 网络拓扑 | Fat-Tree/Spine-Leaf识别 |
| Lab 05 | 性能调优 | 带宽/延迟基准测试 |
| Lab 06 | NCCL | nccl-tests封装、总线带宽分析 |
| Lab 07 | 错误诊断 | 错误计数器分析、根因定位 |
| Lab 08 | 流量控制 | PFC死锁检测、ECN监控 |
| Lab 09 | 网络监控 | Prometheus指标、Grafana面板 |
| Lab 10 | 故障排除 | 综合诊断报告、运维建议 |

## 技术要点

- 使用 `dataclass` 和类型注解定义清晰的数据结构
- 通过 `subprocess` 调用IB诊断工具（ibstat、perfquery、ib_write_bw等）
- 线性回归分析错误趋势，加权投票法进行根因分析
- 并发扫描/测试使用 `ThreadPoolExecutor`
- HTML报告使用内联CSS，无外部依赖
- DOT格式拓扑图可用Graphviz渲染
