# GPU Cluster Scheduler

面向多租户 GPU 集群的智能调度系统，支持拓扑感知调度、Bin Packing、
公平共享配额、抢占策略，以及 GPU 健康管理。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Cluster Scheduler                      │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │   REST API   │   │  Job Queue   │   │  Cluster Monitor │ │
│  │  job_api.py  │   │   (Redis)    │   │ utilization_     │ │
│  │ cluster_api  │   │              │   │ tracker.py       │ │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘ │
│         │                   │                     │           │
│         ▼                   ▼                     ▼           │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                  GPU Scheduler Core                       ││
│  │                                                           ││
│  │  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐ ││
│  │  │topology_aware  │  │ bin_packing  │  │ preemption   │ ││
│  │  │NVLink感知调度  │  │ GPU集中分配   │  │ 优先级抢占   │ ││
│  │  └────────────────┘  └──────────────┘  └──────────────┘ ││
│  └──────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌──────────────────┐   ┌──────────────────────────────────┐│
│  │  Resource Layer  │   │    Tenant Management             ││
│  │ gpu_discovery.py │   │  quota_manager.py                ││
│  │ node_manager.py  │   │  fair_share.py                   ││
│  └──────────────────┘   └──────────────────────────────────┘│
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │              K8s Client Layer (client.py)                 ││
│  │  Watch Nodes/Pods │ Bind Pods │ Taint/Label Nodes        ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
gpu-cluster-scheduler/
├── src/
│   ├── scheduler/           # 调度核心
│   │   ├── gpu_scheduler.py     # 主调度循环
│   │   ├── topology_aware.py    # NVLink 拓扑感知
│   │   ├── bin_packing.py       # Bin Packing 策略
│   │   └── preemption.py        # 抢占逻辑
│   ├── resource/            # 资源管理
│   │   ├── gpu_discovery.py     # GPU 发现与拓扑
│   │   ├── node_manager.py      # 节点状态管理
│   │   └── utilization_tracker.py # 利用率追踪
│   ├── tenant/              # 租户管理
│   │   ├── quota_manager.py     # 配额管理
│   │   └── fair_share.py        # 公平共享算法
│   ├── k8s/                 # K8s 交互
│   │   └── client.py           # K8s API 封装
│   └── api/                 # HTTP API
│       ├── server.py            # FastAPI server
│       ├── job_api.py           # Job CRUD
│       └── cluster_api.py       # 集群信息
├── tests/
│   └── test_scheduler.py
├── deploy/
│   └── helm/
│       └── values.yaml
├── Dockerfile
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动（连接到 K8s 集群）
python -m src.api.server --kubeconfig=$HOME/.kube/config

# 3. 提交 GPU Job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "training-job-1",
    "gpu_count": 4,
    "tenant": "team-a",
    "priority": 5,
    "image": "pytorch-training:latest"
  }'

# 4. 查看集群状态
curl http://localhost:8000/api/v1/cluster/status
```

## 部署到 K8s

```bash
# 使用 Helm
helm install gpu-scheduler deploy/helm/ \
  --namespace kube-system \
  --set scheduler.replicas=2

# 使用 Docker
docker build -t gpu-scheduler:latest .
docker run -v $HOME/.kube/config:/root/.kube/config gpu-scheduler:latest
```

## 配置

关键配置项见 `deploy/helm/values.yaml`：

- **调度策略**：Bin Packing vs Spread、拓扑感知权重
- **租户配额**：每个团队的 GPU 配额和借用限制
- **抢占策略**：优先级阈值、宽限期
- **健康检查**：GPU 故障阈值、检查间隔
