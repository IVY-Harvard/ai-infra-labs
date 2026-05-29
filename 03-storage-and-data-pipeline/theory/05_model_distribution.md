# 05 — 模型分发策略

## 为什么模型分发是工程难题

当你只有一台 8 卡机器时，模型分发看似简单：把文件放在本地 SSD 就行。但真实生产场景要复杂得多：

```
场景 1：模型更新上线
  - 新版本 70B 模型 (140GB) 需要部署到 20 台推理节点
  - 要求：5 分钟内全部节点加载完成
  - 挑战：20 × 140GB = 2.8TB 数据需要分发

场景 2：多团队共享模型
  - 算法团队产出新模型 → 推理团队部署
  - 同一模型被 5 个团队各自下载到各自 NFS
  - 浪费：5 份重复数据 = 700GB × 5 = 3.5TB

场景 3：多集群同步
  - 训练集群产出 Checkpoint → 评估集群加载评测
  - 跨机房传输 420GB Checkpoint
  - 延迟：10Gbps 专线也需要 5+ 分钟

关键指标：
- 分发延迟（TTFB, Time To First Byte）
- 全量加载时间（从文件到 GPU 就绪）
- 带宽利用率
- 存储空间利用率
```

## 分发架构：集中式 vs P2P

### 集中式分发（传统方式）

```
架构：所有节点从中央存储拉取

           ┌─────────────┐
           │  中央存储     │
           │  NFS/S3      │
           └──────┬───────┘
        ┌─────────┼─────────────┐
        │         │             │
        ▼         ▼             ▼
    ┌───────┐ ┌───────┐   ┌───────┐
    │Node 1 │ │Node 2 │   │Node N │
    └───────┘ └───────┘   └───────┘

优点：
  ✓ 简单，无需额外组件
  ✓ 一致性好管理
  ✓ 适合少量节点

缺点：
  ✗ 中央存储带宽瓶颈：N 节点争抢同一出口带宽
  ✗ 线性扩展差：节点数翻倍 → 分发时间翻倍
  ✗ 单点故障

带宽瓶颈计算：
  - 中央存储出口带宽：25Gbps (3.1GB/s)
  - 20 节点同时拉取 140GB 模型
  - 每节点分得带宽：3.1GB/s ÷ 20 = 155MB/s
  - 加载时间：140GB / 155MB/s = 900s = 15 分钟
```

### P2P 分发（BitTorrent-like）

```
架构：节点间互相传输数据块

    ┌───────┐     ┌───────┐
    │Node 1 │◄───►│Node 2 │
    └───┬───┘     └───┬───┘
        │    ╲   ╱    │
        │     ╲ ╱     │
        │      ╳      │
        │     ╱ ╲     │
        │    ╱   ╲    │
    ┌───▼───┐     ┌───▼───┐
    │Node 3 │◄───►│Node 4 │
    └───────┘     └───────┘
         △
         │
    ┌────┴────┐
    │  种子    │  (初始数据源)
    │  Seeder  │
    └─────────┘

工作原理：
1. 将模型文件切分为小块（如 64MB 一块）
2. 种子节点（Seeder）拥有全部块
3. 各节点从种子或其他节点下载不同的块
4. 下载完的块立即可供其他节点下载
5. 随着加入的节点增多，总可用带宽线性增长

优势分析：
  - 种子带宽 3.1GB/s → 20 节点同时下载
  - Node 1 下载了 Block A → 可以分享给 Node 2-20
  - 理论总带宽 ≈ 种子带宽 × log2(节点数)
  - 实际效果：15 分钟 → 3-5 分钟
```

### 混合方案（推荐）

```
架构：分层缓存 + P2P + CDN 思路

┌─────────────────────────────────────────────────┐
│ Layer 1: 本地 NVMe SSD 缓存                      │
│   命中 → 直接加载（秒级）                         │
├─────────────────────────────────────────────────┤
│ Layer 2: 同机架/同 VLAN 节点 P2P                  │
│   同机架节点有缓存 → 内网 P2P 传输（10-30s）      │
├─────────────────────────────────────────────────┤
│ Layer 3: 内网 CDN / Alluxio 集群缓存              │
│   集群内有缓存 → 从缓存节点拉取（1-3min）          │
├─────────────────────────────────────────────────┤
│ Layer 4: 中央对象存储 (S3/MinIO)                  │
│   终极数据源（3-15min）                           │
└─────────────────────────────────────────────────┘

决策逻辑：
if 本地 SSD 有缓存:
    return load_from_local_ssd()  # <5s
elif 同机架节点有缓存:
    return p2p_transfer(peer_node)  # 10-30s
elif Alluxio 缓存命中:
    return load_from_alluxio()  # 1-3min
else:
    return download_from_s3()  # 3-15min
```

## 多级缓存加载

### 设计原则

```
原则 1：最近原则 — 数据尽量靠近 GPU
  本地 SSD > 本机内存 > 同机架 > 跨机架 > 跨机房

原则 2：预加载 — 在需要之前把数据准备好
  调度器知道下一个任务需要什么模型 → 提前预热

原则 3：共享 — 多个任务用同一模型时共享缓存
  多个推理实例加载同一模型 → 用同一份缓存

原则 4：分层淘汰 — 热数据留在快层，冷数据下沉
  最近 24h 使用过 → NVMe
  最近 7 天使用过 → SATA SSD
  超过 7 天 → 远端存储
```

### 缓存 Key 设计

```python
# 模型缓存的唯一标识
cache_key = f"{model_name}/{version}/{format}/{quantization}"

# 示例
"llama-2-70b/v1.2/safetensors/fp16"
"llama-2-70b/v1.2/gguf/q4_k_m"
"mistral-7b/v0.3/bin/bf16"

# 缓存目录结构
/nvme/model-cache/
├── llama-2-70b/
│   ├── v1.2/
│   │   ├── safetensors/fp16/
│   │   │   ├── model-00001-of-00015.safetensors
│   │   │   ├── ...
│   │   │   └── .cache_meta.json  # 缓存元信息
│   │   └── gguf/q4_k_m/
│   │       └── llama-2-70b-q4_k_m.gguf
│   └── v1.1/  # 旧版本
└── mistral-7b/
    └── v0.3/
```

## 模型预加载/预热

### 预加载调度器

```
预加载触发时机：
1. 任务调度时：调度器分配任务到节点 → 同时发送预加载指令
2. 模型发布时：新模型注册 → 向推理集群推送预热指令
3. 定时预热：预测高峰时段需要的模型 → 提前加载
4. 被动缓存：首次请求触发加载 → 后续请求命中缓存

预加载 vs 冷启动 对比：
┌─────────────────┬──────────────────┬──────────────────┐
│                 │ 无预加载（冷启动）│ 有预加载          │
├─────────────────┼──────────────────┼──────────────────┤
│ 模型在远端存储   │ 下载 + 加载      │ 下载 + 加载      │
│ 140GB, 10Gbps   │ ~120s            │ 0s（已预加载）   │
├─────────────────┼──────────────────┼──────────────────┤
│ 模型在本地 SSD   │ 加载到 GPU       │ 加载到 GPU       │
│ 140GB, NVMe     │ ~40s             │ 0s（已在 GPU）   │
├─────────────────┼──────────────────┼──────────────────┤
│ 首次推理延迟     │ 120-160s         │ <1s              │
└─────────────────┴──────────────────┴──────────────────┘
```

## 模型格式与加载速度

### 格式对比

```
┌──────────────┬──────────────┬──────────┬──────────────────────┐
│ 格式          │ 典型用途      │ 加载速度  │ 特点                  │
├──────────────┼──────────────┼──────────┼──────────────────────┤
│ PyTorch .bin │ 训练/通用     │ 慢       │ pickle 序列化          │
│              │              │          │ 安全风险（任意代码执行）│
├──────────────┼──────────────┼──────────┼──────────────────────┤
│ safetensors  │ HuggingFace  │ 快       │ 内存映射（mmap）       │
│              │ 推理/训练    │ 2-5x ↑   │ 安全（无代码执行）     │
│              │              │          │ 支持部分加载          │
├──────────────┼──────────────┼──────────┼──────────────────────┤
│ GGUF         │ llama.cpp    │ 极快      │ 专为推理优化           │
│              │ 量化推理     │          │ 内含量化信息           │
│              │              │          │ 单文件完整模型         │
├──────────────┼──────────────┼──────────┼──────────────────────┤
│ ONNX         │ 跨框架部署   │ 中       │ 跨平台兼容             │
│              │              │          │ 图优化                │
├──────────────┼──────────────┼──────────┼──────────────────────┤
│ TensorRT     │ NVIDIA 推理  │ 最快      │ GPU 专用优化           │
│ Engine       │              │          │ 与硬件绑定             │
└──────────────┴──────────────┴──────────┴──────────────────────┘
```

### safetensors 为什么快

```python
# PyTorch .bin 加载流程：
# 1. 读取文件到内存 (全量读)
# 2. pickle 反序列化 (CPU 密集)
# 3. 创建 tensor 对象
# 4. 拷贝到 GPU
# 总计：文件 IO + CPU 反序列化 + 内存拷贝

# safetensors 加载流程：
# 1. 读取文件头（JSON，几 KB）→ 获取 tensor 偏移量
# 2. mmap 文件（零拷贝映射到内存）
# 3. 按需读取具体 tensor（按偏移量直接定位）
# 4. 直接传输到 GPU（可以 DMA）
# 总计：几乎只有文件 IO

import safetensors
from safetensors.torch import load_file, save_file
import torch
import time

# 加载速度对比
def benchmark_load(filepath):
    """对比 .bin vs .safetensors 加载速度"""
    
    # PyTorch .bin
    t0 = time.time()
    state_dict = torch.load(filepath + ".bin", map_location="cpu")
    bin_time = time.time() - t0
    
    # safetensors
    t0 = time.time()
    state_dict = load_file(filepath + ".safetensors")
    st_time = time.time() - t0
    
    print(f".bin: {bin_time:.2f}s | .safetensors: {st_time:.2f}s | "
          f"speedup: {bin_time/st_time:.1f}x")


# safetensors 支持部分加载（只加载需要的 tensor）
def load_specific_layers(filepath, layer_names):
    """只加载指定层，避免读取整个文件"""
    from safetensors import safe_open
    
    tensors = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for name in layer_names:
            tensors[name] = f.get_tensor(name)
    return tensors
```

### 格式转换实践

```python
from safetensors.torch import save_file, load_file
import torch


def convert_bin_to_safetensors(bin_path, output_path):
    """将 PyTorch .bin 转换为 safetensors"""
    state_dict = torch.load(bin_path, map_location="cpu", 
                            weights_only=True)
    
    # safetensors 不支持非 tensor 类型
    tensor_dict = {
        k: v for k, v in state_dict.items() 
        if isinstance(v, torch.Tensor)
    }
    
    save_file(tensor_dict, output_path)
    
    # 验证
    loaded = load_file(output_path)
    for key in tensor_dict:
        assert torch.equal(tensor_dict[key], loaded[key])
    
    print(f"Converted: {bin_path} -> {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024**3:.2f} GB")
```

## 生产级模型分发系统设计

```
┌─────────────────────────────────────────────────────────────┐
│                    模型注册中心 (Registry)                    │
│   管理模型元信息、版本、格式、存储位置                         │
├─────────────────────────────────────────────────────────────┤
│                    分发引擎 (Distributor)                     │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────┐    │
│   │ 调度策略  │  │ P2P Transfer │  │ 预热管理器        │    │
│   │          │  │ BitTorrent   │  │ Prewarmer         │    │
│   └──────────┘  └──────────────┘  └──────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    缓存层 (Cache Layer)                       │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────┐    │
│   │ 本地 SSD │  │ 节点间缓存    │  │ CDN 缓存          │    │
│   │ (L1)     │  │ (L2)         │  │ (L3)             │    │
│   └──────────┘  └──────────────┘  └──────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    存储后端 (Storage Backend)                 │
│   S3 / MinIO / JuiceFS / NFS                                │
└─────────────────────────────────────────────────────────────┘
```

## 本章小结

- 模型分发的核心挑战是大文件 + 多节点 + 低延迟
- P2P 分发在多节点场景下显著优于集中式拉取
- 多级缓存（本地SSD→节点缓存→远端）是降低加载延迟的关键
- 预加载/预热是消除冷启动延迟的必备手段
- safetensors 格式应作为默认选择：安全、快速、支持部分加载
- 生产环境需要模型注册中心 + 分发引擎 + 缓存层的完整体系

## 延伸阅读

- [safetensors 文档](https://huggingface.co/docs/safetensors/)
- [GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [BitTorrent 协议原理](https://www.bittorrent.org/beps/bep_0003.html)
- [Dragonfly：P2P 文件分发系统](https://d7y.io/)
