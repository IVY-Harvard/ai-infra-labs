# 03 — AI 数据缓存策略

## 缓存为什么是 AI 存储的核心

AI 训练的一个关键特征是**重复访问**：
- 同一个模型文件被反复加载（每次实验、每次调试）
- 训练数据每个 epoch 遍历一次（多 epoch 训练）
- Checkpoint 恢复时读取最近的快照

这意味着大部分数据是「热数据」，如果能缓存在离 GPU 最近的地方，就能大幅减少远端存储的访问延迟。

```
延迟对比（读取 1GB 数据）：
┌─────────────────────────────────────────┐
│ GPU HBM 内存        →  0.001s (1ms)     │
│ 本地 DDR5 内存      →  0.3s             │
│ 本地 NVMe SSD       →  0.3s             │
│ 本地 SATA SSD       →  1.0s             │
│ 本地 HDD            →  8.0s             │
│ 网络存储 (10GbE)    →  1.0s             │
│ 对象存储 (跨区域)   →  5-30s            │
└─────────────────────────────────────────┘
差距可达 1000 倍！
```

## 多级缓存架构设计

### 四级缓存模型

```
┌─────────────────────────────────────────────────────┐
│ L1: GPU/CPU 内存缓存                                 │
│     容量: 10-100GB    延迟: <1ms    命中率目标: 70%+  │
│     实现: PyTorch DataLoader prefetch,               │
│           内存映射(mmap), Pin Memory                  │
├─────────────────────────────────────────────────────┤
│ L2: 本地 NVMe SSD 缓存                               │
│     容量: 500GB-4TB   延迟: <1ms    命中率目标: 95%+  │
│     实现: JuiceFS cache, Alluxio Worker local,       │
│           手动 rsync                                  │
├─────────────────────────────────────────────────────┤
│ L3: 节点间缓存 / 内网存储                             │
│     容量: 10-100TB    延迟: 1-10ms  命中率目标: 99%+  │
│     实现: Alluxio 集群, NFS, 内网 MinIO               │
├─────────────────────────────────────────────────────┤
│ L4: 远端对象存储（数据源）                             │
│     容量: 无限        延迟: 10-100ms                  │
│     实现: S3, OSS, COS, GCS                          │
└─────────────────────────────────────────────────────┘
```

### 各级缓存的典型配置

```python
# L1: PyTorch DataLoader 内存预取配置
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,          # 8 个 Worker 并行预取
    prefetch_factor=4,      # 每个 Worker 预取 4 个 batch
    pin_memory=True,        # 固定内存，加速 GPU 传输
    persistent_workers=True # Worker 常驻，避免重复创建
)
# 内存占用 ≈ 8 workers × 4 prefetch × 32 batch × 样本大小

# L2: JuiceFS 本地 SSD 缓存
# juicefs mount ... --cache-dir /nvme/cache --cache-size 500000

# L3: Alluxio 集群缓存
# alluxio.user.file.readtype.default=CACHE
# alluxio.worker.memory.size=64GB
```

## Alluxio 架构与缓存策略

### Alluxio 是什么

Alluxio 是一个数据编排平台，位于计算框架和存储系统之间：

```
┌─────────────────────────────────────────────────┐
│ 计算层：PyTorch / Spark / Flink / Ray            │
└───────────────────────┬─────────────────────────┘
                        │ POSIX / S3 / HDFS API
┌───────────────────────▼─────────────────────────┐
│                   Alluxio                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Master   │  │ Worker   │  │ Worker       │  │
│  │ (元数据)  │  │ (缓存节点)│  │ (缓存节点)   │  │
│  │          │  │ SSD+RAM  │  │ SSD+RAM      │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└───────────────────────┬─────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────┐
│ 存储层：S3 / HDFS / NFS / MinIO / Ceph           │
└─────────────────────────────────────────────────┘

核心价值：
1. 统一命名空间：多个存储后端挂载到统一路径
2. 智能缓存：热数据缓存到 Worker 节点的 SSD/内存
3. 数据本地性：尽量把数据缓存到计算节点本地
4. 策略灵活：支持多种读写缓存策略
```

### Alluxio 缓存策略详解

```
读策略（ReadType）：
┌──────────────────┬──────────────────────────────────────────┐
│ CACHE            │ 读数据时缓存到 Alluxio（默认）            │
│                  │ 适合：大多数 AI 训练场景                   │
├──────────────────┼──────────────────────────────────────────┤
│ CACHE_PROMOTE    │ 读取时将数据提升到更高层（如 SSD→内存）    │
│                  │ 适合：反复读取的热数据                     │
├──────────────────┼──────────────────────────────────────────┤
│ NO_CACHE         │ 直接从底层存储读取，不缓存                │
│                  │ 适合：一次性扫描的冷数据                   │
└──────────────────┴──────────────────────────────────────────┘

写策略（WriteType）：
┌──────────────────┬──────────────────────────────────────────┐
│ CACHE_THROUGH    │ 同步写入 Alluxio 和底层存储               │
│                  │ 适合：重要数据（Checkpoint）               │
├──────────────────┼──────────────────────────────────────────┤
│ MUST_CACHE       │ 只写入 Alluxio，不写底层存储              │
│                  │ 适合：临时数据（中间结果）                 │
├──────────────────┼──────────────────────────────────────────┤
│ ASYNC_THROUGH    │ 写入 Alluxio 后异步刷到底层存储           │
│                  │ 适合：Checkpoint（低延迟写 + 最终持久化）  │
├──────────────────┼──────────────────────────────────────────┤
│ THROUGH          │ 直接写入底层存储，不缓存                  │
│                  │ 适合：归档数据                            │
└──────────────────┴──────────────────────────────────────────┘

淘汰策略（Eviction）：
┌──────────────────┬──────────────────────────────────────────┐
│ LRU              │ 淘汰最近最少使用的数据（默认）             │
│ LFU              │ 淘汰访问频率最低的数据                    │
│ LRFU             │ LRU + LFU 的混合策略                     │
│ GREEDY           │ 淘汰最大的文件（释放最多空间）             │
└──────────────────┴──────────────────────────────────────────┘
```

### AI 场景下的 Alluxio 最佳配置

```properties
# alluxio-site.properties

# --- Worker 缓存配置 ---
alluxio.worker.memory.size=64GB
alluxio.worker.tieredstore.levels=2

# 第一层：内存（最快，用于最热数据）
alluxio.worker.tieredstore.level0.alias=MEM
alluxio.worker.tieredstore.level0.dirs.path=/dev/shm/alluxio
alluxio.worker.tieredstore.level0.dirs.quota=64GB

# 第二层：SSD（较快，用于热数据）
alluxio.worker.tieredstore.level1.alias=SSD
alluxio.worker.tieredstore.level1.dirs.path=/nvme/alluxio
alluxio.worker.tieredstore.level1.dirs.quota=500GB

# --- 读写策略 ---
alluxio.user.file.readtype.default=CACHE
alluxio.user.file.writetype.default=ASYNC_THROUGH

# --- AI 训练优化 ---
alluxio.user.file.passive.cache.enabled=true
alluxio.user.streaming.reader.chunk.size.bytes=8MB
alluxio.user.local.reader.chunk.size.bytes=8MB
```

## JuiceFS 本地缓存调优

### 缓存配置参数详解

```bash
juicefs mount redis://localhost/1 /mnt/jfs \
  # === 基本缓存配置 ===
  --cache-dir /nvme/jfs-cache       # 缓存目录（用最快的盘）
  --cache-size 500000               # 缓存大小上限 (MB)，500GB
  --free-space-ratio 0.1            # SSD 保留 10% 空间
  
  # === 读缓存优化 ===
  --prefetch 3                      # 顺序读预取 3 个 Block (12MB)
  --cache-partial-only false        # 缓存完整 Block 而非部分
  
  # === 写缓存优化 ===
  --writeback                       # 异步写入模式
  --buffer-size 4096                # 4GB 写缓冲区
  --max-uploads 50                  # 50 个并行上传线程
  --upload-limit 0                  # 不限制上传带宽
  
  # === 缓存有效期 ===
  --cache-ttl 0                     # 0 表示不过期（适合不变的数据集）
  --metacache 3                     # 元数据缓存 3 秒
  --entry-cache 3                   # 目录项缓存 3 秒
  --attr-cache 3                    # 属性缓存 3 秒
```

### 缓存预热策略

```bash
# 方法 1：juicefs warmup（推荐）
# 预热整个模型目录
juicefs warmup /mnt/jfs/models/llama-70b/
# 预热指定文件（支持通配符）
juicefs warmup /mnt/jfs/data/train-*.tar
# 并行预热
juicefs warmup -p 16 /mnt/jfs/models/  # 16 个并行线程

# 方法 2：后台预热脚本
#!/bin/bash
# 在训练任务启动前执行
MODELS=("llama-70b" "llama-13b" "mistral-7b")
for model in "${MODELS[@]}"; do
    juicefs warmup -p 8 "/mnt/jfs/models/${model}/" &
done
wait
echo "All models preloaded to cache"

# 方法 3：定时预热（crontab）
# 每天凌晨 2 点预热明天要用的数据
0 2 * * * juicefs warmup -p 16 /mnt/jfs/data/next_batch/
```

### JuiceFS 缓存监控

```bash
# 实时缓存统计
juicefs stats /mnt/jfs
# 输出示例：
# ---- cache ----
# cache.hit: 156789    # 缓存命中次数
# cache.miss: 234      # 缓存未命中次数
# cache.hitrate: 99.8% # 命中率
# cache.usage: 486GB   # 缓存使用量

# IO 追踪（调试慢操作）
juicefs profile /mnt/jfs --interval 1
# 输出每秒的 IO 操作统计

# Prometheus 指标（生产环境推荐）
# juicefs mount ... --metrics localhost:9567
# 配合 Grafana 仪表板监控
```

## 多级缓存设计模式

### 模式一：单节点多级缓存（你的 8 卡环境）

```
┌─────────────────────────────────────────────────────────┐
│                     GPU 训练进程                          │
│    DataLoader(num_workers=8, prefetch_factor=4)          │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ L1: 系统 Page Cache (内存自动管理, ~128GB 可用)           │
│     Linux 内核自动缓存最近访问的文件页                     │
├─────────────────────────────────────────────────────────┤
│ L2: JuiceFS SSD 缓存 (/nvme/jfs-cache, 500GB)           │
│     显式缓存策略，LRU 淘汰                                │
├─────────────────────────────────────────────────────────┤
│ L3: JuiceFS 后端 (MinIO/S3)                              │
│     完整数据，无限容量                                    │
└─────────────────────────────────────────────────────────┘

优化要点：
1. Page Cache：不要禁用，它是免费的 L1 缓存
2. SSD 缓存：尽量大，容纳当前任务的全部数据
3. 预热：训练前执行 juicefs warmup，确保数据在 L2
```

### 模式二：多节点共享缓存（Alluxio + 对象存储）

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Node 1   │  │ Node 2   │  │ Node 3   │
│ 8×GPU    │  │ 8×GPU    │  │ 8×GPU    │
│ Alluxio  │  │ Alluxio  │  │ Alluxio  │
│ Worker   │  │ Worker   │  │ Worker   │
│ 500GB SSD│  │ 500GB SSD│  │ 500GB SSD│
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
          ┌────────▼────────┐
          │  Alluxio Master │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │    S3 / MinIO   │
          │  (完整数据集)    │
          └─────────────────┘

数据流：
1. Node 1 训练需要 file_A.tar
2. Alluxio Worker 1 检查本地：未命中
3. 查询 Master：Worker 2 有缓存 → 从 Worker 2 拉取（内网传输）
4. 若所有 Worker 都没有 → 从 S3 拉取并缓存到 Worker 1
```

### 模式三：分层缓存（热温冷数据分层）

```
热数据（当前 epoch 训练数据 + 最新模型）
  → NVMe SSD（本地，1TB）
  → 策略：主动预热，不淘汰

温数据（历史 Checkpoint + 近期数据集）
  → SATA SSD 或内网 NFS
  → 策略：LRU 淘汰，30 天保留

冷数据（归档模型 + 历史数据集）
  → S3 对象存储（低频访问层）
  → 策略：自动分层，90 天后转 Glacier

实现：
# JuiceFS 支持多级缓存目录
juicefs mount ... \
  --cache-dir /nvme/hot:/sata-ssd/warm \
  --cache-size 1000000  # 1TB 总缓存
```

## 预读取与预热策略最佳实践

### 训练前预热 Checklist

```bash
#!/bin/bash
# pre_training_warmup.sh — 训练任务启动前执行

set -e

JFS_MOUNT="/mnt/jfs"
MODEL_PATH="${JFS_MOUNT}/models/llama-70b"
DATA_PATH="${JFS_MOUNT}/data/current_training"

echo "=== Step 1: 预热模型文件 ==="
time juicefs warmup -p 16 "${MODEL_PATH}/"
echo "模型预热完成"

echo "=== Step 2: 预热前 N 个 batch 的训练数据 ==="
# 只预热前 100 个 shard，不必全量预热
ls "${DATA_PATH}"/shard-{00000..00099}.tar | \
  xargs -P 16 -I {} juicefs warmup {}
echo "训练数据预热完成"

echo "=== Step 3: 检查缓存状态 ==="
juicefs stats "${JFS_MOUNT}" | grep cache

echo "=== 预热完成，可以启动训练 ==="
```

### 训练中自适应预取

```python
"""在训练循环中实现自适应预取"""
import threading
import subprocess
from collections import deque


class AdaptivePrefetcher:
    """根据训练进度自适应预取下一批数据"""
    
    def __init__(self, data_dir: str, shard_pattern: str, 
                 lookahead: int = 10):
        self.data_dir = data_dir
        self.shard_pattern = shard_pattern
        self.lookahead = lookahead
        self.prefetch_queue = deque()
        self.prefetch_thread = None
    
    def prefetch_next_shards(self, current_shard_idx: int):
        """异步预取未来 N 个 shard"""
        shards_to_prefetch = []
        for i in range(1, self.lookahead + 1):
            shard_path = f"{self.data_dir}/{self.shard_pattern.format(current_shard_idx + i)}"
            shards_to_prefetch.append(shard_path)
        
        def _prefetch():
            for shard in shards_to_prefetch:
                subprocess.run(
                    ["juicefs", "warmup", shard],
                    capture_output=True
                )
        
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            return  # 上一轮预取还没完，跳过
        self.prefetch_thread = threading.Thread(target=_prefetch)
        self.prefetch_thread.start()
```

## Alluxio vs JuiceFS 缓存对比

```
┌─────────────┬──────────────────────┬──────────────────────┐
│ 维度         │ Alluxio              │ JuiceFS 缓存         │
├─────────────┼──────────────────────┼──────────────────────┤
│ 定位         │ 数据编排平台          │ 文件系统内置缓存      │
│ 部署复杂度   │ 高（Master + Worker）│ 低（挂载参数配置）    │
│ 缓存粒度     │ 文件 Block           │ 文件 Block           │
│ 缓存共享     │ ✓（跨节点）          │ ✗（仅本节点）        │
│ 策略灵活性   │ 高（6种读写策略）    │ 中（基本 LRU）       │
│ 适合场景     │ 多节点共享缓存        │ 单节点本地缓存       │
│ 额外开销     │ Java 进程, 内存消耗   │ 几乎无额外开销       │
└─────────────┴──────────────────────┴──────────────────────┘

建议：
- 单节点 8 卡：只用 JuiceFS 缓存即可
- 多节点训练：JuiceFS + Alluxio，利用 Alluxio 跨节点缓存共享
```

## 本章小结

- 多级缓存是 AI 存储性能的关键：内存→SSD→HDD→远端
- JuiceFS 缓存适合单节点场景，配置简单，效果显著
- Alluxio 适合多节点场景，提供跨节点缓存共享
- 预热策略比淘汰策略更重要：训练前主动预热 > 被动缓存
- 监控缓存命中率是调优的起点：目标 > 95%

## 延伸阅读

- [Alluxio 官方文档 - AI/ML 最佳实践](https://docs.alluxio.io/)
- [JuiceFS 缓存管理指南](https://juicefs.com/docs/community/cache/)
- [Linux Page Cache 机制详解](https://www.kernel.org/doc/html/latest/admin-guide/mm/)
