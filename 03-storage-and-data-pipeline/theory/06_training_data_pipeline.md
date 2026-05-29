# 06 — 训练数据管道

## 为什么需要专门的数据管道

你可能习惯了 PyTorch 原生的 DataLoader + Dataset：把数据放在本地磁盘上，Dataset 按索引读取，DataLoader 多进程预取。这在小数据集上工作得很好，但当训练数据扩展到 TB 级别时，会遇到严重问题：

```
问题 1：海量小文件 → 文件系统崩溃
  - 1TB 文本数据 = 数百万个 JSON/TXT 文件
  - ls 一个目录要几十秒（元数据瓶颈）
  - 随机读取 IOPS 远超 NFS 承受能力
  - 每个 epoch 开头的 shuffle 需要读取全部文件列表

问题 2：数据集超过单机磁盘 → 无法本地化
  - 10TB 预训练数据集放不下单块 NVMe
  - 从远端存储流式读取 → 网络带宽成为瓶颈

问题 3：多模态数据 → 格式混杂
  - 文本 + 图片 + 音频 + 视频
  - 不同模态的数据大小差异巨大（1KB 文本 vs 10MB 图片）
  - 需要对齐、配对、统一 Batch

问题 4：数据配比 → 静态 Dataset 不灵活
  - 预训练需要混合多个数据源（网页、书籍、代码、对话）
  - 不同源的配比需要动态调整
  - 每个源的数据量不同，小源需要重复采样
```

## 流式数据加载：核心思想

```
传统 Map-style Dataset：
  数据 → 全部索引化到内存 → 按 index 随机访问
  问题：index 建立慢，内存放不下全量索引

流式 Iterable Dataset：
  数据 → 打成大文件（tar/mds） → 顺序读取 → shuffle buffer
  优势：无需全量索引，顺序 IO 性能高，可跨网络流式读取

对比：
┌──────────────┬──────────────────┬──────────────────────┐
│              │ Map-style        │ Iterable (流式)       │
├──────────────┼──────────────────┼──────────────────────┤
│ 数据格式      │ 独立小文件       │ 打包大文件 (tar/mds)  │
│ 索引方式      │ 全量 index       │ 无需预建索引          │
│ 随机访问      │ ✓               │ ✗（顺序遍历）         │
│ Shuffle      │ 全局 shuffle     │ 局部 shuffle buffer  │
│ 启动速度      │ 慢（建索引）     │ 快（直接读）          │
│ 数据规模      │ < 1TB           │ 无上限               │
│ 远端数据      │ 需全量下载       │ 支持流式读取          │
│ 多 Worker    │ 按 index 分配    │ 按 shard 分配        │
│ 适用场景      │ 小数据集/微调    │ 大规模预训练          │
└──────────────┴──────────────────┴──────────────────────┘
```

## WebDataset：大规模训练的事实标准

### WebDataset 核心概念

WebDataset 将数据打包为标准 tar 文件，每个样本由同名的多个文件组成：

```
数据格式约定：
shard-000000.tar
  ├── sample_00001.jpg     # 图片
  ├── sample_00001.json    # 元数据
  ├── sample_00001.txt     # 文本描述
  ├── sample_00002.jpg
  ├── sample_00002.json
  ├── sample_00002.txt
  └── ...

命名规则：
  {sample_key}.{extension}
  - 相同 sample_key 的文件自动组成一个样本
  - extension 决定数据类型（jpg→图片，txt→文本，json→结构化数据）

Shard 设计：
  - 每个 tar 文件 = 一个 shard
  - 推荐 shard 大小：100MB - 1GB
  - Shard 数量 > Worker 数量 × GPU 数量（确保均匀分配）
  - 例如：8 GPU × 4 Worker = 32 并发 → 至少 128 个 shard
```

### WebDataset 使用示例

```python
import webdataset as wds
import torch
from torch.utils.data import DataLoader


def create_webdataset_pipeline(
    shard_urls: str,       # "s3://bucket/data/shard-{000000..001023}.tar"
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle_buffer: int = 5000,
):
    """创建 WebDataset 流式数据管道"""
    
    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True)
        # Shard 级别 shuffle（每个 epoch 打乱 shard 顺序）
        .shuffle(shuffle_buffer)
        # 样本级别 shuffle（buffer 内随机）
        .decode("pil")
        # 自动解码：jpg→PIL Image, json→dict, txt→str
        .to_tuple("jpg", "json")
        # 指定输出字段顺序
        .map_tuple(
            transform_image,    # 图片增强
            transform_label,    # 标签处理
        )
        .batched(batch_size, partial=False)
        # 组 batch（丢弃不完整的最后一个 batch）
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,        # WebDataset 已经 batch 了
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return dataloader


def transform_image(img):
    """图片预处理"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)


def transform_label(metadata):
    """标签处理"""
    return torch.tensor(metadata["label"], dtype=torch.long)
```

### 创建 WebDataset Shard

```python
import webdataset as wds
import json
import os
from pathlib import Path


def create_shards(
    data_dir: str,
    output_pattern: str,    # "shards/shard-%06d.tar"
    max_shard_size: int = 500_000_000,  # 500MB per shard
):
    """将散落的小文件打成 WebDataset tar shard"""
    
    with wds.ShardWriter(output_pattern, 
                          maxsize=max_shard_size) as sink:
        image_files = sorted(Path(data_dir).glob("*.jpg"))
        
        for img_path in image_files:
            key = img_path.stem  # 文件名（不含扩展名）
            
            # 读取图片
            with open(img_path, "rb") as f:
                image_data = f.read()
            
            # 读取对应的标签文件
            label_path = img_path.with_suffix(".json")
            with open(label_path, "r") as f:
                label_data = json.load(f)
            
            # 写入 shard
            sample = {
                "__key__": key,
                "jpg": image_data,
                "json": label_data,
            }
            sink.write(sample)
    
    print(f"Created shards from {len(image_files)} samples")
```

## MosaicML StreamingDataset

### StreamingDataset vs WebDataset

```
┌───────────────────┬──────────────────────┬──────────────────────┐
│ 特性               │ WebDataset           │ StreamingDataset     │
├───────────────────┼──────────────────────┼──────────────────────┤
│ 数据格式           │ 标准 tar 文件         │ 自定义 MDS 格式       │
│ 远端存储支持       │ S3/GCS/HTTP          │ S3/GCS/OCI/本地      │
│ 确定性 Shuffle     │ 近似（buffer-based） │ 精确（可完全复现）    │
│ Epoch 断点续传     │ 不支持               │ 原生支持             │
│ 多节点去重         │ 需手动分 shard       │ 自动协调             │
│ 样本级随机访问     │ 不支持               │ 支持                 │
│ 数据配比           │ 需自定义             │ 原生 Stream 权重     │
│ 压缩               │ 不支持               │ 支持 (zstd/snappy)  │
│ 适用场景           │ 通用，生态广         │ 大规模预训练         │
└───────────────────┴──────────────────────┴──────────────────────┘

选择建议：
- 已有 tar 格式数据 / HuggingFace 生态 → WebDataset
- 大规模预训练 / 需要精确复现 / 多节点训练 → StreamingDataset
```

### StreamingDataset 使用示例

```python
from streaming import StreamingDataset, StreamingDataLoader
from streaming.base.format.mds.encodings import Encoding
import torch


def create_streaming_dataloader(
    remote_dir: str,       # "s3://bucket/dataset/mds/"
    local_cache: str,      # "/nvme/cache/dataset/"
    batch_size: int = 32,
    num_workers: int = 8,
):
    """创建 StreamingDataset 数据管道"""
    
    dataset = StreamingDataset(
        remote=remote_dir,          # 远端数据源
        local=local_cache,          # 本地缓存目录
        shuffle=True,               # 启用 shuffle
        shuffle_seed=42,            # 确定性 shuffle
        shuffle_block_size=262144,  # shuffle 粒度
        batch_size=batch_size,      # 告知 batch 大小以优化预取
        predownload=8 * batch_size, # 预下载 8 个 batch
        cache_limit="200gb",        # 本地缓存上限
        num_canonical_nodes=8,      # 节点数（影响数据划分）
    )
    
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return dataloader
```

### 创建 MDS 格式数据集

```python
from streaming import MDSWriter
import json
import numpy as np


def write_mds_dataset(
    data_items: list,
    output_dir: str,
    compression: str = "zstd",
):
    """将数据写入 MDS 格式"""
    
    columns = {
        "input_ids": "ndarray:int32",
        "attention_mask": "ndarray:int32",
        "labels": "ndarray:int32",
        "text": "str",
    }
    
    with MDSWriter(
        out=output_dir,
        columns=columns,
        compression=compression,
        size_limit=256 * 1024 * 1024,  # 256MB per shard
    ) as writer:
        for item in data_items:
            sample = {
                "input_ids": np.array(item["input_ids"], dtype=np.int32),
                "attention_mask": np.array(item["attention_mask"], 
                                           dtype=np.int32),
                "labels": np.array(item["labels"], dtype=np.int32),
                "text": item["text"],
            }
            writer.write(sample)
```

## Tokenization 流水线

### 为什么 Tokenization 是瓶颈

```
预训练数据处理流水线：
  原始文本 → 清洗 → 去重 → Tokenization → 打包 → 写入 MDS/tar

Tokenization 的性能瓶颈：
  - 1TB 原始文本 ≈ 2500 亿 token
  - 单线程 Tokenization：~10MB/s → 需要 100000 秒 ≈ 28 小时
  - 这还只是 Tokenization，不含清洗和去重
  - 如果数据处理赶不上训练速度，GPU 就会饿死

加速方案：
  1. 多进程并行：文件级并行 Tokenization
  2. 批量编码：一次编码多条文本
  3. Rust 后端：tiktoken / tokenizers（HuggingFace）底层都是 Rust
  4. 流式处理：不需要全部加载到内存
```

### Tokenization 流水线设计

```python
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import os


def tokenize_file(filepath: str, tokenizer_name: str, 
                  max_length: int = 2048) -> list:
    """对单个文件进行 Tokenization"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )
            results.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "text": text,
            })
    
    return results


def parallel_tokenize(
    input_dir: str,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    num_workers: int = None,
    max_length: int = 2048,
):
    """多进程并行 Tokenization"""
    if num_workers is None:
        num_workers = cpu_count()
    
    files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith((".txt", ".jsonl"))
    ]
    
    worker_fn = partial(
        tokenize_file, 
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, files)):
            output_path = os.path.join(output_dir, f"tokenized_{i:06d}.jsonl")
            with open(output_path, "w") as f:
                for item in result:
                    f.write(json.dumps(item) + "\n")
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(files)} files")
```

## 数据配比（Data Mixing）

### 预训练数据配比策略

```
典型的预训练数据组成（以 Llama-2 为参考）：
┌────────────────┬──────────┬──────────────────────────────┐
│ 数据源          │ 配比     │ 特征                          │
├────────────────┼──────────┼──────────────────────────────┤
│ 网页数据        │ 67%      │ 量大质低，需要严格过滤         │
│ (CommonCrawl)  │          │                              │
├────────────────┼──────────┼──────────────────────────────┤
│ 代码           │ 8%       │ 提升推理和编程能力             │
│ (GitHub)       │          │                              │
├────────────────┼──────────┼──────────────────────────────┤
│ 书籍           │ 8%       │ 长文本理解，知识密度高         │
├────────────────┼──────────┼──────────────────────────────┤
│ 学术论文        │ 5%       │ 科学推理能力                   │
│ (ArXiv)        │          │                              │
├────────────────┼──────────┼──────────────────────────────┤
│ 百科           │ 5%       │ 事实性知识                     │
│ (Wikipedia)    │          │                              │
├────────────────┼──────────┼──────────────────────────────┤
│ 对话/问答       │ 4%       │ 对话能力和指令遵循             │
├────────────────┼──────────┼──────────────────────────────┤
│ 数学            │ 3%       │ 数学推理                      │
└────────────────┴──────────┴──────────────────────────────┘

关键原则：
1. 质量 > 数量：高质量源（书籍/论文）可以过采样
2. 多样性：确保不同领域/语言的覆盖
3. 动态调整：训练中根据 loss 变化调整配比
```

### StreamingDataset 的多源配比

```python
from streaming import StreamingDataset


def create_mixed_dataset(
    cache_dir: str = "/nvme/cache",
):
    """创建多源混合数据集，按权重采样"""
    
    streams = [
        {
            "remote": "s3://data/web/mds/",
            "local": f"{cache_dir}/web/",
            "proportion": 0.67,
        },
        {
            "remote": "s3://data/code/mds/",
            "local": f"{cache_dir}/code/",
            "proportion": 0.08,
        },
        {
            "remote": "s3://data/books/mds/",
            "local": f"{cache_dir}/books/",
            "proportion": 0.08,
        },
        {
            "remote": "s3://data/arxiv/mds/",
            "local": f"{cache_dir}/arxiv/",
            "proportion": 0.05,
        },
        {
            "remote": "s3://data/wiki/mds/",
            "local": f"{cache_dir}/wiki/",
            "proportion": 0.05,
        },
        {
            "remote": "s3://data/qa/mds/",
            "local": f"{cache_dir}/qa/",
            "proportion": 0.04,
        },
        {
            "remote": "s3://data/math/mds/",
            "local": f"{cache_dir}/math/",
            "proportion": 0.03,
        },
    ]
    
    # StreamingDataset 原生支持按 proportion 混合多个 stream
    from streaming import Stream
    stream_objects = [Stream(**s) for s in streams]
    
    dataset = StreamingDataset(
        streams=stream_objects,
        shuffle=True,
        shuffle_seed=42,
        batch_size=32,
    )
    
    return dataset
```

## 多模态数据处理

### 多模态数据的挑战

```
挑战 1：大小差异
  - 文本 token 序列：~2KB
  - 224×224 图片：~150KB
  - 1280×720 视频帧：~2.7MB
  - 同一 batch 内数据大小差 1000 倍

挑战 2：对齐
  - 图文对：一张图片 + 一段描述
  - 视频字幕：视频帧序列 + 时间对齐的文本
  - 对齐关系需要在数据管道中保持

挑战 3：预处理差异
  - 文本：Tokenization
  - 图片：Resize + Normalize + Augmentation
  - 音频：Spectrogram + Normalize

解决方案：WebDataset 天然支持多模态
  同一 sample_key 的多个文件自动关联
```

### 多模态 WebDataset 示例

```python
import webdataset as wds
import torch
from torchvision import transforms
from transformers import AutoTokenizer


def create_multimodal_pipeline(
    shard_urls: str,
    tokenizer_name: str = "openai/clip-vit-base-patch32",
    image_size: int = 224,
    max_text_length: int = 77,
    batch_size: int = 32,
):
    """创建图文多模态数据管道"""
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    
    def process_sample(sample):
        """处理单个图文对样本"""
        image = image_transform(sample["jpg"])
        
        text = sample["txt"].decode("utf-8") if isinstance(
            sample["txt"], bytes) else sample["txt"]
        tokens = tokenizer(
            text,
            max_length=max_text_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }
    
    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True)
        .shuffle(5000)
        .decode("pil")
        .map(process_sample)
        .batched(batch_size, collation_fn=multimodal_collate)
    )
    
    return dataset


def multimodal_collate(batch):
    """多模态 batch 的 collate 函数"""
    return {
        "image": torch.stack([s["image"] for s in batch]),
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
    }
```

## 与大数据框架集成

### Spark 集成：大规模数据预处理

```
使用场景：
  - TB 级原始数据的清洗、去重、格式转换
  - Spark 做重型预处理 → 输出为 WebDataset/MDS → PyTorch 训练消费

流程：
  原始数据 (S3/HDFS)
      │
      ▼
  Spark ETL（清洗、去重、Tokenization）
      │
      ▼
  输出 WebDataset tar / MDS shard (S3)
      │
      ▼
  PyTorch DataLoader 流式读取训练
```

### Ray Data 集成：灵活的数据管道

```python
import ray
from ray.data import read_json, read_parquet


def ray_data_pipeline(input_path: str, output_path: str):
    """使用 Ray Data 构建数据预处理管道"""
    
    # 读取原始数据（自动并行）
    ds = ray.data.read_json(input_path)
    
    # 清洗
    ds = ds.filter(lambda row: len(row["text"]) > 100)
    
    # Tokenization（自动分布式执行）
    ds = ds.map(tokenize_fn, num_cpus=1)
    
    # 写入供训练使用的格式
    ds.write_parquet(output_path)
    
    return ds


def tokenize_fn(row: dict) -> dict:
    """Ray Data map 函数"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    encoded = tokenizer(
        row["text"], max_length=2048, truncation=True, padding=False
    )
    row["input_ids"] = encoded["input_ids"]
    row["attention_mask"] = encoded["attention_mask"]
    return row
```

## 你的 8 卡 H20 环境实践建议

```
数据规模选择方案：

< 100GB 数据（微调场景）：
  → PyTorch 原生 Dataset + DataLoader 足够
  → 数据放本地 NVMe SSD
  → num_workers=8, prefetch_factor=4

100GB - 1TB（中等规模预训练/大规模微调）：
  → WebDataset 打成 tar shard
  → JuiceFS 缓存 + 本地 NVMe SSD
  → 训练前预热到缓存

> 1TB（大规模预训练）：
  → StreamingDataset + S3/MinIO 后端
  → 本地 NVMe 做缓存层
  → 数据预处理用 Ray Data 或 Spark

数据预处理 Checklist：
  1. 把小文件打成 tar/MDS shard（消除元数据瓶颈）
  2. shard 数量 >= 8 GPU × 4 Worker × 4 = 128 个
  3. 每个 shard 100MB - 1GB
  4. Tokenization 用多进程并行（利用全部 CPU 核心）
  5. 预处理结果缓存到本地 NVMe，避免重复计算
```

## 本章小结

- 流式数据加载（WebDataset/StreamingDataset）是大规模训练的必备组件
- WebDataset 使用标准 tar 格式，生态广泛，适合大多数场景
- StreamingDataset 提供精确 shuffle、断点续传和多源配比，适合大规模预训练
- Tokenization 是数据管道的性能瓶颈，需要多进程并行处理
- 多模态数据天然适合 WebDataset 的多文件关联模型
- 数据配比对预训练质量影响巨大，需要精心设计和动态调整

## 延伸阅读

- [WebDataset 文档](https://github.com/webdataset/webdataset)
- [MosaicML StreamingDataset 文档](https://docs.mosaicml.com/projects/streaming/)
- [Ray Data 用户指南](https://docs.ray.io/en/latest/data/data.html)
- [Llama 2 论文中的数据配比](https://arxiv.org/abs/2307.09288)
- [The Pile 数据集设计](https://arxiv.org/abs/2101.00027)
