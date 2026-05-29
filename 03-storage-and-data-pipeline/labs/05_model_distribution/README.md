# Lab 05：模型分发实战

## 实验目标

实现 P2P 模型分发、多级缓存加载和预加载调度，对比不同加载策略的速度。

## 实验内容

### 实验 1：P2P 模型加载器
实现基于分块传输的 P2P 模型分发机制。

### 实验 2：多级缓存加载
实现 本地SSD → 节点缓存 → 远端存储 的多级加载策略。

### 实验 3：预加载调度器
实现任务感知的模型预加载调度。

### 实验 4：加载速度基准测试
对比不同格式（.bin / safetensors / mmap）和不同来源的加载速度。

## 运行方式

```bash
pip install torch safetensors numpy

# P2P 加载器演示
python p2p_model_loader.py --model-dir /path/to/models --chunk-size 64

# 多级缓存
python multi_level_cache.py --cache-dir /nvme/model-cache --remote s3://models/

# 预加载调度器
python preload_scheduler.py --config scheduler_config.json

# 加载速度测试
python load_speed_benchmark.py --model-path /path/to/model
```

## 文件列表

- `p2p_model_loader.py` — P2P 分块传输加载器
- `multi_level_cache.py` — 多级缓存加载策略
- `preload_scheduler.py` — 模型预加载调度器
- `load_speed_benchmark.py` — 加载速度基准测试
