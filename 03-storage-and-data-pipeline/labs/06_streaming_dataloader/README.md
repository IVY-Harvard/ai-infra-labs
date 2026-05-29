# Lab 06：流式数据加载

## 实验目标

实现 WebDataset 和 StreamingDataset 两种流式数据加载方案，对比其吞吐性能。

## 实验内容

### 实验 1：WebDataset 数据创建与加载
将散落的小文件打成 tar shard，使用 WebDataset 流式加载。

### 实验 2：StreamingDataset 数据创建与加载
将数据转换为 MDS 格式，使用 StreamingDataset 加载。

### 实验 3：吞吐量对比测试
对比原生 DataLoader / WebDataset / StreamingDataset 的吞吐。

## 运行方式

```bash
pip install webdataset mosaicml-streaming torch torchvision numpy

# WebDataset 演示
python webdataset_demo.py --data-dir /tmp/wds_demo --num-samples 10000

# StreamingDataset 演示
python streaming_dataset_demo.py --data-dir /tmp/streaming_demo --num-samples 10000

# 吞吐量对比
python throughput_benchmark.py --num-samples 50000 --batch-size 64
```

## 文件列表

- `webdataset_demo.py` — WebDataset 创建与加载
- `streaming_dataset_demo.py` — StreamingDataset 创建与加载
- `throughput_benchmark.py` — 吞吐量基准测试
