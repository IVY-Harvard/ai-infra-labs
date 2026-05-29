# Lab 03：Alluxio 缓存实战

## 实验目标

部署 Alluxio 并配置多级缓存策略，对比不同缓存策略和预加载策略对 AI 训练数据读取的影响。

## 前置条件

- Docker & Docker Compose
- Java 11+（Alluxio 依赖）
- S3/MinIO 后端已就绪
- 本地 NVMe SSD 可用

## 实验内容

### 实验 1：Alluxio 单节点部署与配置
参考 `alluxio_setup.md` 完成部署。

### 实验 2：缓存策略对比
运行 `cache_policy_demo.py` 对比 CACHE/CACHE_PROMOTE/NO_CACHE 三种读策略。

### 实验 3：预加载策略
运行 `preload_strategy.py` 对比不同预加载方案的效果。

## 运行方式

```bash
pip install alluxiofs numpy

# 测试缓存策略
python cache_policy_demo.py --alluxio-host localhost --alluxio-port 19998

# 测试预加载策略
python preload_strategy.py --alluxio-host localhost --data-path /training-data/
```

## 文件列表

- `alluxio_setup.md` — Alluxio 部署指南
- `cache_policy_demo.py` — 缓存策略对比测试
- `preload_strategy.py` — 预加载策略测试
