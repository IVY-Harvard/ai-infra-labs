# Lab 04：Checkpoint 管理

## 实验目标

实现并对比同步、异步、增量三种 Checkpoint 策略，并搭建 GC 自动清理机制。

## 实验内容

### 实验 1：同步 Checkpoint
基础实现，理解 Checkpoint 对训练的阻塞影响。

### 实验 2：异步 Checkpoint
后台线程写入，最小化训练暂停时间。

### 实验 3：增量 Checkpoint
只保存变化部分，大幅减少存储开销。

### 实验 4：GC 策略
自动清理过期 Checkpoint，避免存储爆满。

## 运行方式

```bash
pip install torch numpy

# 同步 Checkpoint 测试
python sync_checkpoint.py --model-size 100 --save-dir /tmp/ckpt/sync

# 异步 Checkpoint 测试
python async_checkpoint.py --model-size 100 --save-dir /tmp/ckpt/async

# 增量 Checkpoint 测试
python incremental_checkpoint.py --model-size 100 --save-dir /tmp/ckpt/incr

# GC 策略测试
python gc_policy.py --checkpoint-dir /tmp/ckpt --keep-latest 3 --keep-best 5
```

## 关键指标对比

| 策略 | 训练暂停时间 | 存储开销 | 实现复杂度 | 适用场景 |
|------|------------|---------|-----------|---------|
| 同步 | 高（分钟级）| 高 | 低 | 开发调试 |
| 异步 | 低（秒级） | 高 | 中 | 生产训练 |
| 增量 | 低 | 低（节省90%）| 高 | 长时间训练 |

## 文件列表

- `sync_checkpoint.py` — 同步 Checkpoint 实现
- `async_checkpoint.py` — 异步 Checkpoint 实现
- `incremental_checkpoint.py` — 增量 Checkpoint 实现
- `gc_policy.py` — GC 策略实现
