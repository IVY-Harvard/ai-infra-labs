# Lab 10: 分布式 Checkpoint 与弹性训练

## 目标

1. 实现分布式 checkpoint 保存和加载
2. 实现异步 checkpoint（非阻塞保存）
3. 理解弹性训练（节点变化时自动恢复）

## 文件说明

| 文件 | 说明 |
|------|------|
| `distributed_checkpoint.py` | 分布式 checkpoint: 每 rank 保存自己的分片 |
| `async_checkpoint.py` | 异步 checkpoint: 在后台线程保存，不阻塞训练 |
| `elastic_training_demo.py` | 弹性训练: 处理节点加入/退出 |

## 运行

```bash
torchrun --nproc_per_node=4 distributed_checkpoint.py
torchrun --nproc_per_node=4 async_checkpoint.py
torchrun --nproc_per_node=4 --rdzv_backend=c10d elastic_training_demo.py
```

## 核心概念

```
传统 Checkpoint: rank 0 收集所有参数 → 保存一个大文件
分布式 Checkpoint: 每个 rank 保存自己的分片 → 并行 I/O, 更快

异步 Checkpoint:
  主线程: 继续训练
  后台线程: 将 state_dict 副本写入磁盘
  关键: 需要 copy state_dict 再传给后台线程（避免数据竞争）
```
