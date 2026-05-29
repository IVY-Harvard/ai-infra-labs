# 04 — Checkpoint 工程化

## Checkpoint 的重要性

Checkpoint 是训练状态的快照。一次 70B 模型训练可能需要数周，如果没有 Checkpoint：
- 硬件故障 → 从头训练（数万美元打水漂）
- 学习率调整 → 无法从中间恢复
- 训练发散 → 无法回滚到好的状态

```
Checkpoint 包含什么：
┌─────────────────────────────────────────────────┐
│ 模型参数（FP16/BF16）      ~140GB (70B 模型)    │
│ 优化器状态（Adam）          ~280GB (2× 参数量)   │
│ 学习率调度器状态             ~1KB                │
│ 随机数种子                   ~1KB                │
│ 训练步数 / Epoch 数         ~1KB                │
│ 数据加载器状态               ~1MB                │
│ 梯度缩放器状态（混合精度）   ~1KB                │
├─────────────────────────────────────────────────┤
│ 总计（70B 模型 + Adam）     ~420GB              │
│ 总计（7B 模型 + Adam）      ~42GB               │
└─────────────────────────────────────────────────┘

8 卡 H20 训练 70B 模型（ZeRO-3 分片后）：
- 每卡存储量：420GB / 8 = 52.5GB
- 所有卡汇聚后总量：420GB
- 每 1000 步写一次，每小时约 2-3 次
```

## 同步 Checkpoint vs 异步 Checkpoint

### 同步 Checkpoint

```
同步 Checkpoint 时间线：
──────┬──────────┬─────────────────────┬──────────┬──────
 训练  │  暂停    │  写入 Checkpoint     │  暂停    │ 训练
 计算  │  AllReduce│  到存储系统          │  同步    │ 继续
──────┴──────────┴─────────────────────┴──────────┴──────
      ← step N-1 →│← Checkpoint 开销  →│← step N →

问题：
- 420GB 写入 NFS (1GB/s)：~420 秒 = 7 分钟！
- 420GB 写入 SSD (3GB/s)：~140 秒 = 2.3 分钟
- 训练完全暂停，GPU 空闲 — 浪费算力
- 每 1000 步暂停一次 = 每小时浪费 5-15 分钟

实现（PyTorch 基础版）：
```

```python
import torch
import torch.distributed as dist
import os
import time


def save_checkpoint_sync(model, optimizer, scheduler, step, path):
    """同步 Checkpoint — 简单但阻塞训练"""
    # 只在 rank 0 保存（或所有 rank 各自保存分片）
    if dist.get_rank() == 0:
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        
        # 写入临时文件再原子重命名（防止写入中断导致文件损坏）
        tmp_path = f"{path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, path)
        print(f"[Rank 0] Checkpoint saved at step {step}")
    
    # 所有 rank 等待 rank 0 完成
    dist.barrier()
```

### 异步 Checkpoint

```
异步 Checkpoint 时间线：
──────┬─────┬──────────┬──────────┬──────────┬──────
 训练  │拷贝 │  训练     │  训练     │  训练     │ 训练
 计算  │到内存│  继续     │  继续     │  继续     │ 继续
──────┴─────┴──────────┴──────────┴──────────┴──────
            │← 后台线程写入存储 →│
            └───────────────────┘

优势：
- 训练暂停时间 = 拷贝到内存的时间（秒级，vs 同步的分钟级）
- GPU 利用率大幅提升
- 后台写入不影响训练

实现关键点：
1. 快速拷贝模型状态到 CPU 内存（pinned memory）
2. 启动后台线程将内存数据写入存储
3. 需要注意内存占用（需要额外一份模型拷贝）
```

```python
import torch
import threading
import shutil
import os
from concurrent.futures import ThreadPoolExecutor


class AsyncCheckpointer:
    """异步 Checkpoint 管理器
    
    核心思路：
    1. 将 GPU 上的模型状态快速拷贝到 CPU pinned memory
    2. 后台线程负责将 CPU 内存中的数据写入磁盘/远端存储
    3. 训练进程不必等待 IO 完成
    """
    
    def __init__(self, save_dir: str, max_concurrent: int = 1):
        self.save_dir = save_dir
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.pending_future = None
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, scheduler, step):
        """异步保存 Checkpoint"""
        # 等待上一次异步写入完成（避免内存爆炸）
        if self.pending_future and not self.pending_future.done():
            print(f"Waiting for previous checkpoint to finish...")
            self.pending_future.result()
        
        # Step 1: 快速拷贝到 CPU（这一步是阻塞的，但很快）
        t0 = time.time()
        cpu_state = {
            'step': step,
            'model_state_dict': {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            },
            'optimizer_state_dict': self._copy_optimizer_state(
                optimizer.state_dict()
            ),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        copy_time = time.time() - t0
        print(f"State copied to CPU in {copy_time:.2f}s")
        
        # Step 2: 后台线程写入存储
        save_path = os.path.join(self.save_dir, f"ckpt_step_{step}.pt")
        self.pending_future = self.executor.submit(
            self._write_to_disk, cpu_state, save_path, step
        )
    
    def _copy_optimizer_state(self, opt_state):
        """深拷贝优化器状态到 CPU"""
        new_state = {'state': {}, 'param_groups': opt_state['param_groups']}
        for k, v in opt_state['state'].items():
            new_state['state'][k] = {
                sk: sv.cpu().clone() if torch.is_tensor(sv) else sv
                for sk, sv in v.items()
            }
        return new_state
    
    @staticmethod
    def _write_to_disk(state_dict, path, step):
        """后台线程：写入磁盘"""
        tmp_path = f"{path}.tmp"
        t0 = time.time()
        torch.save(state_dict, tmp_path)
        os.rename(tmp_path, path)
        write_time = time.time() - t0
        print(f"[Background] Checkpoint step {step} "
              f"written in {write_time:.2f}s")
    
    def wait(self):
        """等待所有挂起的写入完成"""
        if self.pending_future:
            self.pending_future.result()
```

## 增量 Checkpoint

### 为什么需要增量

```
全量 Checkpoint 问题：
- 70B 模型每次写 420GB
- 但模型参数每步变化量 < 0.1%
- 大量重复写入浪费 IO 和存储空间

增量 Checkpoint 思路：
- 第一次：写完整的 Checkpoint（基线）
- 后续：只写变化的部分（增量）
- 恢复时：基线 + 所有增量 = 完整状态

存储节省：
- 全量：每次 420GB × 每小时 3 次 × 24 小时 = 30TB/天
- 增量：420GB(基线) + 每次 5-20GB(增量) × 71 次 ≈ 1.8TB/天
- 节省 ~94% 存储空间
```

```python
import torch
import numpy as np
from typing import Dict, Optional
import os


class IncrementalCheckpointer:
    """增量 Checkpoint
    
    策略：
    1. 保存完整基线 Checkpoint
    2. 后续只保存与基线的 diff（变化的参数）
    3. 恢复时先加载基线，再依次应用增量
    """
    
    def __init__(self, save_dir: str, 
                 change_threshold: float = 1e-6,
                 full_interval: int = 10):
        """
        Args:
            save_dir: 保存目录
            change_threshold: 参数变化阈值（小于此值不保存）
            full_interval: 每 N 次增量后做一次全量
        """
        self.save_dir = save_dir
        self.change_threshold = change_threshold
        self.full_interval = full_interval
        self.baseline_state = None
        self.incremental_count = 0
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, step):
        """保存 Checkpoint（自动决定全量/增量）"""
        current_state = {
            k: v.cpu().clone() 
            for k, v in model.state_dict().items()
        }
        
        if (self.baseline_state is None or 
                self.incremental_count >= self.full_interval):
            # 保存全量基线
            self._save_full(current_state, optimizer, step)
            self.baseline_state = current_state
            self.incremental_count = 0
        else:
            # 保存增量
            self._save_incremental(current_state, optimizer, step)
            self.incremental_count += 1
    
    def _save_full(self, model_state, optimizer, step):
        """保存完整基线"""
        path = os.path.join(self.save_dir, f"full_step_{step}.pt")
        checkpoint = {
            'type': 'full',
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"Full checkpoint saved: {size_mb:.0f}MB at step {step}")
    
    def _save_incremental(self, current_state, optimizer, step):
        """保存增量（只保存变化的参数）"""
        diff = {}
        changed_params = 0
        total_params = 0
        
        for key in current_state:
            total_params += 1
            delta = current_state[key] - self.baseline_state[key]
            
            # 只保存变化超过阈值的参数
            if delta.abs().max().item() > self.change_threshold:
                diff[key] = delta  # 保存差值而非完整参数
                changed_params += 1
        
        path = os.path.join(self.save_dir, f"incr_step_{step}.pt")
        checkpoint = {
            'type': 'incremental',
            'step': step,
            'model_diff': diff,
            'optimizer_state_dict': optimizer.state_dict(),
            'changed_ratio': changed_params / total_params,
        }
        torch.save(checkpoint, path)
        
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"Incremental checkpoint: {size_mb:.0f}MB, "
              f"{changed_params}/{total_params} params changed")
    
    def load(self, step):
        """加载 Checkpoint（自动处理基线+增量）"""
        # 找到最近的全量基线
        full_path = self._find_nearest_full(step)
        checkpoint = torch.load(full_path, weights_only=False)
        model_state = checkpoint['model_state_dict']
        
        # 依次应用增量
        incremental_files = self._find_incrementals_after(
            checkpoint['step'], step
        )
        for incr_path in incremental_files:
            incr = torch.load(incr_path, weights_only=False)
            for key, delta in incr['model_diff'].items():
                model_state[key] += delta
        
        return model_state, checkpoint.get('optimizer_state_dict')
```

## PyTorch DCP（Distributed Checkpoint）

PyTorch 2.0+ 引入了原生的分布式 Checkpoint 支持：

```
传统方式 vs DCP：
┌─────────────────────────────────────────────────────────┐
│ 传统方式（torch.save）：                                  │
│   - 所有 rank 把参数 gather 到 rank 0                     │
│   - rank 0 独自写入完整 Checkpoint                        │
│   - 问题：rank 0 内存不够放完整模型                        │
│   - 问题：其他 rank 空闲等待                               │
│                                                         │
│ PyTorch DCP：                                            │
│   - 每个 rank 独立保存自己的分片                           │
│   - 无需汇聚，分布式并行写入                               │
│   - 支持 resharding：保存时 8 卡，恢复时可以 16 卡         │
│   - 支持异步保存                                          │
└─────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, StateDictOptions
)


def save_with_dcp(model, optimizer, step, checkpoint_dir):
    """使用 PyTorch DCP 保存分布式 Checkpoint"""
    
    # 获取分片状态字典（每个 rank 只获取自己的分片）
    model_state, optimizer_state = get_state_dict(
        model, optimizer,
        options=StateDictOptions(full_state_dict=False)
    )
    
    state_dict = {
        "model": model_state,
        "optimizer": optimizer_state,
        "step": step,
    }
    
    # 分布式写入：每个 rank 并行写自己的分片
    dcp.save(
        state_dict,
        checkpoint_id=f"{checkpoint_dir}/step_{step}",
    )
    
    if dist.get_rank() == 0:
        print(f"DCP checkpoint saved at step {step}")


def load_with_dcp(model, optimizer, checkpoint_dir, step):
    """使用 PyTorch DCP 加载（支持 resharding）"""
    
    # 获取当前模型和优化器的状态字典结构
    model_state, optimizer_state = get_state_dict(
        model, optimizer,
        options=StateDictOptions(full_state_dict=False)
    )
    
    state_dict = {
        "model": model_state,
        "optimizer": optimizer_state,
    }
    
    # DCP 自动处理 resharding
    # 例如：保存时 8 GPU，加载时 16 GPU，DCP 自动重新分片
    dcp.load(
        state_dict,
        checkpoint_id=f"{checkpoint_dir}/step_{step}",
    )
    
    # 将加载的状态设置回模型和优化器
    set_state_dict(
        model, optimizer,
        model_state_dict=state_dict["model"],
        optim_state_dict=state_dict["optimizer"],
        options=StateDictOptions(full_state_dict=False)
    )
```

## Checkpoint 存储开销分析与优化

### 存储开销计算

```
模型规模 vs Checkpoint 大小 vs 每日存储消耗：

┌──────────┬────────────┬──────────────┬───────────────────┐
│ 模型规模  │ 单次 Ckpt  │ 频率(3次/h)  │ 每日存储           │
├──────────┼────────────┼──────────────┼───────────────────┤
│ 7B       │ 42GB       │ 72 次       │ 3.0 TB            │
│ 13B      │ 78GB       │ 72 次       │ 5.6 TB            │
│ 34B      │ 204GB      │ 72 次       │ 14.7 TB           │
│ 70B      │ 420GB      │ 72 次       │ 30.2 TB           │
└──────────┴────────────┴──────────────┴───────────────────┘

一个月训练 70B 模型的 Checkpoint 存储：30TB × 30 = 900TB！
这就是为什么需要 GC（垃圾回收）策略。
```

### 优化手段

```
1. 压缩存储
   - BF16/FP16 代替 FP32（存储量减半）
   - 优化器状态压缩（FP32→FP16，精度损失很小）
   - 文件级压缩（zstd，压缩率约 1.3-1.5x）

2. 减少保存频率
   - 根据 loss 曲线动态调整（loss 下降快时少存）
   - 验证集 loss 改善时才保存

3. 选择性保存
   - 只保存模型参数（不保存优化器状态）→ 减少 2/3
   - 代价：不能恢复训练，只能用于推理

4. 增量保存
   - 只保存变化的参数 → 减少 80-95%
   - 见上文增量 Checkpoint 实现
```

## GC（垃圾回收）策略

```python
import os
import glob
import time
from datetime import datetime, timedelta
from typing import List


class CheckpointGCPolicy:
    """Checkpoint 垃圾回收策略
    
    保留规则（按优先级）：
    1. 始终保留最新 N 个 Checkpoint
    2. 保留验证 loss 最佳的 M 个 Checkpoint
    3. 保留每天/每周的关键节点 Checkpoint
    4. 超过最大保留期的一律删除
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_latest: int = 3,          # 保留最新 3 个
        keep_best: int = 5,            # 保留 loss 最好的 5 个
        keep_daily: int = 7,           # 保留最近 7 天的每日快照
        keep_weekly: int = 4,          # 保留最近 4 周的每周快照
        max_retention_days: int = 30,  # 最多保留 30 天
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_latest = keep_latest
        self.keep_best = keep_best
        self.keep_daily = keep_daily
        self.keep_weekly = keep_weekly
        self.max_retention_days = max_retention_days
    
    def run_gc(self, checkpoints: List[dict]) -> List[str]:
        """执行 GC，返回需要删除的 Checkpoint 路径列表
        
        Args:
            checkpoints: [{"path": str, "step": int, 
                          "val_loss": float, "timestamp": datetime}]
        """
        if not checkpoints:
            return []
        
        # 按时间排序（最新的在前）
        sorted_ckpts = sorted(
            checkpoints, key=lambda x: x['timestamp'], reverse=True
        )
        
        keep_set = set()
        
        # 规则 1：保留最新的 N 个
        for ckpt in sorted_ckpts[:self.keep_latest]:
            keep_set.add(ckpt['path'])
        
        # 规则 2：保留 val_loss 最好的 M 个
        by_loss = sorted(
            [c for c in checkpoints if c.get('val_loss') is not None],
            key=lambda x: x['val_loss']
        )
        for ckpt in by_loss[:self.keep_best]:
            keep_set.add(ckpt['path'])
        
        # 规则 3：保留每日快照
        daily_kept = self._keep_periodic(sorted_ckpts, 
                                          days=1, 
                                          count=self.keep_daily)
        keep_set.update(daily_kept)
        
        # 规则 4：保留每周快照
        weekly_kept = self._keep_periodic(sorted_ckpts, 
                                           days=7, 
                                           count=self.keep_weekly)
        keep_set.update(weekly_kept)
        
        # 确定需要删除的
        now = datetime.now()
        to_delete = []
        for ckpt in checkpoints:
            age_days = (now - ckpt['timestamp']).days
            if (ckpt['path'] not in keep_set or 
                    age_days > self.max_retention_days):
                if ckpt['path'] not in keep_set:
                    to_delete.append(ckpt['path'])
        
        return to_delete
    
    def _keep_periodic(self, sorted_ckpts, days, count):
        """保留每个周期内最早的一个"""
        kept = set()
        now = datetime.now()
        for i in range(count):
            period_start = now - timedelta(days=days * (i + 1))
            period_end = now - timedelta(days=days * i)
            candidates = [
                c for c in sorted_ckpts
                if period_start <= c['timestamp'] < period_end
            ]
            if candidates:
                # 保留该周期内最早的（最具代表性）
                kept.add(candidates[-1]['path'])
        return kept
```

## 本章小结

- 同步 Checkpoint 简单但阻塞训练，异步 Checkpoint 是生产环境的标配
- 增量 Checkpoint 可节省 80-95% 存储空间，适合长时间训练
- PyTorch DCP 是多卡训练的首选方案，原生支持 resharding
- GC 策略必须提前规划：保留最新 + 最优 + 周期快照，其余自动清理
- Checkpoint 优化的目标：最小化训练暂停时间 + 最小化存储开销

## 延伸阅读

- [PyTorch Distributed Checkpoint 文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)
- [DeepSpeed Checkpoint 引擎](https://www.deepspeed.ai/tutorials/pipeline/#saving-and-loading-checkpoints)
- [Nebula: Microsoft 异步 Checkpoint 系统](https://www.microsoft.com/en-us/research/publication/nebula-reliable-low-latency-checkpointing-for-deep-learning/)
