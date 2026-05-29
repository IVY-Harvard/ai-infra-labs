"""
Checkpoint GC（垃圾回收）策略

保留规则：
1. 始终保留最新 N 个
2. 保留验证 loss 最佳的 M 个
3. 保留每日/每周的节点快照
4. 超出保留期的一律删除

用法：
    python gc_policy.py --checkpoint-dir /tmp/ckpt --keep-latest 3 --keep-best 5
    python gc_policy.py --checkpoint-dir /tmp/ckpt --dry-run  # 仅显示将删除什么
"""

import os
import time
import json
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class CheckpointInfo:
    """Checkpoint 信息"""
    path: str
    step: int
    timestamp: datetime
    size_bytes: int
    val_loss: float = None
    is_full: bool = True


class CheckpointGC:
    """Checkpoint 垃圾回收"""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_latest: int = 3,
        keep_best: int = 5,
        keep_daily: int = 7,
        keep_weekly: int = 4,
        max_retention_days: int = 30,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_latest = keep_latest
        self.keep_best = keep_best
        self.keep_daily = keep_daily
        self.keep_weekly = keep_weekly
        self.max_retention_days = max_retention_days

    def scan_checkpoints(self) -> List[CheckpointInfo]:
        """扫描目录中的所有 Checkpoint"""
        checkpoints = []
        ckpt_dir = Path(self.checkpoint_dir)

        if not ckpt_dir.exists():
            return []

        for f in ckpt_dir.rglob("*.pt"):
            # 解析文件名提取 step
            name = f.stem
            step = 0
            is_full = True

            if "step_" in name:
                try:
                    step = int(name.split("step_")[1].split(".")[0])
                except (ValueError, IndexError):
                    pass

            if name.startswith("incr_"):
                is_full = False

            # 读取元数据（如果有 .meta.json）
            meta_path = f.with_suffix(".meta.json")
            val_loss = None
            if meta_path.exists():
                with open(meta_path) as mf:
                    meta = json.load(mf)
                    val_loss = meta.get("val_loss")

            stat = f.stat()
            checkpoints.append(CheckpointInfo(
                path=str(f),
                step=step,
                timestamp=datetime.fromtimestamp(stat.st_mtime),
                size_bytes=stat.st_size,
                val_loss=val_loss,
                is_full=is_full,
            ))

        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    def compute_gc_plan(self, checkpoints: List[CheckpointInfo]) -> Dict:
        """计算 GC 计划"""
        if not checkpoints:
            return {"keep": [], "delete": [], "freed_bytes": 0}

        keep_set: Set[str] = set()
        keep_reasons: Dict[str, List[str]] = {}

        def mark_keep(ckpt: CheckpointInfo, reason: str):
            keep_set.add(ckpt.path)
            if ckpt.path not in keep_reasons:
                keep_reasons[ckpt.path] = []
            keep_reasons[ckpt.path].append(reason)

        # 按时间排序（最新优先）
        by_time = sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

        # 规则 1：保留最新 N 个
        for ckpt in by_time[:self.keep_latest]:
            mark_keep(ckpt, f"latest-{self.keep_latest}")

        # 规则 2：保留 val_loss 最佳的 M 个
        with_loss = [c for c in checkpoints if c.val_loss is not None]
        by_loss = sorted(with_loss, key=lambda x: x.val_loss)
        for ckpt in by_loss[:self.keep_best]:
            mark_keep(ckpt, f"best-loss({ckpt.val_loss:.4f})")

        # 规则 3：保留每日快照
        now = datetime.now()
        for day_offset in range(self.keep_daily):
            day_start = now - timedelta(days=day_offset + 1)
            day_end = now - timedelta(days=day_offset)
            day_ckpts = [
                c for c in checkpoints
                if day_start <= c.timestamp < day_end
            ]
            if day_ckpts:
                # 保留当天最晚的一个
                mark_keep(day_ckpts[0], f"daily-{day_offset+1}d")

        # 规则 4：保留每周快照
        for week_offset in range(self.keep_weekly):
            week_start = now - timedelta(weeks=week_offset + 1)
            week_end = now - timedelta(weeks=week_offset)
            week_ckpts = [
                c for c in checkpoints
                if week_start <= c.timestamp < week_end
            ]
            if week_ckpts:
                mark_keep(week_ckpts[0], f"weekly-{week_offset+1}w")

        # 确定删除列表
        delete_list = []
        freed_bytes = 0
        for ckpt in checkpoints:
            age_days = (now - ckpt.timestamp).days
            if ckpt.path not in keep_set:
                delete_list.append(ckpt)
                freed_bytes += ckpt.size_bytes
            elif age_days > self.max_retention_days and ckpt.path in keep_set:
                # 超过最大保留期但被规则保留的，仍然保留（规则优先）
                pass

        return {
            "keep": [(c, keep_reasons.get(c.path, []))
                     for c in checkpoints if c.path in keep_set],
            "delete": delete_list,
            "freed_bytes": freed_bytes,
        }

    def execute_gc(self, dry_run: bool = False) -> Dict:
        """执行 GC"""
        checkpoints = self.scan_checkpoints()
        plan = self.compute_gc_plan(checkpoints)

        print(f"\n{'='*60}")
        print(f"Checkpoint GC")
        print(f"目录: {self.checkpoint_dir}")
        print(f"扫描到: {len(checkpoints)} 个 Checkpoint")
        print(f"{'='*60}")

        # 显示保留
        print(f"\n保留 ({len(plan['keep'])} 个):")
        for ckpt, reasons in plan["keep"]:
            size_mb = ckpt.size_bytes / 1024 / 1024
            reason_str = ", ".join(reasons)
            print(f"  ✓ step={ckpt.step:>6}, {size_mb:>8.1f}MB, "
                  f"age={self._age_str(ckpt.timestamp)}, [{reason_str}]")

        # 显示删除
        print(f"\n删除 ({len(plan['delete'])} 个):")
        for ckpt in plan["delete"]:
            size_mb = ckpt.size_bytes / 1024 / 1024
            print(f"  ✗ step={ckpt.step:>6}, {size_mb:>8.1f}MB, "
                  f"age={self._age_str(ckpt.timestamp)}, {ckpt.path}")

        freed_mb = plan["freed_bytes"] / 1024 / 1024
        freed_gb = freed_mb / 1024
        print(f"\n释放空间: {freed_mb:.1f}MB ({freed_gb:.2f}GB)")

        if not dry_run:
            confirm = input("\n确认删除? [y/N]: ")
            if confirm.lower() == "y":
                for ckpt in plan["delete"]:
                    os.remove(ckpt.path)
                    # 删除关联的 meta 文件
                    meta_path = Path(ckpt.path).with_suffix(".meta.json")
                    if meta_path.exists():
                        os.remove(meta_path)
                print(f"已删除 {len(plan['delete'])} 个 Checkpoint")
            else:
                print("取消操作")
        else:
            print("\n[Dry Run] 未实际删除任何文件")

        return plan

    @staticmethod
    def _age_str(timestamp: datetime) -> str:
        """格式化时间差"""
        delta = datetime.now() - timestamp
        if delta.days > 0:
            return f"{delta.days}d"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h"
        minutes = delta.seconds // 60
        return f"{minutes}m"


def create_demo_checkpoints(checkpoint_dir: str, num: int = 20):
    """创建演示用的 Checkpoint 文件"""
    import torch
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"创建 {num} 个演示 Checkpoint...")
    now = datetime.now()

    for i in range(num):
        step = (i + 1) * 1000
        # 模拟不同时间创建的 checkpoint
        age_hours = (num - i) * 12  # 每个间隔 12 小时

        # 创建文件
        is_full = (i % 5 == 0)
        prefix = "full" if is_full else "incr"
        filename = f"{prefix}_step_{step}.pt"
        filepath = os.path.join(checkpoint_dir, filename)

        # 写入一些数据
        data = {"step": step, "tensor": torch.randn(256, 256)}
        torch.save(data, filepath)

        # 修改时间戳
        target_time = now - timedelta(hours=age_hours)
        ts = target_time.timestamp()
        os.utime(filepath, (ts, ts))

        # 创建 meta 文件
        meta = {"val_loss": random.uniform(0.5, 3.0), "step": step}
        meta_path = filepath.replace(".pt", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        os.utime(meta_path, (ts, ts))


def main():
    parser = argparse.ArgumentParser(description="Checkpoint GC 策略")
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/ckpt/gc_demo")
    parser.add_argument("--keep-latest", type=int, default=3)
    parser.add_argument("--keep-best", type=int, default=5)
    parser.add_argument("--keep-daily", type=int, default=7)
    parser.add_argument("--keep-weekly", type=int, default=4)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true",
                       help="只显示计划，不实际删除")
    parser.add_argument("--create-demo", action="store_true",
                       help="创建演示数据")
    args = parser.parse_args()

    if args.create_demo:
        create_demo_checkpoints(args.checkpoint_dir)

    gc = CheckpointGC(
        checkpoint_dir=args.checkpoint_dir,
        keep_latest=args.keep_latest,
        keep_best=args.keep_best,
        keep_daily=args.keep_daily,
        keep_weekly=args.keep_weekly,
        max_retention_days=args.max_days,
    )

    gc.execute_gc(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
