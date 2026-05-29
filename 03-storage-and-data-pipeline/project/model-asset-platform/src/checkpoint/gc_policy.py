"""Checkpoint GC（垃圾回收）策略

支持多种保留策略：
- 保留最近 N 个
- 按时间窗口保留（如保留 7 天内的）
- 按指标保留最优 K 个
- 指数间隔保留（step 1,2,4,8,16...）
"""

import os
import time
import shutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class RetentionPolicy(Enum):
    KEEP_LATEST_N = "keep_latest_n"
    TIME_WINDOW = "time_window"
    BEST_METRIC = "best_metric"
    EXPONENTIAL = "exponential"


@dataclass
class GCRule:
    """GC 规则"""
    policy: RetentionPolicy
    # keep_latest_n
    keep_n: int = 5
    # time_window
    window_seconds: float = 7 * 24 * 3600  # 默认 7 天
    # best_metric
    metric_name: str = "loss"
    metric_mode: str = "min"  # min or max
    keep_top_k: int = 3
    # exponential
    base: int = 2


class GCPolicy:
    """Checkpoint 垃圾回收策略管理

    支持组合多种策略，取并集保留。

    用法：
        gc = GCPolicy()
        gc.add_rule(GCRule(policy=RetentionPolicy.KEEP_LATEST_N, keep_n=3))
        gc.add_rule(GCRule(policy=RetentionPolicy.BEST_METRIC,
                           metric_name="loss", keep_top_k=2))
        to_delete = gc.evaluate(checkpoints)
    """

    def __init__(self):
        self.rules: List[GCRule] = []

    def add_rule(self, rule: GCRule):
        """添加保留规则"""
        self.rules.append(rule)

    def evaluate(self, checkpoints: List[Dict]) -> List[Dict]:
        """评估哪些 checkpoint 应被删除

        Args:
            checkpoints: 列表，每个元素包含：
                - step: int
                - timestamp: float
                - local_path: str
                - metrics: Dict[str, float]

        Returns:
            应被删除的 checkpoint 列表
        """
        if not self.rules:
            return []

        # 收集所有规则要保留的 step
        keep_steps = set()

        for rule in self.rules:
            kept = self._apply_rule(rule, checkpoints)
            keep_steps.update(kept)

        # 不在保留集中的即为待删除
        to_delete = [
            ckpt for ckpt in checkpoints
            if ckpt["step"] not in keep_steps
        ]
        return to_delete

    def _apply_rule(self, rule: GCRule,
                    checkpoints: List[Dict]) -> set:
        """应用单条规则，返回应保留的 step 集合"""
        if rule.policy == RetentionPolicy.KEEP_LATEST_N:
            return self._keep_latest_n(checkpoints, rule.keep_n)

        elif rule.policy == RetentionPolicy.TIME_WINDOW:
            return self._time_window(checkpoints, rule.window_seconds)

        elif rule.policy == RetentionPolicy.BEST_METRIC:
            return self._best_metric(
                checkpoints, rule.metric_name,
                rule.metric_mode, rule.keep_top_k
            )

        elif rule.policy == RetentionPolicy.EXPONENTIAL:
            return self._exponential(checkpoints, rule.base)

        return set()

    @staticmethod
    def _keep_latest_n(checkpoints: List[Dict], n: int) -> set:
        """保留最近 N 个"""
        sorted_ckpts = sorted(checkpoints, key=lambda c: c["step"],
                              reverse=True)
        return {c["step"] for c in sorted_ckpts[:n]}

    @staticmethod
    def _time_window(checkpoints: List[Dict],
                     window_seconds: float) -> set:
        """保留时间窗口内的"""
        cutoff = time.time() - window_seconds
        return {
            c["step"] for c in checkpoints
            if c.get("timestamp", 0) >= cutoff
        }

    @staticmethod
    def _best_metric(checkpoints: List[Dict],
                     metric_name: str,
                     mode: str, top_k: int) -> set:
        """保留指标最优的 K 个"""
        # 过滤有该指标的 checkpoint
        with_metric = [
            c for c in checkpoints
            if metric_name in c.get("metrics", {})
        ]

        if not with_metric:
            return set()

        reverse = (mode == "max")
        sorted_ckpts = sorted(
            with_metric,
            key=lambda c: c["metrics"][metric_name],
            reverse=reverse,
        )
        return {c["step"] for c in sorted_ckpts[:top_k]}

    @staticmethod
    def _exponential(checkpoints: List[Dict], base: int) -> set:
        """指数间隔保留

        保留 step 满足 base^k 的 checkpoint。
        例如 base=2 时保留最接近 step 1,2,4,8,16,... 的 checkpoint
        """
        if not checkpoints:
            return set()

        max_step = max(c["step"] for c in checkpoints)
        all_steps = sorted(c["step"] for c in checkpoints)

        keep = set()
        power = 1
        while power <= max_step:
            # 找最接近 power 的 step
            closest = min(all_steps, key=lambda s: abs(s - power))
            keep.add(closest)
            power *= base

        # 始终保留最新
        keep.add(max_step)
        return keep

    def execute_gc(self, checkpoints: List[Dict],
                   dry_run: bool = False) -> Dict:
        """执行 GC 操作

        Args:
            checkpoints: checkpoint 列表
            dry_run: 仅模拟，不实际删除

        Returns:
            GC 结果报告
        """
        to_delete = self.evaluate(checkpoints)

        result = {
            "total": len(checkpoints),
            "to_delete": len(to_delete),
            "to_keep": len(checkpoints) - len(to_delete),
            "freed_bytes": 0,
            "deleted_steps": [],
            "dry_run": dry_run,
        }

        for ckpt in to_delete:
            local_path = ckpt.get("local_path", "")
            if not dry_run and local_path and os.path.isdir(local_path):
                dir_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, filenames in os.walk(local_path)
                    for f in filenames
                )
                result["freed_bytes"] += dir_size
                shutil.rmtree(local_path, ignore_errors=True)
            result["deleted_steps"].append(ckpt["step"])

        return result
