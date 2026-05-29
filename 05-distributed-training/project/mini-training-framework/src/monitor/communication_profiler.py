"""
通信性能分析器
===============
Profile 分布式训练中的通信操作。
"""

import time
from collections import defaultdict
from typing import Dict, List
from contextlib import contextmanager

import torch
import torch.distributed as dist


class CommunicationProfiler:
    """
    通信性能分析器。
    记录每种通信操作的耗时和数据量。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._records: Dict[str, List[Dict]] = defaultdict(list)
        self._active_timers: Dict[str, float] = {}

    @contextmanager
    def profile(self, op_name: str, data_bytes: int = 0):
        """
        通信操作计时上下文管理器。

        用法:
            with profiler.profile("tp_allreduce", tensor.numel() * 2):
                dist.all_reduce(tensor)
        """
        if not self.enabled:
            yield
            return

        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        self._records[op_name].append({
            "time_ms": elapsed * 1000,
            "data_bytes": data_bytes,
            "bandwidth_gbps": data_bytes / elapsed / 1e9 if elapsed > 0 else 0,
        })

    def get_summary(self) -> Dict[str, Dict]:
        """获取各操作的统计摘要"""
        summary = {}
        for op_name, records in self._records.items():
            times = [r["time_ms"] for r in records]
            bws = [r["bandwidth_gbps"] for r in records if r["bandwidth_gbps"] > 0]
            summary[op_name] = {
                "count": len(records),
                "total_time_ms": sum(times),
                "avg_time_ms": sum(times) / len(times) if times else 0,
                "avg_bandwidth_gbps": sum(bws) / len(bws) if bws else 0,
                "total_data_gb": sum(r["data_bytes"] for r in records) / 1e9,
            }
        return summary

    def report(self) -> str:
        """生成可读报告"""
        summary = self.get_summary()
        lines = ["通信性能报告:", "-" * 70]
        lines.append(f"{'操作':<20} {'次数':<8} {'总时间':<12} {'平均时间':<12} {'平均带宽':<14} {'总数据'}")
        lines.append("-" * 70)

        total_time = 0
        for op, stats in sorted(summary.items()):
            total_time += stats["total_time_ms"]
            lines.append(
                f"{op:<20} {stats['count']:<8} "
                f"{stats['total_time_ms']:.1f}ms{'':<4} "
                f"{stats['avg_time_ms']:.2f}ms{'':<4} "
                f"{stats['avg_bandwidth_gbps']:.1f} GB/s{'':<4} "
                f"{stats['total_data_gb']:.2f} GB"
            )

        lines.append("-" * 70)
        lines.append(f"总通信时间: {total_time:.1f} ms")
        return "\n".join(lines)

    def reset(self):
        """重置记录"""
        self._records.clear()
