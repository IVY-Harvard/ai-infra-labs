"""
多卡/多型号对比分析

对比不同 GPU 或不同次运行的基准测试结果。
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ComparisonEntry:
    """对比条目"""
    metric: str
    values: Dict[str, float]  # label -> value
    unit: str = ""
    higher_is_better: bool = True


class BenchmarkComparison:
    """基准测试对比器"""

    def __init__(self):
        self.entries: List[ComparisonEntry] = []

    def add_entry(self, metric: str, values: Dict[str, float],
                  unit: str = "", higher_is_better: bool = True):
        """添加一个对比条目"""
        self.entries.append(ComparisonEntry(
            metric=metric, values=values, unit=unit,
            higher_is_better=higher_is_better,
        ))

    def compare_compute(self, results: Dict[str, Dict]) -> 'BenchmarkComparison':
        """对比多张卡的计算性能"""
        for dtype in ['fp16', 'fp32']:
            values = {}
            for label, data in results.items():
                if 'compute' in data and dtype in data['compute']:
                    # 取最大矩阵的结果
                    best = max(data['compute'][dtype], key=lambda x: x.get('M', 0))
                    values[label] = best.get('tflops', 0)
            if values:
                self.add_entry(f"{dtype.upper()} TFLOPS", values, "TFLOPS")
        return self

    def compare_memory(self, results: Dict[str, Dict]) -> 'BenchmarkComparison':
        """对比多张卡的内存带宽"""
        for label, data in results.items():
            if 'memory' in data and 'copy' in data['memory']:
                best = max(data['memory']['copy'],
                           key=lambda x: x.get('bandwidth_gb_s', 0))
                self.add_entry(
                    "HBM Bandwidth",
                    {label: best.get('bandwidth_gb_s', 0)},
                    "GB/s",
                )
        return self

    def print_comparison(self):
        """打印对比表"""
        if not self.entries:
            print("无对比数据")
            return

        # 收集所有 label
        all_labels = set()
        for entry in self.entries:
            all_labels.update(entry.values.keys())
        labels = sorted(all_labels)

        # 表头
        header = f"{'Metric':<20}"
        for label in labels:
            header += f" | {label:>12}"
        header += f" | {'Best':>12}"
        print(header)
        print("-" * len(header))

        # 数据行
        for entry in self.entries:
            row = f"{entry.metric:<20}"
            values = []
            for label in labels:
                val = entry.values.get(label, 0)
                values.append(val)
                row += f" | {val:>11.1f}{entry.unit[0] if entry.unit else ''}"

            # 标记最优
            if values:
                best_val = max(values) if entry.higher_is_better else min(values)
                best_label = labels[values.index(best_val)]
                row += f" | {best_label:>12}"

            print(row)

    def to_dict(self) -> List[Dict]:
        """导出为字典"""
        return [
            {
                'metric': e.metric,
                'values': e.values,
                'unit': e.unit,
                'higher_is_better': e.higher_is_better,
            }
            for e in self.entries
        ]


def compare_runs(run_files: List[str]) -> BenchmarkComparison:
    """对比多次运行结果"""
    comp = BenchmarkComparison()

    results = {}
    for filepath in run_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        label = data.get('label', filepath)
        results[label] = data

    comp.compare_compute(results)
    comp.compare_memory(results)

    return comp


def compare_gpus(gpu_results: Dict[int, Dict]) -> BenchmarkComparison:
    """对比同一机器内不同 GPU 的结果"""
    comp = BenchmarkComparison()

    labeled = {f"GPU {gpu_id}": data for gpu_id, data in gpu_results.items()}
    comp.compare_compute(labeled)
    comp.compare_memory(labeled)

    return comp


if __name__ == "__main__":
    # Demo
    comp = BenchmarkComparison()
    comp.add_entry("FP16 TFLOPS", {"GPU 0": 139.2, "GPU 1": 140.1, "GPU 2": 138.5}, "TFLOPS")
    comp.add_entry("FP32 TFLOPS", {"GPU 0": 41.2, "GPU 1": 41.5, "GPU 2": 40.8}, "TFLOPS")
    comp.add_entry("HBM BW", {"GPU 0": 3812, "GPU 1": 3825, "GPU 2": 3798}, "GB/s")
    comp.add_entry("Temperature", {"GPU 0": 72, "GPU 1": 74, "GPU 2": 71}, "C", higher_is_better=False)
    comp.print_comparison()
