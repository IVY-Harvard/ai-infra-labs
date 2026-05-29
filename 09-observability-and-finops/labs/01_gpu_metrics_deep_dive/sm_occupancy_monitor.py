"""
SM Occupancy 持续监控器

SM Occupancy 是衡量 GPU 并行度利用效率的核心指标。
Occupancy 低意味着 GPU 无法有效隐藏内存访问延迟，吞吐量会受影响。

本脚本：
1. 持续监控 SM Occupancy 趋势
2. 关联分析 Occupancy 与吞吐量/延迟的关系
3. 当 Occupancy 持续低于阈值时触发告警

用法:
    python sm_occupancy_monitor.py --prometheus-url http://localhost:9090 --alert-threshold 0.3
"""

import argparse
import time
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests


# ============================================================
# 配置
# ============================================================

@dataclass
class AlertConfig:
    """告警配置"""
    occupancy_low_threshold: float = 0.3     # Occupancy 低于此值触发关注
    occupancy_critical_threshold: float = 0.15  # 严重低 Occupancy
    sustained_minutes: int = 5                # 持续多少分钟才触发告警
    correlation_window: int = 20              # 关联分析窗口（采样点数）


# ============================================================
# 数据采集
# ============================================================

class OccupancyCollector:
    """从 Prometheus 采集 SM Occupancy 及关联指标"""

    QUERIES = {
        "sm_occupancy": "DCGM_FI_PROF_SM_OCCUPANCY",
        "sm_active": "DCGM_FI_PROF_SM_ACTIVE",
        "tensor_active": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        "dram_active": "DCGM_FI_PROF_DRAM_ACTIVE",
        # vLLM 推理指标（如果可用）
        "throughput_tps": 'rate(vllm:generation_tokens_total[1m])',
        "ttft_p99": 'histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))',
        "batch_size": "vllm:num_requests_running",
    }

    def __init__(self, prometheus_url: str):
        self.base_url = prometheus_url.rstrip("/")

    def query(self, promql: str) -> dict:
        """执行 Prometheus 查询，返回 {gpu_id: value}"""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/query",
                params={"query": promql},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data["status"] != "success":
                return {}
            result = {}
            for item in data["data"]["result"]:
                gpu_id = item["metric"].get("gpu", item["metric"].get("instance", "0"))
                result[gpu_id] = float(item["value"][1])
            return result
        except Exception:
            return {}

    def collect(self) -> dict:
        """采集所有指标"""
        snapshot = {"timestamp": time.time()}
        for name, promql in self.QUERIES.items():
            snapshot[name] = self.query(promql)
        return snapshot


# ============================================================
# Occupancy 分析引擎
# ============================================================

@dataclass
class OccupancySample:
    """单次采样"""
    timestamp: float
    occupancy: float
    sm_active: float
    tensor_active: float
    dram_active: float
    throughput: Optional[float] = None
    batch_size: Optional[float] = None


class OccupancyAnalyzer:
    """
    SM Occupancy 趋势分析。

    关键洞察：
    - Occupancy 低 + SM Active 高 → 线程配置不当（register/shared memory 压力）
    - Occupancy 低 + SM Active 低 → GPU 空闲或负载太轻
    - Occupancy 高 + 吞吐低 → 可能内存瓶颈严重
    - Occupancy 随 batch_size 增加而提升 → 正常行为
    """

    def __init__(self, window_size: int = 60):
        # 每个 GPU 维护一个滑动窗口
        self.histories: dict[str, deque[OccupancySample]] = {}
        self.window_size = window_size

    def add_sample(self, gpu_id: str, sample: OccupancySample):
        """添加一个采样点"""
        if gpu_id not in self.histories:
            self.histories[gpu_id] = deque(maxlen=self.window_size)
        self.histories[gpu_id].append(sample)

    def analyze(self, gpu_id: str) -> dict:
        """分析单张 GPU 的 Occupancy 趋势"""
        if gpu_id not in self.histories or len(self.histories[gpu_id]) < 3:
            return {"status": "insufficient_data"}

        history = list(self.histories[gpu_id])

        occupancies = [s.occupancy for s in history]
        sm_actives = [s.sm_active for s in history]
        throughputs = [s.throughput for s in history if s.throughput is not None]

        avg_occ = sum(occupancies) / len(occupancies)
        min_occ = min(occupancies)
        max_occ = max(occupancies)
        latest_occ = occupancies[-1]

        # 趋势判断（简单线性回归斜率）
        n = len(occupancies)
        if n >= 5:
            x_mean = (n - 1) / 2
            y_mean = avg_occ
            numerator = sum((i - x_mean) * (occupancies[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator > 0 else 0
            trend = "rising" if slope > 0.005 else "falling" if slope < -0.005 else "stable"
        else:
            slope = 0
            trend = "unknown"

        # 诊断
        diagnosis = self._diagnose(history[-1], avg_occ)

        # Occupancy 与吞吐量关联
        correlation = None
        if len(throughputs) >= 5:
            occ_for_corr = occupancies[-len(throughputs):]
            correlation = self._pearson_correlation(occ_for_corr, throughputs)

        return {
            "gpu_id": gpu_id,
            "current_occupancy": round(latest_occ, 4),
            "avg_occupancy": round(avg_occ, 4),
            "min_occupancy": round(min_occ, 4),
            "max_occupancy": round(max_occ, 4),
            "trend": trend,
            "slope": round(slope, 6),
            "diagnosis": diagnosis,
            "occupancy_throughput_correlation": (
                round(correlation, 3) if correlation is not None else None
            ),
            "sample_count": n,
        }

    def _diagnose(self, latest: OccupancySample, avg_occ: float) -> str:
        """基于 Occupancy 和其他指标给出诊断"""
        if latest.sm_active < 0.05:
            return "GPU 基本空闲"

        if avg_occ < 0.2:
            if latest.sm_active > 0.5:
                return (
                    "Occupancy 极低但 SM Active 较高：可能 kernel 使用了过多 register "
                    "或 shared memory，限制了 warp 并发数。建议 Profile 优化。"
                )
            return "Occupancy 极低且 SM Active 低：负载过轻，考虑增大 batch size。"

        if avg_occ < 0.3:
            return "Occupancy 偏低：可能需要增大 batch size 或调整 max_num_seqs。"

        if avg_occ > 0.7 and latest.dram_active > 0.8:
            return "Occupancy 高但 DRAM Active 也很高：接近 HBM 带宽瓶颈。"

        if 0.4 <= avg_occ <= 0.7:
            return "Occupancy 在合理范围内。"

        return "Occupancy 正常。"

    @staticmethod
    def _pearson_correlation(x: list, y: list) -> float:
        """计算 Pearson 相关系数"""
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        x, y = x[:n], y[:n]
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denom_x = sum((xi - x_mean) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - y_mean) ** 2 for yi in y) ** 0.5
        if denom_x * denom_y == 0:
            return 0.0
        return numerator / (denom_x * denom_y)


# ============================================================
# 告警引擎
# ============================================================

class OccupancyAlerter:
    """Occupancy 告警引擎"""

    def __init__(self, config: AlertConfig):
        self.config = config
        # 记录每个 GPU 的低 Occupancy 持续时间
        self.low_since: dict[str, Optional[float]] = {}

    def check(self, gpu_id: str, analysis: dict) -> Optional[dict]:
        """检查是否需要触发告警"""
        if analysis.get("status") == "insufficient_data":
            return None

        current_occ = analysis["current_occupancy"]
        now = time.time()

        # 检查是否低于阈值
        if current_occ < self.config.occupancy_low_threshold:
            if gpu_id not in self.low_since or self.low_since[gpu_id] is None:
                self.low_since[gpu_id] = now

            duration_minutes = (now - self.low_since[gpu_id]) / 60

            if duration_minutes >= self.config.sustained_minutes:
                severity = (
                    "critical"
                    if current_occ < self.config.occupancy_critical_threshold
                    else "warning"
                )
                return {
                    "gpu_id": gpu_id,
                    "severity": severity,
                    "message": (
                        f"GPU {gpu_id} SM Occupancy 持续 {duration_minutes:.0f} 分钟 "
                        f"低于 {self.config.occupancy_low_threshold} "
                        f"(当前={current_occ:.3f})"
                    ),
                    "diagnosis": analysis["diagnosis"],
                    "trend": analysis["trend"],
                    "suggested_actions": [
                        "增大 vLLM max_num_seqs / max_num_batched_tokens",
                        "检查是否有请求排队但未被调度",
                        "使用 Nsight Compute Profile 分析 kernel 的 register/shared memory 用量",
                    ],
                }
        else:
            # 恢复正常，重置计时器
            self.low_since[gpu_id] = None

        return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SM Occupancy 持续监控器")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--interval", type=int, default=15, help="采集间隔（秒）")
    parser.add_argument("--alert-threshold", type=float, default=0.3)
    parser.add_argument("--sustained-minutes", type=int, default=5)
    parser.add_argument("--output", default=None, help="分析报告输出路径（JSON）")
    args = parser.parse_args()

    collector = OccupancyCollector(args.prometheus_url)
    analyzer = OccupancyAnalyzer(window_size=120)
    alerter = OccupancyAlerter(AlertConfig(
        occupancy_low_threshold=args.alert_threshold,
        sustained_minutes=args.sustained_minutes,
    ))

    print(f"[SM Occupancy Monitor] Prometheus: {args.prometheus_url}")
    print(f"[SM Occupancy Monitor] 告警阈值: {args.alert_threshold}, 持续 {args.sustained_minutes}m")

    while True:
        try:
            snapshot = collector.collect()
            occ_data = snapshot.get("sm_occupancy", {})
            sm_data = snapshot.get("sm_active", {})
            tensor_data = snapshot.get("tensor_active", {})
            dram_data = snapshot.get("dram_active", {})
            tps_data = snapshot.get("throughput_tps", {})
            batch_data = snapshot.get("batch_size", {})

            all_gpus = set(occ_data.keys())
            analyses = {}

            for gpu_id in all_gpus:
                sample = OccupancySample(
                    timestamp=snapshot["timestamp"],
                    occupancy=occ_data.get(gpu_id, 0),
                    sm_active=sm_data.get(gpu_id, 0),
                    tensor_active=tensor_data.get(gpu_id, 0),
                    dram_active=dram_data.get(gpu_id, 0),
                    throughput=tps_data.get(gpu_id),
                    batch_size=batch_data.get(gpu_id),
                )
                analyzer.add_sample(gpu_id, sample)

                analysis = analyzer.analyze(gpu_id)
                analyses[gpu_id] = analysis

                # 告警检查
                alert = alerter.check(gpu_id, analysis)
                if alert:
                    print(f"\n[ALERT][{alert['severity'].upper()}] {alert['message']}")
                    print(f"  诊断: {alert['diagnosis']}")

            # 打印摘要
            ts = datetime.fromtimestamp(snapshot["timestamp"]).strftime("%H:%M:%S")
            for gpu_id, a in analyses.items():
                if a.get("status") == "insufficient_data":
                    continue
                print(
                    f"  [{ts}] GPU {gpu_id}: "
                    f"occ={a['current_occupancy']:.3f} "
                    f"avg={a['avg_occupancy']:.3f} "
                    f"trend={a['trend']}"
                )

            # 保存报告
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(analyses, f, indent=2, ensure_ascii=False)

        except requests.exceptions.ConnectionError:
            print(f"[WARN] 无法连接 Prometheus，将重试...")
        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
