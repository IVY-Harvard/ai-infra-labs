"""
多指标关联分析器 — GPU 推理异常根因定位
==========================================

单指标异常检测回答: "TTFT 是否异常?"
关联分析回答: "TTFT 异常是因为 KV Cache 满了还是 GPU 限频了?"

核心能力:
1. 因果图构建: 哪些指标之间有因果关系
2. 关联异常检测: 正常关联被打破 = 异常
3. 根因排序: 当多个指标同时异常时, 哪个是根因

GPU 推理指标因果图:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  GPU Temperature ─┬→ Thermal Throttle ─→ SM Clock ↓        │
│                   │                       ↓                │
│                   │                    TPOT ↑               │
│                   │                       ↓                │
│  Request Rate ────┼→ Queue Length ────→ TTFT ↑              │
│                   │      ↓                                 │
│                   │   KV Cache ↑ ────→ Preemption ↑        │
│                   │      ↓                ↓                │
│                   │   Swap Count ↑    TTFT ↑↑              │
│                   │                       ↓                │
│                   └─────────────────→ Throughput ↓          │
│                                           ↓                │
│                                    Error Rate ↑             │
└─────────────────────────────────────────────────────────────┘

依赖:
    pip install numpy pandas scipy
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 因果图定义
# ============================================================

@dataclass
class CausalEdge:
    """因果图中的边"""
    source: str       # 原因指标
    target: str       # 结果指标
    direction: str    # "positive" (同向) 或 "negative" (反向)
    lag_seconds: int   # 因果延迟 (秒)
    weight: float     # 因果强度 (0-1)
    description: str  # 人类可读描述


# GPU 推理服务的因果图 (基于领域知识构建)
INFERENCE_CAUSAL_GRAPH: List[CausalEdge] = [
    # === KV Cache 因果链 ===
    CausalEdge("request_rate", "kv_cache_usage", "positive", 30,  0.8,
               "请求增加 → KV Cache 使用率上升"),
    CausalEdge("kv_cache_usage", "preemption_rate", "positive", 5,  0.9,
               "KV Cache 满 → 触发 Preemption"),
    CausalEdge("preemption_rate", "ttft_p99", "positive", 5,  0.85,
               "Preemption → 被抢占请求重新排队 → TTFT 升高"),
    CausalEdge("kv_cache_usage", "swap_count", "positive", 10, 0.7,
               "KV Cache 满 → CPU Swap 发生"),
    CausalEdge("swap_count", "ttft_p99", "positive", 5,  0.6,
               "Swap → PCIe 传输延迟 → TTFT 升高"),

    # === GPU 热管理因果链 ===
    CausalEdge("gpu_temp", "throttle_active", "positive", 0,  0.95,
               "GPU 温度过高 → 触发 Thermal Throttle"),
    CausalEdge("throttle_active", "sm_clock", "negative", 0,  0.9,
               "Throttle → SM Clock 降低"),
    CausalEdge("sm_clock", "tpot_p99", "negative", 0,  0.8,
               "SM Clock 降低 → 每步推理变慢 → TPOT 升高"),
    CausalEdge("tpot_p99", "throughput", "negative", 5,  0.85,
               "TPOT 升高 → 每个请求耗时增加 → 总吞吐下降"),

    # === 排队因果链 ===
    CausalEdge("request_rate", "queue_length", "positive", 0,  0.7,
               "请求到达率增加 → 排队变长"),
    CausalEdge("queue_length", "ttft_p99", "positive", 0,  0.8,
               "排队 → 等待时间增加 → TTFT 升高"),

    # === Batch Size 效应 ===
    CausalEdge("queue_length", "batch_size", "positive", 0,  0.6,
               "排队多 → Scheduler 组更大的 batch"),
    CausalEdge("batch_size", "tpot_p99", "positive", 0,  0.7,
               "Batch 增大 → GPU Memory Bandwidth 竞争 → TPOT 升高"),
    CausalEdge("batch_size", "throughput", "positive", 0,  0.5,
               "Batch 增大 → GPU 利用率提高 → 总吞吐提升 (到一定程度后反转)"),

    # === 吞吐与错误 ===
    CausalEdge("throughput", "error_rate", "negative", 10, 0.4,
               "吞吐下降 → 请求超时增加 → 错误率上升"),

    # === Prefix Cache ===
    CausalEdge("kv_cache_usage", "prefix_cache_hit", "negative", 30, 0.6,
               "KV Cache 压力大 → Prefix Cache Blocks 被淘汰 → 命中率下降"),
    CausalEdge("prefix_cache_hit", "ttft_p99", "negative", 0,  0.5,
               "Prefix Cache 命中率下降 → 重复 Prefill → TTFT 升高"),
]


# ============================================================
# 相关性计算器
# ============================================================

class CorrelationCalculator:
    """计算指标间的动态相关性

    通过滑动窗口计算 Pearson 相关系数和 Spearman 秩相关:
    - Pearson: 线性关系
    - Spearman: 单调关系 (更鲁棒)

    当正常时的高相关性突然降低 → 因果关系被打破 → 异常!
    """

    def __init__(self, window_size: int = 120):
        self.window_size = window_size
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def update(self, metrics: Dict[str, float]):
        """更新指标缓冲"""
        for name, value in metrics.items():
            self._buffers[name].append(value)

    def pearson(self, metric_a: str, metric_b: str) -> Optional[float]:
        """计算 Pearson 相关系数"""
        buf_a = self._buffers.get(metric_a)
        buf_b = self._buffers.get(metric_b)

        if not buf_a or not buf_b or len(buf_a) < 20 or len(buf_b) < 20:
            return None

        min_len = min(len(buf_a), len(buf_b))
        a = np.array(list(buf_a)[-min_len:])
        b = np.array(list(buf_b)[-min_len:])

        std_a, std_b = np.std(a), np.std(b)
        if std_a == 0 or std_b == 0:
            return 0.0

        correlation = np.corrcoef(a, b)[0, 1]
        return round(float(correlation), 4) if not np.isnan(correlation) else 0.0

    def lagged_correlation(
        self, metric_a: str, metric_b: str, lag: int
    ) -> Optional[float]:
        """计算带时延的相关性

        lag > 0: metric_a 领先 metric_b lag 个时间步
        """
        buf_a = self._buffers.get(metric_a)
        buf_b = self._buffers.get(metric_b)

        if not buf_a or not buf_b:
            return None

        min_len = min(len(buf_a), len(buf_b))
        if min_len < 20 + abs(lag):
            return None

        a = np.array(list(buf_a)[-min_len:])
        b = np.array(list(buf_b)[-min_len:])

        if lag > 0:
            a = a[:-lag]
            b = b[lag:]
        elif lag < 0:
            a = a[-lag:]
            b = b[:lag]

        std_a, std_b = np.std(a), np.std(b)
        if std_a == 0 or std_b == 0:
            return 0.0

        correlation = np.corrcoef(a, b)[0, 1]
        return round(float(correlation), 4) if not np.isnan(correlation) else 0.0


# ============================================================
# 关联异常检测器
# ============================================================

class CorrelationAnomalyDetector:
    """检测指标间关联关系的异常变化

    正常状态: KV Cache ↑ 与 Preemption ↑ 高度正相关 (r=0.85)
    异常状态: KV Cache ↑ 但 Preemption 没变 → r 降低 → 关联打破

    关联打破的可能原因:
    - KV Cache 满但 Scheduler 策略变了 (不再 preempt)
    - 数据采集延迟 (metrics 不同步)
    - 系统行为改变 (配置变更、模型更换)
    """

    def __init__(self, causal_graph: List[CausalEdge] = None):
        self.causal_graph = causal_graph or INFERENCE_CAUSAL_GRAPH
        self.calculator = CorrelationCalculator(window_size=120)

        # 每条因果边的历史相关系数
        self._correlation_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=30)  # 保留最近 30 次计算
        )

    def update_and_detect(
        self, metrics: Dict[str, float]
    ) -> List[Dict]:
        """更新指标并检测关联异常

        Returns:
            异常列表, 每个异常包含:
            - edge: 哪条因果关系打破了
            - expected_correlation: 期望的相关性
            - actual_correlation: 实际相关性
            - deviation: 偏离程度
        """
        self.calculator.update(metrics)

        anomalies = []

        for edge in self.causal_graph:
            # 计算当前相关性
            lag_steps = edge.lag_seconds // 15  # 假设 15s 采样间隔
            if lag_steps > 0:
                current_corr = self.calculator.lagged_correlation(
                    edge.source, edge.target, lag_steps
                )
            else:
                current_corr = self.calculator.pearson(edge.source, edge.target)

            if current_corr is None:
                continue

            # 期望方向
            expected_sign = 1.0 if edge.direction == "positive" else -1.0
            signed_corr = current_corr * expected_sign

            # 更新历史
            edge_key = f"{edge.source}->{edge.target}"
            self._correlation_history[edge_key].append(signed_corr)

            # 检测: 相关性是否显著偏离期望
            history = list(self._correlation_history[edge_key])
            if len(history) < 10:
                continue

            hist_mean = np.mean(history)
            hist_std = np.std(history)

            # 关联打破条件:
            # 1. 当前相关性与期望方向相反
            # 2. 或相关性显著低于历史平均
            is_broken = False
            reason = ""

            if signed_corr < 0 and edge.weight > 0.5:
                # 因果方向反转
                is_broken = True
                reason = f"相关性方向反转: 期望{edge.direction} 实际 r={current_corr:.3f}"

            elif hist_std > 0 and (hist_mean - signed_corr) / hist_std > 3:
                # 相关性显著下降
                is_broken = True
                reason = (
                    f"相关性显著下降: 历史均值={hist_mean:.3f}, "
                    f"当前={signed_corr:.3f}, z={((hist_mean - signed_corr) / hist_std):.1f}"
                )

            if is_broken:
                anomalies.append({
                    "edge": edge_key,
                    "source": edge.source,
                    "target": edge.target,
                    "description": edge.description,
                    "expected_direction": edge.direction,
                    "expected_correlation": round(hist_mean, 4),
                    "actual_correlation": current_corr,
                    "deviation": round(hist_mean - signed_corr, 4),
                    "reason": reason,
                    "causal_weight": edge.weight,
                })

        return anomalies


# ============================================================
# 根因分析器
# ============================================================

class RootCauseAnalyzer:
    """多指标异常时的根因排序

    当多个指标同时异常时, 使用因果图推断哪个是根因:
    1. 构建异常传播路径
    2. 拓扑排序找到 "最上游" 的异常
    3. 结合时序信息 (哪个先异常)

    示例:
    同时异常: [KV Cache ↑, Preemption ↑, TTFT ↑, Queue ↑]
    因果图分析:
      KV Cache → Preemption → TTFT  (KV Cache 是根因)
      Queue → TTFT (Queue 也是上游)
      但 Queue → KV Cache? (如果 Queue 先异常 → Queue 是更根本的原因)

    最终判定: Queue Length 增加 → KV Cache 满 → Preemption → TTFT 升高
    根因: 请求流量增加导致排队
    """

    def __init__(self, causal_graph: List[CausalEdge] = None):
        self.causal_graph = causal_graph or INFERENCE_CAUSAL_GRAPH
        # 构建邻接表
        self._adjacency: Dict[str, List[CausalEdge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[CausalEdge]] = defaultdict(list)
        for edge in self.causal_graph:
            self._adjacency[edge.source].append(edge)
            self._reverse_adjacency[edge.target].append(edge)

    def analyze(
        self,
        anomalous_metrics: Dict[str, Dict],
        metric_timeline: Dict[str, float] = None,
    ) -> List[Dict]:
        """分析根因

        Args:
            anomalous_metrics: {metric_name: {score, severity, ...}}
            metric_timeline: {metric_name: first_anomaly_timestamp}
                             (哪个指标先变异常)

        Returns:
            根因候选列表, 按可能性排序
        """
        if not anomalous_metrics:
            return []

        candidates = []
        anomalous_set = set(anomalous_metrics.keys())

        for metric in anomalous_metrics:
            score = self._compute_root_cause_score(
                metric, anomalous_set, metric_timeline
            )
            candidates.append({
                "metric": metric,
                "root_cause_score": round(score, 4),
                "anomaly_score": anomalous_metrics[metric].get("score", 0),
                "severity": anomalous_metrics[metric].get("severity", "info"),
                "downstream_impact": self._get_downstream_metrics(metric, anomalous_set),
                "upstream_causes": self._get_upstream_metrics(metric, anomalous_set),
                "explanation": self._generate_explanation(metric, anomalous_set),
            })

        # 按根因分数排序 (越高越可能是根因)
        candidates.sort(key=lambda x: x["root_cause_score"], reverse=True)
        return candidates

    def _compute_root_cause_score(
        self,
        metric: str,
        anomalous_set: Set[str],
        timeline: Dict[str, float] = None,
    ) -> float:
        """计算某个指标是根因的可能性分数

        评分维度:
        1. 上游依赖: 有多少异常上游指标? (越少越可能是根因)
        2. 下游影响: 有多少异常下游指标? (越多越可能是根因)
        3. 时序优先: 是否最先异常? (越早越可能是根因)
        4. 因果权重: 因果边的权重
        """
        score = 0.0

        # 维度 1: 上游异常数 (越少 → 越可能是源头)
        upstream_anomalies = [
            e.source for e in self._reverse_adjacency.get(metric, [])
            if e.source in anomalous_set
        ]
        if len(upstream_anomalies) == 0:
            score += 3.0  # 没有异常上游 → 很可能是根因
        else:
            score += max(0, 1.0 - len(upstream_anomalies) * 0.5)

        # 维度 2: 下游异常数 (越多 → 影响范围越大)
        downstream_anomalies = [
            e.target for e in self._adjacency.get(metric, [])
            if e.target in anomalous_set
        ]
        score += len(downstream_anomalies) * 0.5

        # 维度 3: 下游因果权重
        for edge in self._adjacency.get(metric, []):
            if edge.target in anomalous_set:
                score += edge.weight * 0.3

        # 维度 4: 时序优先 (最先异常的分数最高)
        if timeline and metric in timeline:
            all_times = sorted(timeline.values())
            if all_times:
                # 越早异常, 分数越高
                rank = all_times.index(timeline[metric])
                score += max(0, 2.0 - rank * 0.5)

        return score

    def _get_downstream_metrics(
        self, metric: str, anomalous_set: Set[str]
    ) -> List[str]:
        """获取下游受影响的异常指标"""
        result = []
        visited = set()
        queue = [metric]
        while queue:
            current = queue.pop(0)
            for edge in self._adjacency.get(current, []):
                if edge.target in anomalous_set and edge.target not in visited:
                    result.append(edge.target)
                    visited.add(edge.target)
                    queue.append(edge.target)
        return result

    def _get_upstream_metrics(
        self, metric: str, anomalous_set: Set[str]
    ) -> List[str]:
        """获取上游异常指标"""
        return [
            e.source for e in self._reverse_adjacency.get(metric, [])
            if e.source in anomalous_set
        ]

    def _generate_explanation(
        self, metric: str, anomalous_set: Set[str]
    ) -> str:
        """生成人类可读的根因解释"""
        upstream = self._get_upstream_metrics(metric, anomalous_set)
        downstream = self._get_downstream_metrics(metric, anomalous_set)

        if not upstream and downstream:
            chain = " → ".join([metric] + downstream[:3])
            return f"根因候选: {metric} 异常, 导致下游 {chain} 连锁异常"
        elif upstream and downstream:
            return (
                f"中间节点: {metric} 受 {', '.join(upstream)} 影响, "
                f"同时影响 {', '.join(downstream[:3])}"
            )
        elif upstream:
            return f"末端节点: {metric} 受 {', '.join(upstream)} 影响导致异常"
        else:
            return f"独立异常: {metric} 无明显上下游关联"


# ============================================================
# 综合分析引擎
# ============================================================

class InferenceCorrelationEngine:
    """GPU 推理指标关联分析引擎

    整合:
    1. 实时关联监控
    2. 关联异常检测
    3. 根因分析

    输出:
    - 当前关联状态 (健康/打破)
    - 异常根因排序
    - 建议操作
    """

    def __init__(self):
        self.correlation_detector = CorrelationAnomalyDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self._anomaly_start_times: Dict[str, float] = {}
        self._active_anomalies: Dict[str, Dict] = {}

    def process(
        self,
        metrics: Dict[str, float],
        metric_anomalies: Dict[str, Dict] = None,
    ) -> Dict:
        """处理一个时间步的数据

        Args:
            metrics: 原始指标值
            metric_anomalies: 来自 statistical/ml detector 的异常结果

        Returns:
            {
                "correlation_anomalies": [...],
                "root_causes": [...],
                "health_score": float,
                "recommendations": [...]
            }
        """
        now = time.time()

        # 1. 关联异常检测
        correlation_anomalies = self.correlation_detector.update_and_detect(metrics)

        # 2. 更新异常时间线
        if metric_anomalies:
            for name, result in metric_anomalies.items():
                if result.get("is_anomaly", False):
                    if name not in self._anomaly_start_times:
                        self._anomaly_start_times[name] = now
                    self._active_anomalies[name] = result
                else:
                    self._anomaly_start_times.pop(name, None)
                    self._active_anomalies.pop(name, None)

        # 3. 根因分析 (如果有多个异常指标)
        root_causes = []
        if len(self._active_anomalies) >= 2:
            root_causes = self.root_cause_analyzer.analyze(
                self._active_anomalies,
                self._anomaly_start_times,
            )

        # 4. 健康分数
        health_score = self._compute_health_score(
            correlation_anomalies, self._active_anomalies
        )

        # 5. 生成建议
        recommendations = self._generate_recommendations(
            root_causes, correlation_anomalies, metrics
        )

        return {
            "correlation_anomalies": correlation_anomalies,
            "root_causes": root_causes[:5],  # Top 5 根因
            "health_score": round(health_score, 2),
            "active_anomaly_count": len(self._active_anomalies),
            "recommendations": recommendations,
        }

    def _compute_health_score(
        self,
        correlation_anomalies: List[Dict],
        active_anomalies: Dict,
    ) -> float:
        """计算系统健康分数 (0-100)

        100 = 完全健康
        0 = 严重异常
        """
        score = 100.0

        # 每个活跃异常扣分
        for name, anomaly in active_anomalies.items():
            severity = anomaly.get("severity", "info")
            if severity == "critical":
                score -= 20
            elif severity == "warning":
                score -= 10
            else:
                score -= 3

        # 每个关联异常扣分
        score -= len(correlation_anomalies) * 5

        return max(0, score)

    def _generate_recommendations(
        self,
        root_causes: List[Dict],
        correlation_anomalies: List[Dict],
        metrics: Dict[str, float],
    ) -> List[str]:
        """基于分析结果生成操作建议"""
        recommendations = []

        if not root_causes and not correlation_anomalies:
            return ["系统运行正常, 无需操作"]

        # 基于根因的建议
        for rc in root_causes[:3]:
            metric = rc["metric"]
            if metric == "kv_cache_usage":
                recommendations.append(
                    "KV Cache 是根因 → 建议: 增加 gpu_memory_utilization 或扩容实例"
                )
            elif metric == "gpu_temp":
                recommendations.append(
                    "GPU 温度是根因 → 建议: 检查散热系统, 必要时降低 max_num_seqs"
                )
            elif metric == "request_rate" or metric == "queue_length":
                recommendations.append(
                    "请求流量是根因 → 建议: 触发 HPA 扩容或启用限流策略"
                )
            elif metric == "preemption_rate":
                recommendations.append(
                    "Preemption 频繁 → 建议: 减小 max_model_len 或增加 KV Cache 空间"
                )
            elif metric in ("tpot_p99", "sm_clock"):
                recommendations.append(
                    "GPU 执行效率下降 → 建议: 检查 NVLink 状态和 GPU 健康"
                )

        # 基于关联异常的建议
        if correlation_anomalies:
            broken_edges = [a["edge"] for a in correlation_anomalies]
            recommendations.append(
                f"检测到 {len(broken_edges)} 条因果关系打破: "
                f"{', '.join(broken_edges[:3])} → 可能有配置变更或硬件异常"
            )

        return recommendations


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== Correlation Analysis & Root Cause Demo ===\n")

    engine = InferenceCorrelationEngine()

    # Phase 1: 正常运行 (建立 baseline)
    print("Phase 1: Normal operation (building baseline)...")
    for i in range(60):
        metrics = {
            "ttft_p99": 0.25 + random.gauss(0, 0.02),
            "tpot_p99": 0.025 + random.gauss(0, 0.002),
            "kv_cache_usage": 0.65 + random.gauss(0, 0.03),
            "throughput": 1500 + random.gauss(0, 80),
            "queue_length": max(0, 2 + random.gauss(0, 1)),
            "preemption_rate": max(0, 0.01 + random.gauss(0, 0.005)),
            "gpu_temp": 65 + random.gauss(0, 1),
            "batch_size": 16 + random.gauss(0, 2),
        }
        result = engine.process(metrics)

    print(f"  Baseline health: {result['health_score']}/100\n")

    # Phase 2: KV Cache 逐渐升高 → 触发 Preemption → TTFT 升高
    print("Phase 2: KV Cache pressure building...")
    for i in range(30):
        kv = min(0.98, 0.65 + i * 0.01 + random.gauss(0, 0.01))
        preemption = max(0, (kv - 0.85) * 10 + random.gauss(0, 0.1))
        ttft = 0.25 + preemption * 0.5 + random.gauss(0, 0.02)

        metrics = {
            "ttft_p99": ttft,
            "tpot_p99": 0.025 + random.gauss(0, 0.002),
            "kv_cache_usage": kv,
            "throughput": max(500, 1500 - preemption * 100 + random.gauss(0, 50)),
            "queue_length": max(0, 2 + preemption * 5 + random.gauss(0, 1)),
            "preemption_rate": preemption,
            "gpu_temp": 65 + random.gauss(0, 1),
            "batch_size": 16 + random.gauss(0, 2),
        }

        anomalies = {}
        if kv > 0.9:
            anomalies["kv_cache_usage"] = {"is_anomaly": True, "score": 0.8, "severity": "warning"}
        if preemption > 1:
            anomalies["preemption_rate"] = {"is_anomaly": True, "score": 0.7, "severity": "warning"}
        if ttft > 1.0:
            anomalies["ttft_p99"] = {"is_anomaly": True, "score": 0.6, "severity": "warning"}

        result = engine.process(metrics, anomalies)

        if i % 10 == 0:
            print(f"  t={i}: kv={kv:.2f} preempt={preemption:.2f} "
                  f"ttft={ttft*1000:.0f}ms health={result['health_score']}/100")

    print(f"\n  Final health: {result['health_score']}/100")
    print(f"  Active anomalies: {result['active_anomaly_count']}")

    if result["root_causes"]:
        print("\n  Root Cause Analysis:")
        for i, rc in enumerate(result["root_causes"][:3]):
            print(f"    #{i+1} {rc['metric']} (score={rc['root_cause_score']:.2f})")
            print(f"        {rc['explanation']}")

    if result["recommendations"]:
        print("\n  Recommendations:")
        for rec in result["recommendations"]:
            print(f"    - {rec}")
