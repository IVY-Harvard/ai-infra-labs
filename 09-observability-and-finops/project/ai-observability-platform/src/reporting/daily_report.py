"""每日运维报告生成器 — 汇总 GPU 集群和推理服务的运行状态

报告章节:
1. 概览摘要 (整体健康状态)
2. GPU 利用率统计 (平均/峰值/低谷)
3. 推理 SLI/SLO 达标情况
4. 异常检测汇总
5. Top 问题与建议

输出格式:
- JSON (供仪表盘消费)
- Markdown 文本 (供邮件/消息推送)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.collectors.gpu_collector import GPUCollector, GPUMetrics
from src.collectors.inference_collector import InferenceCollector, InferenceMetrics
from src.collectors.training_collector import TrainingCollector, TrainingMetrics
from src.analytics.anomaly_engine import AnomalyEngine, Anomaly
from src.alerting.alert_manager import AlertManager

logger = logging.getLogger(__name__)


@dataclass
class SLOTarget:
    """SLO 目标定义"""
    name: str
    metric: str
    target: float
    unit: str = ""
    comparison: str = "lte"  # lte: 值 <= target 为达标, gte: 值 >= target 为达标


@dataclass
class SLOResult:
    """SLO 达标结果"""
    name: str
    target: float
    actual: float
    compliant: bool
    unit: str = ""
    compliance_pct: float = 100.0  # 达标百分比 (时间维度)


@dataclass
class DailyReportData:
    """每日报告数据结构"""
    report_date: str = ""
    generated_at: float = 0

    # GPU 概览
    gpu_count: int = 0
    gpu_utilization_avg: float = 0
    gpu_utilization_peak: float = 0
    gpu_utilization_min: float = 0
    gpu_temperature_avg: float = 0
    gpu_temperature_max: float = 0
    gpu_memory_used_avg_gb: float = 0
    gpu_power_total_watts: float = 0

    # 推理服务
    inference_throughput_avg_tps: float = 0
    inference_kv_cache_avg: float = 0
    inference_requests_total: int = 0
    inference_error_rate: float = 0

    # SLO 达标
    slo_results: List[SLOResult] = field(default_factory=list)

    # 异常
    anomalies_total: int = 0
    anomalies_critical: int = 0
    anomalies_warning: int = 0
    top_anomaly_metrics: List[str] = field(default_factory=list)

    # 告警
    alerts_fired: int = 0
    alerts_resolved: int = 0
    alerts_still_firing: int = 0

    # 建议
    recommendations: List[str] = field(default_factory=list)


class DailyReportGenerator:
    """每日运维报告生成器"""

    # 默认 SLO 目标
    DEFAULT_SLOS = [
        SLOTarget("TTFT P99", "ttft_p99_ms", 2000, "ms", "lte"),
        SLOTarget("TPOT P99", "tpot_p99_ms", 50, "ms", "lte"),
        SLOTarget("可用性", "availability", 99.9, "%", "gte"),
        SLOTarget("错误率", "error_rate", 0.1, "%", "lte"),
        SLOTarget("GPU 利用率", "gpu_utilization", 60, "%", "gte"),
    ]

    def __init__(
        self,
        gpu_collector: Optional[GPUCollector] = None,
        inference_collector: Optional[InferenceCollector] = None,
        training_collector: Optional[TrainingCollector] = None,
        anomaly_engine: Optional[AnomalyEngine] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        self.gpu_collector = gpu_collector or GPUCollector(use_nvml=False)
        self.inference_collector = inference_collector or InferenceCollector()
        self.training_collector = training_collector or TrainingCollector()
        self.anomaly_engine = anomaly_engine or AnomalyEngine()
        self.alert_manager = alert_manager or AlertManager()
        self.slo_targets = list(self.DEFAULT_SLOS)

    def add_slo_target(self, target: SLOTarget):
        """添加自定义 SLO 目标"""
        self.slo_targets.append(target)

    def generate(self) -> DailyReportData:
        """生成每日报告数据

        聚合各采集器和分析器的数据, 生成统一报告
        """
        report = DailyReportData(
            report_date=time.strftime("%Y-%m-%d"),
            generated_at=time.time(),
        )

        # 采集 GPU 指标
        self._collect_gpu_summary(report)

        # 采集推理指标
        self._collect_inference_summary(report)

        # 检查 SLO 达标
        self._evaluate_slos(report)

        # 汇总异常
        self._collect_anomaly_summary(report)

        # 汇总告警
        self._collect_alert_summary(report)

        # 生成建议
        self._generate_recommendations(report)

        logger.info(
            f"每日报告已生成: date={report.report_date}, "
            f"GPU利用率={report.gpu_utilization_avg:.1f}%, "
            f"异常={report.anomalies_total}, 告警={report.alerts_fired}"
        )

        return report

    def _collect_gpu_summary(self, report: DailyReportData):
        """采集 GPU 指标摘要"""
        gpu_metrics = self.gpu_collector.collect()
        if not gpu_metrics:
            return

        report.gpu_count = len(gpu_metrics)
        utils = [m.utilization_pct for m in gpu_metrics]
        temps = [m.temperature_c for m in gpu_metrics]

        report.gpu_utilization_avg = sum(utils) / len(utils)
        report.gpu_utilization_peak = max(utils)
        report.gpu_utilization_min = min(utils)
        report.gpu_temperature_avg = sum(temps) / len(temps)
        report.gpu_temperature_max = max(temps)
        report.gpu_memory_used_avg_gb = (
            sum(m.memory_used_gb for m in gpu_metrics) / len(gpu_metrics)
        )
        report.gpu_power_total_watts = sum(m.power_watts for m in gpu_metrics)

    def _collect_inference_summary(self, report: DailyReportData):
        """采集推理服务指标摘要"""
        metrics = self.inference_collector.collect_mock()

        report.inference_throughput_avg_tps = metrics.throughput_tps
        report.inference_kv_cache_avg = metrics.kv_cache_usage
        report.inference_requests_total = metrics.request_success_total + metrics.request_failure_total

        total = report.inference_requests_total
        if total > 0:
            report.inference_error_rate = metrics.request_failure_total / total * 100
        else:
            report.inference_error_rate = 0

    def _evaluate_slos(self, report: DailyReportData):
        """评估 SLO 达标情况"""
        # 构建指标映射
        metrics_map = {
            "ttft_p99_ms": self.inference_collector.collect_mock().ttft_p99_ms,
            "tpot_p99_ms": self.inference_collector.collect_mock().tpot_p99_ms,
            "availability": 99.95,  # 模拟值
            "error_rate": report.inference_error_rate,
            "gpu_utilization": report.gpu_utilization_avg,
        }

        for slo in self.slo_targets:
            actual = metrics_map.get(slo.metric, 0)

            if slo.comparison == "lte":
                compliant = actual <= slo.target
            else:
                compliant = actual >= slo.target

            # 模拟达标百分比
            if compliant:
                compliance_pct = 99.0 + (100 - 99) * 0.8  # 99.8%
            else:
                ratio = actual / slo.target if slo.target > 0 else 0
                if slo.comparison == "lte":
                    compliance_pct = max(0, 100 - (ratio - 1) * 100)
                else:
                    compliance_pct = min(100, ratio * 100)

            report.slo_results.append(SLOResult(
                name=slo.name,
                target=slo.target,
                actual=actual,
                compliant=compliant,
                unit=slo.unit,
                compliance_pct=round(compliance_pct, 2),
            ))

    def _collect_anomaly_summary(self, report: DailyReportData):
        """汇总异常检测结果"""
        anomalies = self.anomaly_engine.get_recent_anomalies(limit=500)

        report.anomalies_total = len(anomalies)
        report.anomalies_critical = sum(1 for a in anomalies if a.severity.value == "critical")
        report.anomalies_warning = sum(1 for a in anomalies if a.severity.value == "warning")

        # 统计异常最多的指标
        metric_counts: Dict[str, int] = {}
        for a in anomalies:
            metric_counts[a.metric_name] = metric_counts.get(a.metric_name, 0) + 1

        top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)
        report.top_anomaly_metrics = [m[0] for m in top_metrics[:5]]

    def _collect_alert_summary(self, report: DailyReportData):
        """汇总告警状态"""
        stats = self.alert_manager.get_stats()
        report.alerts_fired = stats.get("fired", 0)
        report.alerts_resolved = stats.get("resolved", 0)
        report.alerts_still_firing = stats.get("total_firing", 0)

    def _generate_recommendations(self, report: DailyReportData):
        """根据报告数据生成建议"""
        recommendations = []

        # GPU 利用率建议
        if report.gpu_utilization_avg < 50:
            recommendations.append(
                f"GPU 平均利用率偏低 ({report.gpu_utilization_avg:.0f}%), "
                f"建议考虑缩容或合并工作负载以节约成本"
            )
        elif report.gpu_utilization_avg > 90:
            recommendations.append(
                f"GPU 平均利用率过高 ({report.gpu_utilization_avg:.0f}%), "
                f"建议扩容以保留性能裕量"
            )

        # 温度建议
        if report.gpu_temperature_max > 80:
            recommendations.append(
                f"GPU 最高温度 {report.gpu_temperature_max}°C, 接近热保护阈值, "
                f"检查散热系统"
            )

        # SLO 达标建议
        for slo in report.slo_results:
            if not slo.compliant:
                recommendations.append(
                    f"SLO 未达标: {slo.name} (目标={slo.target}{slo.unit}, "
                    f"实际={slo.actual:.2f}{slo.unit}), 需要优先处理"
                )

        # 异常建议
        if report.anomalies_critical > 0:
            recommendations.append(
                f"存在 {report.anomalies_critical} 个 CRITICAL 异常, "
                f"请检查根因并处理"
            )

        if not recommendations:
            recommendations.append("系统运行正常, 无需特别关注")

        report.recommendations = recommendations

    def to_json(self, report: DailyReportData) -> Dict:
        """将报告转为 JSON 格式"""
        return {
            "report_date": report.report_date,
            "generated_at": report.generated_at,
            "gpu_summary": {
                "count": report.gpu_count,
                "utilization_avg_pct": round(report.gpu_utilization_avg, 1),
                "utilization_peak_pct": report.gpu_utilization_peak,
                "utilization_min_pct": report.gpu_utilization_min,
                "temperature_avg_c": round(report.gpu_temperature_avg, 1),
                "temperature_max_c": report.gpu_temperature_max,
                "memory_used_avg_gb": round(report.gpu_memory_used_avg_gb, 2),
                "power_total_watts": round(report.gpu_power_total_watts, 1),
            },
            "inference_summary": {
                "throughput_avg_tps": round(report.inference_throughput_avg_tps, 1),
                "kv_cache_avg": round(report.inference_kv_cache_avg, 3),
                "requests_total": report.inference_requests_total,
                "error_rate_pct": round(report.inference_error_rate, 3),
            },
            "slo_compliance": [
                {
                    "name": s.name,
                    "target": s.target,
                    "actual": round(s.actual, 2),
                    "unit": s.unit,
                    "compliant": s.compliant,
                    "compliance_pct": s.compliance_pct,
                }
                for s in report.slo_results
            ],
            "anomalies": {
                "total": report.anomalies_total,
                "critical": report.anomalies_critical,
                "warning": report.anomalies_warning,
                "top_metrics": report.top_anomaly_metrics,
            },
            "alerts": {
                "fired": report.alerts_fired,
                "resolved": report.alerts_resolved,
                "still_firing": report.alerts_still_firing,
            },
            "recommendations": report.recommendations,
        }

    def to_markdown(self, report: DailyReportData) -> str:
        """将报告转为 Markdown 文本"""
        lines = [
            f"# AI 平台每日运维报告",
            f"**日期:** {report.report_date}",
            "",
            "---",
            "",
            "## 1. GPU 集群概览",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| GPU 数量 | {report.gpu_count} |",
            f"| 平均利用率 | {report.gpu_utilization_avg:.1f}% |",
            f"| 峰值利用率 | {report.gpu_utilization_peak}% |",
            f"| 最低利用率 | {report.gpu_utilization_min}% |",
            f"| 平均温度 | {report.gpu_temperature_avg:.1f}°C |",
            f"| 最高温度 | {report.gpu_temperature_max}°C |",
            f"| 平均显存使用 | {report.gpu_memory_used_avg_gb:.2f} GB |",
            f"| 总功耗 | {report.gpu_power_total_watts:.0f} W |",
            "",
            "## 2. 推理服务状态",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 吞吐量 | {report.inference_throughput_avg_tps:.1f} tokens/s |",
            f"| KV Cache 使用率 | {report.inference_kv_cache_avg:.1%} |",
            f"| 请求总数 | {report.inference_requests_total} |",
            f"| 错误率 | {report.inference_error_rate:.3f}% |",
            "",
            "## 3. SLO 达标情况",
            "",
            f"| SLO | 目标 | 实际 | 达标 | 达标率 |",
            f"|-----|------|------|------|--------|",
        ]

        for s in report.slo_results:
            status = "YES" if s.compliant else "**NO**"
            lines.append(
                f"| {s.name} | {s.target}{s.unit} | {s.actual:.2f}{s.unit} "
                f"| {status} | {s.compliance_pct}% |"
            )

        lines.extend([
            "",
            "## 4. 异常检测",
            "",
            f"- 异常总数: {report.anomalies_total}",
            f"- CRITICAL: {report.anomalies_critical}",
            f"- WARNING: {report.anomalies_warning}",
        ])

        if report.top_anomaly_metrics:
            lines.append(f"- Top 异常指标: {', '.join(report.top_anomaly_metrics)}")

        lines.extend([
            "",
            "## 5. 告警统计",
            "",
            f"- 触发: {report.alerts_fired}",
            f"- 已解除: {report.alerts_resolved}",
            f"- 仍在触发: {report.alerts_still_firing}",
            "",
            "## 6. 建议",
            "",
        ])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append("")
        lines.append(f"---")
        lines.append(f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.generated_at))}*")

        return "\n".join(lines)
