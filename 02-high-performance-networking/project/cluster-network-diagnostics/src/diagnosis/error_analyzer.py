"""
错误分析器模块

分析IB/RoCE错误计数器的时间序列数据，检测趋势（CRC错误增长、
符号错误增长等），分类根因（坏线缆、故障收发器、交换机问题），
输出诊断报告。
"""

import logging
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import datetime

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorTrend(Enum):
    """错误趋势"""
    STABLE = "stable"           # 稳定，无增长
    SLOW_INCREASE = "slow_increase"  # 缓慢增长
    FAST_INCREASE = "fast_increase"  # 快速增长
    SPIKE = "spike"             # 突发性飙升
    DECREASING = "decreasing"  # 下降（通常是计数器重置）


class RootCause(Enum):
    """故障根因分类"""
    BAD_CABLE = "bad_cable"                     # 线缆故障
    FAULTY_TRANSCEIVER = "faulty_transceiver"   # 光模块/收发器故障
    SWITCH_PORT_ISSUE = "switch_port_issue"     # 交换机端口问题
    HCA_ISSUE = "hca_issue"                     # HCA硬件问题
    CONGESTION = "congestion"                   # 拥塞
    CONFIGURATION = "configuration"             # 配置问题
    ENVIRONMENTAL = "environmental"             # 环境因素（温度等）
    FIRMWARE_BUG = "firmware_bug"               # 固件缺陷
    UNKNOWN = "unknown"


@dataclass
class ErrorSample:
    """单次错误计数器采样"""
    timestamp: str
    counter_name: str
    value: int
    node: str = ""
    device: str = ""
    port: int = 0


@dataclass
class ErrorTrendAnalysis:
    """错误趋势分析结果"""
    counter_name: str
    node: str
    device: str
    port: int
    trend: ErrorTrend
    current_value: int
    rate_per_hour: float  # 每小时增长率
    samples: List[ErrorSample] = field(default_factory=list)
    regression_slope: float = 0.0
    r_squared: float = 0.0


@dataclass
class RootCauseAnalysis:
    """根因分析结果"""
    root_cause: RootCause
    confidence: float  # 0.0-1.0 置信度
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """诊断报告"""
    report_id: str
    generated_at: str
    time_window_hours: float
    node: str
    device: str
    port: int
    severity: ErrorSeverity
    trend_analyses: List[ErrorTrendAnalysis] = field(default_factory=list)
    root_cause_analysis: Optional[RootCauseAnalysis] = None
    summary: str = ""
    action_items: List[str] = field(default_factory=list)


class ErrorAnalyzer:
    """
    IB/RoCE错误分析器

    对历史错误计数器数据进行时间序列分析，检测异常趋势，
    执行根因分析，生成诊断报告。
    """

    # 各类错误的分析阈值
    ANALYSIS_CONFIG = {
        "SymbolErrorCounter": {
            "rate_warning": 1.0,    # 每小时1个
            "rate_critical": 10.0,   # 每小时10个
            "root_causes": [RootCause.BAD_CABLE, RootCause.FAULTY_TRANSCEIVER],
        },
        "LinkErrorRecoveryCounter": {
            "rate_warning": 0.5,
            "rate_critical": 5.0,
            "root_causes": [RootCause.BAD_CABLE, RootCause.SWITCH_PORT_ISSUE],
        },
        "LinkDownedCounter": {
            "rate_warning": 0.1,
            "rate_critical": 1.0,
            "root_causes": [RootCause.BAD_CABLE, RootCause.SWITCH_PORT_ISSUE, RootCause.HCA_ISSUE],
        },
        "PortRcvErrors": {
            "rate_warning": 1.0,
            "rate_critical": 10.0,
            "root_causes": [RootCause.BAD_CABLE, RootCause.CONGESTION],
        },
        "LocalLinkIntegrityErrors": {
            "rate_warning": 0.5,
            "rate_critical": 5.0,
            "root_causes": [RootCause.HCA_ISSUE, RootCause.FIRMWARE_BUG],
        },
        "ExcessiveBufferOverrunErrors": {
            "rate_warning": 0.1,
            "rate_critical": 1.0,
            "root_causes": [RootCause.CONGESTION, RootCause.CONFIGURATION],
        },
        "PortXmitDiscards": {
            "rate_warning": 10.0,
            "rate_critical": 100.0,
            "root_causes": [RootCause.CONGESTION, RootCause.CONFIGURATION],
        },
    }

    def __init__(self, config: dict):
        """
        初始化错误分析器

        Args:
            config: 配置字典
        """
        self.analysis_window_hours = config.get("analysis_window_hours", 24)
        self.sample_interval_minutes = config.get("sample_interval_minutes", 5)
        self.history_store_path = config.get("history_store_path", "/var/lib/cluster-diag/error_history.json")
        # 历史数据存储: {node:device:port -> {counter_name -> [ErrorSample]}}
        self.error_history: Dict[str, Dict[str, List[ErrorSample]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record_sample(self, node: str, device: str, port: int,
                      counters: Dict[str, int]) -> None:
        """
        记录一次错误计数器采样

        Args:
            node: 节点名
            device: 设备名
            port: 端口号
            counters: 计数器名称到值的映射
        """
        key = f"{node}:{device}:{port}"
        timestamp = datetime.datetime.now().isoformat()

        for counter_name, value in counters.items():
            sample = ErrorSample(
                timestamp=timestamp,
                counter_name=counter_name,
                value=value,
                node=node,
                device=device,
                port=port,
            )
            self.error_history[key][counter_name].append(sample)

        logger.debug(f"记录采样: {key}, {len(counters)} 个计数器")

    def _parse_timestamp(self, ts: str) -> datetime.datetime:
        """解析ISO格式时间戳"""
        try:
            return datetime.datetime.fromisoformat(ts)
        except ValueError:
            return datetime.datetime.now()

    def _compute_rate(self, samples: List[ErrorSample]) -> float:
        """
        计算错误增长率（每小时）

        使用最后两个采样点间的差值计算即时增长率
        """
        if len(samples) < 2:
            return 0.0

        latest = samples[-1]
        previous = samples[-2]

        value_diff = latest.value - previous.value
        if value_diff <= 0:
            return 0.0

        t1 = self._parse_timestamp(previous.timestamp)
        t2 = self._parse_timestamp(latest.timestamp)
        time_diff_hours = (t2 - t1).total_seconds() / 3600.0

        if time_diff_hours <= 0:
            return 0.0

        return value_diff / time_diff_hours

    def _linear_regression(self, samples: List[ErrorSample]) -> Tuple[float, float, float]:
        """
        对采样数据进行线性回归

        Args:
            samples: 采样列表

        Returns:
            (斜率, 截距, R²) 元组
        """
        if len(samples) < 3:
            return 0.0, 0.0, 0.0

        # 将时间转换为相对小时数
        base_time = self._parse_timestamp(samples[0].timestamp)
        x_values = []
        y_values = []

        for s in samples:
            t = self._parse_timestamp(s.timestamp)
            hours = (t - base_time).total_seconds() / 3600.0
            x_values.append(hours)
            y_values.append(float(s.value))

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # 计算R²
        ss_res = sum(
            (y - (slope * x + intercept)) ** 2
            for x, y in zip(x_values, y_values)
        )
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, intercept, r_squared

    def _detect_trend(self, samples: List[ErrorSample],
                      counter_name: str) -> ErrorTrendAnalysis:
        """
        检测单个计数器的错误趋势

        Args:
            samples: 该计数器的历史采样
            counter_name: 计数器名称

        Returns:
            ErrorTrendAnalysis对象
        """
        if not samples:
            return ErrorTrendAnalysis(
                counter_name=counter_name,
                node="", device="", port=0,
                trend=ErrorTrend.STABLE,
                current_value=0,
                rate_per_hour=0.0,
            )

        latest = samples[-1]
        rate = self._compute_rate(samples)
        slope, intercept, r_squared = self._linear_regression(samples)

        # 判断趋势
        config = self.ANALYSIS_CONFIG.get(counter_name, {})
        rate_warning = config.get("rate_warning", 1.0)
        rate_critical = config.get("rate_critical", 10.0)

        if rate <= 0:
            if len(samples) >= 2 and latest.value < samples[-2].value:
                trend = ErrorTrend.DECREASING
            else:
                trend = ErrorTrend.STABLE
        elif rate >= rate_critical:
            # 检查是否为突发（最近增长率远超平均增长率）
            if len(samples) >= 5:
                avg_rate = slope  # 使用回归斜率作为平均增长率
                if avg_rate > 0 and rate > avg_rate * 5:
                    trend = ErrorTrend.SPIKE
                else:
                    trend = ErrorTrend.FAST_INCREASE
            else:
                trend = ErrorTrend.FAST_INCREASE
        elif rate >= rate_warning:
            trend = ErrorTrend.SLOW_INCREASE
        else:
            trend = ErrorTrend.STABLE

        return ErrorTrendAnalysis(
            counter_name=counter_name,
            node=latest.node,
            device=latest.device,
            port=latest.port,
            trend=trend,
            current_value=latest.value,
            rate_per_hour=round(rate, 4),
            samples=samples[-10:],  # 保留最近10个采样
            regression_slope=round(slope, 6),
            r_squared=round(r_squared, 4),
        )

    def _analyze_root_cause(self, trend_analyses: List[ErrorTrendAnalysis]) -> RootCauseAnalysis:
        """
        基于多个错误趋势分析结果推断根因

        使用加权投票机制综合多个错误指标来判断最可能的根因
        """
        cause_scores: Dict[RootCause, float] = defaultdict(float)
        evidence: List[str] = []

        for analysis in trend_analyses:
            if analysis.trend in (ErrorTrend.STABLE, ErrorTrend.DECREASING):
                continue

            config = self.ANALYSIS_CONFIG.get(analysis.counter_name, {})
            possible_causes = config.get("root_causes", [RootCause.UNKNOWN])

            # 根据趋势严重程度给权重
            weight = {
                ErrorTrend.SLOW_INCREASE: 1.0,
                ErrorTrend.FAST_INCREASE: 3.0,
                ErrorTrend.SPIKE: 5.0,
            }.get(analysis.trend, 0.5)

            for cause in possible_causes:
                cause_scores[cause] += weight

            evidence.append(
                f"{analysis.counter_name}: 趋势={analysis.trend.value}, "
                f"当前值={analysis.current_value}, "
                f"增长率={analysis.rate_per_hour}/h"
            )

        if not cause_scores:
            return RootCauseAnalysis(
                root_cause=RootCause.UNKNOWN,
                confidence=0.0,
                evidence=["所有错误计数器均稳定"],
                recommendations=["当前无需采取行动"],
            )

        # 选择得分最高的根因
        total_score = sum(cause_scores.values())
        best_cause = max(cause_scores, key=cause_scores.get)
        confidence = cause_scores[best_cause] / total_score if total_score > 0 else 0.0

        # 生成建议
        recommendations = self._generate_recommendations(best_cause, confidence)

        return RootCauseAnalysis(
            root_cause=best_cause,
            confidence=round(confidence, 3),
            evidence=evidence,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, cause: RootCause, confidence: float) -> List[str]:
        """根据根因生成运维建议"""
        recs_map = {
            RootCause.BAD_CABLE: [
                "检查并重新插拔线缆两端",
                "使用ibcablegen验证线缆完整性",
                "如果问题持续，更换线缆",
                "记录线缆型号和批次，排查是否为批量问题",
            ],
            RootCause.FAULTY_TRANSCEIVER: [
                "检查光模块温度和光功率读数",
                "对比同型号光模块的读数判断是否异常",
                "尝试更换光模块",
                "确认光模块与交换机/HCA的兼容性",
            ],
            RootCause.SWITCH_PORT_ISSUE: [
                "将线缆迁移到交换机的备用端口测试",
                "检查交换机固件版本是否需要升级",
                "查看交换机端口的完整错误日志",
                "联系交换机厂商技术支持",
            ],
            RootCause.HCA_ISSUE: [
                "检查HCA固件版本是否为最新",
                "运行mlxfwmanager检查固件状态",
                "检查PCIe链路状态（lspci -vv）",
                "考虑更换HCA卡",
            ],
            RootCause.CONGESTION: [
                "检查网络流量分布是否均衡",
                "验证ECN/PFC配置是否正确",
                "考虑调整路由策略或ECMP配置",
                "分析应用层通信模式，优化集合通信拓扑",
            ],
            RootCause.CONFIGURATION: [
                "检查端口速率和MTU配置一致性",
                "验证子网管理器（SM）配置",
                "确认PKey和QoS设置正确",
                "检查Adaptive Routing配置",
            ],
            RootCause.ENVIRONMENTAL: [
                "检查机房温度和湿度",
                "确认设备散热正常",
                "检查供电稳定性",
            ],
            RootCause.FIRMWARE_BUG: [
                "查看供应商已知问题列表",
                "升级到最新稳定版固件",
                "收集诊断日志提交给供应商",
            ],
        }

        recs = recs_map.get(cause, ["收集更多诊断信息进一步分析"])
        if confidence < 0.5:
            recs.append("注意：根因判断置信度较低，建议综合其他信息判断")
        return recs

    def analyze_link(self, node: str, device: str, port: int) -> DiagnosisReport:
        """
        对指定链路执行完整的错误分析

        Args:
            node: 节点名
            device: 设备名
            port: 端口号

        Returns:
            DiagnosisReport诊断报告
        """
        key = f"{node}:{device}:{port}"
        logger.info(f"开始分析链路错误: {key}")

        report = DiagnosisReport(
            report_id=f"diag-{key}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            generated_at=datetime.datetime.now().isoformat(),
            time_window_hours=self.analysis_window_hours,
            node=node,
            device=device,
            port=port,
            severity=ErrorSeverity.INFO,
        )

        # 获取该链路的历史数据
        link_history = self.error_history.get(key, {})
        if not link_history:
            report.summary = f"链路 {key} 无历史数据可供分析"
            logger.warning(report.summary)
            return report

        # 对每个计数器进行趋势分析
        max_severity = ErrorSeverity.INFO
        for counter_name, samples in link_history.items():
            if counter_name not in self.ANALYSIS_CONFIG:
                continue

            # 过滤时间窗口内的样本
            cutoff = datetime.datetime.now() - datetime.timedelta(
                hours=self.analysis_window_hours
            )
            filtered = [
                s for s in samples
                if self._parse_timestamp(s.timestamp) >= cutoff
            ]

            if not filtered:
                continue

            trend_analysis = self._detect_trend(filtered, counter_name)
            report.trend_analyses.append(trend_analysis)

            # 更新严重程度
            config = self.ANALYSIS_CONFIG[counter_name]
            if trend_analysis.rate_per_hour >= config["rate_critical"]:
                max_severity = ErrorSeverity.CRITICAL
            elif trend_analysis.rate_per_hour >= config["rate_warning"]:
                if max_severity.value not in ("error", "critical"):
                    max_severity = ErrorSeverity.WARNING

        report.severity = max_severity

        # 根因分析
        report.root_cause_analysis = self._analyze_root_cause(report.trend_analyses)

        # 生成摘要
        active_trends = [
            t for t in report.trend_analyses
            if t.trend not in (ErrorTrend.STABLE, ErrorTrend.DECREASING)
        ]
        if active_trends:
            report.summary = (
                f"链路 {key} 在过去 {self.analysis_window_hours} 小时内检测到 "
                f"{len(active_trends)} 个异常趋势。"
                f"最可能的根因: {report.root_cause_analysis.root_cause.value} "
                f"(置信度: {report.root_cause_analysis.confidence:.0%})"
            )
            report.action_items = report.root_cause_analysis.recommendations[:3]
        else:
            report.summary = f"链路 {key} 在过去 {self.analysis_window_hours} 小时内运行正常"

        logger.info(f"错误分析完成: {key}, 严重程度: {max_severity.value}")
        return report

    def analyze_cluster(self) -> List[DiagnosisReport]:
        """
        对集群中所有有历史数据的链路进行错误分析

        Returns:
            所有链路的诊断报告列表
        """
        reports = []
        for key in self.error_history:
            parts = key.split(":")
            if len(parts) == 3:
                node, device, port = parts[0], parts[1], int(parts[2])
                report = self.analyze_link(node, device, port)
                reports.append(report)

        # 按严重程度排序
        severity_order = {
            ErrorSeverity.CRITICAL: 0,
            ErrorSeverity.ERROR: 1,
            ErrorSeverity.WARNING: 2,
            ErrorSeverity.INFO: 3,
        }
        reports.sort(key=lambda r: severity_order.get(r.severity, 99))

        logger.info(
            f"集群错误分析完成: {len(reports)} 条链路, "
            f"严重 {sum(1 for r in reports if r.severity == ErrorSeverity.CRITICAL)}, "
            f"告警 {sum(1 for r in reports if r.severity == ErrorSeverity.WARNING)}"
        )
        return reports

    def save_history(self, path: Optional[str] = None) -> None:
        """
        保存错误历史数据到文件

        Args:
            path: 文件路径，默认使用配置中的路径
        """
        save_path = path or self.history_store_path
        data = {}
        for key, counters in self.error_history.items():
            data[key] = {}
            for counter_name, samples in counters.items():
                data[key][counter_name] = [
                    {
                        "timestamp": s.timestamp,
                        "value": s.value,
                        "node": s.node,
                        "device": s.device,
                        "port": s.port,
                    }
                    for s in samples
                ]

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"错误历史数据已保存到: {save_path}")
        except IOError as e:
            logger.error(f"保存错误历史数据失败: {e}")

    def load_history(self, path: Optional[str] = None) -> None:
        """
        从文件加载错误历史数据

        Args:
            path: 文件路径
        """
        load_path = path or self.history_store_path
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, counters in data.items():
                for counter_name, samples_data in counters.items():
                    for s in samples_data:
                        sample = ErrorSample(
                            timestamp=s["timestamp"],
                            counter_name=counter_name,
                            value=s["value"],
                            node=s.get("node", ""),
                            device=s.get("device", ""),
                            port=s.get("port", 0),
                        )
                        self.error_history[key][counter_name].append(sample)

            logger.info(f"已加载错误历史数据: {load_path}, {len(data)} 条链路")
        except FileNotFoundError:
            logger.warning(f"错误历史文件不存在: {load_path}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"解析错误历史文件失败: {e}")
