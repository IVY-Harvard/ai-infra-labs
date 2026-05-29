"""
健康诊断模块测试

测试链路健康检查器、错误分析器和PFC死锁检测器，
使用模拟的计数器数据和样本数据。
"""

import unittest
from unittest.mock import patch, MagicMock
import datetime

from src.diagnosis.link_health_checker import (
    LinkHealthChecker, LinkHealthStatus, PortCounters, CableInfo,
    LinkHealthResult, ClusterHealthReport
)
from src.diagnosis.error_analyzer import (
    ErrorAnalyzer, ErrorSample, ErrorTrend, ErrorSeverity,
    RootCause, ErrorTrendAnalysis, DiagnosisReport
)
from src.diagnosis.pfc_deadlock_detector import (
    PFCDeadlockDetector, PFCCounters, PFCSnapshot, PFCStatus,
    PFCAnalysisResult
)


class TestLinkHealthChecker(unittest.TestCase):
    """链路健康检查器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "nodes": [
                {"hostname": "node01", "ip": "192.168.1.1"},
            ],
            "ssh_user": "root",
            "ssh_key": "~/.ssh/id_rsa",
            "max_workers": 2,
            "thresholds": {},
        }
        self.checker = LinkHealthChecker(self.config)

    def test_parse_perfquery(self):
        """测试解析perfquery输出"""
        output = """# Port counters: Lid 1 port 1 (CapMask: 0x5200)
PortSelect:......................1
CounterSelect:...................0x0000
SymbolErrorCounter:..............15
LinkErrorRecoveryCounter:........3
LinkDownedCounter:...............1
PortRcvErrors:...................0
PortRcvRemotePhysicalErrors:.....0
PortRcvSwitchRelayErrors:........0
ExcessiveBufferOverrunErrors:....0
PortXmitDiscards:................5
PortXmitConstraintErrors:........0
PortRcvConstraintErrors:.........0
LocalLinkIntegrityErrors:........2
VL15Dropped:.....................0
PortXmitData:....................1234567890
PortRcvData:.....................9876543210
PortXmitPkts:....................123456
PortRcvPkts:.....................654321
"""
        counters = self.checker._parse_perfquery(output)

        self.assertEqual(counters.symbol_errors, 15)
        self.assertEqual(counters.link_error_recovery, 3)
        self.assertEqual(counters.link_downed, 1)
        self.assertEqual(counters.port_rcv_errors, 0)
        self.assertEqual(counters.port_xmit_discards, 5)
        self.assertEqual(counters.local_link_integrity_errors, 2)
        self.assertEqual(counters.port_xmit_data, 1234567890)
        self.assertEqual(counters.port_rcv_pkts, 654321)

    def test_parse_cable_info(self):
        """测试解析线缆信息"""
        output = """Cable Type: Optical
Vendor name: Mellanox
Part number: MFS1S00-H003E
Serial number: MT2117X12345
Cable length: 3m
Temperature: 42.5 C
TX Power: -1.25 dBm
RX Power: -3.50 dBm
"""
        cable = self.checker._parse_cable_info(output)

        self.assertEqual(cable.vendor, "Mellanox")
        self.assertEqual(cable.part_number, "MFS1S00-H003E")
        self.assertEqual(cable.serial_number, "MT2117X12345")
        self.assertAlmostEqual(cable.temperature, 42.5)
        self.assertAlmostEqual(cable.tx_power_dbm, -1.25)
        self.assertAlmostEqual(cable.rx_power_dbm, -3.50)

    def test_check_link_health_healthy(self):
        """测试健康链路的判断"""
        counters = PortCounters(symbol_errors=0, link_error_recovery=0)
        cable = CableInfo(temperature=35.0, rx_power_dbm=-5.0)

        status, issues, recs = self.checker._check_link_health(counters, cable)

        self.assertEqual(status, LinkHealthStatus.HEALTHY)
        self.assertEqual(len(issues), 0)

    def test_check_link_health_warning(self):
        """测试告警状态判断"""
        counters = PortCounters(symbol_errors=15, link_error_recovery=0)
        cable = CableInfo()

        status, issues, recs = self.checker._check_link_health(counters, cable)

        self.assertEqual(status, LinkHealthStatus.WARNING)
        self.assertGreater(len(issues), 0)
        self.assertIn("符号错误数偏高", issues[0])

    def test_check_link_health_critical(self):
        """测试严重状态判断"""
        counters = PortCounters(
            symbol_errors=200,
            link_error_recovery=100,
            port_rcv_errors=150,
        )
        cable = CableInfo(temperature=80.0, rx_power_dbm=-15.0)

        status, issues, recs = self.checker._check_link_health(counters, cable)

        self.assertEqual(status, LinkHealthStatus.CRITICAL)
        self.assertGreater(len(issues), 2)
        self.assertGreater(len(recs), 0)

    def test_check_link_health_cable_temp(self):
        """测试线缆温度告警"""
        counters = PortCounters()
        cable = CableInfo(temperature=70.0)

        status, issues, recs = self.checker._check_link_health(counters, cable)

        self.assertEqual(status, LinkHealthStatus.WARNING)
        self.assertTrue(any("温度" in i for i in issues))

    def test_port_counters_total_errors(self):
        """测试总错误计数"""
        counters = PortCounters(
            symbol_errors=10,
            link_error_recovery=5,
            link_downed=2,
            port_rcv_errors=3,
            port_rcv_remote_physical_errors=1,
            excessive_buffer_overrun=0,
            local_link_integrity_errors=4,
        )
        self.assertEqual(counters.total_errors, 25)

    def test_escalate_status(self):
        """测试状态升级逻辑"""
        # WARNING -> CRITICAL = CRITICAL
        result = LinkHealthChecker._escalate_status(
            LinkHealthStatus.WARNING, LinkHealthStatus.CRITICAL
        )
        self.assertEqual(result, LinkHealthStatus.CRITICAL)

        # CRITICAL -> WARNING = CRITICAL (不降级)
        result = LinkHealthChecker._escalate_status(
            LinkHealthStatus.CRITICAL, LinkHealthStatus.WARNING
        )
        self.assertEqual(result, LinkHealthStatus.CRITICAL)


class TestErrorAnalyzer(unittest.TestCase):
    """错误分析器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "analysis_window_hours": 24,
            "sample_interval_minutes": 5,
            "history_store_path": "/tmp/test_error_history.json",
        }
        self.analyzer = ErrorAnalyzer(self.config)

    def test_record_sample(self):
        """测试记录采样"""
        counters = {
            "SymbolErrorCounter": 10,
            "LinkErrorRecoveryCounter": 2,
        }
        self.analyzer.record_sample("node01", "mlx5_0", 1, counters)

        key = "node01:mlx5_0:1"
        self.assertIn(key, self.analyzer.error_history)
        self.assertEqual(len(self.analyzer.error_history[key]["SymbolErrorCounter"]), 1)
        self.assertEqual(
            self.analyzer.error_history[key]["SymbolErrorCounter"][0].value, 10
        )

    def test_compute_rate(self):
        """测试增长率计算"""
        now = datetime.datetime.now()
        samples = [
            ErrorSample(
                timestamp=(now - datetime.timedelta(hours=1)).isoformat(),
                counter_name="SymbolErrorCounter",
                value=10,
            ),
            ErrorSample(
                timestamp=now.isoformat(),
                counter_name="SymbolErrorCounter",
                value=20,
            ),
        ]

        rate = self.analyzer._compute_rate(samples)
        # 1小时内从10增长到20，速率约为10/h
        self.assertAlmostEqual(rate, 10.0, places=0)

    def test_compute_rate_no_increase(self):
        """测试无增长时的速率"""
        now = datetime.datetime.now()
        samples = [
            ErrorSample(
                timestamp=(now - datetime.timedelta(hours=1)).isoformat(),
                counter_name="SymbolErrorCounter",
                value=10,
            ),
            ErrorSample(
                timestamp=now.isoformat(),
                counter_name="SymbolErrorCounter",
                value=10,
            ),
        ]

        rate = self.analyzer._compute_rate(samples)
        self.assertEqual(rate, 0.0)

    def test_linear_regression(self):
        """测试线性回归"""
        now = datetime.datetime.now()
        # 创建线性增长的数据: 每小时增加10
        samples = []
        for i in range(5):
            samples.append(ErrorSample(
                timestamp=(now - datetime.timedelta(hours=4-i)).isoformat(),
                counter_name="test",
                value=i * 10,
            ))

        slope, intercept, r_squared = self.analyzer._linear_regression(samples)

        # 斜率应约为10/h
        self.assertAlmostEqual(slope, 10.0, places=0)
        # R²应接近1（完美线性）
        self.assertGreater(r_squared, 0.95)

    def test_detect_trend_stable(self):
        """测试稳定趋势检测"""
        now = datetime.datetime.now()
        samples = [
            ErrorSample(
                timestamp=(now - datetime.timedelta(hours=1)).isoformat(),
                counter_name="SymbolErrorCounter",
                value=5, node="n1", device="d1", port=1,
            ),
            ErrorSample(
                timestamp=now.isoformat(),
                counter_name="SymbolErrorCounter",
                value=5, node="n1", device="d1", port=1,
            ),
        ]

        analysis = self.analyzer._detect_trend(samples, "SymbolErrorCounter")
        self.assertEqual(analysis.trend, ErrorTrend.STABLE)

    def test_detect_trend_fast_increase(self):
        """测试快速增长趋势检测"""
        now = datetime.datetime.now()
        samples = [
            ErrorSample(
                timestamp=(now - datetime.timedelta(hours=1)).isoformat(),
                counter_name="SymbolErrorCounter",
                value=0, node="n1", device="d1", port=1,
            ),
            ErrorSample(
                timestamp=now.isoformat(),
                counter_name="SymbolErrorCounter",
                value=50, node="n1", device="d1", port=1,
            ),
        ]

        analysis = self.analyzer._detect_trend(samples, "SymbolErrorCounter")
        self.assertEqual(analysis.trend, ErrorTrend.FAST_INCREASE)
        self.assertAlmostEqual(analysis.rate_per_hour, 50.0, places=0)

    def test_analyze_root_cause_bad_cable(self):
        """测试根因分析 - 坏线缆"""
        trend_analyses = [
            ErrorTrendAnalysis(
                counter_name="SymbolErrorCounter",
                node="n1", device="d1", port=1,
                trend=ErrorTrend.FAST_INCREASE,
                current_value=100,
                rate_per_hour=20.0,
            ),
            ErrorTrendAnalysis(
                counter_name="LinkErrorRecoveryCounter",
                node="n1", device="d1", port=1,
                trend=ErrorTrend.SLOW_INCREASE,
                current_value=10,
                rate_per_hour=2.0,
            ),
        ]

        rca = self.analyzer._analyze_root_cause(trend_analyses)
        # 两个计数器都指向BAD_CABLE
        self.assertEqual(rca.root_cause, RootCause.BAD_CABLE)
        self.assertGreater(rca.confidence, 0.3)
        self.assertGreater(len(rca.recommendations), 0)

    def test_analyze_root_cause_all_stable(self):
        """测试根因分析 - 全部稳定"""
        trend_analyses = [
            ErrorTrendAnalysis(
                counter_name="SymbolErrorCounter",
                node="n1", device="d1", port=1,
                trend=ErrorTrend.STABLE,
                current_value=0,
                rate_per_hour=0.0,
            ),
        ]

        rca = self.analyzer._analyze_root_cause(trend_analyses)
        self.assertEqual(rca.root_cause, RootCause.UNKNOWN)
        self.assertEqual(rca.confidence, 0.0)

    def test_analyze_link_no_history(self):
        """测试分析无历史数据的链路"""
        report = self.analyzer.analyze_link("node01", "mlx5_0", 1)

        self.assertIn("无历史数据", report.summary)
        self.assertEqual(report.severity, ErrorSeverity.INFO)

    def test_analyze_link_with_data(self):
        """测试分析有历史数据的链路"""
        now = datetime.datetime.now()
        # 记录多个采样点
        for i in range(5):
            counters = {"SymbolErrorCounter": i * 20}
            # 手动设置时间戳
            key = "node01:mlx5_0:1"
            sample = ErrorSample(
                timestamp=(now - datetime.timedelta(hours=4-i)).isoformat(),
                counter_name="SymbolErrorCounter",
                value=i * 20,
                node="node01", device="mlx5_0", port=1,
            )
            self.analyzer.error_history[key]["SymbolErrorCounter"].append(sample)

        report = self.analyzer.analyze_link("node01", "mlx5_0", 1)

        self.assertIsNotNone(report.root_cause_analysis)
        self.assertGreater(len(report.trend_analyses), 0)


class TestPFCDeadlockDetector(unittest.TestCase):
    """PFC死锁检测器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "nodes": [
                {"hostname": "node01", "ip": "192.168.1.1",
                 "rdma_devices": [{"name": "mlx5_0", "ports": [1]}]},
                {"hostname": "node02", "ip": "192.168.1.2",
                 "rdma_devices": [{"name": "mlx5_0", "ports": [1]}]},
            ],
            "ssh_user": "root",
            "ssh_key": "~/.ssh/id_rsa",
            "pfc_thresholds": {
                "pause_rate_warning": 100,
                "pause_rate_storm": 1000,
            },
            "roce_priorities": [3],
        }
        self.detector = PFCDeadlockDetector(self.config)

    def test_parse_ethtool_pfc(self):
        """测试解析ethtool PFC输出"""
        output = """tx_prio0_pause: 0
tx_prio1_pause: 0
tx_prio2_pause: 0
tx_prio3_pause: 1500
tx_prio4_pause: 0
tx_prio5_pause: 0
tx_prio6_pause: 0
tx_prio7_pause: 0
rx_prio0_pause: 0
rx_prio1_pause: 0
rx_prio2_pause: 0
rx_prio3_pause: 800
rx_prio4_pause: 0
rx_prio5_pause: 0
rx_prio6_pause: 0
rx_prio7_pause: 0
tx_prio3_pause_duration: 50000
rx_prio3_pause_duration: 30000
"""
        counters = PFCCounters(node="node01", device="mlx5_0", port=1)
        result = self.detector._parse_ethtool_pfc(output, counters)

        self.assertEqual(result.tx_pause[3], 1500)
        self.assertEqual(result.rx_pause[3], 800)
        self.assertEqual(result.tx_pause_duration[3], 50000)
        self.assertEqual(result.rx_pause_duration[3], 30000)
        # 其他优先级应为0
        self.assertEqual(result.tx_pause[0], 0)
        self.assertEqual(result.rx_pause[7], 0)

    def test_compute_pfc_rates(self):
        """测试PFC速率计算"""
        now = datetime.datetime.now()
        prev_time = (now - datetime.timedelta(seconds=10)).isoformat()
        curr_time = now.isoformat()

        prev_snapshot = PFCSnapshot(timestamp=prev_time)
        curr_snapshot = PFCSnapshot(timestamp=curr_time)

        key = "node01:mlx5_0:1"
        prev_snapshot.counters[key] = PFCCounters(
            node="node01", device="mlx5_0", port=1,
            tx_pause={i: 0 for i in range(8)},
            rx_pause={i: 0 for i in range(8)},
        )
        prev_snapshot.counters[key].tx_pause[3] = 1000

        curr_snapshot.counters[key] = PFCCounters(
            node="node01", device="mlx5_0", port=1,
            tx_pause={i: 0 for i in range(8)},
            rx_pause={i: 0 for i in range(8)},
        )
        curr_snapshot.counters[key].tx_pause[3] = 2000

        rates = self.detector._compute_pfc_rates(prev_snapshot, curr_snapshot)

        # 10秒内增加了1000个暂停帧，速率=100/s
        self.assertIn(key, rates)
        self.assertAlmostEqual(rates[key][3], 100.0, places=0)

    def test_detect_pfc_storm(self):
        """测试PFC风暴检测"""
        rates = {
            "node01:mlx5_0:1": {i: 0.0 for i in range(8)},
        }
        rates["node01:mlx5_0:1"][3] = 5000.0  # 超过storm阈值

        storms = self.detector._detect_pfc_storm(rates)

        self.assertEqual(len(storms), 1)
        self.assertIn("PFC风暴", storms[0])
        self.assertIn("node01", storms[0])

    def test_detect_pfc_storm_normal(self):
        """测试正常PFC无风暴"""
        rates = {
            "node01:mlx5_0:1": {i: 0.0 for i in range(8)},
        }
        rates["node01:mlx5_0:1"][3] = 50.0  # 低于告警阈值

        storms = self.detector._detect_pfc_storm(rates)
        self.assertEqual(len(storms), 0)

    def test_detect_cyclic_dependency(self):
        """测试循环依赖检测"""
        snapshot = PFCSnapshot(timestamp=datetime.datetime.now().isoformat())

        # 节点A和B都同时发送和接收大量PFC暂停
        snapshot.counters["node01:mlx5_0:1"] = PFCCounters(
            node="node01", device="mlx5_0", port=1,
            tx_pause={3: 500, **{i: 0 for i in range(8) if i != 3}},
            rx_pause={3: 500, **{i: 0 for i in range(8) if i != 3}},
        )
        snapshot.counters["node02:mlx5_0:1"] = PFCCounters(
            node="node02", device="mlx5_0", port=1,
            tx_pause={3: 600, **{i: 0 for i in range(8) if i != 3}},
            rx_pause={3: 600, **{i: 0 for i in range(8) if i != 3}},
        )

        patterns = self.detector._detect_cyclic_dependency(snapshot)

        self.assertGreater(len(patterns), 0)
        self.assertIn("node01", patterns[0].involved_nodes)
        self.assertIn("node02", patterns[0].involved_nodes)

    def test_analyze_insufficient_snapshots(self):
        """测试快照不足时的分析"""
        # 只有一个快照
        self.detector.snapshots = [
            PFCSnapshot(timestamp=datetime.datetime.now().isoformat())
        ]

        result = self.detector.analyze()

        self.assertEqual(result.overall_status, PFCStatus.NORMAL)
        self.assertGreater(len(result.recommendations), 0)
        self.assertIn("至少两个PFC快照", result.recommendations[0])

    def test_analyze_normal(self):
        """测试正常状态分析"""
        now = datetime.datetime.now()

        prev = PFCSnapshot(timestamp=(now - datetime.timedelta(seconds=10)).isoformat())
        curr = PFCSnapshot(timestamp=now.isoformat())

        key = "node01:mlx5_0:1"
        prev.counters[key] = PFCCounters(
            node="node01", device="mlx5_0", port=1,
            tx_pause={i: 0 for i in range(8)},
            rx_pause={i: 0 for i in range(8)},
        )
        curr.counters[key] = PFCCounters(
            node="node01", device="mlx5_0", port=1,
            tx_pause={i: 0 for i in range(8)},
            rx_pause={i: 0 for i in range(8)},
        )

        self.detector.snapshots = [prev, curr]
        result = self.detector.analyze()

        self.assertEqual(result.overall_status, PFCStatus.NORMAL)
        self.assertEqual(len(result.storm_alerts), 0)

    def test_detect_sustained_pause(self):
        """测试持续暂停检测"""
        rates = {
            "node01:mlx5_0:1": {i: 0.0 for i in range(8)},
        }
        rates["node01:mlx5_0:1"][3] = 200.0  # 超过告警阈值

        # 模拟连续3个周期
        for _ in range(3):
            hotspots = self.detector._detect_sustained_pause(rates)

        # 第3次应该检测到持续暂停
        self.assertGreater(len(hotspots), 0)
        self.assertIn("持续PFC暂停", hotspots[0])


if __name__ == "__main__":
    unittest.main()
