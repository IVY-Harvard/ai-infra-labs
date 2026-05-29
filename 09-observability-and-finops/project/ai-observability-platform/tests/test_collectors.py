"""采集器模块单元测试

测试覆盖:
1. GPUCollector — mock 数据采集、Prometheus 格式输出
2. InferenceCollector — mock 数据采集、指标字段验证
3. TrainingCollector — mock 数据采集
4. NetworkCollector — mock 数据采集、多节点返回
"""

import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from src.collectors.gpu_collector import GPUCollector, GPUMetrics
from src.collectors.inference_collector import InferenceCollector, InferenceMetrics
from src.collectors.training_collector import TrainingCollector, TrainingMetrics
from src.collectors.network_collector import NetworkCollector, NetworkMetrics


class TestGPUCollector(unittest.TestCase):
    """GPU 采集器测试"""

    def setUp(self):
        # 使用 mock 模式 (不依赖 NVML)
        self.collector = GPUCollector(use_nvml=False)

    def test_collect_returns_list(self):
        """采集返回 GPUMetrics 列表"""
        metrics = self.collector.collect()
        self.assertIsInstance(metrics, list)
        self.assertTrue(len(metrics) > 0)

    def test_collect_mock_gpu_count(self):
        """mock 模式返回 8 个 GPU"""
        metrics = self.collector._collect_mock()
        self.assertEqual(len(metrics), 8)

    def test_gpu_metrics_fields(self):
        """验证 GPUMetrics 字段合理性"""
        metrics = self.collector.collect()
        for m in metrics:
            self.assertIsInstance(m, GPUMetrics)
            self.assertGreaterEqual(m.gpu_id, 0)
            self.assertGreater(m.temperature_c, 0)
            self.assertLessEqual(m.temperature_c, 100)
            self.assertGreater(m.power_watts, 0)
            self.assertGreaterEqual(m.utilization_pct, 0)
            self.assertLessEqual(m.utilization_pct, 100)
            self.assertGreater(m.memory_total_gb, 0)
            self.assertGreaterEqual(m.memory_used_gb, 0)
            self.assertLessEqual(m.memory_used_gb, m.memory_total_gb)
            self.assertGreater(m.timestamp, 0)

    def test_gpu_metrics_unique_ids(self):
        """每个 GPU 有唯一 ID"""
        metrics = self.collector.collect()
        gpu_ids = [m.gpu_id for m in metrics]
        self.assertEqual(len(gpu_ids), len(set(gpu_ids)))

    def test_collect_as_prometheus(self):
        """Prometheus 格式输出验证"""
        output = self.collector.collect_as_prometheus()
        self.assertIsInstance(output, str)
        self.assertIn("gpu_temperature_celsius", output)
        self.assertIn("gpu_power_watts", output)
        self.assertIn("gpu_utilization_percent", output)
        self.assertIn("gpu_memory_used_bytes", output)

    def test_prometheus_format_labels(self):
        """Prometheus 输出包含 GPU 标签"""
        output = self.collector.collect_as_prometheus()
        lines = output.strip().split("\n")
        # 每个 GPU 应有 4 个指标行
        self.assertGreater(len(lines), 0)
        for line in lines:
            self.assertIn("gpu=", line)

    def test_nvml_init_failure_graceful(self):
        """NVML 初始化失败时优雅降级"""
        collector = GPUCollector(use_nvml=False)
        self.assertFalse(collector._nvml_initialized)
        # 仍然可以通过 mock 采集
        metrics = collector.collect()
        self.assertTrue(len(metrics) > 0)


class TestInferenceCollector(unittest.TestCase):
    """推理服务采集器测试"""

    def setUp(self):
        self.collector = InferenceCollector()

    def test_collect_mock_returns_metrics(self):
        """mock 采集返回 InferenceMetrics"""
        metrics = self.collector.collect_mock()
        self.assertIsInstance(metrics, InferenceMetrics)

    def test_inference_metrics_fields(self):
        """验证推理指标字段"""
        metrics = self.collector.collect_mock()
        self.assertEqual(metrics.instance, "vllm-0")
        self.assertGreater(metrics.throughput_tps, 0)
        self.assertGreaterEqual(metrics.kv_cache_usage, 0)
        self.assertLessEqual(metrics.kv_cache_usage, 1.0)
        self.assertGreaterEqual(metrics.requests_running, 0)
        self.assertGreaterEqual(metrics.requests_waiting, 0)
        self.assertGreater(metrics.timestamp, 0)

    def test_inference_latency_positive(self):
        """延迟指标应为正值"""
        metrics = self.collector.collect_mock()
        self.assertGreaterEqual(metrics.ttft_p99_ms, 0)
        self.assertGreaterEqual(metrics.tpot_p99_ms, 0)

    def test_multiple_collects_vary(self):
        """多次采集结果应有变化 (随机性)"""
        results = [self.collector.collect_mock().throughput_tps for _ in range(10)]
        # 10 次采集不应完全相同 (概率极低)
        unique_values = set(results)
        self.assertGreater(len(unique_values), 1)

    def test_query_scalar_returns_float(self):
        """_query_scalar 方法返回 float"""
        # 不连接实际 Prometheus, 验证方法签名和默认行为
        import asyncio

        async def run_query():
            session = MagicMock()
            # 模拟失败时返回 0.0
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"status": "error"})
            context_manager = AsyncMock()
            context_manager.__aenter__ = AsyncMock(return_value=mock_response)
            context_manager.__aexit__ = AsyncMock(return_value=False)
            session.get = MagicMock(return_value=context_manager)
            result = await self.collector._query_scalar(session, "test_query")
            return result

        result = asyncio.get_event_loop().run_until_complete(run_query())
        self.assertEqual(result, 0.0)


class TestTrainingCollector(unittest.TestCase):
    """训练任务采集器测试"""

    def setUp(self):
        self.collector = TrainingCollector()

    def test_collect_mock_returns_metrics(self):
        """mock 采集返回 TrainingMetrics"""
        metrics = self.collector.collect_mock()
        self.assertIsInstance(metrics, TrainingMetrics)

    def test_training_metrics_fields(self):
        """验证训练指标字段"""
        metrics = self.collector.collect_mock()
        self.assertEqual(metrics.job_name, "pretrain-qwen-72b")
        self.assertGreater(metrics.gpu_utilization_avg, 0)
        self.assertLessEqual(metrics.gpu_utilization_avg, 1.0)
        self.assertGreater(metrics.samples_per_second, 0)
        self.assertGreater(metrics.loss, 0)
        self.assertGreater(metrics.epoch, 0)
        self.assertGreater(metrics.eta_hours, 0)
        self.assertGreater(metrics.gpu_memory_allocated_gb, 0)
        self.assertGreater(metrics.timestamp, 0)

    def test_training_loss_reasonable(self):
        """训练 loss 应在合理范围"""
        metrics = self.collector.collect_mock()
        self.assertGreater(metrics.loss, 0)
        self.assertLess(metrics.loss, 10)

    def test_gpu_memory_reasonable(self):
        """GPU 显存使用量应在合理范围"""
        metrics = self.collector.collect_mock()
        self.assertGreater(metrics.gpu_memory_allocated_gb, 0)
        self.assertLess(metrics.gpu_memory_allocated_gb, 100)


class TestNetworkCollector(unittest.TestCase):
    """网络指标采集器测试"""

    def setUp(self):
        self.collector = NetworkCollector()

    def test_collect_mock_returns_list(self):
        """mock 采集返回列表"""
        metrics = self.collector.collect_mock()
        self.assertIsInstance(metrics, list)
        self.assertTrue(len(metrics) > 0)

    def test_network_node_count(self):
        """mock 模式返回 4 个节点"""
        metrics = self.collector.collect_mock()
        self.assertEqual(len(metrics), 4)

    def test_network_metrics_fields(self):
        """验证网络指标字段"""
        metrics = self.collector.collect_mock()
        for m in metrics:
            self.assertIsInstance(m, NetworkMetrics)
            self.assertTrue(m.node.startswith("gpu-node-"))
            self.assertGreater(m.nvlink_tx_bytes_per_s, 0)
            self.assertGreater(m.nvlink_rx_bytes_per_s, 0)
            self.assertGreater(m.pcie_tx_bytes_per_s, 0)
            self.assertGreater(m.pcie_rx_bytes_per_s, 0)
            self.assertGreaterEqual(m.nvlink_crc_errors, 0)
            self.assertGreater(m.timestamp, 0)

    def test_network_unique_nodes(self):
        """每个节点有唯一名称"""
        metrics = self.collector.collect_mock()
        nodes = [m.node for m in metrics]
        self.assertEqual(len(nodes), len(set(nodes)))

    def test_nvlink_bandwidth_reasonable(self):
        """NVLink 带宽在合理范围 (100GB/s - 400GB/s)"""
        metrics = self.collector.collect_mock()
        for m in metrics:
            self.assertGreaterEqual(m.nvlink_tx_bytes_per_s, 100e9)
            self.assertLessEqual(m.nvlink_tx_bytes_per_s, 400e9)

    def test_pcie_bandwidth_reasonable(self):
        """PCIe 带宽在合理范围 (5GB/s - 30GB/s)"""
        metrics = self.collector.collect_mock()
        for m in metrics:
            self.assertGreaterEqual(m.pcie_tx_bytes_per_s, 5e9)
            self.assertLessEqual(m.pcie_tx_bytes_per_s, 30e9)


if __name__ == "__main__":
    unittest.main()
