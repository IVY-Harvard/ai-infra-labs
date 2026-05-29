"""
基准测试模块测试

测试带宽测试、延迟分析和NCCL基准测试的输出解析和结果聚合逻辑。
"""

import unittest
from unittest.mock import patch, MagicMock
import json

from src.benchmark.bandwidth_tester import (
    BandwidthTester, BandwidthResult, AllPairsResult,
    BandwidthTestType, TransportType, LINK_SPEED_MAX_BW
)
from src.benchmark.latency_profiler import (
    LatencyProfiler, LatencyResult, LatencyProfile,
    LatencyTestType
)
from src.benchmark.nccl_benchmark import (
    NCCLBenchmark, NCCLTestResult, NCCLTestPoint,
    CollectiveOp, NCCLBenchmarkSuite
)


class TestBandwidthTester(unittest.TestCase):
    """带宽测试器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "nodes": [
                {"hostname": "node01", "ip": "192.168.1.1"},
                {"hostname": "node02", "ip": "192.168.1.2"},
            ],
            "ssh_user": "root",
            "ssh_key": "~/.ssh/id_rsa",
            "bandwidth_benchmark": {
                "message_sizes": [65536, 1048576],
                "iterations": 1000,
                "transport": "RC",
                "device": "mlx5_0",
                "ib_port": 1,
                "link_speed": "HDR",
            },
        }
        self.tester = BandwidthTester(self.config)

    def test_parse_bw_output(self):
        """测试解析ib_write_bw输出"""
        output = """************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF
 Number of qps   : 1
 Connection type  : RC
 TX depth         : 128
 CQ Moderation    : 100
 Mtu              : 4096[B]
 Link type        : IB
 Max inline data  : 0[B]
 rdma_cm QPs      : OFF
 Data ex. method  : Ethernet
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 65536      5000            195.45             194.82               0.371628
---------------------------------------------------------------------------------------
"""
        result = self.tester._parse_bw_output(output)

        self.assertIsNotNone(result)
        self.assertEqual(result["message_size"], 65536)
        self.assertEqual(result["iterations"], 5000)
        self.assertAlmostEqual(result["bw_peak_gbps"], 195.45, places=2)
        self.assertAlmostEqual(result["bw_avg_gbps"], 194.82, places=2)
        self.assertAlmostEqual(result["msg_rate_mpps"], 0.371628, places=4)

    def test_parse_bw_output_invalid(self):
        """测试解析无效输出"""
        output = "Connection refused\nError: could not connect"
        result = self.tester._parse_bw_output(output)
        self.assertIsNone(result)

    def test_build_bw_command_server(self):
        """测试构建服务端命令"""
        cmd = self.tester._build_bw_command(
            BandwidthTestType.WRITE, 65536, is_server=True, port=18515
        )
        self.assertIn("ib_write_bw", cmd)
        self.assertIn("-d mlx5_0", cmd)
        self.assertIn("-s 65536", cmd)
        self.assertIn("-p 18515", cmd)
        self.assertIn("--report_gbits", cmd)
        self.assertNotIn("192.168", cmd)  # 服务端不应有IP

    def test_build_bw_command_client(self):
        """测试构建客户端命令"""
        cmd = self.tester._build_bw_command(
            BandwidthTestType.READ, 1048576, is_server=False,
            server_ip="192.168.1.2", port=18516
        )
        self.assertIn("ib_read_bw", cmd)
        self.assertIn("-s 1048576", cmd)
        self.assertIn("192.168.1.2", cmd)
        self.assertIn("-p 18516", cmd)

    def test_compute_aggregate_stats(self):
        """测试聚合统计计算"""
        all_pairs = AllPairsResult(
            test_time="2024-01-01",
            test_type=BandwidthTestType.WRITE,
            node_count=3,
            pair_count=3,
        )

        # 模拟3对结果
        bandwidths = [180.0, 190.0, 195.0]
        for i, bw in enumerate(bandwidths):
            all_pairs.results.append(BandwidthResult(
                source_node=f"node{i}",
                dest_node=f"node{i+1}",
                test_type=BandwidthTestType.WRITE,
                transport=TransportType.RC,
                message_size=65536,
                iterations=1000,
                bandwidth_mbps=bw * 1000 / 8,
                bandwidth_gbps=bw,
                theoretical_max_gbps=200.0,
                efficiency=bw / 200.0,
                duration_seconds=10.0,
                success=True,
            ))

        self.tester._compute_aggregate_stats(all_pairs)

        self.assertAlmostEqual(all_pairs.avg_bandwidth_gbps, 188.33, places=1)
        self.assertEqual(all_pairs.min_bandwidth_gbps, 180.0)
        self.assertEqual(all_pairs.max_bandwidth_gbps, 195.0)
        self.assertGreater(all_pairs.avg_efficiency, 0.9)

    def test_compute_aggregate_stats_with_underperformers(self):
        """测试识别性能不达标的节点对"""
        all_pairs = AllPairsResult(
            test_time="2024-01-01",
            test_type=BandwidthTestType.WRITE,
            node_count=2,
            pair_count=2,
        )

        # 一个正常，一个不达标
        all_pairs.results = [
            BandwidthResult(
                source_node="node01", dest_node="node02",
                test_type=BandwidthTestType.WRITE,
                transport=TransportType.RC,
                message_size=65536, iterations=1000,
                bandwidth_mbps=0, bandwidth_gbps=190.0,
                theoretical_max_gbps=200.0,
                efficiency=0.95, duration_seconds=10.0, success=True,
            ),
            BandwidthResult(
                source_node="node03", dest_node="node04",
                test_type=BandwidthTestType.WRITE,
                transport=TransportType.RC,
                message_size=65536, iterations=1000,
                bandwidth_mbps=0, bandwidth_gbps=100.0,
                theoretical_max_gbps=200.0,
                efficiency=0.50, duration_seconds=10.0, success=True,
            ),
        ]

        self.tester._compute_aggregate_stats(all_pairs)

        # 效率50%应该被标记为不达标
        self.assertEqual(len(all_pairs.underperforming_pairs), 1)
        self.assertIn("node03", all_pairs.underperforming_pairs[0])

    def test_link_speed_mapping(self):
        """测试链路速度到带宽的映射"""
        self.assertEqual(LINK_SPEED_MAX_BW["HDR"], 200.0)
        self.assertEqual(LINK_SPEED_MAX_BW["NDR"], 400.0)
        self.assertEqual(LINK_SPEED_MAX_BW["EDR"], 100.0)


class TestLatencyProfiler(unittest.TestCase):
    """延迟分析器测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "nodes": [
                {"hostname": "node01", "ip": "192.168.1.1"},
                {"hostname": "node02", "ip": "192.168.1.2"},
            ],
            "ssh_user": "root",
            "latency_benchmark": {
                "message_sizes": [2, 64, 512],
                "iterations": 5000,
                "device": "mlx5_0",
            },
        }
        self.profiler = LatencyProfiler(self.config)

    def test_parse_lat_output(self):
        """测试解析ib_write_lat输出"""
        output = """************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write Latency Test
 Connection type  : RC
 Number of qps   : 1
 Mtu              : 4096[B]
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
 2       10000          1.15           12.34        1.28               1.35           0.42            2.56                   5.89
---------------------------------------------------------------------------------------
"""
        result = self.profiler._parse_lat_output(output)

        self.assertIsNotNone(result)
        self.assertEqual(result["message_size"], 2)
        self.assertEqual(result["iterations"], 10000)
        self.assertAlmostEqual(result["min_us"], 1.15, places=2)
        self.assertAlmostEqual(result["max_us"], 12.34, places=2)
        self.assertAlmostEqual(result["avg_us"], 1.35, places=2)
        self.assertAlmostEqual(result["p99_us"], 2.56, places=2)
        self.assertAlmostEqual(result["p999_us"], 5.89, places=2)

    def test_parse_lat_output_invalid(self):
        """测试解析无效输出"""
        output = "Error: cannot connect to server"
        result = self.profiler._parse_lat_output(output)
        self.assertIsNone(result)

    def test_compute_percentiles(self):
        """测试分位数计算"""
        # 100个样本，从1到100
        values = list(range(1, 101))

        percentiles = self.profiler.compute_percentiles(values)

        self.assertAlmostEqual(percentiles["p50"], 50.5, places=0)
        self.assertAlmostEqual(percentiles["p99"], 99.0, places=0)

    def test_compute_percentiles_empty(self):
        """测试空列表的分位数"""
        percentiles = self.profiler.compute_percentiles([])
        self.assertEqual(percentiles["p50"], 0)

    def test_compute_percentiles_single(self):
        """测试单元素列表"""
        percentiles = self.profiler.compute_percentiles([5.0])
        self.assertEqual(percentiles["p50"], 5.0)
        self.assertEqual(percentiles["p99"], 5.0)

    def test_build_lat_command_server(self):
        """测试构建延迟测试服务端命令"""
        cmd = self.profiler._build_lat_command(
            LatencyTestType.WRITE, 2, is_server=True, port=19515
        )
        self.assertIn("ib_write_lat", cmd)
        self.assertIn("-s 2", cmd)
        self.assertIn("-p 19515", cmd)

    def test_build_lat_command_client(self):
        """测试构建延迟测试客户端命令"""
        cmd = self.profiler._build_lat_command(
            LatencyTestType.READ, 64, is_server=False,
            server_ip="10.0.0.1", port=19516
        )
        self.assertIn("ib_read_lat", cmd)
        self.assertIn("10.0.0.1", cmd)

    def test_compute_profile_stats(self):
        """测试延迟概况统计"""
        profile = LatencyProfile(
            test_time="2024-01-01",
            test_type=LatencyTestType.WRITE,
            node_count=3,
            pair_count=3,
            message_size=2,
        )

        # 模拟结果
        latencies = [1.2, 1.5, 5.0]  # 最后一个是离群点
        for i, lat in enumerate(latencies):
            profile.results.append(LatencyResult(
                source_node=f"node{i}",
                dest_node=f"node{i+1}",
                test_type=LatencyTestType.WRITE,
                message_size=2,
                iterations=5000,
                avg_latency_us=lat,
                p50_latency_us=lat,
                p99_latency_us=lat * 2,
                p999_latency_us=lat * 3,
                success=True,
            ))

        self.profiler._compute_profile_stats(profile)

        # 平均值约为 (1.2+1.5+5.0)/3 = 2.57
        self.assertAlmostEqual(profile.cluster_avg_latency_us, 2.57, places=1)
        # 5.0 > 2.57 * 2 = 5.14... 实际上不超过，所以调整测试
        # 让离群点更明显
        profile.results[-1].avg_latency_us = 10.0
        self.profiler._compute_profile_stats(profile)
        # 现在平均约4.23, 10.0 > 4.23*2 = 8.47 -> 是离群点
        self.assertGreater(len(profile.outlier_pairs), 0)

    def test_correlate_with_topology(self):
        """测试延迟与拓扑距离关联"""
        profile = LatencyProfile(
            test_time="2024-01-01",
            test_type=LatencyTestType.WRITE,
            node_count=3,
            pair_count=2,
            message_size=2,
        )
        profile.results = [
            LatencyResult(
                source_node="node01", dest_node="node02",
                test_type=LatencyTestType.WRITE,
                message_size=2, iterations=5000,
                avg_latency_us=1.5, success=True,
            ),
            LatencyResult(
                source_node="node01", dest_node="node03",
                test_type=LatencyTestType.WRITE,
                message_size=2, iterations=5000,
                avg_latency_us=3.0, success=True,
            ),
        ]

        distances = {
            ("node01", "node02"): 2,   # 2跳
            ("node01", "node03"): 4,   # 4跳
        }

        self.profiler.correlate_with_topology(profile, distances)

        self.assertEqual(profile.results[0].hop_count, 2)
        self.assertEqual(profile.results[1].hop_count, 4)
        self.assertIn(2, profile.latency_by_hops)
        self.assertIn(4, profile.latency_by_hops)


class TestNCCLBenchmark(unittest.TestCase):
    """NCCL基准测试测试"""

    def setUp(self):
        """测试初始化"""
        self.config = {
            "nodes": [
                {"hostname": "node01", "ip": "192.168.1.1"},
                {"hostname": "node02", "ip": "192.168.1.2"},
            ],
            "ssh_user": "root",
            "nccl_benchmark": {
                "nccl_tests_path": "/usr/local/bin",
                "min_bytes": "8",
                "max_bytes": "1G",
                "iterations": 20,
                "gpus_per_node": 8,
            },
        }
        self.benchmark = NCCLBenchmark(self.config)

    def test_parse_nccl_output(self):
        """测试解析NCCL测试输出"""
        output = """# nThread 1 nGpus 1 minBytes 8 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12345 on    node01 device  0 [0x86] NVIDIA A100-SXM4-80GB
#  Rank  1 Group  0 Pid  12346 on    node01 device  1 [0x87] NVIDIA A100-SXM4-80GB
#  Rank  2 Group  0 Pid  23456 on    node02 device  0 [0x86] NVIDIA A100-SXM4-80GB
#  Rank  3 Group  0 Pid  23457 on    node02 device  1 [0x87] NVIDIA A100-SXM4-80GB
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    27.02    0.00    0.00      0    26.74    0.00    0.00      0
        1024           256     float     sum      -1    28.15    0.04    0.03      0    27.89    0.04    0.03      0
     1048576        262144     float     sum      -1   124.50    8.42    6.32      0   123.80    8.47    6.35      0
    67108864      16777216     float     sum      -1  1234.50   54.35   40.76      0  1230.20   54.56   40.92      0
   536870912     134217728     float     sum      -1  9876.50   54.36   40.77      0  9870.20   54.39   40.79      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 17.58
#
"""
        result = self.benchmark._parse_nccl_output(output, CollectiveOp.ALL_REDUCE)

        self.assertEqual(result.operation, CollectiveOp.ALL_REDUCE)
        self.assertEqual(result.num_gpus, 4)  # 4 ranks
        self.assertEqual(len(result.test_points), 5)

        # 验证峰值
        self.assertAlmostEqual(result.peak_bus_bw_gbps, 40.92, places=1)
        self.assertEqual(result.peak_message_size, 67108864)

        # 验证单个测试点
        last_point = result.test_points[-1]
        self.assertEqual(last_point.message_size, 536870912)
        self.assertAlmostEqual(last_point.bus_bandwidth_gbps, 40.79, places=1)

    def test_parse_nccl_output_empty(self):
        """测试解析空输出"""
        output = "Error: NCCL failed to initialize"
        result = self.benchmark._parse_nccl_output(output, CollectiveOp.ALL_REDUCE)

        self.assertEqual(len(result.test_points), 0)
        self.assertEqual(result.peak_bus_bw_gbps, 0.0)

    def test_human_readable_size(self):
        """测试大小格式化"""
        self.assertEqual(NCCLBenchmark._human_readable_size(8), "8B")
        self.assertEqual(NCCLBenchmark._human_readable_size(1024), "1KB")
        self.assertEqual(NCCLBenchmark._human_readable_size(1048576), "1MB")
        self.assertEqual(NCCLBenchmark._human_readable_size(1073741824), "1GB")
        self.assertEqual(NCCLBenchmark._human_readable_size(67108864), "64MB")

    def test_build_hostfile_content(self):
        """测试hostfile生成"""
        content = self.benchmark._build_hostfile_content()
        lines = content.strip().split("\n")

        self.assertEqual(len(lines), 2)
        self.assertIn("node01", lines[0])
        self.assertIn("slots=8", lines[0])
        self.assertIn("node02", lines[1])

    def test_build_env_string(self):
        """测试环境变量字符串构建"""
        env_str = self.benchmark._build_env_string({"NCCL_ALGO": "Ring"})

        self.assertIn("-x NCCL_ALGO=Ring", env_str)
        self.assertIn("-x NCCL_DEBUG=INFO", env_str)
        self.assertIn("-x NCCL_IB_DISABLE=0", env_str)

    def test_build_nccl_test_command(self):
        """测试NCCL测试命令构建"""
        cmd = self.benchmark._build_nccl_test_command(
            CollectiveOp.ALL_REDUCE, num_gpus=16
        )

        self.assertIn("mpirun", cmd)
        self.assertIn("-np 16", cmd)
        self.assertIn("all_reduce_perf", cmd)
        self.assertIn("-b 8", cmd)
        self.assertIn("-e 1G", cmd)
        self.assertIn("-n 20", cmd)
        self.assertIn("-d float", cmd)

    def test_build_nccl_test_command_different_ops(self):
        """测试不同集合操作的命令构建"""
        for op, binary in [
            (CollectiveOp.ALL_GATHER, "all_gather_perf"),
            (CollectiveOp.REDUCE_SCATTER, "reduce_scatter_perf"),
            (CollectiveOp.BROADCAST, "broadcast_perf"),
        ]:
            cmd = self.benchmark._build_nccl_test_command(op, num_gpus=8)
            self.assertIn(binary, cmd)

    @patch.object(NCCLBenchmark, '_run_remote')
    def test_run_test(self, mock_run):
        """测试运行NCCL测试"""
        nccl_output = """# nThread 1 nGpus 1 minBytes 8 maxBytes 1073741824
#
# Using devices
#  Rank  0 Group  0 Pid  1 on node01 device  0 [0x86] NVIDIA A100-SXM4-80GB
#
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
     1048576        262144     float     sum      -1   100.00   10.49    7.86      0   100.00   10.49    7.86      0
"""
        mock_run.side_effect = [
            ("", "", 0),  # write hostfile
            (nccl_output, "", 0),  # run test
        ]

        result = self.benchmark.run_test(CollectiveOp.ALL_REDUCE, num_gpus=1)

        self.assertTrue(result.success)
        self.assertEqual(len(result.test_points), 1)
        self.assertGreater(result.peak_bus_bw_gbps, 0)


if __name__ == "__main__":
    unittest.main()
