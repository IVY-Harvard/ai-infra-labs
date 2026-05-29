"""
延迟测量模块

使用ib_write_lat/ib_read_lat测量RDMA延迟。
计算P50/P99/P999分位数，识别延迟离群点，
关联拓扑距离分析延迟与跳数的关系。
"""

import logging
import subprocess
import re
import json
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

logger = logging.getLogger(__name__)


class LatencyTestType(Enum):
    """延迟测试类型"""
    WRITE = "write"    # ib_write_lat
    READ = "read"      # ib_read_lat
    SEND = "send"      # ib_send_lat


@dataclass
class LatencyResult:
    """单次延迟测试结果"""
    source_node: str
    dest_node: str
    test_type: LatencyTestType
    message_size: int
    iterations: int
    # 延迟指标（微秒）
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0
    avg_latency_us: float = 0.0
    median_latency_us: float = 0.0
    p50_latency_us: float = 0.0
    p90_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    p999_latency_us: float = 0.0
    stddev_us: float = 0.0
    # 拓扑信息
    hop_count: int = -1
    # 元信息
    timestamp: str = ""
    success: bool = True
    error_message: str = ""
    raw_samples: List[float] = field(default_factory=list)


@dataclass
class LatencyProfile:
    """延迟分析概况"""
    test_time: str
    test_type: LatencyTestType
    node_count: int
    pair_count: int
    message_size: int
    results: List[LatencyResult] = field(default_factory=list)
    # 聚合统计
    cluster_avg_latency_us: float = 0.0
    cluster_p50_us: float = 0.0
    cluster_p99_us: float = 0.0
    cluster_p999_us: float = 0.0
    # 延迟与跳数的相关性
    latency_by_hops: Dict[int, List[float]] = field(default_factory=dict)
    # 异常节点对
    outlier_pairs: List[str] = field(default_factory=list)


class LatencyProfiler:
    """
    RDMA延迟分析器

    执行节点间延迟测量，收集细粒度延迟样本，
    计算分位数统计，分析延迟与拓扑距离的关系。
    """

    def __init__(self, config: dict):
        """
        初始化延迟分析器

        Args:
            config: 配置字典
        """
        self.nodes = config.get("nodes", [])
        self.ssh_user = config.get("ssh_user", "root")
        self.ssh_key = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout = config.get("ssh_timeout", 30)
        self.test_timeout = config.get("latency_test_timeout", 120)
        self.max_workers = config.get("max_workers", 4)

        # 测试参数
        lat_config = config.get("latency_benchmark", {})
        self.message_sizes = lat_config.get("message_sizes", [2, 64, 512, 4096])
        self.iterations = lat_config.get("iterations", 10000)
        self.device = lat_config.get("device", "mlx5_0")
        self.ib_port = lat_config.get("ib_port", 1)
        self.gid_index = lat_config.get("gid_index", 3)
        self.server_port_base = lat_config.get("server_port_base", 19515)

        # 离群点检测阈值
        self.outlier_factor = lat_config.get("outlier_factor", 2.0)  # 超过平均值N倍视为离群

    def _run_remote(self, host: str, command: str,
                    timeout: Optional[int] = None) -> Tuple[str, str, int]:
        """在远程主机上执行命令"""
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.ssh_timeout}",
            "-o", "BatchMode=yes",
            "-i", self.ssh_key,
            f"{self.ssh_user}@{host}",
            command
        ]
        effective_timeout = timeout or self.test_timeout
        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True,
                timeout=effective_timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"命令超时 ({effective_timeout}s): {host}")
            return "", "Timeout", -1
        except Exception as e:
            logger.error(f"命令执行异常 {host}: {e}")
            return "", str(e), -1

    def _build_lat_command(self, test_type: LatencyTestType,
                           message_size: int, is_server: bool,
                           server_ip: str = "",
                           port: int = 19515) -> str:
        """
        构建ib_*_lat命令

        Args:
            test_type: 测试类型
            message_size: 消息大小
            is_server: 是否为服务端
            server_ip: 服务端IP
            port: 端口号

        Returns:
            命令字符串
        """
        cmd_map = {
            LatencyTestType.WRITE: "ib_write_lat",
            LatencyTestType.READ: "ib_read_lat",
            LatencyTestType.SEND: "ib_send_lat",
        }
        binary = cmd_map[test_type]

        cmd_parts = [
            binary,
            f"-d {self.device}",
            f"-i {self.ib_port}",
            f"-s {message_size}",
            f"-n {self.iterations}",
            f"-p {port}",
            "-F",
        ]

        if self.gid_index >= 0:
            cmd_parts.append(f"-x {self.gid_index}")

        if not is_server:
            cmd_parts.append(server_ip)

        return " ".join(cmd_parts)

    def _parse_lat_output(self, output: str) -> Optional[Dict]:
        """
        解析ib_*_lat输出

        典型输出格式:
            #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
            2       10000          1.23           15.67        1.45               1.52           0.34            2.89                   5.12

        Returns:
            解析后的字典
        """
        for line in output.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    msg_size = int(parts[0])
                    iterations = int(parts[1])
                    t_min = float(parts[2])
                    t_max = float(parts[3])
                    t_typical = float(parts[4])
                    t_avg = float(parts[5])
                    t_stdev = float(parts[6]) if len(parts) > 6 else 0.0
                    p99 = float(parts[7]) if len(parts) > 7 else t_max
                    p999 = float(parts[8]) if len(parts) > 8 else t_max

                    return {
                        "message_size": msg_size,
                        "iterations": iterations,
                        "min_us": t_min,
                        "max_us": t_max,
                        "typical_us": t_typical,
                        "avg_us": t_avg,
                        "stdev_us": t_stdev,
                        "p99_us": p99,
                        "p999_us": p999,
                    }
                except (ValueError, IndexError):
                    continue
        return None

    def _parse_lat_histogram(self, output: str) -> List[float]:
        """
        解析延迟直方图/原始样本数据

        如果ib_*_lat配合--output=histogram使用，
        可以获取更细粒度的延迟分布数据
        """
        samples = []
        in_histogram = False

        for line in output.split("\n"):
            line = line.strip()
            # 检测直方图区域
            if "histogram" in line.lower() or "latency" in line.lower():
                in_histogram = True
                continue
            if in_histogram:
                match = re.match(r"([\d.]+)\s+(\d+)", line)
                if match:
                    latency = float(match.group(1))
                    count = int(match.group(2))
                    samples.extend([latency] * count)

        return samples

    def compute_percentiles(self, values: List[float]) -> Dict[str, float]:
        """
        计算分位数统计

        Args:
            values: 延迟值列表

        Returns:
            分位数字典
        """
        if not values:
            return {"p50": 0, "p90": 0, "p99": 0, "p999": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(p: float) -> float:
            idx = p * (n - 1)
            lower = int(math.floor(idx))
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

        return {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "p999": percentile(0.999),
        }

    def test_pair(self, src_node: dict, dst_node: dict,
                  test_type: LatencyTestType,
                  message_size: int,
                  port_offset: int = 0) -> LatencyResult:
        """
        对一对节点执行延迟测试

        Args:
            src_node: 源节点
            dst_node: 目标节点
            test_type: 测试类型
            message_size: 消息大小
            port_offset: 端口偏移

        Returns:
            LatencyResult
        """
        src_host = src_node.get("ip", src_node.get("hostname", ""))
        dst_host = dst_node.get("ip", dst_node.get("hostname", ""))
        src_name = src_node.get("hostname", src_host)
        dst_name = dst_node.get("hostname", dst_host)
        port = self.server_port_base + port_offset

        logger.info(
            f"开始延迟测试: {src_name} -> {dst_name}, "
            f"消息大小={message_size}B"
        )

        result = LatencyResult(
            source_node=src_name,
            dest_node=dst_name,
            test_type=test_type,
            message_size=message_size,
            iterations=self.iterations,
            timestamp=datetime.datetime.now().isoformat(),
        )

        # 启动服务端
        server_cmd = self._build_lat_command(
            test_type, message_size, is_server=True, port=port
        )
        bg_cmd = f"nohup {server_cmd} > /tmp/ib_lat_server_{port}.log 2>&1 & echo $!"
        stdout, stderr, rc = self._run_remote(dst_host, bg_cmd)
        if rc != 0:
            result.success = False
            result.error_message = f"服务端启动失败: {stderr}"
            return result

        server_pid = stdout.strip()
        time.sleep(2)

        # 启动客户端
        client_cmd = self._build_lat_command(
            test_type, message_size, is_server=False,
            server_ip=dst_host, port=port
        )
        stdout, stderr, rc = self._run_remote(
            src_host, client_cmd, timeout=self.test_timeout
        )

        # 清理服务端
        self._run_remote(dst_host, f"kill {server_pid} 2>/dev/null")

        if rc != 0:
            result.success = False
            result.error_message = f"客户端执行失败: {stderr}"
            logger.error(f"延迟测试失败 {src_name}->{dst_name}: {stderr}")
            return result

        # 解析结果
        parsed = self._parse_lat_output(stdout)
        if parsed:
            result.min_latency_us = parsed["min_us"]
            result.max_latency_us = parsed["max_us"]
            result.avg_latency_us = parsed["avg_us"]
            result.median_latency_us = parsed["typical_us"]
            result.p50_latency_us = parsed["typical_us"]
            result.p99_latency_us = parsed["p99_us"]
            result.p999_latency_us = parsed["p999_us"]
            result.stddev_us = parsed["stdev_us"]

            # 尝试解析直方图
            samples = self._parse_lat_histogram(stdout)
            if samples:
                result.raw_samples = samples
                percentiles = self.compute_percentiles(samples)
                result.p50_latency_us = percentiles["p50"]
                result.p90_latency_us = percentiles["p90"]
                result.p99_latency_us = percentiles["p99"]
                result.p999_latency_us = percentiles["p999"]

            logger.info(
                f"延迟测试完成: {src_name}->{dst_name} "
                f"avg={result.avg_latency_us:.2f}us "
                f"p99={result.p99_latency_us:.2f}us"
            )
        else:
            result.success = False
            result.error_message = "无法解析测试输出"

        return result

    def profile_all_pairs(self, test_type: LatencyTestType = LatencyTestType.WRITE,
                          message_size: Optional[int] = None) -> LatencyProfile:
        """
        对所有节点对进行延迟分析

        Args:
            test_type: 测试类型
            message_size: 消息大小

        Returns:
            LatencyProfile
        """
        if message_size is None:
            message_size = min(self.message_sizes)  # 延迟测试默认用小消息

        profile = LatencyProfile(
            test_time=datetime.datetime.now().isoformat(),
            test_type=test_type,
            node_count=len(self.nodes),
            pair_count=0,
            message_size=message_size,
        )

        # 生成节点对
        pairs = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                pairs.append((self.nodes[i], self.nodes[j], len(pairs)))
        profile.pair_count = len(pairs)

        logger.info(
            f"开始全对延迟测试: {len(self.nodes)} 节点, {len(pairs)} 对"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for src, dst, offset in pairs:
                future = executor.submit(
                    self.test_pair, src, dst, test_type, message_size, offset
                )
                futures[future] = f"{src.get('hostname', '')}->{dst.get('hostname', '')}"

            for future in as_completed(futures):
                pair_name = futures[future]
                try:
                    result = future.result()
                    profile.results.append(result)
                except Exception as e:
                    logger.error(f"延迟测试异常 {pair_name}: {e}")

        # 计算聚合统计
        self._compute_profile_stats(profile)

        return profile

    def profile_message_size_sweep(self, src_node: dict, dst_node: dict,
                                    test_type: LatencyTestType = LatencyTestType.WRITE
                                    ) -> List[LatencyResult]:
        """
        对一对节点进行消息大小扫描的延迟测试

        Returns:
            不同消息大小的延迟结果列表
        """
        results = []
        for i, msg_size in enumerate(sorted(self.message_sizes)):
            result = self.test_pair(
                src_node, dst_node, test_type, msg_size, port_offset=i
            )
            results.append(result)

        return results

    def _compute_profile_stats(self, profile: LatencyProfile) -> None:
        """计算延迟概况统计"""
        successful = [r for r in profile.results if r.success]
        if not successful:
            return

        # 聚合平均延迟
        avg_values = [r.avg_latency_us for r in successful]
        profile.cluster_avg_latency_us = sum(avg_values) / len(avg_values)

        # 聚合分位数（用所有对的平均延迟值计算）
        percentiles = self.compute_percentiles(avg_values)
        profile.cluster_p50_us = percentiles["p50"]
        profile.cluster_p99_us = percentiles["p99"]
        profile.cluster_p999_us = percentiles["p999"]

        # 按跳数分组
        for r in successful:
            if r.hop_count >= 0:
                if r.hop_count not in profile.latency_by_hops:
                    profile.latency_by_hops[r.hop_count] = []
                profile.latency_by_hops[r.hop_count].append(r.avg_latency_us)

        # 检测离群点
        threshold = profile.cluster_avg_latency_us * self.outlier_factor
        for r in successful:
            if r.avg_latency_us > threshold:
                profile.outlier_pairs.append(
                    f"{r.source_node}->{r.dest_node}: "
                    f"avg={r.avg_latency_us:.2f}us "
                    f"(集群平均={profile.cluster_avg_latency_us:.2f}us)"
                )

        logger.info(
            f"延迟分析统计: "
            f"集群平均={profile.cluster_avg_latency_us:.2f}us, "
            f"P50={profile.cluster_p50_us:.2f}us, "
            f"P99={profile.cluster_p99_us:.2f}us, "
            f"离群点={len(profile.outlier_pairs)}"
        )

    def correlate_with_topology(self, profile: LatencyProfile,
                                topology_distances: Dict[Tuple[str, str], int]) -> None:
        """
        将延迟结果与拓扑距离进行关联分析

        Args:
            profile: 延迟分析概况
            topology_distances: 拓扑距离信息 {(src, dst): hop_count}
        """
        for result in profile.results:
            pair = (result.source_node, result.dest_node)
            rev_pair = (result.dest_node, result.source_node)

            hops = topology_distances.get(pair, topology_distances.get(rev_pair, -1))
            result.hop_count = hops

        # 重新按跳数分组
        profile.latency_by_hops = {}
        for r in profile.results:
            if r.success and r.hop_count >= 0:
                if r.hop_count not in profile.latency_by_hops:
                    profile.latency_by_hops[r.hop_count] = []
                profile.latency_by_hops[r.hop_count].append(r.avg_latency_us)

        # 输出关联分析结果
        for hops in sorted(profile.latency_by_hops.keys()):
            latencies = profile.latency_by_hops[hops]
            avg = sum(latencies) / len(latencies)
            logger.info(f"跳数 {hops}: 平均延迟 {avg:.2f}us ({len(latencies)} 对)")

    def export_results(self, profile: LatencyProfile, output_path: str) -> None:
        """导出延迟测试结果"""
        data = {
            "test_time": profile.test_time,
            "test_type": profile.test_type.value,
            "message_size": profile.message_size,
            "node_count": profile.node_count,
            "pair_count": profile.pair_count,
            "statistics": {
                "cluster_avg_latency_us": round(profile.cluster_avg_latency_us, 4),
                "cluster_p50_us": round(profile.cluster_p50_us, 4),
                "cluster_p99_us": round(profile.cluster_p99_us, 4),
                "cluster_p999_us": round(profile.cluster_p999_us, 4),
            },
            "latency_by_hops": {
                str(k): {
                    "avg": round(sum(v) / len(v), 4),
                    "count": len(v),
                }
                for k, v in profile.latency_by_hops.items()
            },
            "outlier_pairs": profile.outlier_pairs,
            "results": [
                {
                    "source": r.source_node,
                    "dest": r.dest_node,
                    "avg_us": round(r.avg_latency_us, 4),
                    "p50_us": round(r.p50_latency_us, 4),
                    "p99_us": round(r.p99_latency_us, 4),
                    "p999_us": round(r.p999_latency_us, 4),
                    "min_us": round(r.min_latency_us, 4),
                    "max_us": round(r.max_latency_us, 4),
                    "hop_count": r.hop_count,
                    "success": r.success,
                }
                for r in profile.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"延迟测试结果已导出: {output_path}")
