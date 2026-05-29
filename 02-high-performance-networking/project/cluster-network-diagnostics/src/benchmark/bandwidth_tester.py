"""
带宽测试模块

使用ib_write_bw/ib_read_bw进行多节点RDMA带宽基准测试。
支持全对(all-pairs)和指定节点对测试，收集每条链路和聚合带宽，
与理论最大值对比计算带宽利用率。
"""

import logging
import subprocess
import re
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

logger = logging.getLogger(__name__)


class BandwidthTestType(Enum):
    """带宽测试类型"""
    WRITE = "write"    # ib_write_bw
    READ = "read"      # ib_read_bw
    SEND = "send"      # ib_send_bw


class TransportType(Enum):
    """传输类型"""
    RC = "RC"   # Reliable Connection
    UC = "UC"   # Unreliable Connection
    UD = "UD"   # Unreliable Datagram


@dataclass
class BandwidthResult:
    """单次带宽测试结果"""
    source_node: str
    dest_node: str
    test_type: BandwidthTestType
    transport: TransportType
    message_size: int        # 消息大小（字节）
    iterations: int          # 迭代次数
    bandwidth_mbps: float    # 带宽 (MB/s)
    bandwidth_gbps: float    # 带宽 (Gb/s)
    theoretical_max_gbps: float  # 理论最大带宽
    efficiency: float        # 带宽利用率 (0.0-1.0)
    duration_seconds: float  # 测试持续时间
    timestamp: str = ""
    error_message: str = ""
    success: bool = True


@dataclass
class AllPairsResult:
    """全对带宽测试结果"""
    test_time: str
    test_type: BandwidthTestType
    node_count: int
    pair_count: int
    results: List[BandwidthResult] = field(default_factory=list)
    # 聚合统计
    avg_bandwidth_gbps: float = 0.0
    min_bandwidth_gbps: float = 0.0
    max_bandwidth_gbps: float = 0.0
    median_bandwidth_gbps: float = 0.0
    stddev_bandwidth_gbps: float = 0.0
    avg_efficiency: float = 0.0
    bisection_bandwidth_gbps: float = 0.0
    # 异常链路
    underperforming_pairs: List[str] = field(default_factory=list)


# 链路速度到理论最大带宽的映射（Gb/s, 4x宽度）
LINK_SPEED_MAX_BW = {
    "SDR": 8.0,
    "DDR": 16.0,
    "QDR": 32.0,
    "FDR": 54.5,
    "EDR": 100.0,
    "HDR": 200.0,
    "NDR": 400.0,
    "XDR": 800.0,
    # 也支持直接的速率值
    "100": 100.0,
    "200": 200.0,
    "400": 400.0,
}


class BandwidthTester:
    """
    RDMA带宽测试器

    协调多节点间的带宽测试，收集和分析结果。
    使用perftest工具套件（ib_write_bw、ib_read_bw等）执行实际测试。
    """

    def __init__(self, config: dict):
        """
        初始化带宽测试器

        Args:
            config: 配置字典
        """
        self.nodes = config.get("nodes", [])
        self.ssh_user = config.get("ssh_user", "root")
        self.ssh_key = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout = config.get("ssh_timeout", 30)
        self.test_timeout = config.get("bandwidth_test_timeout", 120)
        self.max_workers = config.get("max_workers", 4)

        # 测试参数
        bench_config = config.get("bandwidth_benchmark", {})
        self.message_sizes = bench_config.get("message_sizes", [65536, 1048576, 4194304])
        self.iterations = bench_config.get("iterations", 5000)
        self.transport = TransportType(bench_config.get("transport", "RC"))
        self.device = bench_config.get("device", "mlx5_0")
        self.ib_port = bench_config.get("ib_port", 1)
        self.gid_index = bench_config.get("gid_index", 3)  # RoCE v2
        self.server_port_base = bench_config.get("server_port_base", 18515)
        self.link_speed = bench_config.get("link_speed", "HDR")
        self.theoretical_max = LINK_SPEED_MAX_BW.get(self.link_speed, 200.0)
        # 低于阈值视为性能不达标
        self.efficiency_threshold = bench_config.get("efficiency_threshold", 0.85)

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

    def _build_bw_command(self, test_type: BandwidthTestType,
                          message_size: int, is_server: bool,
                          server_ip: str = "",
                          port: int = 18515) -> str:
        """
        构建ib_*_bw命令

        Args:
            test_type: 测试类型
            message_size: 消息大小
            is_server: 是否为服务端
            server_ip: 服务端IP（客户端需要）
            port: 监听端口

        Returns:
            完整命令字符串
        """
        cmd_map = {
            BandwidthTestType.WRITE: "ib_write_bw",
            BandwidthTestType.READ: "ib_read_bw",
            BandwidthTestType.SEND: "ib_send_bw",
        }
        binary = cmd_map[test_type]

        cmd_parts = [
            binary,
            f"-d {self.device}",
            f"-i {self.ib_port}",
            f"-s {message_size}",
            f"-n {self.iterations}",
            f"-p {port}",
            f"--connection={self.transport.value}",
            "--report_gbits",  # 以Gb/s报告
            "-F",  # 不检查CPU affinity
        ]

        # RoCE需要指定GID index
        if self.gid_index >= 0:
            cmd_parts.append(f"-x {self.gid_index}")

        if not is_server:
            cmd_parts.append(server_ip)

        return " ".join(cmd_parts)

    def _parse_bw_output(self, output: str) -> Optional[Dict]:
        """
        解析ib_*_bw输出

        典型输出格式:
            #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
            65536      5000            98.12              97.85                0.186701

        Returns:
            解析后的结果字典，或None
        """
        for line in output.split("\n"):
            line = line.strip()
            # 跳过注释和标题行
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # 匹配数据行
            parts = line.split()
            if len(parts) >= 4:
                try:
                    msg_size = int(parts[0])
                    iterations = int(parts[1])
                    bw_peak = float(parts[2])
                    bw_avg = float(parts[3])
                    msg_rate = float(parts[4]) if len(parts) > 4 else 0.0

                    return {
                        "message_size": msg_size,
                        "iterations": iterations,
                        "bw_peak_gbps": bw_peak,
                        "bw_avg_gbps": bw_avg,
                        "msg_rate_mpps": msg_rate,
                    }
                except (ValueError, IndexError):
                    continue

        return None

    def test_pair(self, src_node: dict, dst_node: dict,
                  test_type: BandwidthTestType,
                  message_size: int,
                  port_offset: int = 0) -> BandwidthResult:
        """
        对一对节点执行带宽测试

        Args:
            src_node: 源节点信息 {"hostname": "...", "ip": "..."}
            dst_node: 目标节点信息
            test_type: 测试类型
            message_size: 消息大小
            port_offset: 端口偏移量（避免并发冲突）

        Returns:
            BandwidthResult
        """
        src_host = src_node.get("ip", src_node.get("hostname", ""))
        dst_host = dst_node.get("ip", dst_node.get("hostname", ""))
        src_name = src_node.get("hostname", src_host)
        dst_name = dst_node.get("hostname", dst_host)
        port = self.server_port_base + port_offset

        logger.info(
            f"开始带宽测试: {src_name} -> {dst_name}, "
            f"类型={test_type.value}, 消息大小={message_size}"
        )

        result = BandwidthResult(
            source_node=src_name,
            dest_node=dst_name,
            test_type=test_type,
            transport=self.transport,
            message_size=message_size,
            iterations=self.iterations,
            bandwidth_mbps=0.0,
            bandwidth_gbps=0.0,
            theoretical_max_gbps=self.theoretical_max,
            efficiency=0.0,
            duration_seconds=0.0,
            timestamp=datetime.datetime.now().isoformat(),
        )

        # 启动服务端（在目标节点上）
        server_cmd = self._build_bw_command(
            test_type, message_size, is_server=True, port=port
        )
        # 服务端在后台运行
        bg_server_cmd = f"nohup {server_cmd} > /tmp/ib_bw_server_{port}.log 2>&1 & echo $!"
        stdout, stderr, rc = self._run_remote(dst_host, bg_server_cmd)
        if rc != 0:
            result.success = False
            result.error_message = f"启动服务端失败: {stderr}"
            logger.error(result.error_message)
            return result

        server_pid = stdout.strip()
        logger.debug(f"服务端已启动: {dst_name} PID={server_pid}")

        # 等待服务端就绪
        time.sleep(2)

        # 启动客户端（在源节点上）
        client_cmd = self._build_bw_command(
            test_type, message_size, is_server=False,
            server_ip=dst_host, port=port
        )

        start_time = time.time()
        stdout, stderr, rc = self._run_remote(
            src_host, client_cmd, timeout=self.test_timeout
        )
        elapsed = time.time() - start_time

        # 清理服务端
        self._run_remote(dst_host, f"kill {server_pid} 2>/dev/null")

        if rc != 0:
            result.success = False
            result.error_message = f"客户端执行失败: {stderr}"
            logger.error(f"带宽测试失败 {src_name}->{dst_name}: {stderr}")
            return result

        # 解析结果
        parsed = self._parse_bw_output(stdout)
        if parsed:
            result.bandwidth_gbps = parsed["bw_avg_gbps"]
            result.bandwidth_mbps = parsed["bw_avg_gbps"] * 1000 / 8  # Gb/s -> MB/s
            result.duration_seconds = elapsed
            result.efficiency = (
                result.bandwidth_gbps / self.theoretical_max
                if self.theoretical_max > 0 else 0.0
            )

            logger.info(
                f"带宽测试完成: {src_name}->{dst_name} "
                f"{result.bandwidth_gbps:.2f} Gb/s "
                f"(效率 {result.efficiency:.1%})"
            )
        else:
            result.success = False
            result.error_message = "无法解析测试输出"
            logger.error(f"解析测试输出失败: {src_name}->{dst_name}")

        return result

    def test_all_pairs(self, test_type: BandwidthTestType = BandwidthTestType.WRITE,
                       message_size: Optional[int] = None) -> AllPairsResult:
        """
        执行全对带宽测试

        对集群中所有节点对进行带宽测试

        Args:
            test_type: 测试类型
            message_size: 消息大小，默认使用配置中最大的

        Returns:
            AllPairsResult
        """
        if message_size is None:
            message_size = max(self.message_sizes)

        all_pairs = AllPairsResult(
            test_time=datetime.datetime.now().isoformat(),
            test_type=test_type,
            node_count=len(self.nodes),
            pair_count=0,
        )

        # 生成所有节点对
        pairs = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                pairs.append((self.nodes[i], self.nodes[j], len(pairs)))

        all_pairs.pair_count = len(pairs)
        logger.info(
            f"开始全对带宽测试: {len(self.nodes)} 节点, "
            f"{len(pairs)} 对, 消息大小={message_size}"
        )

        # 并发执行（注意控制并发度避免资源竞争）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for src, dst, offset in pairs:
                future = executor.submit(
                    self.test_pair, src, dst, test_type, message_size, offset
                )
                pair_name = f"{src.get('hostname', '')}->{dst.get('hostname', '')}"
                futures[future] = pair_name

            for future in as_completed(futures):
                pair_name = futures[future]
                try:
                    result = future.result()
                    all_pairs.results.append(result)
                except Exception as e:
                    logger.error(f"测试异常 {pair_name}: {e}")

        # 计算聚合统计
        self._compute_aggregate_stats(all_pairs)

        return all_pairs

    def test_message_size_sweep(self, src_node: dict, dst_node: dict,
                                test_type: BandwidthTestType = BandwidthTestType.WRITE
                                ) -> List[BandwidthResult]:
        """
        对一对节点执行消息大小扫描测试

        从小到大遍历所有配置的消息大小

        Args:
            src_node: 源节点
            dst_node: 目标节点
            test_type: 测试类型

        Returns:
            不同消息大小的测试结果列表
        """
        results = []
        for i, msg_size in enumerate(sorted(self.message_sizes)):
            result = self.test_pair(
                src_node, dst_node, test_type, msg_size, port_offset=i
            )
            results.append(result)
            logger.info(
                f"消息大小 {msg_size}: "
                f"{result.bandwidth_gbps:.2f} Gb/s"
            )

        return results

    def _compute_aggregate_stats(self, all_pairs: AllPairsResult) -> None:
        """计算全对测试的聚合统计"""
        successful = [r for r in all_pairs.results if r.success]
        if not successful:
            logger.warning("没有成功的测试结果")
            return

        bw_values = [r.bandwidth_gbps for r in successful]
        bw_values.sort()

        all_pairs.avg_bandwidth_gbps = sum(bw_values) / len(bw_values)
        all_pairs.min_bandwidth_gbps = bw_values[0]
        all_pairs.max_bandwidth_gbps = bw_values[-1]
        all_pairs.median_bandwidth_gbps = bw_values[len(bw_values) // 2]

        # 标准差
        mean = all_pairs.avg_bandwidth_gbps
        variance = sum((x - mean) ** 2 for x in bw_values) / len(bw_values)
        all_pairs.stddev_bandwidth_gbps = variance ** 0.5

        # 平均效率
        eff_values = [r.efficiency for r in successful]
        all_pairs.avg_efficiency = sum(eff_values) / len(eff_values)

        # 对分带宽（所有并发链路的总带宽 / 2）
        all_pairs.bisection_bandwidth_gbps = sum(bw_values) / 2

        # 识别性能不达标的节点对
        for result in successful:
            if result.efficiency < self.efficiency_threshold:
                pair_name = f"{result.source_node}->{result.dest_node}"
                all_pairs.underperforming_pairs.append(
                    f"{pair_name}: {result.bandwidth_gbps:.2f} Gb/s "
                    f"(效率 {result.efficiency:.1%})"
                )

        logger.info(
            f"全对测试统计: "
            f"平均={all_pairs.avg_bandwidth_gbps:.2f} Gb/s, "
            f"最小={all_pairs.min_bandwidth_gbps:.2f} Gb/s, "
            f"最大={all_pairs.max_bandwidth_gbps:.2f} Gb/s, "
            f"效率={all_pairs.avg_efficiency:.1%}, "
            f"不达标={len(all_pairs.underperforming_pairs)}"
        )

    def export_results(self, results: AllPairsResult, output_path: str) -> None:
        """
        导出测试结果到JSON文件

        Args:
            results: 测试结果
            output_path: 输出文件路径
        """
        data = {
            "test_time": results.test_time,
            "test_type": results.test_type.value,
            "node_count": results.node_count,
            "pair_count": results.pair_count,
            "statistics": {
                "avg_bandwidth_gbps": round(results.avg_bandwidth_gbps, 4),
                "min_bandwidth_gbps": round(results.min_bandwidth_gbps, 4),
                "max_bandwidth_gbps": round(results.max_bandwidth_gbps, 4),
                "median_bandwidth_gbps": round(results.median_bandwidth_gbps, 4),
                "stddev_bandwidth_gbps": round(results.stddev_bandwidth_gbps, 4),
                "avg_efficiency": round(results.avg_efficiency, 4),
                "bisection_bandwidth_gbps": round(results.bisection_bandwidth_gbps, 4),
            },
            "underperforming_pairs": results.underperforming_pairs,
            "results": [
                {
                    "source": r.source_node,
                    "dest": r.dest_node,
                    "message_size": r.message_size,
                    "bandwidth_gbps": round(r.bandwidth_gbps, 4),
                    "efficiency": round(r.efficiency, 4),
                    "success": r.success,
                    "error": r.error_message,
                }
                for r in results.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"带宽测试结果已导出: {output_path}")
