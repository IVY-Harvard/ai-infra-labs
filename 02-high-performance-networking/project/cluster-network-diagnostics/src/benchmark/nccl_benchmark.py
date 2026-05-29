"""
NCCL基准测试模块

封装nccl-tests（all_reduce_perf、all_gather_perf等）进行集合通信
性能基准测试。支持变化消息大小、GPU数量、算法选择。
计算总线带宽(bus bandwidth)和算法带宽(algorithm bandwidth)。
"""

import logging
import subprocess
import re
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import datetime

logger = logging.getLogger(__name__)


class CollectiveOp(Enum):
    """NCCL集合通信操作类型"""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALL_TO_ALL = "alltoall"
    SENDRECV = "sendrecv"


class NCCLAlgorithm(Enum):
    """NCCL通信算法"""
    TREE = "Tree"
    RING = "Ring"
    COLLNET_DIRECT = "CollNetDirect"
    COLLNET_CHAIN = "CollNetChain"
    NVLS = "NVLS"
    NVLS_TREE = "NVLSTree"


class NCCLProtocol(Enum):
    """NCCL通信协议"""
    LL = "LL"       # Low Latency
    LL128 = "LL128" # Low Latency 128B
    SIMPLE = "Simple"


@dataclass
class NCCLTestPoint:
    """单个消息大小的NCCL测试结果"""
    message_size: int           # 消息大小（字节）
    message_size_human: str     # 人类可读的大小
    count: int                  # 元素数量
    data_type: str              # 数据类型
    reduction_op: str           # 归约操作
    # 带外性能指标
    time_us: float = 0.0       # 耗时（微秒）
    algo_bandwidth_gbps: float = 0.0   # 算法带宽（GB/s）
    bus_bandwidth_gbps: float = 0.0    # 总线带宽（GB/s）
    errors: int = 0             # 验证错误数


@dataclass
class NCCLTestResult:
    """一次完整的NCCL测试结果"""
    operation: CollectiveOp
    num_gpus: int
    num_nodes: int
    gpus_per_node: int
    algorithm: str = ""
    protocol: str = ""
    timestamp: str = ""
    duration_seconds: float = 0.0
    test_points: List[NCCLTestPoint] = field(default_factory=list)
    # 峰值性能
    peak_bus_bw_gbps: float = 0.0
    peak_algo_bw_gbps: float = 0.0
    peak_message_size: int = 0
    # 环境信息
    nccl_version: str = ""
    cuda_version: str = ""
    nccl_env_vars: Dict[str, str] = field(default_factory=dict)
    # 状态
    success: bool = True
    error_message: str = ""
    raw_output: str = ""


@dataclass
class NCCLBenchmarkSuite:
    """NCCL基准测试套件（多个测试的集合）"""
    suite_time: str
    cluster_name: str
    total_gpus: int
    results: List[NCCLTestResult] = field(default_factory=list)
    summary: Dict[str, Dict] = field(default_factory=dict)


# 集合操作到nccl-tests可执行文件的映射
NCCL_TEST_BINARIES = {
    CollectiveOp.ALL_REDUCE: "all_reduce_perf",
    CollectiveOp.ALL_GATHER: "all_gather_perf",
    CollectiveOp.REDUCE_SCATTER: "reduce_scatter_perf",
    CollectiveOp.BROADCAST: "broadcast_perf",
    CollectiveOp.REDUCE: "reduce_perf",
    CollectiveOp.ALL_TO_ALL: "alltoall_perf",
    CollectiveOp.SENDRECV: "sendrecv_perf",
}


class NCCLBenchmark:
    """
    NCCL集合通信基准测试

    封装nccl-tests工具套件，支持:
    - 多种集合操作（AllReduce, AllGather等）
    - 消息大小从小到大的扫描
    - 多种算法和协议的对比
    - 多节点多GPU的分布式测试
    - 总线带宽和算法带宽的计算和分析
    """

    def __init__(self, config: dict):
        """
        初始化NCCL基准测试

        Args:
            config: 配置字典
        """
        self.nodes = config.get("nodes", [])
        self.ssh_user = config.get("ssh_user", "root")
        self.ssh_key = config.get("ssh_key", "~/.ssh/id_rsa")
        self.ssh_timeout = config.get("ssh_timeout", 30)

        # NCCL测试参数
        nccl_config = config.get("nccl_benchmark", {})
        self.nccl_tests_path = nccl_config.get("nccl_tests_path", "/usr/local/bin")
        self.min_bytes = nccl_config.get("min_bytes", "8")
        self.max_bytes = nccl_config.get("max_bytes", "8G")
        self.step_factor = nccl_config.get("step_factor", 2)
        self.iterations = nccl_config.get("iterations", 20)
        self.warmup_iterations = nccl_config.get("warmup_iterations", 5)
        self.data_type = nccl_config.get("data_type", "float")
        self.reduction_op = nccl_config.get("reduction_op", "sum")
        self.gpus_per_node = nccl_config.get("gpus_per_node", 8)
        self.test_timeout = nccl_config.get("timeout", 600)
        self.cluster_name = config.get("cluster_name", "default")

        # NCCL环境变量
        self.nccl_env = nccl_config.get("nccl_env", {
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "NCCL_NET_GDR_LEVEL": "5",
            "NCCL_IB_HCA": "mlx5",
            "NCCL_SOCKET_IFNAME": "eth0",
        })

        # MPI相关配置
        self.mpi_path = nccl_config.get("mpi_path", "/usr/local/bin/mpirun")
        self.hostfile = nccl_config.get("hostfile", "")

    def _build_hostfile_content(self) -> str:
        """
        生成MPI hostfile内容

        Returns:
            hostfile内容字符串
        """
        lines = []
        for node in self.nodes:
            hostname = node.get("hostname", node.get("ip", ""))
            slots = node.get("gpus", self.gpus_per_node)
            lines.append(f"{hostname} slots={slots}")
        return "\n".join(lines)

    def _build_env_string(self, extra_env: Optional[Dict] = None) -> str:
        """构建环境变量字符串"""
        env = {**self.nccl_env}
        if extra_env:
            env.update(extra_env)

        parts = []
        for key, value in env.items():
            parts.append(f"-x {key}={value}")
        return " ".join(parts)

    def _build_nccl_test_command(self, operation: CollectiveOp,
                                  num_gpus: int,
                                  extra_env: Optional[Dict] = None,
                                  custom_args: Optional[str] = None) -> str:
        """
        构建完整的NCCL测试命令

        Args:
            operation: 集合操作类型
            num_gpus: 总GPU数量
            extra_env: 额外环境变量
            custom_args: 自定义参数

        Returns:
            完整命令字符串
        """
        binary = NCCL_TEST_BINARIES.get(operation, "all_reduce_perf")
        binary_path = os.path.join(self.nccl_tests_path, binary)

        num_nodes = len(self.nodes)
        nproc = num_gpus

        # 构建MPI命令
        cmd_parts = [
            self.mpi_path,
            f"-np {nproc}",
            f"--hostfile /tmp/nccl_hostfile",
            "--allow-run-as-root",
            "--bind-to none",
            "-map-by slot",
            self._build_env_string(extra_env),
        ]

        # NCCL测试参数
        test_args = [
            binary_path,
            f"-b {self.min_bytes}",
            f"-e {self.max_bytes}",
            f"-f {self.step_factor}",
            f"-g 1",  # 每个进程使用1个GPU
            f"-n {self.iterations}",
            f"-w {self.warmup_iterations}",
            f"-t 1",  # 线程数
            f"-d {self.data_type}",
            f"-o {self.reduction_op}",
            "-c 1",  # 数据校验
        ]

        if custom_args:
            test_args.append(custom_args)

        return " ".join(cmd_parts + test_args)

    def _parse_nccl_output(self, output: str, operation: CollectiveOp) -> NCCLTestResult:
        """
        解析nccl-tests输出

        典型输出格式:
            # nThread 1 nGpus 1 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20
            # Using devices
            #  Rank  0 Group  0 Pid 12345 on node01 device  0 [0x86] NVIDIA A100-SXM4-80GB
            #
            #                                                              out-of-place                       in-place
            #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
            #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
                8             2     float     sum      -1    27.02    0.00    0.00      0    26.74    0.00    0.00      0
              16             4     float     sum      -1    26.55    0.00    0.00      0    26.42    0.00    0.00      0
            ...
            67108864      16777216     float     sum      -1   1234.5   54.35   50.95      0   1230.2   54.56   51.15      0
        """
        result = NCCLTestResult(
            operation=operation,
            num_gpus=0,
            num_nodes=len(self.nodes),
            gpus_per_node=self.gpus_per_node,
            timestamp=datetime.datetime.now().isoformat(),
            raw_output=output,
        )

        # 解析GPU数量和版本信息
        gpu_match = re.search(r"nGpus\s+(\d+)", output)
        if gpu_match:
            result.gpus_per_node = int(gpu_match.group(1))

        # 计算总GPU数
        rank_matches = re.findall(r"Rank\s+\d+", output)
        if rank_matches:
            result.num_gpus = len(rank_matches)
        else:
            result.num_gpus = len(self.nodes) * self.gpus_per_node

        # 解析NCCL版本
        ver_match = re.search(r"NCCL version\s+([\d.]+)", output)
        if ver_match:
            result.nccl_version = ver_match.group(1)

        # 解析使用的算法和协议
        algo_match = re.search(r"Using\s+(Tree|Ring|CollNet\w*|NVLS\w*)", output)
        if algo_match:
            result.algorithm = algo_match.group(1)

        proto_match = re.search(r"protocol\s+(LL|LL128|Simple)", output)
        if proto_match:
            result.protocol = proto_match.group(1)

        # 解析每一行的测试结果
        for line in output.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 尝试匹配数据行
            # 格式: size count type redop root time algbw busbw #wrong [time algbw busbw #wrong]
            parts = line.split()
            if len(parts) >= 9:
                try:
                    size = int(parts[0])
                    count = int(parts[1])
                    dtype = parts[2]
                    redop = parts[3]

                    # 使用in-place结果（如果有的话，在后半部分）
                    if len(parts) >= 13:
                        # 有in-place和out-of-place两组结果
                        time_us = float(parts[8])
                        algo_bw = float(parts[9])
                        bus_bw = float(parts[10])
                        errors = int(parts[11])
                    else:
                        time_us = float(parts[5])
                        algo_bw = float(parts[6])
                        bus_bw = float(parts[7])
                        errors = int(parts[8]) if len(parts) > 8 else 0

                    point = NCCLTestPoint(
                        message_size=size,
                        message_size_human=self._human_readable_size(size),
                        count=count,
                        data_type=dtype,
                        reduction_op=redop,
                        time_us=time_us,
                        algo_bandwidth_gbps=algo_bw,
                        bus_bandwidth_gbps=bus_bw,
                        errors=errors,
                    )
                    result.test_points.append(point)
                except (ValueError, IndexError):
                    continue

        # 计算峰值性能
        if result.test_points:
            peak_point = max(result.test_points, key=lambda p: p.bus_bandwidth_gbps)
            result.peak_bus_bw_gbps = peak_point.bus_bandwidth_gbps
            result.peak_algo_bw_gbps = peak_point.algo_bandwidth_gbps
            result.peak_message_size = peak_point.message_size

        return result

    @staticmethod
    def _human_readable_size(size_bytes: int) -> str:
        """将字节数转换为人类可读格式"""
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        unit_idx = 0
        while size >= 1024 and unit_idx < len(units) - 1:
            size /= 1024
            unit_idx += 1
        if size == int(size):
            return f"{int(size)}{units[unit_idx]}"
        return f"{size:.1f}{units[unit_idx]}"

    def run_test(self, operation: CollectiveOp,
                 num_gpus: Optional[int] = None,
                 extra_env: Optional[Dict] = None) -> NCCLTestResult:
        """
        运行单个NCCL集合操作测试

        Args:
            operation: 集合操作类型
            num_gpus: GPU数量（默认使用所有节点的所有GPU）
            extra_env: 额外的NCCL环境变量

        Returns:
            NCCLTestResult
        """
        if num_gpus is None:
            num_gpus = len(self.nodes) * self.gpus_per_node

        logger.info(
            f"开始NCCL测试: {operation.value}, "
            f"{num_gpus} GPU ({len(self.nodes)} 节点)"
        )

        # 准备hostfile
        master_node = self.nodes[0]
        master_ip = master_node.get("ip", master_node.get("hostname", ""))

        hostfile_content = self._build_hostfile_content()
        write_hf_cmd = f"cat > /tmp/nccl_hostfile << 'EOF'\n{hostfile_content}\nEOF"
        self._run_remote(master_ip, write_hf_cmd)

        # 构建测试命令
        test_cmd = self._build_nccl_test_command(operation, num_gpus, extra_env)

        # 执行测试
        import time
        start_time = time.time()
        stdout, stderr, rc = self._run_remote(
            master_ip, test_cmd, timeout=self.test_timeout
        )
        elapsed = time.time() - start_time

        if rc != 0:
            logger.error(f"NCCL测试执行失败: {stderr}")
            result = NCCLTestResult(
                operation=operation,
                num_gpus=num_gpus,
                num_nodes=len(self.nodes),
                gpus_per_node=self.gpus_per_node,
                timestamp=datetime.datetime.now().isoformat(),
                success=False,
                error_message=stderr,
            )
            return result

        # 解析结果
        result = self._parse_nccl_output(stdout, operation)
        result.duration_seconds = elapsed
        result.nccl_env_vars = self.nccl_env

        logger.info(
            f"NCCL测试完成: {operation.value}, "
            f"峰值总线带宽={result.peak_bus_bw_gbps:.2f} GB/s, "
            f"消息大小={self._human_readable_size(result.peak_message_size)}"
        )

        return result

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
            return "", "Timeout", -1
        except Exception as e:
            return "", str(e), -1

    def run_full_suite(self, operations: Optional[List[CollectiveOp]] = None) -> NCCLBenchmarkSuite:
        """
        运行完整的NCCL基准测试套件

        Args:
            operations: 要测试的集合操作列表，默认测试AllReduce和AllGather

        Returns:
            NCCLBenchmarkSuite
        """
        if operations is None:
            operations = [
                CollectiveOp.ALL_REDUCE,
                CollectiveOp.ALL_GATHER,
                CollectiveOp.REDUCE_SCATTER,
            ]

        total_gpus = len(self.nodes) * self.gpus_per_node
        suite = NCCLBenchmarkSuite(
            suite_time=datetime.datetime.now().isoformat(),
            cluster_name=self.cluster_name,
            total_gpus=total_gpus,
        )

        logger.info(
            f"开始NCCL完整基准测试套件: "
            f"{len(operations)} 操作, {total_gpus} GPU"
        )

        for op in operations:
            result = self.run_test(op)
            suite.results.append(result)

            # 生成每个操作的摘要
            suite.summary[op.value] = {
                "peak_bus_bw_gbps": result.peak_bus_bw_gbps,
                "peak_algo_bw_gbps": result.peak_algo_bw_gbps,
                "peak_message_size": self._human_readable_size(result.peak_message_size),
                "success": result.success,
                "duration_seconds": result.duration_seconds,
            }

        logger.info("NCCL基准测试套件完成")
        return suite

    def compare_algorithms(self, operation: CollectiveOp = CollectiveOp.ALL_REDUCE
                           ) -> Dict[str, NCCLTestResult]:
        """
        对比不同NCCL算法的性能

        Args:
            operation: 要测试的集合操作

        Returns:
            算法名到测试结果的映射
        """
        algorithms = {
            "Tree": {"NCCL_ALGO": "Tree"},
            "Ring": {"NCCL_ALGO": "Ring"},
        }

        results = {}
        for algo_name, env_override in algorithms.items():
            logger.info(f"测试算法: {algo_name}")
            result = self.run_test(operation, extra_env=env_override)
            result.algorithm = algo_name
            results[algo_name] = result

        # 输出对比
        logger.info("算法对比结果:")
        for algo, result in results.items():
            logger.info(
                f"  {algo}: 峰值总线带宽={result.peak_bus_bw_gbps:.2f} GB/s"
            )

        return results

    def export_results(self, suite: NCCLBenchmarkSuite, output_path: str) -> None:
        """导出测试结果到JSON"""
        data = {
            "suite_time": suite.suite_time,
            "cluster_name": suite.cluster_name,
            "total_gpus": suite.total_gpus,
            "summary": suite.summary,
            "results": [],
        }

        for result in suite.results:
            result_data = {
                "operation": result.operation.value,
                "num_gpus": result.num_gpus,
                "num_nodes": result.num_nodes,
                "algorithm": result.algorithm,
                "protocol": result.protocol,
                "peak_bus_bw_gbps": result.peak_bus_bw_gbps,
                "peak_algo_bw_gbps": result.peak_algo_bw_gbps,
                "peak_message_size": result.peak_message_size,
                "nccl_version": result.nccl_version,
                "duration_seconds": result.duration_seconds,
                "success": result.success,
                "test_points": [
                    {
                        "size": p.message_size,
                        "size_human": p.message_size_human,
                        "time_us": p.time_us,
                        "algo_bw_gbps": p.algo_bandwidth_gbps,
                        "bus_bw_gbps": p.bus_bandwidth_gbps,
                        "errors": p.errors,
                    }
                    for p in result.test_points
                ],
            }
            data["results"].append(result_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"NCCL测试结果已导出: {output_path}")
