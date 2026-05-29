#!/usr/bin/env python3
"""Inter-Node Bandwidth Matrix - 测量集群节点间网络带宽"""

import subprocess
import argparse
import json
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple


class BandwidthMatrix:
    """节点间带宽矩阵测量"""

    def __init__(self, hosts: List[str], mode: str = "tcp", port: int = 5201):
        self.hosts = hosts
        self.mode = mode
        self.port = port
        self.matrix: List[List[float]] = [
            [0.0] * len(hosts) for _ in range(len(hosts))
        ]

    def run_iperf3_test(self, server: str, client: str,
                        duration: int = 10) -> Optional[float]:
        """在两节点间运行 iperf3 测试"""
        # 启动服务端
        server_cmd = f"ssh {server} 'iperf3 -s -p {self.port} -1 -D'"
        subprocess.run(server_cmd, shell=True, capture_output=True)
        time.sleep(1)

        # 运行客户端
        client_cmd = [
            "ssh", client,
            f"iperf3 -c {server} -p {self.port} -t {duration} -J"
        ]
        try:
            result = subprocess.run(
                client_cmd, capture_output=True, text=True, timeout=duration + 30
            )
            data = json.loads(result.stdout)
            bw_bps = data["end"]["sum_received"]["bits_per_second"]
            return bw_bps / 1e9  # Gbps
        except (json.JSONDecodeError, KeyError, subprocess.TimeoutExpired):
            return None

    def run_rdma_test(self, server: str, client: str) -> Optional[float]:
        """在两节点间运行 RDMA 带宽测试"""
        # 启动服务端
        server_cmd = f"ssh {server} 'ib_write_bw -d mlx5_0 -s 65536 -n 1000 --report_gbits &'"
        subprocess.run(server_cmd, shell=True, capture_output=True)
        time.sleep(2)

        # 运行客户端
        client_cmd = [
            "ssh", client,
            f"ib_write_bw -d mlx5_0 -s 65536 -n 1000 --report_gbits {server}"
        ]
        try:
            result = subprocess.run(
                client_cmd, capture_output=True, text=True, timeout=60
            )
            # 解析最后一行带宽数据
            for line in reversed(result.stdout.split("\n")):
                match = re.search(r"([\d.]+)\s*$", line.strip())
                if match:
                    return float(match.group(1))
        except subprocess.TimeoutExpired:
            pass
        return None

    def measure_pair(self, src_idx: int, dst_idx: int) -> Tuple[int, int, float]:
        """测量一对节点间的带宽"""
        src, dst = self.hosts[src_idx], self.hosts[dst_idx]
        print(f"  测试 {src} -> {dst}...")

        if self.mode == "rdma":
            bw = self.run_rdma_test(dst, src)
        else:
            bw = self.run_iperf3_test(dst, src)

        return src_idx, dst_idx, bw or 0.0

    def measure_all(self, parallel: int = 4):
        """测量所有节点对间的带宽"""
        pairs = [
            (i, j) for i in range(len(self.hosts))
            for j in range(len(self.hosts)) if i != j
        ]

        print(f"[*] 测量 {len(pairs)} 对节点间带宽 (并行度: {parallel})")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(self.measure_pair, i, j): (i, j)
                for i, j in pairs
            }
            for future in as_completed(futures):
                src_idx, dst_idx, bw = future.result()
                self.matrix[src_idx][dst_idx] = bw

    def print_matrix(self):
        """打印带宽矩阵"""
        print("\n节点间带宽矩阵 (Gbps):")
        print("=" * (15 + 12 * len(self.hosts)))

        # 表头
        header = f"{'':15s}" + "".join(f"{h:>12s}" for h in self.hosts)
        print(header)
        print("-" * len(header))

        for i, host in enumerate(self.hosts):
            row = f"{host:15s}"
            for j in range(len(self.hosts)):
                if i == j:
                    row += f"{'---':>12s}"
                elif self.matrix[i][j] > 0:
                    row += f"{self.matrix[i][j]:>11.2f} "
                else:
                    row += f"{'N/A':>12s}"
            print(row)

        # 统计摘要
        all_bw = [self.matrix[i][j] for i in range(len(self.hosts))
                  for j in range(len(self.hosts)) if i != j and self.matrix[i][j] > 0]
        if all_bw:
            print(f"\n统计: 最小={min(all_bw):.2f} Gbps, "
                  f"最大={max(all_bw):.2f} Gbps, "
                  f"平均={sum(all_bw)/len(all_bw):.2f} Gbps")

    def export_heatmap_data(self, output_path: str):
        """导出热力图数据"""
        data = {
            "hosts": self.hosts,
            "matrix": self.matrix,
            "mode": self.mode,
            "unit": "Gbps",
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[+] 热力图数据已导出: {output_path}")


def load_hosts(hosts_file: str) -> List[str]:
    """从文件加载主机列表"""
    with open(hosts_file, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="节点间带宽矩阵测量")
    parser.add_argument("--hosts", required=True,
                        help="主机列表文件 (每行一个主机名/IP)")
    parser.add_argument("--mode", choices=["tcp", "rdma"], default="tcp",
                        help="传输模式 (默认: tcp)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="并行测试数 (默认: 4)")
    parser.add_argument("--port", type=int, default=5201,
                        help="iperf3 端口 (默认: 5201)")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    hosts = load_hosts(args.hosts)
    if len(hosts) < 2:
        print("错误: 至少需要 2 个节点")
        return

    print(f"[*] 节点数: {len(hosts)}, 模式: {args.mode}")
    print(f"[*] 节点列表: {', '.join(hosts)}")

    bm = BandwidthMatrix(hosts, mode=args.mode, port=args.port)
    bm.measure_all(parallel=args.parallel)
    bm.print_matrix()

    if args.output:
        bm.export_heatmap_data(args.output)


if __name__ == "__main__":
    main()
