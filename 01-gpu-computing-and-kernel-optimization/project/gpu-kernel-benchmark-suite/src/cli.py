"""
GPU Kernel Benchmark Suite — CLI 入口

Usage:
    python -m src.cli benchmark --all
    python -m src.cli benchmark --compute --gpu 0
    python -m src.cli benchmark --memory
    python -m src.cli benchmark --p2p
    python -m src.cli report --output report.html
    python -m src.cli compare run1.json run2.json
"""

import argparse
import json
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any

import torch


def run_benchmark(args) -> Dict[str, Any]:
    """运行基准测试"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_gpus': torch.cuda.device_count(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
    }

    gpu_ids = args.gpu if args.gpu else list(range(torch.cuda.device_count()))

    if args.all or args.compute:
        print("\n" + "=" * 60)
        print("Compute Benchmark")
        print("=" * 60)
        from src.kernels.compute_benchmark import ComputeBenchmark

        results['compute'] = {}
        for gpu_id in gpu_ids:
            bench = ComputeBenchmark(device_id=gpu_id)
            gpu_results = bench.run_all()
            bench.print_results(gpu_results)

            # 序列化
            for dtype, res_list in gpu_results.items():
                if dtype not in results['compute']:
                    results['compute'][dtype] = []
                for r in res_list:
                    results['compute'][dtype].append({
                        'gpu_id': gpu_id,
                        'dtype': r.dtype,
                        'M': r.M, 'N': r.N, 'K': r.K,
                        'time_ms': r.time_ms,
                        'tflops': r.tflops,
                        'peak_tflops': r.peak_tflops,
                        'utilization': r.utilization,
                    })

    if args.all or args.memory:
        print("\n" + "=" * 60)
        print("Memory Bandwidth Benchmark")
        print("=" * 60)
        from src.kernels.memory_benchmark import MemoryBenchmark

        results['memory'] = {}
        for gpu_id in gpu_ids[:1]:  # 内存测试取一张卡
            bench = MemoryBenchmark(device_id=gpu_id)
            mem_results = bench.run_all()
            bench.print_results(mem_results)

            for category, res_list in mem_results.items():
                results['memory'][category] = [
                    {
                        'test_name': r.test_name,
                        'data_size_mb': r.data_size_mb,
                        'time_ms': r.time_ms,
                        'bandwidth_gb_s': r.bandwidth_gb_s,
                        'utilization': r.utilization,
                    } for r in res_list
                ]

    if args.all or args.p2p:
        print("\n" + "=" * 60)
        print("P2P / NVLink Benchmark")
        print("=" * 60)
        from src.kernels.p2p_benchmark import P2PBenchmark

        bench = P2PBenchmark()
        if bench.num_gpus >= 2:
            p2p_results = bench.benchmark_all_pairs()
            bench.print_results(p2p_results)

            results['p2p'] = [
                [{'src': r.src_gpu, 'dst': r.dst_gpu,
                  'bandwidth_gb_s': r.bandwidth_gb_s,
                  'can_access': r.can_access}
                 for r in row]
                for row in p2p_results
            ]

    # 保存结果
    output_file = args.output or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_file}")

    return results


def run_report(args):
    """生成 HTML 报告"""
    from src.reporter.html_report import generate_benchmark_report

    input_file = args.input
    if not input_file:
        # 查找最新的 benchmark 结果
        json_files = [f for f in os.listdir('.') if f.startswith('benchmark_') and f.endswith('.json')]
        if not json_files:
            print("未找到基准测试结果文件。请先运行 benchmark。")
            return
        input_file = sorted(json_files)[-1]
        print(f"使用最新结果: {input_file}")

    with open(input_file, 'r') as f:
        results = json.load(f)

    output = args.output or 'report.html'
    generate_benchmark_report(results, output)


def run_compare(args):
    """对比多次结果"""
    from src.reporter.comparison import compare_runs

    if len(args.files) < 2:
        print("至少需要 2 个结果文件进行对比")
        return

    comp = compare_runs(args.files)
    comp.print_comparison()


def main():
    parser = argparse.ArgumentParser(description='GPU Kernel Benchmark Suite')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # benchmark 子命令
    bench_parser = subparsers.add_parser('benchmark', help='运行基准测试')
    bench_parser.add_argument('--all', action='store_true', help='运行全部测试')
    bench_parser.add_argument('--compute', action='store_true', help='计算性能测试')
    bench_parser.add_argument('--memory', action='store_true', help='内存带宽测试')
    bench_parser.add_argument('--p2p', action='store_true', help='P2P/NVLink 测试')
    bench_parser.add_argument('--gpu', type=int, nargs='*', help='指定 GPU ID')
    bench_parser.add_argument('--output', type=str, help='结果输出文件')

    # report 子命令
    report_parser = subparsers.add_parser('report', help='生成报告')
    report_parser.add_argument('--input', type=str, help='输入 JSON 文件')
    report_parser.add_argument('--output', type=str, help='输出 HTML 文件')

    # compare 子命令
    compare_parser = subparsers.add_parser('compare', help='对比结果')
    compare_parser.add_argument('files', nargs='+', help='要对比的 JSON 文件')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("错误: CUDA 不可用")
        sys.exit(1)

    if args.command == 'benchmark':
        if not (args.all or args.compute or args.memory or args.p2p):
            args.all = True
        run_benchmark(args)
    elif args.command == 'report':
        run_report(args)
    elif args.command == 'compare':
        run_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
