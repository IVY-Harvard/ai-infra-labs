"""
GPU Benchmark Suite 测试

运行: pytest tests/test_benchmarks.py -v
"""

import pytest
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 跳过条件
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs"
)


class TestComputeBenchmark:
    """计算基准测试"""

    @requires_cuda
    def test_fp32_gemm(self):
        from src.kernels.compute_benchmark import ComputeBenchmark
        bench = ComputeBenchmark(device_id=0)
        results = bench.benchmark_fp32(sizes=[512])

        assert len(results) == 1
        assert results[0].tflops > 0
        assert results[0].time_ms > 0
        assert 0 < results[0].utilization <= 100

    @requires_cuda
    def test_fp16_gemm(self):
        from src.kernels.compute_benchmark import ComputeBenchmark
        bench = ComputeBenchmark(device_id=0)
        results = bench.benchmark_fp16(sizes=[512])

        assert len(results) == 1
        assert results[0].tflops > 0
        # FP16 应该比 FP32 快
        fp32_results = bench.benchmark_fp32(sizes=[512])
        assert results[0].tflops > fp32_results[0].tflops * 0.8

    @requires_cuda
    def test_larger_matrix_higher_utilization(self):
        """大矩阵应该有更高的利用率"""
        from src.kernels.compute_benchmark import ComputeBenchmark
        bench = ComputeBenchmark(device_id=0)

        small = bench.benchmark_fp16(sizes=[256])[0]
        large = bench.benchmark_fp16(sizes=[4096])[0]

        # 大矩阵利用率更高（至少相等）
        assert large.utilization >= small.utilization * 0.5


class TestMemoryBenchmark:
    """内存带宽测试"""

    @requires_cuda
    def test_copy_bandwidth(self):
        from src.kernels.memory_benchmark import MemoryBenchmark
        bench = MemoryBenchmark(device_id=0)
        results = bench.benchmark_copy(sizes_mb=[128])

        assert len(results) == 1
        # HBM 带宽至少应该 > 100 GB/s（任何 GPU）
        assert results[0].bandwidth_gb_s > 100

    @requires_cuda
    def test_h2d_bandwidth(self):
        from src.kernels.memory_benchmark import MemoryBenchmark
        bench = MemoryBenchmark(device_id=0)
        results = bench.benchmark_h2d(sizes_mb=[64])

        assert len(results) == 1
        # PCIe 带宽至少应该 > 5 GB/s
        assert results[0].bandwidth_gb_s > 5
        # 但不应该超过 PCIe 理论上限太多
        assert results[0].bandwidth_gb_s < 100


class TestP2PBenchmark:
    """P2P 带宽测试"""

    @requires_multi_gpu
    def test_p2p_access(self):
        from src.kernels.p2p_benchmark import P2PBenchmark
        bench = P2PBenchmark()
        access_matrix = bench.check_p2p_access()

        # 对角线应该全是 True
        for i in range(bench.num_gpus):
            assert access_matrix[i][i] is True

    @requires_multi_gpu
    def test_p2p_bandwidth(self):
        from src.kernels.p2p_benchmark import P2PBenchmark
        bench = P2PBenchmark()
        result = bench.benchmark_pair(0, 1, size_mb=64)

        assert result.bandwidth_gb_s > 0


class TestGPUProfiler:
    """GPU Profiler 测试"""

    @requires_cuda
    def test_get_status(self):
        from src.profiler.gpu_profiler import GPUProfiler
        profiler = GPUProfiler()
        status = profiler.get_status(0)

        assert status.device_id == 0
        assert status.memory_total_gb > 0
        assert status.temperature >= 0 or status.temperature == -1

    @requires_cuda
    def test_get_all_status(self):
        from src.profiler.gpu_profiler import GPUProfiler
        profiler = GPUProfiler()
        statuses = profiler.get_all_status()

        assert len(statuses) == torch.cuda.device_count()


class TestHTMLReporter:
    """HTML 报告测试"""

    def test_generate_empty_report(self, tmp_path):
        from src.reporter.html_report import HTMLReporter
        reporter = HTMLReporter()
        reporter.add_header("Test Report")
        output = str(tmp_path / "test_report.html")
        reporter.generate(output)

        assert os.path.exists(output)
        with open(output, 'r') as f:
            content = f.read()
        assert 'Test Report' in content

    def test_generate_with_table(self, tmp_path):
        from src.reporter.html_report import HTMLReporter
        reporter = HTMLReporter()
        reporter.add_header("Test")
        reporter.add_table("Data", ["A", "B"], [["1", "2"], ["3", "4"]])
        output = str(tmp_path / "test_table.html")
        reporter.generate(output)

        with open(output, 'r') as f:
            content = f.read()
        assert '<table>' in content


class TestComparison:
    """对比功能测试"""

    def test_basic_comparison(self):
        from src.reporter.comparison import BenchmarkComparison
        comp = BenchmarkComparison()
        comp.add_entry("TFLOPS", {"GPU0": 140.0, "GPU1": 138.0}, "TFLOPS")

        entries = comp.to_dict()
        assert len(entries) == 1
        assert entries[0]['values']['GPU0'] == 140.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
