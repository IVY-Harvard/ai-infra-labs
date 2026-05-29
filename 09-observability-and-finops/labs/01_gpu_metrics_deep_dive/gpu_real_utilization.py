"""
GPU 真实利用率计算器

nvidia-smi 的 GPU-Util 只告诉你"有没有东西在跑"，
本脚本基于 DCGM Profiling Metrics 计算真正的 GPU 计算效率。

用法:
    python gpu_real_utilization.py --prometheus-url http://localhost:9090 --interval 30
"""

import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import requests


# ============================================================
# 数据模型
# ============================================================

@dataclass
class GPUMetrics:
    """单张 GPU 的 DCGM 指标快照"""
    gpu_id: str
    timestamp: float

    # Profiling metrics
    sm_active: float = 0.0          # DCGM_FI_PROF_SM_ACTIVE
    sm_occupancy: float = 0.0       # DCGM_FI_PROF_SM_OCCUPANCY
    tensor_active: float = 0.0      # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
    dram_active: float = 0.0        # DCGM_FI_PROF_DRAM_ACTIVE
    fp32_active: float = 0.0        # DCGM_FI_PROF_PIPE_FP32_ACTIVE
    fp16_active: float = 0.0        # DCGM_FI_PROF_PIPE_FP16_ACTIVE
    nvlink_tx_bytes: float = 0.0    # DCGM_FI_PROF_NVLINK_TX_BYTES
    nvlink_rx_bytes: float = 0.0    # DCGM_FI_PROF_NVLINK_RX_BYTES

    # Health metrics
    gpu_temp: float = 0.0
    memory_temp: float = 0.0
    power_usage: float = 0.0
    fb_used_mb: float = 0.0
    fb_free_mb: float = 0.0
    ecc_sbe: int = 0
    ecc_dbe: int = 0
    pcie_replay: int = 0
    throttle_reasons: int = 0


@dataclass
class RealUtilization:
    """真实利用率计算结果"""
    gpu_id: str
    timestamp: str

    # 核心效率指标
    compute_efficiency: float        # Tensor Core × SM Active
    memory_bandwidth_efficiency: float  # DRAM Active
    sm_parallel_efficiency: float    # SM Occupancy
    nvlink_utilization: float        # NVLink 利用率

    # 综合评分
    overall_score: float             # 0-100 综合效率分
    phase_estimate: str              # 推测当前阶段: prefill / decode / idle / mixed

    # 健康度
    health_score: float              # 0-100 健康评分
    health_issues: list = field(default_factory=list)

    # 对比：nvidia-smi 风格利用率
    naive_utilization: float         # 近似 nvidia-smi GPU-Util


# ============================================================
# Prometheus 查询层
# ============================================================

class PrometheusClient:
    """Prometheus 查询客户端"""

    # DCGM 指标名称 → GPUMetrics 字段映射
    METRIC_MAP = {
        "DCGM_FI_PROF_SM_ACTIVE": "sm_active",
        "DCGM_FI_PROF_SM_OCCUPANCY": "sm_occupancy",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": "tensor_active",
        "DCGM_FI_PROF_DRAM_ACTIVE": "dram_active",
        "DCGM_FI_PROF_PIPE_FP32_ACTIVE": "fp32_active",
        "DCGM_FI_PROF_PIPE_FP16_ACTIVE": "fp16_active",
        "DCGM_FI_PROF_NVLINK_TX_BYTES": "nvlink_tx_bytes",
        "DCGM_FI_PROF_NVLINK_RX_BYTES": "nvlink_rx_bytes",
        "DCGM_FI_DEV_GPU_TEMP": "gpu_temp",
        "DCGM_FI_DEV_MEMORY_TEMP": "memory_temp",
        "DCGM_FI_DEV_POWER_USAGE": "power_usage",
        "DCGM_FI_DEV_FB_USED": "fb_used_mb",
        "DCGM_FI_DEV_FB_FREE": "fb_free_mb",
        "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "ecc_sbe",
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": "ecc_dbe",
        "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": "pcie_replay",
        "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": "throttle_reasons",
    }

    def __init__(self, prometheus_url: str):
        self.base_url = prometheus_url.rstrip("/")

    def query_instant(self, promql: str) -> list:
        """执行即时查询"""
        resp = requests.get(
            f"{self.base_url}/api/v1/query",
            params={"query": promql},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data["data"]["result"]

    def fetch_all_gpu_metrics(self) -> dict[str, GPUMetrics]:
        """拉取所有 GPU 的全部 DCGM 指标，返回 {gpu_id: GPUMetrics}"""
        now = time.time()
        gpu_map: dict[str, GPUMetrics] = {}

        for metric_name, field_name in self.METRIC_MAP.items():
            results = self.query_instant(metric_name)
            for item in results:
                gpu_id = item["metric"].get("gpu", item["metric"].get("GPU_I_ID", "unknown"))
                if gpu_id not in gpu_map:
                    gpu_map[gpu_id] = GPUMetrics(gpu_id=gpu_id, timestamp=now)
                value = float(item["value"][1])
                setattr(gpu_map[gpu_id], field_name, value)

        return gpu_map


# ============================================================
# 真实利用率计算引擎
# ============================================================

class RealUtilizationCalculator:
    """
    基于 DCGM Profiling Metrics 的真实 GPU 利用率计算。

    核心思想：
      nvidia-smi utilization ≈ GR_ENGINE_ACTIVE（有无 kernel 在跑）
      真实计算效率 = Tensor Core Active × SM Active（Tensor Core 真正在干活的比例）
      真实带宽效率 = DRAM Active（HBM 带宽的利用率）
    """

    # H20 参考值（用于归一化和评分）
    H20_NVLINK_BW = 900e9   # 900 GB/s 双向
    H20_TDP = 400            # 400W TDP
    H20_HBM_GB = 96          # 96 GB HBM3

    def calculate(self, metrics: GPUMetrics) -> RealUtilization:
        """计算单张 GPU 的真实利用率"""

        # ------ 1. 计算效率 ------
        # Tensor Core Active × SM Active = 真正的 AI 计算利用率
        compute_efficiency = metrics.tensor_active * metrics.sm_active

        # ------ 2. 内存带宽效率 ------
        memory_bandwidth_efficiency = metrics.dram_active

        # ------ 3. SM 并行效率 ------
        sm_parallel_efficiency = metrics.sm_occupancy

        # ------ 4. NVLink 利用率 ------
        total_nvlink = metrics.nvlink_tx_bytes + metrics.nvlink_rx_bytes
        nvlink_utilization = min(total_nvlink / self.H20_NVLINK_BW, 1.0)

        # ------ 5. 阶段推测 ------
        phase = self._estimate_phase(metrics)

        # ------ 6. 综合效率评分 ------
        overall_score = self._compute_overall_score(
            compute_efficiency, memory_bandwidth_efficiency,
            sm_parallel_efficiency, phase,
        )

        # ------ 7. 健康度评分 ------
        health_score, health_issues = self._compute_health_score(metrics)

        # ------ 8. 近似 nvidia-smi utilization ------
        # GR_ENGINE_ACTIVE ≈ nvidia-smi utilization
        # 我们用 SM Active 近似（更精确一些）
        naive_utilization = metrics.sm_active

        return RealUtilization(
            gpu_id=metrics.gpu_id,
            timestamp=datetime.fromtimestamp(metrics.timestamp).isoformat(),
            compute_efficiency=round(compute_efficiency, 4),
            memory_bandwidth_efficiency=round(memory_bandwidth_efficiency, 4),
            sm_parallel_efficiency=round(sm_parallel_efficiency, 4),
            nvlink_utilization=round(nvlink_utilization, 4),
            overall_score=round(overall_score, 1),
            phase_estimate=phase,
            health_score=round(health_score, 1),
            health_issues=health_issues,
            naive_utilization=round(naive_utilization, 4),
        )

    def _estimate_phase(self, m: GPUMetrics) -> str:
        """
        根据指标模式推测当前处于哪个阶段。

        Prefill: Tensor Active 高, DRAM Active 中等 (compute-bound)
        Decode:  Tensor Active 低, DRAM Active 高   (memory-bound)
        Idle:    所有指标都低
        Mixed:   Continuous batching 下 Prefill + Decode 混合
        """
        if m.sm_active < 0.05:
            return "idle"
        if m.tensor_active > 0.4 and m.dram_active < 0.5:
            return "prefill_dominant"
        if m.tensor_active < 0.2 and m.dram_active > 0.5:
            return "decode_dominant"
        return "mixed"

    def _compute_overall_score(
        self,
        compute_eff: float,
        mem_bw_eff: float,
        sm_parallel: float,
        phase: str,
    ) -> float:
        """
        综合效率评分 (0-100)。

        评分逻辑：
          - idle → 0 分
          - prefill_dominant → 侧重 compute_efficiency
          - decode_dominant → 侧重 memory_bandwidth_efficiency
          - mixed → 取两者的加权最大值
        """
        if phase == "idle":
            return 0.0

        if phase == "prefill_dominant":
            # Prefill: 70% 看计算效率, 20% 看 SM 并行度, 10% 看带宽
            raw = compute_eff * 0.7 + sm_parallel * 0.2 + mem_bw_eff * 0.1
        elif phase == "decode_dominant":
            # Decode: 50% 看带宽, 30% 看 SM 并行度, 20% 看计算效率
            raw = mem_bw_eff * 0.5 + sm_parallel * 0.3 + compute_eff * 0.2
        else:
            # Mixed: 取 Prefill 和 Decode 评分的最大值
            prefill_score = compute_eff * 0.7 + sm_parallel * 0.2 + mem_bw_eff * 0.1
            decode_score = mem_bw_eff * 0.5 + sm_parallel * 0.3 + compute_eff * 0.2
            raw = max(prefill_score, decode_score)

        return min(raw * 100, 100.0)

    def _compute_health_score(self, m: GPUMetrics) -> tuple[float, list]:
        """
        GPU 健康度评分 (0-100)，参考 theory/02 的评分模型。
        """
        score = 100.0
        issues = []

        # 温度
        if m.gpu_temp > 85:
            penalty = (m.gpu_temp - 85) * 3
            score -= penalty
            issues.append(f"GPU 温度偏高: {m.gpu_temp}°C")
        if m.memory_temp > 95:
            penalty = (m.memory_temp - 95) * 5
            score -= penalty
            issues.append(f"HBM 温度偏高: {m.memory_temp}°C")

        # ECC 错误
        if m.ecc_dbe > 0:
            score -= 50
            issues.append(f"检测到不可纠正 ECC 错误 (DBE={m.ecc_dbe})，建议退役 GPU")
        if m.ecc_sbe > 10:
            score -= 10
            issues.append(f"单比特 ECC 错误偏多: {m.ecc_sbe}")

        # 限频
        thermal_throttle_bits = 0x08 | 0x20 | 0x40 | 0x80
        if m.throttle_reasons & thermal_throttle_bits:
            score -= 20
            issues.append(f"检测到热/功率限频 (bitmap=0x{m.throttle_reasons:02x})")

        # PCIe 重传
        if m.pcie_replay > 10:
            score -= 15
            issues.append(f"PCIe 重传率偏高: {m.pcie_replay}/s")

        return max(0.0, score), issues


# ============================================================
# 报告生成
# ============================================================

def generate_report(results: list[RealUtilization]) -> dict:
    """生成集群级 GPU 利用率报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "gpu_count": len(results),
        "summary": {},
        "gpus": [],
    }

    if not results:
        return report

    # 汇总统计
    compute_effs = [r.compute_efficiency for r in results]
    mem_bw_effs = [r.memory_bandwidth_efficiency for r in results]
    overall_scores = [r.overall_score for r in results]
    naive_utils = [r.naive_utilization for r in results]

    report["summary"] = {
        "avg_compute_efficiency": round(sum(compute_effs) / len(compute_effs), 4),
        "avg_memory_bw_efficiency": round(sum(mem_bw_effs) / len(mem_bw_effs), 4),
        "avg_overall_score": round(sum(overall_scores) / len(overall_scores), 1),
        "avg_naive_utilization": round(sum(naive_utils) / len(naive_utils), 4),
        "utilization_gap": round(
            sum(naive_utils) / len(naive_utils) - sum(compute_effs) / len(compute_effs), 4
        ),
        "comment": (
            "utilization_gap 反映 nvidia-smi 利用率与真实计算效率的差距。"
            "差距越大，说明 GPU 'looks busy but isn't really computing'。"
        ),
    }

    for r in results:
        gpu_data = {
            "gpu_id": r.gpu_id,
            "naive_utilization": f"{r.naive_utilization:.1%}",
            "real_compute_efficiency": f"{r.compute_efficiency:.1%}",
            "memory_bw_efficiency": f"{r.memory_bandwidth_efficiency:.1%}",
            "sm_parallel_efficiency": f"{r.sm_parallel_efficiency:.1%}",
            "overall_score": r.overall_score,
            "phase": r.phase_estimate,
            "health_score": r.health_score,
            "health_issues": r.health_issues,
        }
        report["gpus"].append(gpu_data)

    return report


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GPU 真实利用率计算器")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--interval", type=int, default=30, help="采集间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只采集一次后退出")
    parser.add_argument("--output", default=None, help="输出文件路径（JSON）")
    args = parser.parse_args()

    client = PrometheusClient(args.prometheus_url)
    calculator = RealUtilizationCalculator()

    print(f"[GPU Real Utilization] 连接 Prometheus: {args.prometheus_url}")
    print(f"[GPU Real Utilization] 采集间隔: {args.interval}s")

    while True:
        try:
            gpu_metrics = client.fetch_all_gpu_metrics()
            results = [calculator.calculate(m) for m in gpu_metrics.values()]
            report = generate_report(results)

            # 打印摘要
            summary = report["summary"]
            print(f"\n{'='*60}")
            print(f"[{report['timestamp']}] GPU 利用率报告 ({report['gpu_count']} GPUs)")
            print(f"  nvidia-smi 风格利用率: {summary.get('avg_naive_utilization', 0):.1%}")
            print(f"  真实计算效率:          {summary.get('avg_compute_efficiency', 0):.1%}")
            print(f"  内存带宽效率:          {summary.get('avg_memory_bw_efficiency', 0):.1%}")
            print(f"  综合评分:              {summary.get('avg_overall_score', 0):.1f}/100")
            print(f"  利用率落差:            {summary.get('utilization_gap', 0):.1%}")

            for gpu in report["gpus"]:
                status = "OK" if gpu["health_score"] >= 80 else "WARN"
                print(f"  GPU {gpu['gpu_id']}: score={gpu['overall_score']:.0f} "
                      f"phase={gpu['phase']} health={gpu['health_score']:.0f} [{status}]")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)

        except requests.exceptions.ConnectionError:
            print(f"[WARN] 无法连接 Prometheus ({args.prometheus_url})，将重试...")
        except Exception as e:
            print(f"[ERROR] {e}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
