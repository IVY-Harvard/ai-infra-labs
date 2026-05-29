"""
GPU 推理容量计算器
=================

基于硬件规格、模型参数、工作负载特征计算:
1. KV Cache 容量: 最大并发请求数 × 序列长度
2. 吞吐上限: 理论与实测的 tokens/s
3. SLO 约束容量: 满足延迟 SLO 的最大负载
4. 成本效率: tokens/GPU-hour, $/million_tokens

依赖: numpy
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 硬件与模型规格
# ============================================================

@dataclass
class GPUSpec:
    """GPU 硬件规格"""
    name: str
    memory_gb: float               # 显存容量 (GB)
    memory_bandwidth_tb_s: float   # 显存带宽 (TB/s)
    fp16_tflops: float             # FP16 算力 (TFLOPS)
    fp8_tflops: float              # FP8 算力 (TFLOPS)
    nvlink_bandwidth_gb_s: float   # NVLink 双向带宽 (GB/s)
    pcie_bandwidth_gb_s: float     # PCIe 带宽 (GB/s)
    tdp_watts: float               # TDP (W)
    cost_per_hour: float           # 按需价格 ($/h)


@dataclass
class ModelSpec:
    """LLM 模型规格"""
    name: str
    total_params_b: float          # 总参数量 (Billion)
    num_layers: int                # Transformer 层数
    hidden_dim: int                # 隐藏维度
    num_attention_heads: int       # 注意力头数
    num_kv_heads: int              # KV 头数 (GQA)
    head_dim: int                  # 每个头的维度
    max_seq_len: int               # 最大序列长度
    vocab_size: int                # 词表大小
    dtype_bytes: int = 2           # 数据类型字节数 (FP16=2, FP8=1)


@dataclass
class WorkloadProfile:
    """工作负载特征"""
    avg_prompt_tokens: int = 1024     # 平均输入 token 数
    avg_output_tokens: int = 256      # 平均输出 token 数
    max_prompt_tokens: int = 8192     # 最大输入长度
    max_output_tokens: int = 2048     # 最大输出长度
    concurrent_requests: int = 64      # 目标并发数
    target_qps: float = 10.0          # 目标 QPS
    slo_ttft_p99_s: float = 5.0       # TTFT SLO (秒)
    slo_tpot_p99_ms: float = 80.0     # TPOT SLO (毫秒)
    prefix_cache_hit_rate: float = 0.0 # 预估 Prefix Cache 命中率


# 预定义规格
GPU_SPECS = {
    "H20": GPUSpec(
        name="NVIDIA H20",
        memory_gb=96,
        memory_bandwidth_tb_s=4.0,
        fp16_tflops=148,
        fp8_tflops=296,
        nvlink_bandwidth_gb_s=900,
        pcie_bandwidth_gb_s=64,
        tdp_watts=400,
        cost_per_hour=4.0,
    ),
    "H100": GPUSpec(
        name="NVIDIA H100 SXM",
        memory_gb=80,
        memory_bandwidth_tb_s=3.35,
        fp16_tflops=989,
        fp8_tflops=1979,
        nvlink_bandwidth_gb_s=900,
        pcie_bandwidth_gb_s=64,
        tdp_watts=700,
        cost_per_hour=8.0,
    ),
    "A100_80G": GPUSpec(
        name="NVIDIA A100 80GB",
        memory_gb=80,
        memory_bandwidth_tb_s=2.0,
        fp16_tflops=312,
        fp8_tflops=624,
        nvlink_bandwidth_gb_s=600,
        pcie_bandwidth_gb_s=64,
        tdp_watts=400,
        cost_per_hour=4.0,
    ),
}

MODEL_SPECS = {
    "Qwen2.5-72B": ModelSpec(
        name="Qwen2.5-72B",
        total_params_b=72.7,
        num_layers=80,
        hidden_dim=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=32768,
        vocab_size=152064,
    ),
    "Qwen2.5-7B": ModelSpec(
        name="Qwen2.5-7B",
        total_params_b=7.6,
        num_layers=28,
        hidden_dim=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        head_dim=128,
        max_seq_len=32768,
        vocab_size=152064,
    ),
    "Llama3-70B": ModelSpec(
        name="Llama3-70B",
        total_params_b=70.6,
        num_layers=80,
        hidden_dim=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=8192,
        vocab_size=128256,
    ),
}


# ============================================================
# 容量计算器
# ============================================================

class CapacityCalculator:
    """GPU 推理容量计算器

    计算各维度的容量上限:
    1. KV Cache 容量 (最常见瓶颈)
    2. 计算吞吐上限
    3. SLO 约束容量
    4. 成本效率

    使用方式:
        calc = CapacityCalculator(
            gpu=GPU_SPECS["H20"],
            model=MODEL_SPECS["Qwen2.5-72B"],
            tp_size=8,
        )
        report = calc.full_capacity_report(workload)
    """

    def __init__(
        self,
        gpu: GPUSpec,
        model: ModelSpec,
        tp_size: int = 8,
        gpu_memory_utilization: float = 0.9,
        num_instances: int = 1,
    ):
        self.gpu = gpu
        self.model = model
        self.tp_size = tp_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_instances = num_instances

    # ========================================================
    # KV Cache 容量计算
    # ========================================================

    def kv_cache_per_token_bytes(self) -> float:
        """单个 token 的 KV Cache 大小 (per GPU)

        公式:
        kv_per_token = num_layers × 2(K+V) × (num_kv_heads / tp_size) × head_dim × dtype_bytes
        """
        kv_heads_per_gpu = self.model.num_kv_heads / self.tp_size
        # 如果 GQA 的 kv_heads < tp_size, 则每个 GPU 至少有 1 个
        kv_heads_per_gpu = max(1, kv_heads_per_gpu)

        per_token = (
            self.model.num_layers
            * 2  # K and V
            * kv_heads_per_gpu
            * self.model.head_dim
            * self.model.dtype_bytes
        )
        return per_token

    def available_kv_cache_memory_gb(self) -> float:
        """可用于 KV Cache 的 GPU 显存 (GB)

        Total GPU Memory
        ├── Model Weights (per GPU)
        ├── Activation Memory
        ├── Framework Overhead
        └── KV Cache (剩余部分)
        """
        total = self.gpu.memory_gb * self.gpu_memory_utilization

        # 模型权重 (per GPU with TP)
        model_weights_gb = (
            self.model.total_params_b * 1e9 * self.model.dtype_bytes
            / self.tp_size / 1e9
        )

        # Activation memory (近似)
        activation_gb = 4.0  # 约 4GB for batch processing

        # Framework overhead (CUDA context, buffers, etc.)
        overhead_gb = 3.0

        available = total - model_weights_gb - activation_gb - overhead_gb
        return max(0, available)

    def max_total_tokens(self) -> int:
        """KV Cache 能容纳的最大总 token 数 (per GPU)"""
        available_bytes = self.available_kv_cache_memory_gb() * 1e9
        per_token_bytes = self.kv_cache_per_token_bytes()
        return int(available_bytes / per_token_bytes)

    def max_concurrent_requests(self, workload: WorkloadProfile) -> int:
        """最大并发请求数 (基于 KV Cache)

        并发请求数 = 总可用 tokens / 每请求平均 tokens
        每请求 tokens = prompt_tokens + output_tokens (最大值)
        """
        avg_seq_len = workload.avg_prompt_tokens + workload.avg_output_tokens
        max_tokens = self.max_total_tokens()
        return int(max_tokens / avg_seq_len)

    # ========================================================
    # 吞吐计算
    # ========================================================

    def theoretical_decode_throughput(self) -> float:
        """理论 Decode 吞吐 (tokens/s, 单请求)

        Decode 是 Memory Bandwidth Bound:
        throughput ≈ bandwidth / (2 × model_params_per_gpu × dtype_bytes)

        Factor 2: 读模型权重 + 读写 KV Cache
        """
        params_per_gpu = self.model.total_params_b * 1e9 / self.tp_size
        bytes_per_token = 2 * params_per_gpu * self.model.dtype_bytes

        bandwidth_bytes_s = self.gpu.memory_bandwidth_tb_s * 1e12
        tokens_per_s = bandwidth_bytes_s / bytes_per_token
        return tokens_per_s

    def estimated_batch_throughput(self, batch_size: int) -> float:
        """估算 Batch 吞吐 (tokens/s, 所有请求合计)

        Batch 吞吐近似:
        - 小 batch (< 16): 接近线性增长
        - 中 batch (16-128): 次线性 (memory bandwidth 开始饱和)
        - 大 batch (> 128): 增长放缓, 受限于 compute

        经验公式 (基于 H20 + Qwen2.5-72B 实测拟合):
        total_tps ≈ single_tps × batch × efficiency(batch)
        efficiency(batch) = 1 / (1 + 0.002 × batch)
        """
        single_tps = self.theoretical_decode_throughput()
        # 效率随 batch 递减 (实测拟合)
        efficiency = 1.0 / (1.0 + 0.002 * batch_size)
        return single_tps * batch_size * efficiency

    def max_throughput_tokens_per_s(self, workload: WorkloadProfile) -> float:
        """最大吞吐 (考虑所有约束)"""
        max_batch = self.max_concurrent_requests(workload)
        # 实际 batch 不超过 max_num_seqs (默认 256)
        effective_batch = min(max_batch, 256)
        total_tps = self.estimated_batch_throughput(effective_batch)
        # 乘以实例数
        return total_tps * self.num_instances

    # ========================================================
    # SLO 约束容量
    # ========================================================

    def slo_constrained_batch_size(self, workload: WorkloadProfile) -> int:
        """满足 TPOT SLO 的最大 batch size

        TPOT ≈ 1 / (throughput / batch_size)
        目标: TPOT_p99 < slo_tpot_p99_ms

        实测经验 (H20 + Qwen2.5-72B):
        TPOT_p99(ms) ≈ 15 + 0.3 × batch_size (简化线性模型)
        """
        # 反解: batch_size = (target_tpot - base_tpot) / per_batch_increment
        base_tpot_ms = 15.0   # batch=1 时的 TPOT
        per_batch_ms = 0.3    # 每增加 1 个并发的 TPOT 增量

        max_batch = int((workload.slo_tpot_p99_ms - base_tpot_ms) / per_batch_ms)
        return max(1, max_batch)

    def slo_constrained_qps(self, workload: WorkloadProfile) -> float:
        """满足 SLO 的最大 QPS

        QPS = throughput / avg_output_tokens
        受限于:
        1. TPOT SLO → 限制 batch size → 限制 throughput
        2. TTFT SLO → 限制排队时间
        """
        max_batch = self.slo_constrained_batch_size(workload)
        throughput = self.estimated_batch_throughput(max_batch)
        qps = throughput / workload.avg_output_tokens

        # 乘以实例数
        return qps * self.num_instances

    # ========================================================
    # 成本效率
    # ========================================================

    def cost_per_million_tokens(self, workload: WorkloadProfile) -> Dict[str, float]:
        """计算每百万 token 成本

        $/1M tokens = (GPU_cost_per_hour × num_gpus) / (throughput_per_hour / 1M)
        """
        throughput_per_s = self.max_throughput_tokens_per_s(workload)
        throughput_per_hour = throughput_per_s * 3600

        total_gpu_cost_per_hour = (
            self.gpu.cost_per_hour * self.tp_size * self.num_instances
        )

        cost_per_1m_output = (
            total_gpu_cost_per_hour / (throughput_per_hour / 1e6)
            if throughput_per_hour > 0 else float('inf')
        )

        # Prompt tokens 成本 (Prefill 比 Decode 快 ~10x)
        prefill_efficiency = 10.0
        cost_per_1m_input = cost_per_1m_output / prefill_efficiency

        return {
            "output_tokens_per_1m_usd": round(cost_per_1m_output, 4),
            "input_tokens_per_1m_usd": round(cost_per_1m_input, 4),
            "blended_per_1m_usd": round(
                (cost_per_1m_input * workload.avg_prompt_tokens
                 + cost_per_1m_output * workload.avg_output_tokens)
                / (workload.avg_prompt_tokens + workload.avg_output_tokens),
                4
            ),
            "gpu_cost_per_hour": round(total_gpu_cost_per_hour, 2),
            "tokens_per_gpu_hour": round(throughput_per_hour / (self.tp_size * self.num_instances), 0),
        }

    # ========================================================
    # 完整容量报告
    # ========================================================

    def full_capacity_report(self, workload: WorkloadProfile) -> Dict:
        """生成完整容量评估报告"""
        kv_per_token = self.kv_cache_per_token_bytes()
        available_kv_gb = self.available_kv_cache_memory_gb()
        max_tokens = self.max_total_tokens()
        max_concurrent = self.max_concurrent_requests(workload)
        theoretical_tps = self.theoretical_decode_throughput()
        max_tps = self.max_throughput_tokens_per_s(workload)
        slo_batch = self.slo_constrained_batch_size(workload)
        slo_qps = self.slo_constrained_qps(workload)
        costs = self.cost_per_million_tokens(workload)

        # 判断瓶颈
        kv_limited_batch = max_concurrent
        slo_limited_batch = slo_batch
        bottleneck = "kv_cache" if kv_limited_batch < slo_limited_batch else "slo"

        report = {
            "configuration": {
                "gpu": self.gpu.name,
                "model": self.model.name,
                "tp_size": self.tp_size,
                "num_instances": self.num_instances,
                "total_gpus": self.tp_size * self.num_instances,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            },
            "kv_cache_capacity": {
                "per_token_bytes": kv_per_token,
                "available_memory_gb": round(available_kv_gb, 2),
                "max_total_tokens": max_tokens,
                "max_concurrent_requests": max_concurrent,
                "avg_seq_len_assumed": workload.avg_prompt_tokens + workload.avg_output_tokens,
            },
            "throughput_capacity": {
                "theoretical_single_request_tps": round(theoretical_tps, 1),
                "estimated_max_batch_tps": round(max_tps, 1),
                "max_qps_unlimited": round(max_tps / workload.avg_output_tokens, 2),
            },
            "slo_constrained": {
                "max_batch_for_tpot_slo": slo_batch,
                "max_qps_with_slo": round(slo_qps, 2),
                "tpot_slo_ms": workload.slo_tpot_p99_ms,
                "ttft_slo_s": workload.slo_ttft_p99_s,
            },
            "cost_efficiency": costs,
            "bottleneck_analysis": {
                "primary_bottleneck": bottleneck,
                "kv_cache_limited_batch": kv_limited_batch,
                "slo_limited_batch": slo_limited_batch,
                "recommendation": (
                    "增加 GPU 显存或减小 max_model_len" if bottleneck == "kv_cache"
                    else "优化推理效率或放宽 SLO"
                ),
            },
        }

        return report

    def instances_needed(self, workload: WorkloadProfile) -> int:
        """计算满足目标 QPS 需要多少个实例"""
        single_instance = CapacityCalculator(
            gpu=self.gpu, model=self.model,
            tp_size=self.tp_size, num_instances=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        single_qps = single_instance.slo_constrained_qps(workload)
        if single_qps <= 0:
            return float('inf')
        return int(np.ceil(workload.target_qps / single_qps))


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    import json

    print("=== GPU Inference Capacity Calculator ===\n")

    # 场景: 8×H20 + Qwen2.5-72B (TP=8), 单实例
    calc = CapacityCalculator(
        gpu=GPU_SPECS["H20"],
        model=MODEL_SPECS["Qwen2.5-72B"],
        tp_size=8,
        gpu_memory_utilization=0.9,
        num_instances=1,
    )

    workload = WorkloadProfile(
        avg_prompt_tokens=1024,
        avg_output_tokens=256,
        max_prompt_tokens=8192,
        max_output_tokens=2048,
        target_qps=10.0,
        slo_ttft_p99_s=5.0,
        slo_tpot_p99_ms=80.0,
    )

    report = calc.full_capacity_report(workload)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 需要多少实例?
    instances = calc.instances_needed(workload)
    print(f"\n目标 QPS={workload.target_qps} 需要 {instances} 个实例")
    print(f"  = {instances * 8} 张 H20 GPU")
    print(f"  = ${instances * 8 * 4.0:.0f}/hour")
