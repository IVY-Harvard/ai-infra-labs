"""
KV Cache 显存计算器

输入模型参数和推理配置，计算 KV Cache 的显存占用。
帮助理解为什么 KV Cache 是推理系统的核心瓶颈。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int  # GQA: < num_attention_heads; MHA: = num_attention_heads
    head_dim: int
    hidden_size: int
    vocab_size: int

    @property
    def num_params_billions(self) -> float:
        """粗略估算参数量 (B)"""
        # 近似: 12 * L * d^2 (包含 attention + FFN)
        params = 12 * self.num_layers * self.hidden_size ** 2
        params += self.vocab_size * self.hidden_size * 2  # embedding + lm_head
        return params / 1e9


@dataclass
class InferenceConfig:
    """推理配置"""
    seq_len: int = 4096          # 总序列长度 (prompt + output)
    batch_size: int = 32         # 并发请求数
    tp_size: int = 1             # Tensor Parallelism 大小
    kv_cache_dtype: str = "fp16" # KV Cache 数据类型

    @property
    def dtype_bytes(self) -> int:
        dtype_map = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
        return dtype_map.get(self.kv_cache_dtype, 2)


@dataclass
class GPUConfig:
    """GPU 配置"""
    name: str
    memory_gb: float
    bandwidth_tb_s: float
    flops_tflops: float

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gb * 1024**3)


# ============ 预定义模型 ============

MODELS = {
    "llama2-7b": ModelConfig(
        name="LLaMA-2-7B",
        num_layers=32, num_attention_heads=32, num_kv_heads=32,
        head_dim=128, hidden_size=4096, vocab_size=32000
    ),
    "llama2-13b": ModelConfig(
        name="LLaMA-2-13B",
        num_layers=40, num_attention_heads=40, num_kv_heads=40,
        head_dim=128, hidden_size=5120, vocab_size=32000
    ),
    "llama2-70b": ModelConfig(
        name="LLaMA-2-70B",
        num_layers=80, num_attention_heads=64, num_kv_heads=8,
        head_dim=128, hidden_size=8192, vocab_size=32000
    ),
    "llama3-8b": ModelConfig(
        name="LLaMA-3-8B",
        num_layers=32, num_attention_heads=32, num_kv_heads=8,
        head_dim=128, hidden_size=4096, vocab_size=128256
    ),
    "llama3-70b": ModelConfig(
        name="LLaMA-3-70B",
        num_layers=80, num_attention_heads=64, num_kv_heads=8,
        head_dim=128, hidden_size=8192, vocab_size=128256
    ),
    "qwen2.5-72b": ModelConfig(
        name="Qwen-2.5-72B",
        num_layers=80, num_attention_heads=64, num_kv_heads=8,
        head_dim=128, hidden_size=8192, vocab_size=152064
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B",
        num_layers=32, num_attention_heads=32, num_kv_heads=8,
        head_dim=128, hidden_size=4096, vocab_size=32000
    ),
}

GPUS = {
    "h20": GPUConfig(name="NVIDIA H20", memory_gb=96, bandwidth_tb_s=4.0, flops_tflops=148),
    "h100": GPUConfig(name="NVIDIA H100", memory_gb=80, bandwidth_tb_s=3.35, flops_tflops=989),
    "a100-80g": GPUConfig(name="NVIDIA A100 80G", memory_gb=80, bandwidth_tb_s=2.0, flops_tflops=312),
    "a100-40g": GPUConfig(name="NVIDIA A100 40G", memory_gb=40, bandwidth_tb_s=1.55, flops_tflops=312),
    "l40s": GPUConfig(name="NVIDIA L40S", memory_gb=48, bandwidth_tb_s=0.864, flops_tflops=366),
}


def calculate_kv_cache_per_token(model: ModelConfig, dtype_bytes: float) -> float:
    """计算每个 token 的 KV Cache 大小 (bytes)"""
    # 2 (K+V) × layers × kv_heads × head_dim × dtype_bytes
    return 2 * model.num_layers * model.num_kv_heads * model.head_dim * dtype_bytes


def calculate_kv_cache_total(model: ModelConfig, config: InferenceConfig) -> dict:
    """计算总 KV Cache 显存占用"""
    per_token = calculate_kv_cache_per_token(model, config.dtype_bytes)

    # 每个请求的 KV Cache
    per_request = per_token * config.seq_len

    # 总 KV Cache (考虑 TP 分片)
    total = per_request * config.batch_size / config.tp_size

    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "per_request_bytes": per_request,
        "per_request_mb": per_request / (1024**2),
        "per_request_gb": per_request / (1024**3),
        "total_bytes": total,
        "total_gb": total / (1024**3),
    }


def calculate_model_size(model: ModelConfig, dtype_bytes: int = 2) -> float:
    """计算模型权重大小 (bytes)"""
    # 近似计算
    params = model.num_params_billions * 1e9
    return params * dtype_bytes


def analyze_gpu_capacity(
    model: ModelConfig,
    gpu: GPUConfig,
    config: InferenceConfig,
    model_dtype_bytes: int = 2,
) -> dict:
    """分析 GPU 容量和最大 batch size"""

    # 模型权重 (考虑 TP)
    model_size = calculate_model_size(model, model_dtype_bytes) / config.tp_size

    # 可用于 KV Cache 的显存 (留 10% overhead)
    available_memory = gpu.memory_bytes * 0.9 - model_size

    # 每个请求的 KV Cache (在该 GPU 上)
    per_token = calculate_kv_cache_per_token(model, config.dtype_bytes)
    per_request = per_token * config.seq_len / config.tp_size

    # 最大 batch size
    max_batch = int(available_memory / per_request) if per_request > 0 else 0

    # 最大吞吐估算 (假设 Decode 完全 bandwidth-bound)
    model_read_time = model_size / (gpu.bandwidth_tb_s * 1e12)  # seconds
    tokens_per_second_per_request = 1 / model_read_time
    max_throughput = tokens_per_second_per_request * min(max_batch, config.batch_size)

    return {
        "model_size_gb": model_size / (1024**3),
        "available_kv_cache_gb": available_memory / (1024**3),
        "per_request_kv_gb": per_request / (1024**3),
        "max_batch_size": max_batch,
        "estimated_throughput_tok_s": max_throughput,
        "kv_cache_utilization": (config.batch_size * per_request) / available_memory
            if available_memory > 0 else float('inf'),
    }


def print_analysis(model_name: str, gpu_name: str, config: InferenceConfig):
    """打印完整分析报告"""
    model = MODELS[model_name]
    gpu = GPUS[gpu_name]

    print("=" * 70)
    print(f"  KV Cache Analysis: {model.name} on {gpu.name}")
    print("=" * 70)

    # 模型信息
    print(f"\n{'Model Configuration':=^70}")
    print(f"  Model: {model.name} (~{model.num_params_billions:.1f}B params)")
    print(f"  Layers: {model.num_layers}")
    print(f"  Attention Heads: {model.num_attention_heads}")
    print(f"  KV Heads: {model.num_kv_heads} ({'GQA' if model.num_kv_heads < model.num_attention_heads else 'MHA'})")
    print(f"  Head Dim: {model.head_dim}")
    print(f"  Hidden Size: {model.hidden_size}")

    # 推理配置
    print(f"\n{'Inference Configuration':=^70}")
    print(f"  Sequence Length: {config.seq_len:,}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Tensor Parallelism: {config.tp_size}")
    print(f"  KV Cache Dtype: {config.kv_cache_dtype}")

    # KV Cache 计算
    kv = calculate_kv_cache_total(model, config)
    print(f"\n{'KV Cache Size':=^70}")
    print(f"  Per Token: {kv['per_token_bytes']:,.0f} bytes ({kv['per_token_kb']:.2f} KB)")
    print(f"  Per Request (seq={config.seq_len:,}): {kv['per_request_mb']:.2f} MB ({kv['per_request_gb']:.3f} GB)")
    print(f"  Total (batch={config.batch_size}): {kv['total_gb']:.2f} GB")

    # GPU 容量分析
    analysis = analyze_gpu_capacity(model, gpu, config)
    print(f"\n{'GPU Capacity Analysis ({gpu.name})':=^70}")
    print(f"  GPU Memory: {gpu.memory_gb} GB")
    print(f"  Model Weights (per GPU): {analysis['model_size_gb']:.2f} GB")
    print(f"  Available for KV Cache: {analysis['available_kv_cache_gb']:.2f} GB")
    print(f"  Max Batch Size: {analysis['max_batch_size']}")
    print(f"  Current KV Utilization: {analysis['kv_cache_utilization']*100:.1f}%")
    print(f"  Est. Max Throughput: {analysis['estimated_throughput_tok_s']:.0f} tok/s")

    if analysis['kv_cache_utilization'] > 1.0:
        print(f"\n  ⚠️  WARNING: Current batch size ({config.batch_size}) exceeds capacity!")
        print(f"     Max supported: {analysis['max_batch_size']} requests")

    # GQA 节省分析
    if model.num_kv_heads < model.num_attention_heads:
        ratio = model.num_attention_heads / model.num_kv_heads
        print(f"\n{'GQA Savings':=^70}")
        print(f"  GQA Ratio: {model.num_attention_heads} heads / {model.num_kv_heads} KV heads = {ratio:.0f}x")
        hypothetical_mha = kv['total_gb'] * ratio
        print(f"  If MHA (no GQA): {hypothetical_mha:.2f} GB KV Cache")
        print(f"  With GQA: {kv['total_gb']:.2f} GB KV Cache")
        print(f"  Savings: {hypothetical_mha - kv['total_gb']:.2f} GB ({(1-1/ratio)*100:.0f}%)")


def compare_sequence_lengths(model_name: str, gpu_name: str, tp_size: int = 8):
    """对比不同序列长度的影响"""
    model = MODELS[model_name]
    gpu = GPUS[gpu_name]

    print(f"\n{'='*70}")
    print(f"  Sequence Length Impact: {model.name} on {gpu.name} (TP={tp_size})")
    print(f"{'='*70}")
    print(f"\n  {'Seq Len':<10} {'Per Req':<12} {'Max Batch':<12} {'Max Throughput':<15}")
    print(f"  {'-'*49}")

    for seq_len in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        config = InferenceConfig(seq_len=seq_len, batch_size=1, tp_size=tp_size)
        kv = calculate_kv_cache_total(model, config)
        analysis = analyze_gpu_capacity(model, gpu, config)

        if analysis['max_batch_size'] > 0:
            config_max = InferenceConfig(seq_len=seq_len, batch_size=analysis['max_batch_size'], tp_size=tp_size)
            analysis_max = analyze_gpu_capacity(model, gpu, config_max)
            throughput = analysis_max['estimated_throughput_tok_s']
        else:
            throughput = 0

        print(f"  {seq_len:<10,} {kv['per_request_gb']:.3f} GB     {analysis['max_batch_size']:<12} {throughput:.0f} tok/s")


def compare_kv_dtypes(model_name: str, gpu_name: str, seq_len: int = 4096, tp_size: int = 8):
    """对比不同 KV Cache 数据类型的影响"""
    model = MODELS[model_name]
    gpu = GPUS[gpu_name]

    print(f"\n{'='*70}")
    print(f"  KV Cache Dtype Impact: {model.name} (seq={seq_len}, TP={tp_size})")
    print(f"{'='*70}")
    print(f"\n  {'Dtype':<8} {'Per Req':<12} {'Max Batch':<12} {'Batch Ratio':<12}")
    print(f"  {'-'*44}")

    base_batch = None
    for dtype in ["fp32", "fp16", "fp8", "int8", "int4"]:
        config = InferenceConfig(seq_len=seq_len, batch_size=1, tp_size=tp_size, kv_cache_dtype=dtype)
        kv = calculate_kv_cache_total(model, config)
        analysis = analyze_gpu_capacity(model, gpu, config)

        if base_batch is None:
            base_batch = analysis['max_batch_size']
        ratio = analysis['max_batch_size'] / base_batch if base_batch > 0 else 0

        print(f"  {dtype:<8} {kv['per_request_gb']:.3f} GB     {analysis['max_batch_size']:<12} {ratio:.2f}x")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("          KV Cache Memory Calculator for LLM Inference")
    print("=" * 70)

    # 场景 1: LLaMA-2-70B on 8×H20
    print_analysis("llama2-70b", "h20", InferenceConfig(
        seq_len=4096, batch_size=64, tp_size=8, kv_cache_dtype="fp16"
    ))

    # 场景 2: 长上下文影响
    compare_sequence_lengths("llama2-70b", "h20", tp_size=8)

    # 场景 3: KV Cache 量化影响
    compare_kv_dtypes("llama2-70b", "h20", seq_len=4096, tp_size=8)

    # 场景 4: 对比不同模型
    print(f"\n{'='*70}")
    print(f"  Model Comparison (seq=4096, batch=32, TP=8, FP16)")
    print(f"{'='*70}")
    print(f"\n  {'Model':<20} {'KV/Token':<12} {'KV/Req':<12} {'Total KV':<12} {'Max Batch':<10}")
    print(f"  {'-'*66}")

    for model_name in ["llama2-7b", "llama2-70b", "llama3-8b", "llama3-70b", "qwen2.5-72b"]:
        model = MODELS[model_name]
        config = InferenceConfig(seq_len=4096, batch_size=32, tp_size=8)
        kv = calculate_kv_cache_total(model, config)
        analysis = analyze_gpu_capacity(model, GPUS["h20"], config)

        print(f"  {model.name:<20} {kv['per_token_kb']:.1f} KB     "
              f"{kv['per_request_mb']:.1f} MB    "
              f"{kv['total_gb']:.2f} GB     "
              f"{analysis['max_batch_size']}")

    print("\n" + "=" * 70)
    print("  Key Takeaways:")
    print("  1. KV Cache 大小与 seq_len 线性增长 → 长上下文是大挑战")
    print("  2. GQA (8 KV heads vs 64) 节省 8x KV Cache → 关键优化")
    print("  3. FP8 KV Cache 让 max batch 翻倍 → 吞吐翻倍")
    print("  4. 128K 上下文单请求占 ~40GB → 几乎无法 batch")
    print("=" * 70)
