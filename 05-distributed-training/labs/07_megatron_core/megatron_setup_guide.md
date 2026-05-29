# Megatron-Core 安装和配置指南

## 环境准备

### 安装依赖

```bash
# 基础环境
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation

# Megatron-Core (pip)
pip install megatron-core

# 或从源码安装（推荐，获取最新特性）
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

### 验证安装

```python
import megatron.core as mcore
print(mcore.__version__)

from megatron.core.transformer import TransformerConfig
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    sequence_parallel=True,
)
print(config)
```

## Megatron-Core 核心概念

### 1. 并行组初始化

```python
from megatron.core import parallel_state

# 初始化所有并行组
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    # 自动计算 data_parallel_size = world_size / (tp * pp)
)

# 获取当前 rank 的并行信息
tp_rank = parallel_state.get_tensor_model_parallel_rank()
pp_rank = parallel_state.get_pipeline_model_parallel_rank()
dp_rank = parallel_state.get_data_parallel_rank()
tp_group = parallel_state.get_tensor_model_parallel_group()
```

### 2. TransformerConfig

```python
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,        # GQA
    ffn_hidden_size=14336,        # SwiGLU
    hidden_dropout=0.0,
    attention_dropout=0.0,
    
    # 并行配置
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    sequence_parallel=True,        # 开启 SP
    
    # 精度
    bf16=True,
    params_dtype=torch.bfloat16,
    
    # 优化
    recompute_granularity='selective',  # Activation Checkpointing
    use_flash_attention=True,
)
```

### 3. Sequence Parallelism 工作原理

```
                    TP Region                        Non-TP Region
                    (GEMM 部分)                      (LayerNorm, Dropout)
                    
Without SP:    X ∈ [B, S, H]                    X ∈ [B, S, H] (完整)
               切分: W → W/tp_size              不切分
               
With SP:       X ∈ [B, S, H]                    X ∈ [B, S/tp_size, H] (按S切分)
               切分: W → W/tp_size              切分 sequence 维度
               
通信:
  Without SP: AllReduce (after Row Parallel)
  With SP:    ReduceScatter → [B, S/tp, H]  →  LayerNorm  → AllGather → [B, S, H]
              
  总通信量相同（AllReduce = ReduceScatter + AllGather），但激活值显存减少！
```

### 4. 典型 8×H20 配置

```bash
# 7B 模型, TP=4, PP=1, DP=2
DISTRIBUTED_ARGS="
    --nproc_per_node 8
    --nnodes 1
"

MEGATRON_ARGS="
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-flash-attn
    --bf16
    --micro-batch-size 2
    --global-batch-size 32
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length 2048
    --max-position-embeddings 2048
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py $MEGATRON_ARGS
```

## 常见问题

### Q: SP 开启后 LayerNorm 怎么处理？
A: LayerNorm 是在 hidden 维度上做归一化，不依赖 sequence 维度，
   所以 [B, S/tp, H] 上可以正常计算。

### Q: Dropout 在 SP 下需要注意什么？
A: Dropout 在 non-TP region（sequence 已切分）本地做，
   在 TP region（sequence 完整）需要同一 TP group 使用相同 seed。

### Q: SP 的显存节省有多大？
A: LayerNorm + Dropout 的激活值从 [B, S, H] 变为 [B, S/tp, H]，
   节省 (tp_size - 1) / tp_size ≈ 75% (TP=4) 的这部分激活值。
