# 模块 06：大模型微调与对齐

## 模块概述

本模块系统讲解大语言模型（LLM）微调训练的完整知识体系，从理论基础到工程落地。面向已有模型部署经验、但尚未进行过微调训练的工程师，提供从零到一的微调实战能力。

**目标硬件环境：** 8 × NVIDIA H20 GPU（96GB HBM3 × 8 = 768GB 总显存）

## 前置要求

- 完成前序模块（Transformer 架构、推理部署）
- 熟悉 PyTorch 基本操作
- 有模型推理部署经验（vLLM/TGI 等）
- 了解 Linux 多卡环境配置

## 学习路线

```
Week 1: 理论基础 + 基础实验
├── 微调全景概览（theory/01）
├── LoRA 深度解析（theory/02）
├── 全量微调 baseline（labs/01）
├── LoRA/QLoRA 实战（labs/02）
└── Unsloth 加速（labs/03）

Week 2: 数据工程 + 工具链
├── 微调数据工程（theory/03）
├── 数据准备实战（labs/05）
├── LLaMA-Factory 使用（labs/04）
└── 多卡分布式微调（labs/09）

Week 3: 对齐训练
├── RLHF 全流程（theory/04）
├── DPO 与替代方案（theory/05）
├── RLHF 实战（labs/06）
└── DPO 训练（labs/07）

Week 4: 评估 + 生产化
├── 评估体系（theory/06）
├── 生产化流水线（theory/07）
├── 评估框架实战（labs/08）
├── 生产流水线（labs/10）
└── 企业项目：微调平台（project/）
```

## 目录结构

```
06-finetuning-and-alignment/
├── README.md                          # 本文件
├── theory/                            # 理论知识
│   ├── 01_finetuning_landscape.md     # 微调全景
│   ├── 02_lora_deep_dive.md           # LoRA 深度解析
│   ├── 03_data_engineering.md         # 数据工程
│   ├── 04_rlhf_pipeline.md            # RLHF 全流程
│   ├── 05_dpo_and_alternatives.md     # DPO 及替代方案
│   ├── 06_evaluation_framework.md     # 评估体系
│   └── 07_production_pipeline.md      # 生产化流水线
├── labs/                              # 实验代码
│   ├── 01_full_finetuning/            # 全量微调
│   ├── 02_lora_qlora/                 # LoRA/QLoRA
│   ├── 03_unsloth_acceleration/       # Unsloth 加速
│   ├── 04_llama_factory/              # LLaMA-Factory
│   ├── 05_data_preparation/           # 数据准备
│   ├── 06_rlhf_pipeline/             # RLHF 实战
│   ├── 07_dpo_training/              # DPO 训练
│   ├── 08_evaluation_framework/       # 评估框架
│   ├── 09_multi_gpu_finetune/         # 多卡微调
│   └── 10_production_pipeline/        # 生产流水线
└── project/
    └── finetuning-platform/           # 企业级微调平台
```

## 硬件环境说明

### H20 GPU 特点

| 特性 | H20 规格 | 对微调的意义 |
|------|---------|-------------|
| 显存 | 96GB HBM3 | 可训练 70B 模型（QLoRA）|
| 带宽 | 4TB/s HBM | 大 batch 训练效率高 |
| NVLink | 900 GB/s | 多卡通信快，适合 FSDP |
| FP16 | ~150 TFLOPS | 训练吞吐量大 |
| 总显存 | 768GB (8卡) | 全参微调 30B+ 可行 |

### 各规模模型微调显存估算

| 模型规模 | 全量微调 | LoRA (r=64) | QLoRA (4bit+LoRA) |
|---------|---------|------------|-------------------|
| 7B     | ~120GB  | ~32GB      | ~12GB            |
| 13B    | ~220GB  | ~56GB      | ~22GB            |
| 30B    | ~520GB  | ~130GB     | ~48GB            |
| 70B    | ~1.2TB  | ~280GB     | ~96GB            |

> 8 × H20 (768GB) 可以：
> - 全量微调 30B 以下模型
> - LoRA 微调 70B 模型（需 DeepSpeed ZeRO-3）
> - QLoRA 单卡微调 70B 模型

## 环境安装

```bash
# 创建 conda 环境
conda create -n finetune python=3.10 -y
conda activate finetune

# PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 核心微调库
pip install transformers==4.44.0 datasets==2.20.0 accelerate==0.33.0
pip install peft==0.12.0 trl==0.9.6 bitsandbytes==0.43.0

# 评估相关
pip install lm-eval==0.4.3 rouge-score nltk jieba

# 工具链
pip install deepspeed==0.14.4 flash-attn==2.6.3 wandb

# Unsloth（可选，需额外安装）
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# LLaMA-Factory（可选）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[metrics]"
```

## 核心概念速查

| 概念 | 说明 |
|------|------|
| SFT (Supervised Fine-Tuning) | 有监督微调，用指令-回答对训练模型 |
| LoRA | 低秩适配，冻结原模型只训练少量参数 |
| QLoRA | 4-bit 量化 + LoRA，极低显存微调 |
| RLHF | 基于人类反馈的强化学习对齐 |
| DPO | 直接偏好优化，无需训练 Reward Model |
| PEFT | 参数高效微调的统称 |
| DeepSpeed ZeRO | 分布式训练优化，按 stage 切分状态 |
| FSDP | PyTorch 原生的全切片数据并行 |
