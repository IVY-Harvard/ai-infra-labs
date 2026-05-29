# Lab 09: 多卡分布式微调

## 目标
- 掌握多 GPU LoRA 微调
- 理解 DeepSpeed ZeRO 各 stage 的区别
- 在 8 × H20 上微调大模型

## 硬件
- 4-8 × H20

## 运行方式
```bash
# 多卡 LoRA（使用 accelerate）
accelerate launch --num_processes 8 distributed_lora.py

# DeepSpeed ZeRO-2 微调
deepspeed --num_gpus 8 deepspeed_finetune.py --deepspeed ds_config.json
```
