# LLaMA-Factory WebUI 使用指南

## 启动 WebUI

```bash
# 进入 LLaMA-Factory 目录
cd LLaMA-Factory

# 启动（默认端口 7860）
llamafactory-cli webui

# 自定义配置启动
CUDA_VISIBLE_DEVICES=0,1 GRADIO_SERVER_PORT=7861 llamafactory-cli webui
```

## 界面说明

### Tab 1: Train（训练）

#### 模型配置区
- **Model Name**: 选择模型，如 `Qwen/Qwen2-7B`
- **Finetuning Method**: `lora` / `full` / `freeze`
- **Quantization**: `none` / `4bit` / `8bit`
- **Template**: 选择对话模板（qwen / llama3 / chatglm 等）

#### LoRA 配置区
- **LoRA Rank**: 默认 8，推荐 64
- **LoRA Alpha**: 默认 16，推荐 2*rank
- **LoRA Target**: 选择要训练的模块
- **LoRA Dropout**: 默认 0

#### 训练配置区
- **Dataset**: 选择注册过的数据集
- **Max Length**: 最大序列长度（2048）
- **Epochs**: 训练轮数（3）
- **Batch Size**: 每卡 batch（4）
- **Gradient Accumulation**: 梯度累积步数（4）
- **Learning Rate**: 学习率（2e-4）
- **LR Scheduler**: cosine / linear

#### 高级选项
- **Gradient Checkpointing**: 推荐开启
- **Flash Attention**: 推荐开启
- **DeepSpeed Stage**: 0/2/3
- **Mixed Precision**: bf16 / fp16

### Tab 2: Evaluate（评估）
- 选择训练好的 checkpoint
- 选择评测数据集
- 查看评测结果

### Tab 3: Chat（推理测试）
- 加载模型 + LoRA
- 交互式对话
- 调节生成参数

### Tab 4: Export（导出）
- 选择基座 + LoRA
- 合并并导出
- 支持 safetensors 格式

## 最佳实践

### 推荐的工作流程
```
1. WebUI 快速实验
   - 小数据集 + 少 epoch
   - 快速验证数据格式和模型选择

2. 确定方案后转为配置文件
   - 从 WebUI 导出配置
   - 添加更多自定义参数

3. 命令行正式训练
   - 大数据集 + 完整训练
   - 支持断点续训
   - 方便脚本化和 CI/CD
```

### 性能调优技巧
```
1. 开启 Flash Attention 2
   - 速度提升 20-50%
   - 显存减少 20%

2. 使用 Unsloth 加速
   - 在 Advanced 选项中选择
   - 额外 2-4x 加速

3. 多卡训练
   - 设置 DeepSpeed Stage 2（LoRA）
   - 设置 DeepSpeed Stage 3（全量微调）

4. 数据加载优化
   - preprocessing_num_workers: 16
   - dataloader_num_workers: 8
```
