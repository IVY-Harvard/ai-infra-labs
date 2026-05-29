# LLaMA-Factory 安装与环境配置

## 安装步骤

### 1. 基础安装
```bash
# 克隆仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装（含评测依赖）
pip install -e ".[metrics]"

# 验证安装
llamafactory-cli version
```

### 2. 可选依赖
```bash
# Flash Attention 2（推荐）
pip install flash-attn --no-build-isolation

# DeepSpeed（多卡训练）
pip install deepspeed

# Unsloth 加速
pip install unsloth
```

## 数据集注册

### 自定义数据集格式

LLaMA-Factory 在 `data/dataset_info.json` 中管理数据集注册信息。

#### ShareGPT 格式（推荐）
```json
// data/my_dataset.json
[
    {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "你好！有什么可以帮助你的？"}
        ]
    }
]
```

注册到 `dataset_info.json`:
```json
{
    "my_dataset": {
        "file_name": "my_dataset.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        }
    }
}
```

#### Alpaca 格式
```json
[
    {
        "instruction": "翻译以下文本",
        "input": "Hello World",
        "output": "你好世界"
    }
]
```

注册:
```json
{
    "my_alpaca": {
        "file_name": "my_alpaca.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
}
```

## WebUI 使用指南

### 启动 WebUI
```bash
# 基本启动
llamafactory-cli webui

# 指定端口
GRADIO_SERVER_PORT=7861 llamafactory-cli webui

# 允许外部访问
GRADIO_SERVER_NAME=0.0.0.0 llamafactory-cli webui
```

### WebUI 功能说明

```
1. 模型选择
   - 选择基座模型（支持 100+ 模型）
   - 选择微调方法（Full/LoRA/QLoRA）
   - 选择精度（FP16/BF16/INT4/INT8）

2. 数据配置
   - 选择已注册的数据集
   - 设置最大序列长度
   - 预览数据样本

3. 训练参数
   - 学习率、Epoch、Batch Size
   - LoRA 相关参数（Rank, Alpha, Target）
   - 分布式训练设置

4. 开始训练
   - 实时查看训练 Loss 曲线
   - GPU 使用率监控
   - 训练完成后自动保存

5. 推理测试
   - 加载训练好的模型
   - 交互式对话测试
   - 参数调节（temperature, top_p 等）

6. 模型导出
   - 合并 LoRA 权重
   - 导出完整模型
   - 支持量化导出
```

## 常见问题

### Q: 如何使用多卡训练？
```bash
# 方法1: 使用 accelerate
accelerate launch --num_processes 8 \
    -m llamafactory.launcher train configs/my_config.yaml

# 方法2: 在配置中指定 DeepSpeed
# 添加 deepspeed: ds_z2_config.json
```

### Q: 如何断点续训？
```yaml
# 在配置中添加
resume_from_checkpoint: true
output_dir: ./output/my_training  # 指向之前的输出目录
```

### Q: 如何评测训练后的模型？
```bash
llamafactory-cli eval configs/eval_config.yaml
```
