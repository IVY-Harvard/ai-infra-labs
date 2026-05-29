# Lab 07：Tokenization 流水线

## 实验目标

构建高效的 Tokenization 流水线，包括单线程基准、多进程并行和批量处理优化。

## 实验内容

### 实验 1：基础 Tokenization 流水线
实现完整的 文本清洗 → Tokenization → 打包 流水线。

### 实验 2：多进程并行 Tokenization
利用全部 CPU 核心并行处理多个文件。

### 实验 3：Token 统计分析
统计 token 分布、序列长度、词汇覆盖等信息。

## 运行方式

```bash
pip install transformers tokenizers datasets numpy

# 基础流水线
python tokenizer_pipeline.py --input-dir /path/to/texts --output-dir /tmp/tokenized

# 并行 Tokenization
python parallel_tokenization.py --input-dir /path/to/texts --num-workers 16

# Token 统计
python token_statistics.py --tokenized-dir /tmp/tokenized
```

## 文件列表

- `tokenizer_pipeline.py` — 基础 Tokenization 流水线
- `parallel_tokenization.py` — 多进程并行处理
- `token_statistics.py` — Token 分布统计工具
