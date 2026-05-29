# Lab 05: TensorRT-LLM 实践

## 目标

- 构建 TensorRT-LLM engine
- 与 vLLM 进行性能对比
- 理解编译时优化的效果

## 前置要求

```bash
pip install tensorrt-llm -U --extra-index-url https://pypi.nvidia.com
```

## 实验内容

1. **build_engine.py** — 构建 TRT-LLM engine
2. **benchmark_vs_vllm.py** — 与 vLLM 性能对比
