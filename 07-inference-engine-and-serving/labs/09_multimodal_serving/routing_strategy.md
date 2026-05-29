# 多模态路由策略

## 核心问题
不同模态的请求有不同的计算特征，需要智能路由。

## 路由策略

```
Request → Content-Type Detection
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
    Text Only  Image+Text  Audio+Text
        │       │           │
        ▼       ▼           ▼
    LLM Engine  VLM Engine  Audio LLM
    (vLLM)     (vLLM+VE)   (dedicated)
```

## 关键考虑

1. **Vision Encoder 的开销**: 图片编码 ~50-200ms，比纯文本 Prefill 慢
2. **资源隔离**: VLM 请求的 Prefill 更重，可能阻塞文本请求
3. **GPU 分配**: Vision Encoder 可以放在单独 GPU 上
4. **批处理**: 图片请求不太适合和纯文本混合 batch
