# Lab 07: torch.compile 实战

## 实验目的

理解 PyTorch 2.0 的编译流程：
1. `torch.compile` 的基本用法和不同模式
2. TorchDynamo 如何追踪计算图
3. Inductor 如何生成 Triton kernel
4. Graph Break 是什么、为什么发生、如何避免

## 前置要求

- PyTorch 2.0+
- 已读 theory/05（torch.compile 部分）

## 运行

```bash
python compile_demo.py
python custom_backend.py
```

## 关键思考

1. torch.compile 在什么场景下加速最明显？
2. 为什么有些代码会导致 graph break？
3. Inductor 生成的 Triton 代码和手写的有什么区别？
