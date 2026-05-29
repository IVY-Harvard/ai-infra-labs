# Lab 02: Paged Attention 实现

## 目标

- 实现 Block 分配器（PagedAttention 的核心）
- 理解 Block Table 映射机制
- 实现 Copy-on-Write
- 对比分页 vs 连续分配的显存效率

## 实验内容

1. **block_manager.py** — Block 分配器
   - allocate/free Block
   - Block Table 管理
   - Copy-on-Write 支持
   - 引用计数

2. **paged_kv_cache.py** — 分页 KV Cache 实现
   - 基于 Block 的 KV Cache 存储
   - 按需分配新 Block
   - 支持动态增长

3. **fragmentation_demo.py** — 碎片问题对比演示
   - 连续分配 vs 分页分配的碎片率对比
   - 可视化内存布局

## 运行方式

```bash
python block_manager.py       # Block 管理器测试
python paged_kv_cache.py      # 分页 KV Cache 测试
python fragmentation_demo.py  # 碎片对比演示
```

## 关键思考

- 为什么 Block 不需要物理连续？
- Copy-on-Write 在哪些场景下节省最大？
- Block Size 的大小如何影响碎片率？
