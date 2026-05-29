# BlockManager 源码分析

## 源码位置

`vllm/core/block_manager.py` (或 `vllm/core/block/block_manager.py`)

## 核心接口

```python
class BlockSpaceManager:
    """管理 GPU 和 CPU 上的物理 Block"""
    
    def can_allocate(self, seq_group) -> AllocStatus:
        """检查能否为新序列分配初始 Block"""
    
    def allocate(self, seq_group):
        """分配初始 Block (Prefill 阶段)"""
    
    def can_append_slots(self, seq_group) -> bool:
        """检查能否为所有序列分配新 slot"""
    
    def append_slots(self, seq, num_lookahead_slots=0):
        """为新 token 分配 slot (Decode 阶段)"""
    
    def free(self, seq):
        """释放序列的所有 Block"""
    
    def can_swap_in(self, seq_group) -> bool:
        """检查能否 swap in"""
    
    def swap_in(self, seq_group) -> Dict[int, int]:
        """CPU → GPU"""
    
    def swap_out(self, seq_group) -> Dict[int, int]:
        """GPU → CPU"""
    
    def get_block_table(self, seq) -> List[int]:
        """获取序列的物理 Block ID 列表"""
```

## allocate() 分析

```python
def allocate(self, seq_group):
    """为新序列分配初始 Block"""
    seq = seq_group.get_seqs()[0]
    num_prompt_tokens = seq.get_len()
    
    # 计算需要多少 Block
    num_blocks = (num_prompt_tokens + self.block_size - 1) // self.block_size
    
    # 从 free pool 中分配
    block_table = []
    for _ in range(num_blocks):
        block = self.gpu_allocator.allocate()
        block_table.append(block)
    
    # 记录映射: seq_id → block_table
    self.block_tables[seq.seq_id] = block_table
```

关键点:
- 只分配 `ceil(prompt_len / block_size)` 个 Block
- 不是 `max_seq_len / block_size` 个！
- 这就是"按需分配"的体现

## append_slots() 分析

```python
def append_slots(self, seq):
    """为新 token 分配 slot"""
    block_table = self.block_tables[seq.seq_id]
    
    if len(block_table) == 0:
        # 新 Block
        block = self.gpu_allocator.allocate()
        block_table.append(block)
        return
    
    last_block = block_table[-1]
    
    # 检查最后一个 Block 是否已满
    if self._is_block_full(last_block, seq):
        # 需要新 Block
        block = self.gpu_allocator.allocate()
        block_table.append(block)
    
    # 如果 Block 被共享 (ref_count > 1), 需要 CoW
    elif last_block.ref_count > 1:
        new_block = self.gpu_allocator.allocate()
        self.gpu_allocator.copy(last_block, new_block)
        last_block.ref_count -= 1
        block_table[-1] = new_block
```

## Copy-on-Write 实现

```python
def fork(self, parent_seq, child_seq):
    """Fork: 共享 Block (用于 beam search)"""
    parent_table = self.block_tables[parent_seq.seq_id]
    child_table = []
    
    for block in parent_table:
        block.ref_count += 1  # 引用计数 +1
        child_table.append(block)  # 指向同一个物理 Block
    
    self.block_tables[child_seq.seq_id] = child_table

# 写入时:
# append_slots() 检查 ref_count > 1 → CoW
# 只在实际需要写入时才复制
# prompt 部分永远不需要复制 (只读)
```

## Prefix Caching 扩展

```python
class BlockSpaceManagerV2(BlockSpaceManager):
    """支持 Prefix Caching 的 BlockManager"""
    
    def __init__(self, ...):
        self.prefix_cache = PrefixCache()  # hash → block
    
    def allocate(self, seq_group):
        seq = seq_group.get_seqs()[0]
        tokens = seq.get_token_ids()
        
        block_table = []
        for i in range(0, len(tokens), self.block_size):
            chunk = tokens[i:i+self.block_size]
            block_hash = hash(tuple(chunk))
            
            # 检查是否有缓存
            cached_block = self.prefix_cache.get(block_hash)
            if cached_block is not None:
                # Cache hit! 复用已有 Block
                cached_block.ref_count += 1
                block_table.append(cached_block)
            else:
                # Cache miss, 分配新 Block
                block = self.gpu_allocator.allocate()
                self.prefix_cache.put(block_hash, block)
                block_table.append(block)
        
        self.block_tables[seq.seq_id] = block_table
```

## 阅读建议

1. 先理解 `PhysicalTokenBlock` 数据结构
2. 看 `allocate()` 理解初始分配
3. 看 `append_slots()` 理解增量分配
4. 看 `fork()` 理解 CoW
5. 看 `swap_in/out()` 理解抢占恢复
6. 最后看 Prefix Caching 的实现

## 实践练习

在你的 8×H20 环境上:

```python
# 启动 vLLM 并打印 BlockManager 统计
import vllm

llm = vllm.LLM("meta-llama/Llama-2-7b-hf")
# 查看内部状态:
engine = llm.llm_engine
scheduler = engine.scheduler[0]
block_mgr = scheduler.block_manager

print(f"Total GPU blocks: {block_mgr.gpu_allocator.get_num_total_blocks()}")
print(f"Free GPU blocks: {block_mgr.gpu_allocator.get_num_free_blocks()}")
```
