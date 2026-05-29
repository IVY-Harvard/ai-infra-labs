"""
Block Manager — PagedAttention 的核心组件

实现物理 Block 的分配、释放、Copy-on-Write。
这是 vLLM BlockManager 的简化版本，展示核心原理。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class BlockStatus(Enum):
    FREE = "free"
    ALLOCATED = "allocated"


@dataclass
class PhysicalBlock:
    """物理 Block — 对应 GPU 显存中的一块固定大小空间"""
    block_id: int
    block_size: int  # 每个 block 能存多少 token
    ref_count: int = 0  # 引用计数 (用于 Copy-on-Write)

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0

    def __repr__(self):
        return f"PhysBlock(id={self.block_id}, refs={self.ref_count})"


@dataclass
class LogicalBlock:
    """逻辑 Block — 序列视角的 Block"""
    block_number: int  # 逻辑编号 (0, 1, 2, ...)
    physical_block: Optional[PhysicalBlock] = None
    num_tokens: int = 0  # 当前已填充的 token 数

    @property
    def is_full(self) -> bool:
        if self.physical_block is None:
            return False
        return self.num_tokens >= self.physical_block.block_size

    @property
    def remaining_slots(self) -> int:
        if self.physical_block is None:
            return 0
        return self.physical_block.block_size - self.num_tokens


class BlockManager:
    """
    Block 管理器

    管理物理 Block 池的分配和释放。
    类比操作系统的物理页框管理器。
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_gpu_blocks: Optional[int] = None,
        num_cpu_blocks: int = 0,
    ):
        """
        Args:
            num_blocks: GPU 上的物理 Block 总数
            block_size: 每个 Block 的 token 容量
            num_cpu_blocks: CPU 上的 Block 数 (用于 swap)
        """
        self.block_size = block_size
        self.num_gpu_blocks = num_blocks
        self.num_cpu_blocks = num_cpu_blocks

        # 初始化 GPU Block 池
        self.gpu_blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        ]
        self.gpu_free_blocks: List[PhysicalBlock] = list(self.gpu_blocks)

        # 初始化 CPU Block 池 (用于 swap)
        self.cpu_blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i + num_blocks, block_size=block_size)
            for i in range(num_cpu_blocks)
        ]
        self.cpu_free_blocks: List[PhysicalBlock] = list(self.cpu_blocks)

        # Block Table: seq_id -> List[LogicalBlock]
        self.block_tables: Dict[int, List[LogicalBlock]] = {}

        print(f"[BlockManager] Initialized:")
        print(f"  GPU Blocks: {num_blocks} × {block_size} tokens = {num_blocks * block_size} total tokens")
        print(f"  CPU Blocks: {num_cpu_blocks} × {block_size} tokens")

    @property
    def num_free_gpu_blocks(self) -> int:
        return len(self.gpu_free_blocks)

    @property
    def num_free_cpu_blocks(self) -> int:
        return len(self.cpu_free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """检查是否有足够的空闲 Block 来分配指定数量的 token"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.num_free_gpu_blocks >= num_blocks_needed

    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """
        为序列分配初始 Block (用于 Prefill 阶段)

        只分配 ceil(num_tokens / block_size) 个 Block，
        不是预分配 max_seq_len！这就是按需分配的核心。
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if self.num_free_gpu_blocks < num_blocks_needed:
            return False

        logical_blocks = []
        remaining_tokens = num_tokens

        for i in range(num_blocks_needed):
            # 从空闲池中取一个物理 Block
            physical_block = self.gpu_free_blocks.pop()
            physical_block.ref_count = 1

            # 创建逻辑 Block
            tokens_in_block = min(remaining_tokens, self.block_size)
            logical_block = LogicalBlock(
                block_number=i,
                physical_block=physical_block,
                num_tokens=tokens_in_block,
            )
            logical_blocks.append(logical_block)
            remaining_tokens -= tokens_in_block

        self.block_tables[seq_id] = logical_blocks
        return True

    def append_slot(self, seq_id: int) -> bool:
        """
        为序列分配一个新的 token slot (用于 Decode 阶段)

        如果当前最后一个 Block 还有空间 → 直接用
        如果已满 → 分配新 Block
        """
        if seq_id not in self.block_tables:
            return False

        blocks = self.block_tables[seq_id]

        if not blocks:
            # 没有 Block，分配第一个
            return self._allocate_new_block(seq_id)

        last_block = blocks[-1]

        if last_block.is_full:
            # 最后一个 Block 已满，需要新 Block
            return self._allocate_new_block(seq_id)
        else:
            # 还有空间，检查是否需要 Copy-on-Write
            if last_block.physical_block.ref_count > 1:
                # 共享的 Block，需要 CoW
                return self._copy_on_write(seq_id, len(blocks) - 1)
            else:
                # 独占的 Block，直接写入
                last_block.num_tokens += 1
                return True

    def _allocate_new_block(self, seq_id: int) -> bool:
        """分配一个新的物理 Block"""
        if self.num_free_gpu_blocks == 0:
            return False

        physical_block = self.gpu_free_blocks.pop()
        physical_block.ref_count = 1

        new_logical = LogicalBlock(
            block_number=len(self.block_tables[seq_id]),
            physical_block=physical_block,
            num_tokens=1,
        )
        self.block_tables[seq_id].append(new_logical)
        return True

    def _copy_on_write(self, seq_id: int, block_idx: int) -> bool:
        """
        Copy-on-Write: 复制共享 Block

        当多个序列共享同一个物理 Block 时，
        写入前需要先复制一份独立的副本。
        """
        if self.num_free_gpu_blocks == 0:
            return False

        blocks = self.block_tables[seq_id]
        old_logical = blocks[block_idx]
        old_physical = old_logical.physical_block

        # 分配新的物理 Block
        new_physical = self.gpu_free_blocks.pop()
        new_physical.ref_count = 1

        # 减少旧 Block 的引用计数
        old_physical.ref_count -= 1
        if old_physical.ref_count == 0:
            self.gpu_free_blocks.append(old_physical)

        # 更新逻辑 Block 指向新物理 Block
        # (实际实现中这里需要复制 KV 数据)
        new_logical = LogicalBlock(
            block_number=block_idx,
            physical_block=new_physical,
            num_tokens=old_logical.num_tokens + 1,
        )
        blocks[block_idx] = new_logical
        return True

    def fork(self, parent_seq_id: int, child_seq_id: int) -> bool:
        """
        Fork 序列 (用于 Beam Search / Parallel Sampling)

        子序列共享父序列的所有 Block (引用计数 +1)
        写入时才复制 (Copy-on-Write)
        """
        if parent_seq_id not in self.block_tables:
            return False

        parent_blocks = self.block_tables[parent_seq_id]
        child_blocks = []

        for logical_block in parent_blocks:
            # 增加物理 Block 的引用计数 (共享!)
            logical_block.physical_block.ref_count += 1

            # 子序列的逻辑 Block 指向同一个物理 Block
            child_logical = LogicalBlock(
                block_number=logical_block.block_number,
                physical_block=logical_block.physical_block,
                num_tokens=logical_block.num_tokens,
            )
            child_blocks.append(child_logical)

        self.block_tables[child_seq_id] = child_blocks
        return True

    def free(self, seq_id: int):
        """释放序列的所有 Block"""
        if seq_id not in self.block_tables:
            return

        for logical_block in self.block_tables[seq_id]:
            physical_block = logical_block.physical_block
            physical_block.ref_count -= 1
            if physical_block.ref_count == 0:
                self.gpu_free_blocks.append(physical_block)

        del self.block_tables[seq_id]

    def swap_out(self, seq_id: int) -> Optional[Dict[int, int]]:
        """
        Swap Out: GPU → CPU

        将序列的 KV Cache Block 从 GPU 交换到 CPU。
        返回: {gpu_block_id: cpu_block_id} 映射
        """
        if seq_id not in self.block_tables:
            return None

        blocks = self.block_tables[seq_id]
        mapping = {}

        for logical_block in blocks:
            gpu_block = logical_block.physical_block
            if self.num_free_cpu_blocks == 0:
                # CPU 空间也不够，回滚
                # (简化: 这里不处理回滚)
                return None

            cpu_block = self.cpu_free_blocks.pop()
            cpu_block.ref_count = 1
            mapping[gpu_block.block_id] = cpu_block.block_id

            # 释放 GPU Block
            gpu_block.ref_count -= 1
            if gpu_block.ref_count == 0:
                self.gpu_free_blocks.append(gpu_block)

            # 更新逻辑 Block 指向 CPU Block
            logical_block.physical_block = cpu_block

        return mapping

    def swap_in(self, seq_id: int) -> Optional[Dict[int, int]]:
        """
        Swap In: CPU → GPU

        将序列的 KV Cache Block 从 CPU 交换回 GPU。
        """
        if seq_id not in self.block_tables:
            return None

        blocks = self.block_tables[seq_id]
        mapping = {}

        for logical_block in blocks:
            cpu_block = logical_block.physical_block
            if self.num_free_gpu_blocks == 0:
                return None

            gpu_block = self.gpu_free_blocks.pop()
            gpu_block.ref_count = 1
            mapping[cpu_block.block_id] = gpu_block.block_id

            # 释放 CPU Block
            cpu_block.ref_count -= 1
            if cpu_block.ref_count == 0:
                self.cpu_free_blocks.append(cpu_block)

            logical_block.physical_block = gpu_block

        return mapping

    def get_block_table(self, seq_id: int) -> Optional[List[int]]:
        """获取序列的 Block Table (物理 Block ID 列表)"""
        if seq_id not in self.block_tables:
            return None
        return [lb.physical_block.block_id for lb in self.block_tables[seq_id]]

    def get_stats(self) -> dict:
        """获取统计信息"""
        total_tokens_allocated = sum(
            sum(lb.num_tokens for lb in blocks)
            for blocks in self.block_tables.values()
        )
        total_blocks_allocated = self.num_gpu_blocks - self.num_free_gpu_blocks

        return {
            "total_gpu_blocks": self.num_gpu_blocks,
            "free_gpu_blocks": self.num_free_gpu_blocks,
            "used_gpu_blocks": total_blocks_allocated,
            "gpu_utilization": total_blocks_allocated / self.num_gpu_blocks,
            "total_sequences": len(self.block_tables),
            "total_tokens": total_tokens_allocated,
            "avg_blocks_per_seq": total_blocks_allocated / max(len(self.block_tables), 1),
            "memory_efficiency": total_tokens_allocated / (total_blocks_allocated * self.block_size)
                if total_blocks_allocated > 0 else 0,
        }

    def print_memory_map(self):
        """可视化 Block 分配情况"""
        print(f"\n  GPU Block Map ({self.num_gpu_blocks} blocks, block_size={self.block_size}):")
        print(f"  {'─'*60}")

        # 创建 block_id → seq_id 映射
        block_to_seq = {}
        for seq_id, blocks in self.block_tables.items():
            for lb in blocks:
                block_to_seq[lb.physical_block.block_id] = (seq_id, lb.num_tokens)

        # 打印每行 16 个 Block
        blocks_per_row = 16
        for row_start in range(0, self.num_gpu_blocks, blocks_per_row):
            row_end = min(row_start + blocks_per_row, self.num_gpu_blocks)
            line = f"  [{row_start:3d}] "
            for i in range(row_start, row_end):
                if i in block_to_seq:
                    seq_id, _ = block_to_seq[i]
                    line += f"S{seq_id}"
                else:
                    line += "··"
                line += " "
            print(line)


def test_basic_operations():
    """测试基本操作"""
    print("\n" + "=" * 70)
    print("  Test: Basic Block Manager Operations")
    print("=" * 70)

    bm = BlockManager(num_blocks=64, block_size=16, num_cpu_blocks=32)

    # 分配序列
    print("\n  [1] Allocating sequences...")
    assert bm.allocate(seq_id=0, num_tokens=50)  # 需要 4 个 Block
    assert bm.allocate(seq_id=1, num_tokens=30)  # 需要 2 个 Block
    assert bm.allocate(seq_id=2, num_tokens=100) # 需要 7 个 Block

    print(f"  Seq 0 block table: {bm.get_block_table(0)}")
    print(f"  Seq 1 block table: {bm.get_block_table(1)}")
    print(f"  Seq 2 block table: {bm.get_block_table(2)}")

    stats = bm.get_stats()
    print(f"\n  Stats: {stats['used_gpu_blocks']} blocks used, "
          f"{stats['free_gpu_blocks']} free, "
          f"efficiency={stats['memory_efficiency']:.2%}")

    # Append (Decode)
    print("\n  [2] Appending tokens (Decode)...")
    for _ in range(20):
        bm.append_slot(seq_id=0)

    print(f"  Seq 0 after 20 more tokens: {bm.get_block_table(0)}")
    print(f"  Seq 0 total tokens: {sum(lb.num_tokens for lb in bm.block_tables[0])}")

    # Free
    print("\n  [3] Freeing sequence 1...")
    bm.free(seq_id=1)
    stats = bm.get_stats()
    print(f"  After free: {stats['free_gpu_blocks']} blocks free")

    bm.print_memory_map()


def test_copy_on_write():
    """测试 Copy-on-Write"""
    print("\n" + "=" * 70)
    print("  Test: Copy-on-Write (Parallel Sampling)")
    print("=" * 70)

    bm = BlockManager(num_blocks=64, block_size=16)

    # 分配父序列 (prompt)
    bm.allocate(seq_id=0, num_tokens=48)  # 3 个 Block
    print(f"\n  Parent (seq 0) blocks: {bm.get_block_table(0)}")
    print(f"  Free blocks before fork: {bm.num_free_gpu_blocks}")

    # Fork 出 3 个子序列 (parallel sampling, n=4)
    bm.fork(parent_seq_id=0, child_seq_id=1)
    bm.fork(parent_seq_id=0, child_seq_id=2)
    bm.fork(parent_seq_id=0, child_seq_id=3)

    print(f"\n  After forking 3 children:")
    print(f"  Seq 0 blocks: {bm.get_block_table(0)}")
    print(f"  Seq 1 blocks: {bm.get_block_table(1)}")
    print(f"  Seq 2 blocks: {bm.get_block_table(2)}")
    print(f"  Seq 3 blocks: {bm.get_block_table(3)}")
    print(f"  Free blocks after fork: {bm.num_free_gpu_blocks}")
    print(f"  (Fork is FREE! Just increment ref_count)")

    # 检查引用计数
    parent_blocks = bm.block_tables[0]
    print(f"\n  Ref counts of shared blocks:")
    for lb in parent_blocks:
        print(f"    Block {lb.physical_block.block_id}: ref_count = {lb.physical_block.ref_count}")

    # 子序列开始独立生成 → 触发 CoW
    print(f"\n  Seq 1 starts generating (triggers CoW on last block)...")
    bm.append_slot(seq_id=1)
    print(f"  Seq 1 blocks after append: {bm.get_block_table(1)}")
    print(f"  Free blocks after CoW: {bm.num_free_gpu_blocks}")

    stats = bm.get_stats()
    print(f"\n  Memory efficiency: {stats['memory_efficiency']:.2%}")
    print(f"  Without CoW, would need: {3 * 3} blocks (3 copies × 3 blocks)")
    print(f"  With CoW, using: {stats['used_gpu_blocks']} blocks")
    print(f"  Savings: {9 - stats['used_gpu_blocks']} blocks!")


def test_swap():
    """测试 Swap (GPU ↔ CPU)"""
    print("\n" + "=" * 70)
    print("  Test: Swap Out/In (Preemption)")
    print("=" * 70)

    bm = BlockManager(num_blocks=32, block_size=16, num_cpu_blocks=32)

    # 分配多个序列，填满 GPU
    for i in range(10):
        bm.allocate(seq_id=i, num_tokens=48)  # 每个 3 blocks, 共 30/32

    stats = bm.get_stats()
    print(f"\n  After filling GPU: {stats['used_gpu_blocks']}/{bm.num_gpu_blocks} blocks used")
    print(f"  Free GPU blocks: {stats['free_gpu_blocks']}")

    # 新请求来了但 GPU 不够 → 需要抢占
    print(f"\n  New request arrives, needs 5 blocks, only {stats['free_gpu_blocks']} free!")
    print(f"  → Preempt seq 0 (swap out to CPU)")

    mapping = bm.swap_out(seq_id=0)
    print(f"  Swap out mapping (GPU→CPU): {mapping}")
    print(f"  Free GPU blocks after swap: {bm.num_free_gpu_blocks}")

    # 现在可以分配新请求
    bm.allocate(seq_id=100, num_tokens=70)
    print(f"  New request (seq 100) allocated: {bm.get_block_table(100)}")

    # 稍后恢复被抢占的请求
    print(f"\n  Later: swap in seq 0 back to GPU...")
    bm.free(seq_id=100)  # 先释放一些空间
    mapping_in = bm.swap_in(seq_id=0)
    print(f"  Swap in mapping (CPU→GPU): {mapping_in}")
    print(f"  Seq 0 restored: {bm.get_block_table(0)}")


if __name__ == "__main__":
    test_basic_operations()
    test_copy_on_write()
    test_swap()

    print("\n" + "=" * 70)
    print("  Key Takeaways:")
    print("  1. Blocks are fixed-size → no external fragmentation")
    print("  2. On-demand allocation → minimal internal fragmentation")
    print("  3. Copy-on-Write → shared prefixes are free")
    print("  4. Swap → graceful handling of memory pressure")
    print("=" * 70)
