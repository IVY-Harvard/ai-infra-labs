"""
Block Manager — 显存块管理器

管理 GPU 上 KV Cache 的物理 Block 分配。
实现 PagedAttention 的核心：按需分配、CoW、Swap。
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class PhysicalBlock:
    """物理 Block"""
    block_id: int
    block_size: int
    ref_count: int = 0

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """Block 分配器（管理一种设备上的 Block 池）"""

    def __init__(self, num_blocks: int, block_size: int, device: str = "gpu"):
        self.device = device
        self.block_size = block_size
        self.all_blocks = [
            PhysicalBlock(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        ]
        self.free_blocks: List[PhysicalBlock] = list(self.all_blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_total_blocks(self) -> int:
        return len(self.all_blocks)

    def allocate(self) -> Optional[PhysicalBlock]:
        if not self.free_blocks:
            return None
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock):
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            self.free_blocks.append(block)


class BlockManager:
    """
    Block 管理器

    管理 GPU 和 CPU Block 池。
    为序列分配/释放 Block，支持 CoW 和 Swap。
    """

    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 256,
        num_cpu_blocks: int = 64,
    ):
        self.block_size = block_size
        self.gpu_allocator = BlockAllocator(num_gpu_blocks, block_size, "gpu")
        self.cpu_allocator = BlockAllocator(num_cpu_blocks, block_size, "cpu")

        # seq_id → List[PhysicalBlock]
        self.block_tables: Dict[int, List[PhysicalBlock]] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        """检查是否有足够 Block"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return self.gpu_allocator.num_free_blocks >= num_blocks

    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """为序列分配初始 Block (Prefill)"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size

        if self.gpu_allocator.num_free_blocks < num_blocks:
            return False

        blocks = []
        for _ in range(num_blocks):
            block = self.gpu_allocator.allocate()
            if block is None:
                # 回滚
                for b in blocks:
                    self.gpu_allocator.free(b)
                return False
            blocks.append(block)

        self.block_tables[seq_id] = blocks
        return True

    def can_append_slot(self) -> bool:
        """检查是否能追加一个 slot"""
        # 最坏情况: 需要一个新 Block
        return self.gpu_allocator.num_free_blocks >= 1

    def append_slot(self, seq_id: int, token_pos: int) -> bool:
        """为新 token 分配 slot (Decode)"""
        if seq_id not in self.block_tables:
            return False

        blocks = self.block_tables[seq_id]

        # 检查当前最后一个 Block 是否已满
        current_block_idx = token_pos // self.block_size
        pos_in_block = token_pos % self.block_size

        if current_block_idx >= len(blocks):
            # 需要新 Block
            block = self.gpu_allocator.allocate()
            if block is None:
                return False
            blocks.append(block)

        # 处理 CoW
        block = blocks[current_block_idx]
        if block.ref_count > 1:
            # Copy-on-Write
            new_block = self.gpu_allocator.allocate()
            if new_block is None:
                return False
            block.ref_count -= 1
            if block.ref_count <= 0:
                self.gpu_allocator.free(block)
            blocks[current_block_idx] = new_block

        return True

    def free(self, seq_id: int):
        """释放序列的所有 Block"""
        if seq_id not in self.block_tables:
            return

        for block in self.block_tables[seq_id]:
            self.gpu_allocator.free(block)

        del self.block_tables[seq_id]

    def fork(self, parent_seq_id: int, child_seq_id: int) -> bool:
        """Fork (CoW)"""
        if parent_seq_id not in self.block_tables:
            return False

        parent_blocks = self.block_tables[parent_seq_id]
        child_blocks = []

        for block in parent_blocks:
            block.ref_count += 1
            child_blocks.append(block)

        self.block_tables[child_seq_id] = child_blocks
        return True

    def swap_out(self, seq_id: int) -> Optional[Dict[int, int]]:
        """GPU → CPU"""
        if seq_id not in self.block_tables:
            return None

        mapping = {}
        new_blocks = []

        for gpu_block in self.block_tables[seq_id]:
            cpu_block = self.cpu_allocator.allocate()
            if cpu_block is None:
                # 回滚
                for b in new_blocks:
                    self.cpu_allocator.free(b)
                return None

            mapping[gpu_block.block_id] = cpu_block.block_id
            self.gpu_allocator.free(gpu_block)
            new_blocks.append(cpu_block)

        self.block_tables[seq_id] = new_blocks
        return mapping

    def swap_in(self, seq_id: int) -> Optional[Dict[int, int]]:
        """CPU → GPU"""
        if seq_id not in self.block_tables:
            return None

        mapping = {}
        new_blocks = []

        for cpu_block in self.block_tables[seq_id]:
            gpu_block = self.gpu_allocator.allocate()
            if gpu_block is None:
                for b in new_blocks:
                    self.gpu_allocator.free(b)
                return None

            mapping[cpu_block.block_id] = gpu_block.block_id
            self.cpu_allocator.free(cpu_block)
            new_blocks.append(gpu_block)

        self.block_tables[seq_id] = new_blocks
        return mapping

    def get_block_table(self, seq_id: int) -> Optional[List[int]]:
        """获取物理 Block ID 列表"""
        if seq_id not in self.block_tables:
            return None
        return [b.block_id for b in self.block_tables[seq_id]]

    @property
    def num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.num_free_blocks

    @property
    def num_total_gpu_blocks(self) -> int:
        return self.gpu_allocator.num_total_blocks
