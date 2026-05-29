"""BlockManager 测试"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.block_manager import BlockManager, PhysicalBlock


class TestBlockManager:
    """BlockManager 单元测试"""

    def setup_method(self):
        self.bm = BlockManager(block_size=16, num_gpu_blocks=64, num_cpu_blocks=32)

    def test_allocate_basic(self):
        """测试基本分配"""
        assert self.bm.can_allocate(32)
        assert self.bm.allocate(seq_id=0, num_tokens=32)

        # 32 tokens / 16 block_size = 2 blocks
        block_table = self.bm.get_block_table(0)
        assert block_table is not None
        assert len(block_table) == 2
        assert self.bm.num_free_gpu_blocks == 62

    def test_allocate_insufficient(self):
        """测试显存不足"""
        # 分配所有 Block
        for i in range(32):  # 64 blocks / 2 per seq = 32 seqs
            assert self.bm.allocate(seq_id=i, num_tokens=32)

        # 现在没有空闲 Block
        assert self.bm.num_free_gpu_blocks == 0
        assert not self.bm.can_allocate(16)
        assert not self.bm.allocate(seq_id=100, num_tokens=16)

    def test_free(self):
        """测试释放"""
        self.bm.allocate(seq_id=0, num_tokens=32)
        assert self.bm.num_free_gpu_blocks == 62

        self.bm.free(seq_id=0)
        assert self.bm.num_free_gpu_blocks == 64
        assert self.bm.get_block_table(0) is None

    def test_append_slot(self):
        """测试追加 slot"""
        self.bm.allocate(seq_id=0, num_tokens=16)  # 1 block, 正好满

        # 追加 token 17 → 需要新 Block
        assert self.bm.append_slot(seq_id=0, token_pos=16)
        assert len(self.bm.get_block_table(0)) == 2

    def test_fork_cow(self):
        """测试 Fork (Copy-on-Write)"""
        self.bm.allocate(seq_id=0, num_tokens=32)
        initial_free = self.bm.num_free_gpu_blocks

        # Fork 不应该分配新 Block (共享!)
        assert self.bm.fork(parent_seq_id=0, child_seq_id=1)
        assert self.bm.num_free_gpu_blocks == initial_free  # 没有新分配!

        # 检查引用计数
        parent_blocks = self.bm.block_tables[0]
        for block in parent_blocks:
            assert block.ref_count == 2  # parent + child

    def test_swap_out_in(self):
        """测试 Swap"""
        self.bm.allocate(seq_id=0, num_tokens=32)
        gpu_free_before = self.bm.num_free_gpu_blocks

        # Swap out
        mapping = self.bm.swap_out(seq_id=0)
        assert mapping is not None
        assert self.bm.num_free_gpu_blocks > gpu_free_before

        # Swap in
        mapping_in = self.bm.swap_in(seq_id=0)
        assert mapping_in is not None
        assert self.bm.num_free_gpu_blocks == gpu_free_before

    def test_block_table_correctness(self):
        """测试 Block Table 正确性"""
        self.bm.allocate(seq_id=0, num_tokens=48)  # 3 blocks
        table = self.bm.get_block_table(0)

        assert len(table) == 3
        # Block IDs 应该唯一
        assert len(set(table)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
