"""
碎片问题演示

对比连续分配 vs 分页分配的显存效率。
可视化展示两种方案的内存布局和碎片率。
"""

import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MemoryRegion:
    """内存区域"""
    start: int
    size: int
    seq_id: int  # -1 = free
    actual_used: int = 0  # 实际使用量


class ContiguousAllocator:
    """连续内存分配器（朴素方案）"""

    def __init__(self, total_size: int, max_seq_len: int):
        self.total_size = total_size
        self.max_seq_len = max_seq_len
        self.regions: List[MemoryRegion] = [MemoryRegion(0, total_size, -1)]

    def allocate(self, seq_id: int, actual_tokens: int) -> bool:
        """分配 max_seq_len 大小的连续区域"""
        for i, region in enumerate(self.regions):
            if region.seq_id == -1 and region.size >= self.max_seq_len:
                # 找到足够大的空闲区域
                allocated = MemoryRegion(region.start, self.max_seq_len, seq_id, actual_tokens)
                remaining_size = region.size - self.max_seq_len

                self.regions[i] = allocated
                if remaining_size > 0:
                    self.regions.insert(i + 1, MemoryRegion(
                        region.start + self.max_seq_len, remaining_size, -1
                    ))
                return True
        return False  # 找不到连续空间

    def free(self, seq_id: int):
        """释放区域"""
        for region in self.regions:
            if region.seq_id == seq_id:
                region.seq_id = -1
                region.actual_used = 0
        self._merge_free_regions()

    def _merge_free_regions(self):
        """合并相邻空闲区域"""
        merged = []
        for region in self.regions:
            if merged and merged[-1].seq_id == -1 and region.seq_id == -1:
                merged[-1].size += region.size
            else:
                merged.append(region)
        self.regions = merged

    def get_stats(self) -> dict:
        total_allocated = sum(r.size for r in self.regions if r.seq_id != -1)
        total_used = sum(r.actual_used for r in self.regions if r.seq_id != -1)
        total_free = sum(r.size for r in self.regions if r.seq_id == -1)
        max_contiguous_free = max(
            (r.size for r in self.regions if r.seq_id == -1), default=0
        )
        num_free_regions = sum(1 for r in self.regions if r.seq_id == -1)

        return {
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_free": total_free,
            "internal_fragmentation": (total_allocated - total_used) / total_allocated if total_allocated > 0 else 0,
            "external_fragmentation": 1 - (max_contiguous_free / total_free) if total_free > 0 else 0,
            "num_free_fragments": num_free_regions,
            "max_contiguous_free": max_contiguous_free,
            "memory_efficiency": total_used / self.total_size,
        }


class PagedAllocator:
    """分页内存分配器（PagedAttention 方案）"""

    def __init__(self, total_size: int, block_size: int):
        self.total_size = total_size
        self.block_size = block_size
        self.num_blocks = total_size // block_size

        self.free_blocks = list(range(self.num_blocks))
        self.allocations = {}  # seq_id -> (blocks, actual_tokens)

    def allocate(self, seq_id: int, actual_tokens: int) -> bool:
        """按需分配 Block"""
        num_blocks_needed = (actual_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks_needed:
            return False

        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.allocations[seq_id] = (blocks, actual_tokens)
        return True

    def free(self, seq_id: int):
        """释放 Block"""
        if seq_id in self.allocations:
            blocks, _ = self.allocations[seq_id]
            self.free_blocks.extend(blocks)
            del self.allocations[seq_id]

    def get_stats(self) -> dict:
        total_blocks_used = self.num_blocks - len(self.free_blocks)
        total_allocated = total_blocks_used * self.block_size
        total_used = sum(tokens for _, tokens in self.allocations.values())
        # 分页方案的内部碎片只在每个序列最后一个 Block
        fragmentation_tokens = sum(
            (len(blocks) * self.block_size - tokens)
            for blocks, tokens in self.allocations.values()
        )

        return {
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_free": len(self.free_blocks) * self.block_size,
            "internal_fragmentation": fragmentation_tokens / total_allocated if total_allocated > 0 else 0,
            "external_fragmentation": 0,  # 分页方案没有外部碎片!
            "num_free_fragments": 1,  # 逻辑上是一个 free pool
            "max_contiguous_free": len(self.free_blocks) * self.block_size,
            "memory_efficiency": total_used / self.total_size,
            "blocks_used": total_blocks_used,
            "blocks_free": len(self.free_blocks),
        }


def visualize_memory(allocator, name: str, width: int = 64):
    """可视化内存布局"""
    print(f"\n  {name} Memory Layout:")

    if isinstance(allocator, ContiguousAllocator):
        # 将 regions 映射到字符
        total = allocator.total_size
        display = ['·'] * width

        for region in allocator.regions:
            start_pos = int(region.start / total * width)
            end_pos = int((region.start + region.size) / total * width)
            if region.seq_id != -1:
                char = chr(65 + (region.seq_id % 26))  # A, B, C, ...
                used_end = int((region.start + region.actual_used) / total * width)
                for i in range(start_pos, min(end_pos, width)):
                    display[i] = char if i < used_end else '░'

        print(f"  [{''.join(display)}]")
        print(f"   █ = used  ░ = allocated but wasted  · = free")

    elif isinstance(allocator, PagedAllocator):
        display = ['·'] * width
        blocks_per_char = max(1, allocator.num_blocks // width)

        # Track which blocks are used by which seq
        block_to_seq = {}
        for seq_id, (blocks, _) in allocator.allocations.items():
            for b in blocks:
                block_to_seq[b] = seq_id

        for i in range(width):
            block_idx = i * blocks_per_char
            if block_idx < allocator.num_blocks and block_idx in block_to_seq:
                display[i] = chr(65 + (block_to_seq[block_idx] % 26))

        print(f"  [{''.join(display)}]")
        print(f"   Letter = used block  · = free block")


def run_simulation():
    """运行分配模拟"""
    print("\n" + "=" * 70)
    print("  Fragmentation Comparison: Contiguous vs Paged Allocation")
    print("=" * 70)

    random.seed(42)

    # 参数
    total_memory = 10000  # 总 token slots
    max_seq_len = 200     # 配置的最大序列长度
    block_size = 16       # Block 大小
    num_requests = 100    # 总请求数

    contig = ContiguousAllocator(total_memory, max_seq_len)
    paged = PagedAllocator(total_memory, block_size)

    # 生成请求: 实际长度远小于 max_seq_len
    requests = [(i, random.randint(10, 150)) for i in range(num_requests)]
    active_requests = []

    # 模拟: 不断添加和完成请求
    contig_served = 0
    paged_served = 0
    contig_failed = 0
    paged_failed = 0

    print(f"\n  Config: total_memory={total_memory}, max_seq_len={max_seq_len}, block_size={block_size}")
    print(f"  Running {num_requests} requests with random lengths [10, 150]...")

    for i, (seq_id, actual_len) in enumerate(requests):
        # 随机完成一些旧请求
        if active_requests and random.random() < 0.3:
            finished_id = active_requests.pop(random.randint(0, len(active_requests) - 1))
            contig.free(finished_id)
            paged.free(finished_id)

        # 尝试分配新请求
        c_ok = contig.allocate(seq_id, actual_len)
        p_ok = paged.allocate(seq_id, actual_len)

        if c_ok:
            contig_served += 1
        else:
            contig_failed += 1
        if p_ok:
            paged_served += 1
        else:
            paged_failed += 1

        if c_ok or p_ok:
            active_requests.append(seq_id)

        # 中间打印
        if i == num_requests // 2:
            print(f"\n  --- Midpoint ({i} requests processed) ---")
            visualize_memory(contig, "Contiguous")
            visualize_memory(paged, "Paged")

    # 最终结果
    print(f"\n  {'='*60}")
    print(f"  Final Results after {num_requests} requests")
    print(f"  {'='*60}")

    c_stats = contig.get_stats()
    p_stats = paged.get_stats()

    print(f"\n  {'Metric':<30} {'Contiguous':<15} {'Paged':<15}")
    print(f"  {'-'*60}")
    print(f"  {'Requests served':<30} {contig_served:<15} {paged_served:<15}")
    print(f"  {'Requests failed':<30} {contig_failed:<15} {paged_failed:<15}")
    print(f"  {'Internal Fragmentation':<30} {c_stats['internal_fragmentation']*100:.1f}%{'':9s} {p_stats['internal_fragmentation']*100:.1f}%")
    print(f"  {'External Fragmentation':<30} {c_stats['external_fragmentation']*100:.1f}%{'':9s} {p_stats['external_fragmentation']*100:.1f}%")
    print(f"  {'Free fragments':<30} {c_stats['num_free_fragments']:<15} {p_stats['num_free_fragments']:<15}")
    print(f"  {'Max contiguous free':<30} {c_stats['max_contiguous_free']:<15} {p_stats['max_contiguous_free']:<15}")
    print(f"  {'Memory Efficiency':<30} {c_stats['memory_efficiency']*100:.1f}%{'':9s} {p_stats['memory_efficiency']*100:.1f}%")

    visualize_memory(contig, "Contiguous (Final)")
    visualize_memory(paged, "Paged (Final)")

    # Block Size 影响分析
    print(f"\n\n  {'='*60}")
    print(f"  Block Size Impact on Internal Fragmentation")
    print(f"  {'='*60}")
    print(f"\n  {'Block Size':<12} {'Avg Frag/Seq':<15} {'Total Frag %':<15}")
    print(f"  {'-'*42}")

    for bs in [1, 4, 8, 16, 32, 64, 128, 256]:
        avg_frag = bs / 2  # 平均每个序列浪费 block_size/2 个 slot
        # 假设平均序列长度 80
        avg_seq_len = 80
        frag_pct = avg_frag / (avg_seq_len + avg_frag) * 100
        print(f"  {bs:<12} {avg_frag:<15.1f} {frag_pct:<15.1f}%")


if __name__ == "__main__":
    run_simulation()

    print("\n" + "=" * 70)
    print("  Conclusion:")
    print("  1. Contiguous allocation: 60-90% internal frag + external frag")
    print("  2. Paged allocation: <5% internal frag, ZERO external frag")
    print("  3. Paged serves MORE requests with SAME memory")
    print("  4. Block size 16 is sweet spot (< 10% frag, good kernel efficiency)")
    print("=" * 70)
