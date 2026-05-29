"""
Scheduler — 请求调度器

实现 Continuous Batching 的核心调度逻辑。
管理 waiting/running/swapped 三个队列。
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque, Tuple

from .sequence import (
    Sequence, SequenceGroup, SequenceStatus,
    SequenceGroupMetadata, SequenceData, SamplingParams,
)
from .block_manager import BlockManager


@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_num_seqs: int = 256          # 最大并发序列数
    max_num_batched_tokens: int = 2048  # 每步最大 token 数
    max_model_len: int = 4096        # 最大序列长度


@dataclass
class SchedulerOutput:
    """调度器输出"""
    scheduled_seq_groups: List[SequenceGroup]
    num_prefill_groups: int
    num_decode_groups: int
    blocks_to_swap_in: Dict[int, int]   # cpu_block → gpu_block
    blocks_to_swap_out: Dict[int, int]  # gpu_block → cpu_block
    blocks_to_copy: Dict[int, int]      # src_block → dst_block


class Scheduler:
    """
    请求调度器

    核心职责：
    1. 管理请求队列 (waiting/running/swapped)
    2. 每步决定哪些请求参与计算
    3. 与 BlockManager 协调显存分配
    4. 处理抢占逻辑
    """

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager

        # 三个队列
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: Deque[SequenceGroup] = deque()
        self.swapped: Deque[SequenceGroup] = deque()

        self._seq_id_counter = 0

    def add_seq_group(self, seq_group: SequenceGroup):
        """添加新的序列组到等待队列"""
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str):
        """取消请求"""
        for queue in [self.waiting, self.running, self.swapped]:
            for seq_group in list(queue):
                if seq_group.request_id == request_id:
                    for seq in seq_group.seqs:
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.block_manager.free(seq.seq_id)
                    queue.remove(seq_group)
                    return

    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def schedule(self) -> SchedulerOutput:
        """
        执行一步调度

        返回本步要执行的序列组和相关操作。
        """
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, int] = {}

        scheduled_groups: List[SequenceGroup] = []
        num_prefill = 0
        num_decode = 0

        # Phase 1: 处理 running 中的序列
        running_list = list(self.running)
        self.running.clear()

        for seq_group in running_list:
            # 检查完成
            if seq_group.is_finished:
                for seq in seq_group.seqs:
                    self.block_manager.free(seq.seq_id)
                continue

            # 检查能否继续 (分配新 slot)
            can_continue = True
            for seq in seq_group.get_unfinished_seqs():
                if not self.block_manager.can_append_slot():
                    can_continue = False
                    break

            if can_continue:
                # 分配新 slot
                for seq in seq_group.get_unfinished_seqs():
                    self.block_manager.append_slot(seq.seq_id, seq.total_len)
                self.running.append(seq_group)
                scheduled_groups.append(seq_group)
                num_decode += 1
            else:
                # 抢占: swap out 到 CPU
                self._preempt(seq_group, blocks_to_swap_out)

        # Phase 2: 尝试恢复 swapped 序列
        while self.swapped:
            seq_group = self.swapped[0]
            # 检查 GPU 是否有足够空间 swap in
            num_blocks_needed = sum(
                len(self.block_manager.block_tables.get(seq.seq_id, []))
                for seq in seq_group.get_unfinished_seqs()
            )
            if self.block_manager.num_free_gpu_blocks >= num_blocks_needed:
                self.swapped.popleft()
                for seq in seq_group.get_unfinished_seqs():
                    mapping = self.block_manager.swap_in(seq.seq_id)
                    if mapping:
                        blocks_to_swap_in.update(mapping)
                    seq.status = SequenceStatus.RUNNING
                self.running.append(seq_group)
                scheduled_groups.append(seq_group)
                num_decode += 1
            else:
                break

        # Phase 3: 调度新请求 (Prefill)
        num_batched_tokens = sum(
            seq.total_len for sg in scheduled_groups
            for seq in sg.get_unfinished_seqs()
        )

        while self.waiting:
            seq_group = self.waiting[0]

            # 检查约束
            if len(self.running) >= self.config.max_num_seqs:
                break

            # 检查 token 预算
            prompt_len = seq_group.seqs[0].prompt_len
            if num_batched_tokens + prompt_len > self.config.max_num_batched_tokens:
                break

            # 检查 Block
            if not self.block_manager.can_allocate(prompt_len):
                break

            # 调度!
            self.waiting.popleft()
            for seq in seq_group.seqs:
                self.block_manager.allocate(seq.seq_id, prompt_len)
                seq.status = SequenceStatus.RUNNING
            self.running.append(seq_group)
            scheduled_groups.append(seq_group)
            num_prefill += 1
            num_batched_tokens += prompt_len

        return SchedulerOutput(
            scheduled_seq_groups=scheduled_groups,
            num_prefill_groups=num_prefill,
            num_decode_groups=num_decode,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )

    def _preempt(self, seq_group: SequenceGroup, swap_mapping: Dict[int, int]):
        """抢占序列: swap out 到 CPU"""
        for seq in seq_group.get_unfinished_seqs():
            mapping = self.block_manager.swap_out(seq.seq_id)
            if mapping:
                swap_mapping.update(mapping)
            seq.status = SequenceStatus.SWAPPED
        self.swapped.appendleft(seq_group)

    def get_num_unfinished_seqs(self) -> int:
        return (len(self.waiting) + len(self.running) + len(self.swapped))
