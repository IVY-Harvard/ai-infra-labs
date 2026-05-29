"""Scheduler 测试"""
import pytest
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.scheduler import Scheduler, SchedulerConfig
from src.core.block_manager import BlockManager
from src.core.sequence import (
    Sequence, SequenceGroup, SequenceStatus, SamplingParams
)


class TestScheduler:
    """Scheduler 单元测试"""

    def setup_method(self):
        self.block_manager = BlockManager(block_size=16, num_gpu_blocks=64)
        self.config = SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=512)
        self.scheduler = Scheduler(self.config, self.block_manager)
        self._seq_counter = 0

    def _create_seq_group(self, prompt_len: int = 32, max_tokens: int = 64):
        self._seq_counter += 1
        seq = Sequence(
            seq_id=self._seq_counter,
            prompt_token_ids=list(range(prompt_len)),
            sampling_params=SamplingParams(max_tokens=max_tokens),
        )
        return SequenceGroup(
            request_id=f"req-{self._seq_counter}",
            seqs=[seq],
            sampling_params=seq.sampling_params,
            arrival_time=time.time(),
        )

    def test_add_and_schedule(self):
        """测试添加请求和调度"""
        sg = self._create_seq_group(prompt_len=32)
        self.scheduler.add_seq_group(sg)

        assert len(self.scheduler.waiting) == 1
        assert self.scheduler.has_unfinished_seqs()

        # 调度
        output = self.scheduler.schedule()
        assert output.num_prefill_groups == 1
        assert len(self.scheduler.running) == 1
        assert len(self.scheduler.waiting) == 0

    def test_continuous_batching(self):
        """测试连续批处理"""
        # 添加多个请求
        for _ in range(5):
            self.scheduler.add_seq_group(self._create_seq_group(prompt_len=16))

        # 第一步: 应该调度所有请求 Prefill
        output = self.scheduler.schedule()
        assert output.num_prefill_groups == 5
        assert len(self.scheduler.running) == 5

        # 模拟完成一个请求
        first_seq = self.scheduler.running[0].seqs[0]
        first_seq.status = SequenceStatus.FINISHED_STOPPED

        # 添加新请求
        self.scheduler.add_seq_group(self._create_seq_group(prompt_len=16))

        # 第二步: 应该处理完成的 + 新请求
        output = self.scheduler.schedule()
        # 1 个完成释放, 4 个继续 decode, 1 个新 prefill
        assert output.num_decode_groups == 4
        assert output.num_prefill_groups == 1

    def test_max_seqs_limit(self):
        """测试最大序列数限制"""
        # 配置只允许 4 个
        self.config.max_num_seqs = 4

        for _ in range(10):
            self.scheduler.add_seq_group(self._create_seq_group(prompt_len=16))

        output = self.scheduler.schedule()
        # 应该只调度 4 个
        assert output.num_prefill_groups == 4
        assert len(self.scheduler.waiting) == 6

    def test_abort(self):
        """测试取消请求"""
        sg = self._create_seq_group()
        self.scheduler.add_seq_group(sg)
        self.scheduler.schedule()

        self.scheduler.abort_seq_group(sg.request_id)
        assert not self.scheduler.has_unfinished_seqs()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
