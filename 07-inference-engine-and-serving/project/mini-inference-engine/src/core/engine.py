"""
LLMEngine — 推理引擎主入口

协调 Scheduler、Worker、ModelRunner 完成推理。
对应 vLLM 的 LLMEngine。
"""

import time
import uuid
from typing import List, Optional, Dict, Iterator
from dataclasses import dataclass

from .sequence import (
    Sequence, SequenceGroup, SequenceStatus,
    SamplingParams, SamplerOutput,
)
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from .block_manager import BlockManager


@dataclass
class EngineConfig:
    """引擎配置"""
    model_name: str = "gpt2"
    block_size: int = 16
    num_gpu_blocks: int = 256
    num_cpu_blocks: int = 64
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.9


class LLMEngine:
    """
    LLM 推理引擎

    主循环：
    1. 接收请求 (add_request)
    2. 调度 (scheduler.schedule)
    3. 执行 (worker.execute_model)
    4. 处理输出 (process_outputs)
    5. 返回结果
    """

    def __init__(self, config: EngineConfig):
        self.config = config

        # 初始化 Block Manager
        self.block_manager = BlockManager(
            block_size=config.block_size,
            num_gpu_blocks=config.num_gpu_blocks,
            num_cpu_blocks=config.num_cpu_blocks,
        )

        # 初始化 Scheduler
        scheduler_config = SchedulerConfig(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=config.max_model_len,
        )
        self.scheduler = Scheduler(scheduler_config, self.block_manager)

        # 初始化 Model (延迟加载)
        self.model_runner = None
        self.tokenizer = None

        # 请求跟踪
        self._request_counter = 0
        self._seq_counter = 0

        # 指标
        self.num_completed_requests = 0
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0

    def load_model(self):
        """加载模型和 tokenizer"""
        from ..model.model_runner import ModelRunner

        self.model_runner = ModelRunner(
            model_name=self.config.model_name,
            block_size=self.config.block_size,
            num_gpu_blocks=self.config.num_gpu_blocks,
        )
        self.model_runner.load_model()
        self.tokenizer = self.model_runner.tokenizer

    def add_request(
        self,
        request_id: Optional[str],
        prompt: str,
        sampling_params: SamplingParams,
    ) -> str:
        """添加新请求"""
        if request_id is None:
            request_id = f"req-{uuid.uuid4().hex[:8]}"

        # Tokenize
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt_token_ids = self.tokenizer.encode(prompt)

        # 截断
        if len(prompt_token_ids) > self.config.max_model_len:
            prompt_token_ids = prompt_token_ids[:self.config.max_model_len]

        # 创建 Sequence
        self._seq_counter += 1
        seq = Sequence(
            seq_id=self._seq_counter,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        # 创建 SequenceGroup
        seq_group = SequenceGroup(
            request_id=request_id,
            seqs=[seq],
            sampling_params=sampling_params,
            arrival_time=time.time(),
        )

        # 加入调度器
        self.scheduler.add_seq_group(seq_group)
        self.total_prompt_tokens += len(prompt_token_ids)

        return request_id

    def step(self) -> List[Dict]:
        """
        执行一步推理

        Returns:
            完成的请求结果列表
        """
        # 1. 调度
        scheduler_output = self.scheduler.schedule()

        if not scheduler_output.scheduled_seq_groups:
            return []

        # 2. 执行模型
        if self.model_runner is not None:
            sampler_output = self.model_runner.execute_model(
                scheduler_output,
                self.block_manager,
            )
        else:
            # 没有模型时，模拟输出
            sampler_output = self._mock_execute(scheduler_output)

        # 3. 处理输出
        results = self._process_outputs(scheduler_output, sampler_output)

        return results

    def _mock_execute(self, scheduler_output: SchedulerOutput) -> SamplerOutput:
        """模拟模型输出 (用于测试)"""
        import random
        token_ids = []
        for seq_group in scheduler_output.scheduled_seq_groups:
            for seq in seq_group.get_unfinished_seqs():
                # 随机生成一个 token
                token_ids.append(random.randint(0, 50256))
        return SamplerOutput(token_ids=token_ids)

    def _process_outputs(
        self,
        scheduler_output: SchedulerOutput,
        sampler_output: SamplerOutput,
    ) -> List[Dict]:
        """处理模型输出，更新序列状态"""
        results = []
        token_idx = 0

        for seq_group in scheduler_output.scheduled_seq_groups:
            for seq in seq_group.get_unfinished_seqs():
                if token_idx >= len(sampler_output.token_ids):
                    break

                new_token = sampler_output.token_ids[token_idx]
                token_idx += 1

                # 追加 token
                seq.append_token(new_token)
                self.total_generated_tokens += 1

                # 检查停止条件
                if self._check_stop(seq):
                    continue

            # 收集已完成的请求结果
            if seq_group.is_finished:
                result = self._make_result(seq_group)
                results.append(result)
                self.num_completed_requests += 1

        return results

    def _check_stop(self, seq: Sequence) -> bool:
        """检查是否应该停止生成"""
        # 达到最大长度
        if seq.output_len >= seq.sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH
            return True

        # EOS token
        if self.tokenizer and seq.get_last_token_id() == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return True

        # 超过模型最大长度
        if seq.total_len >= self.config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH
            return True

        return False

    def _make_result(self, seq_group: SequenceGroup) -> Dict:
        """生成请求结果"""
        seq = seq_group.seqs[0]
        output_text = ""
        if self.tokenizer:
            output_text = self.tokenizer.decode(
                seq.output_token_ids, skip_special_tokens=True
            )

        return {
            "request_id": seq_group.request_id,
            "output_text": output_text,
            "output_token_ids": seq.output_token_ids,
            "prompt_len": seq.prompt_len,
            "output_len": seq.output_len,
            "finish_reason": "length" if seq.status == SequenceStatus.FINISHED_LENGTH else "stop",
        }

    def abort_request(self, request_id: str):
        """取消请求"""
        self.scheduler.abort_seq_group(request_id)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_seqs()

    def get_stats(self) -> Dict:
        """获取引擎统计"""
        return {
            "num_waiting": len(self.scheduler.waiting),
            "num_running": len(self.scheduler.running),
            "num_swapped": len(self.scheduler.swapped),
            "num_completed": self.num_completed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "gpu_blocks_used": self.block_manager.num_total_gpu_blocks - self.block_manager.num_free_gpu_blocks,
            "gpu_blocks_free": self.block_manager.num_free_gpu_blocks,
            "kv_cache_utilization": 1 - (self.block_manager.num_free_gpu_blocks / self.block_manager.num_total_gpu_blocks),
        }
