"""
Sequence / SequenceGroup 数据结构

定义推理引擎中请求和序列的核心数据结构。
对应 vLLM 的 vllm/sequence.py
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict


class SequenceStatus(Enum):
    """序列状态"""
    WAITING = "waiting"       # 在等待队列中
    RUNNING = "running"       # 正在 GPU 上执行
    SWAPPED = "swapped"       # KV Cache 被 swap 到 CPU
    FINISHED_STOPPED = "finished_stopped"    # 正常完成 (EOS/stop)
    FINISHED_LENGTH = "finished_length"      # 达到最大长度
    FINISHED_ABORTED = "finished_aborted"    # 被取消

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    """采样参数"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    stop: Optional[List[str]] = None
    stream: bool = False
    n: int = 1  # 生成数量


@dataclass
class Sequence:
    """
    单个序列

    代表一个 token 序列，包含 prompt + generated tokens。
    每个序列有独立的 KV Cache Block 映射。
    """
    seq_id: int
    prompt_token_ids: List[int]
    sampling_params: SamplingParams

    # 生成的 token IDs
    output_token_ids: List[int] = field(default_factory=list)

    # 状态
    status: SequenceStatus = SequenceStatus.WAITING

    # Block table: 逻辑 block index → 物理 block id
    # 由 BlockManager 管理
    logical_token_blocks: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished

    def append_token(self, token_id: int):
        """追加一个生成的 token"""
        self.output_token_ids.append(token_id)

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]


@dataclass
class SequenceGroup:
    """
    序列组

    一个用户请求可能包含多个序列 (n > 1 或 beam search)。
    所有序列共享同一个 prompt。
    """
    request_id: str
    seqs: List[Sequence]
    sampling_params: SamplingParams
    arrival_time: float

    @property
    def is_finished(self) -> bool:
        return all(seq.is_finished for seq in self.seqs)

    @property
    def num_seqs(self) -> int:
        return len(self.seqs)

    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        if status is None:
            return self.seqs
        return [seq for seq in self.seqs if seq.status == status]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs if not seq.is_finished]


@dataclass
class SequenceGroupMetadata:
    """
    传递给 ModelRunner 的序列组元数据

    包含执行模型推理所需的所有信息。
    """
    request_id: str
    is_prompt: bool  # True = Prefill, False = Decode
    seq_data: Dict[int, "SequenceData"]  # seq_id → data
    sampling_params: SamplingParams
    block_tables: Dict[int, List[int]]  # seq_id → [physical_block_ids]


@dataclass
class SequenceData:
    """序列数据 (传给 ModelRunner)"""
    token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.token_ids) - len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return len(self.token_ids)


@dataclass
class SamplerOutput:
    """采样输出"""
    token_ids: List[int]  # 每个序列的输出 token
    probs: Optional[List[float]] = None
