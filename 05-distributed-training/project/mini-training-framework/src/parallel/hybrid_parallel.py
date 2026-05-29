"""
混合并行编排器 — TP + PP + DP 统一管理
=========================================
核心类 HybridParallelEngine:
  1. 根据配置创建 TP/PP/DP 通信组
  2. 将模型层分配到各 PP stage
  3. 对每个 stage 内部使用 TP
  4. stage 间使用 PP 调度
  5. 可选的 DP 梯度同步
"""

import os
import argparse
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .tensor_parallel import TPTransformerLayer
from .pipeline_parallel import PipelineStage, PipelineSchedule1F1B


@dataclass
class ParallelConfig:
    """并行配置"""
    tp_size: int = 4
    pp_size: int = 2
    dp_size: int = 1
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    ffn_size: int = 0  # 默认 4 * hidden_size
    vocab_size: int = 32000
    max_seq_len: int = 2048
    micro_batch_size: int = 2
    num_micro_batches: int = 8
    learning_rate: float = 3e-4
    precision: str = "bf16"
    max_steps: int = 1000

    def __post_init__(self):
        if self.ffn_size == 0:
            self.ffn_size = self.hidden_size * 4

    @classmethod
    def from_yaml(cls, path: str) -> "ParallelConfig":
        if not HAS_YAML:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        flat = {}
        for section in cfg.values():
            if isinstance(section, dict):
                flat.update(section)
        # 映射 YAML 字段名
        field_map = {
            "tensor_parallel_size": "tp_size",
            "pipeline_parallel_size": "pp_size",
            "data_parallel_size": "dp_size",
        }
        mapped = {}
        for k, v in flat.items():
            mapped[field_map.get(k, k)] = v
        return cls(**{k: v for k, v in mapped.items() if hasattr(cls, k)})


class ParallelGroupManager:
    """
    管理 3D 并行组的创建和查询。

    GPU 编号排列: rank = dp * (tp * pp) + pp * tp + tp_idx
    """

    def __init__(self, tp_size: int, pp_size: int, dp_size: int):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        assert tp_size * pp_size * dp_size == self.world_size, \
            f"TP({tp_size})×PP({pp_size})×DP({dp_size})={tp_size*pp_size*dp_size} != world_size({self.world_size})"

        self.tp_rank = self.rank % tp_size
        self.pp_rank = (self.rank // tp_size) % pp_size
        self.dp_rank = self.rank // (tp_size * pp_size)

        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
        self.tp_ranks = []
        self.pp_ranks = []
        self.dp_ranks = []

        self._create_groups()

    def _create_groups(self):
        tp, pp, dp = self.tp_size, self.pp_size, self.dp_size

        for d in range(dp):
            for p in range(pp):
                ranks = [d * tp * pp + p * tp + t for t in range(tp)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.tp_group = group
                    self.tp_ranks = ranks

        for d in range(dp):
            for t in range(tp):
                ranks = [d * tp * pp + p * tp + t for p in range(pp)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.pp_group = group
                    self.pp_ranks = ranks

        for p in range(pp):
            for t in range(tp):
                ranks = [d * tp * pp + p * tp + t for d in range(dp)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.dp_group = group
                    self.dp_ranks = ranks


class HybridParallelEngine:
    """
    混合并行训练引擎。
    编排 TP + PP + DP 的完整训练循环。
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.groups = ParallelGroupManager(config.tp_size, config.pp_size, config.dp_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        # 创建本 stage 的模型
        self.stage = self._build_stage()
        self.optimizer = torch.optim.AdamW(
            self.stage.parameters(), lr=config.learning_rate, weight_decay=0.01
        )

        # PP 调度器
        self.scheduler = PipelineSchedule1F1B(
            stage=self.stage,
            pp_rank=self.groups.pp_rank,
            pp_size=config.pp_size,
            pp_group=self.groups.pp_group,
            pp_ranks=self.groups.pp_ranks,
            num_micro_batches=config.num_micro_batches,
        )

    def _build_stage(self) -> PipelineStage:
        """构建当前 PP stage 的模型"""
        cfg = self.config
        layers_per_stage = cfg.num_layers // cfg.pp_size
        start_layer = self.groups.pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage

        layers = nn.ModuleList()
        for _ in range(layers_per_stage):
            layer = TPTransformerLayer(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                ffn_size=cfg.ffn_size,
                tp_size=cfg.tp_size,
                tp_rank=self.groups.tp_rank,
                tp_group=self.groups.tp_group,
            )
            layers.append(layer)

        stage = PipelineStage(layers).to(self.device)

        if self.groups.rank == 0:
            params = sum(p.numel() for p in stage.parameters())
            total_params = params * cfg.tp_size * cfg.pp_size
            print(f"  Stage {self.groups.pp_rank}: {params/1e6:.1f}M params/GPU, "
                  f"~{total_params/1e6:.0f}M total")

        return stage

    def train_step(self, data_iter) -> float:
        """执行一步训练（包含 PP 调度）"""
        self.optimizer.zero_grad()

        activation_shape = (
            self.config.micro_batch_size,
            self.config.max_seq_len,
            self.config.hidden_size,
        )

        # 定义 loss 函数
        def loss_fn(output):
            return output.mean()  # 简化: 实际应用中替换为真实 loss

        # PP 1F1B 调度
        avg_loss = self.scheduler.run(
            data_iter=data_iter,
            loss_fn=loss_fn,
            device=self.device,
            activation_shape=activation_shape,
            dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float32,
        )

        # DP 梯度同步
        if self.config.dp_size > 1:
            for param in self.stage.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=self.groups.dp_group)
                    param.grad /= self.config.dp_size

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(self.stage.parameters(), 1.0)
        self.optimizer.step()

        return avg_loss

    def summary(self):
        """打印配置总结"""
        cfg = self.config
        g = self.groups
        if g.rank == 0:
            print(f"\n{'='*60}")
            print(f"Hybrid Parallel Engine")
            print(f"{'='*60}")
            print(f"  TP={cfg.tp_size}, PP={cfg.pp_size}, DP={cfg.dp_size}")
            print(f"  Model: H={cfg.hidden_size}, L={cfg.num_layers}, Heads={cfg.num_heads}")
            print(f"  Batch: micro={cfg.micro_batch_size}, micro_batches={cfg.num_micro_batches}")
            global_batch = cfg.micro_batch_size * cfg.num_micro_batches * cfg.dp_size
            print(f"  Global batch size: {global_batch}")
            bubble = (cfg.pp_size - 1) / (cfg.num_micro_batches + cfg.pp_size - 1)
            print(f"  PP Bubble rate: {bubble:.1%}")
            print(f"  Precision: {cfg.precision}")
            print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# 可作为独立脚本运行
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mini Training Framework")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    if args.config and HAS_YAML:
        config = ParallelConfig.from_yaml(args.config)
    else:
        world_size = dist.get_world_size()
        dp = world_size // (args.tp * args.pp)
        config = ParallelConfig(
            tp_size=args.tp, pp_size=args.pp, dp_size=dp,
            hidden_size=1024, num_layers=8, num_heads=16,
            max_seq_len=256, micro_batch_size=4,
            num_micro_batches=8, max_steps=args.steps,
        )

    engine = HybridParallelEngine(config)
    engine.summary()

    # 模拟训练
    rank = dist.get_rank()
    for step in range(config.max_steps):
        # 简化: 生成随机数据
        def data_gen():
            while True:
                yield (torch.randn(config.micro_batch_size, config.max_seq_len,
                                   config.hidden_size, device=engine.device),)
        data_iter = data_gen()
        loss = engine.train_step(data_iter)

        if rank == 0 and step % 5 == 0:
            print(f"  Step {step} | Loss {loss:.4f}")

    if rank == 0:
        print("\n训练完成！")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
