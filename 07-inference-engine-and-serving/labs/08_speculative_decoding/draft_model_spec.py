"""
Draft Model 投机解码实现

演示投机解码的核心逻辑:
1. Draft model 快速生成 K 个候选 token
2. Target model 一次性验证
3. Rejection sampling 保证输出分布不变
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import time


class SpeculativeDecoder:
    """
    投机解码器

    使用小模型 (draft) 猜测, 大模型 (target) 验证。
    通过 rejection sampling 保证输出质量 = 纯 target 模型。
    """

    def __init__(
        self,
        target_model=None,
        draft_model=None,
        num_speculative_tokens: int = 5,
        tokenizer=None,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.tokenizer = tokenizer

        # 统计
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0

    def speculative_decode_step(
        self,
        input_ids: torch.Tensor,  # [1, seq_len]
    ) -> Tuple[List[int], int]:
        """
        一轮投机解码

        Returns:
            (accepted_tokens, num_accepted)
        """
        device = input_ids.device
        K = self.num_speculative_tokens

        # ===== Step 1: Draft Model 生成 K 个候选 =====
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids.clone()

        for _ in range(K):
            with torch.no_grad():
                draft_logits = self.draft_model(current_ids).logits[:, -1, :]
                draft_prob = F.softmax(draft_logits, dim=-1)

            # 从 draft 分布采样
            draft_token = torch.multinomial(draft_prob, num_samples=1)
            draft_tokens.append(draft_token.item())
            draft_probs.append(draft_prob[0])

            # 追加到输入
            current_ids = torch.cat([current_ids, draft_token], dim=-1)

        # ===== Step 2: Target Model 一次性验证 =====
        # 把所有候选 token 拼接后, target model 做一次前向
        verify_ids = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=device)
        ], dim=-1)

        with torch.no_grad():
            target_logits = self.target_model(verify_ids).logits
            # 取对应位置的概率
            # target_logits[:, -K-1:-1, :] 对应 K 个位置的目标分布

        # ===== Step 3: Rejection Sampling =====
        accepted_tokens = []
        for i in range(K):
            pos = input_ids.shape[1] + i - 1  # 对应位置
            target_prob = F.softmax(target_logits[:, pos, :], dim=-1)[0]
            draft_prob = draft_probs[i]
            draft_token = draft_tokens[i]

            # Rejection sampling
            p = target_prob[draft_token].item()
            q = draft_prob[draft_token].item()

            if q == 0:
                break

            # Accept with probability min(1, p/q)
            acceptance_prob = min(1.0, p / q)
            if torch.rand(1).item() < acceptance_prob:
                accepted_tokens.append(draft_token)
            else:
                # Reject! 从修正分布重新采样
                # P_resample = normalize(max(0, P_target - P_draft))
                residual = torch.clamp(target_prob - draft_prob, min=0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    residual = residual / residual_sum
                    resampled = torch.multinomial(residual, num_samples=1)
                    accepted_tokens.append(resampled.item())
                else:
                    resampled = torch.multinomial(target_prob, num_samples=1)
                    accepted_tokens.append(resampled.item())
                break

        # 如果所有 K 个都被接受, 还需要从 target 再采样一个
        if len(accepted_tokens) == K:
            last_pos = input_ids.shape[1] + K - 1
            last_target_prob = F.softmax(target_logits[:, last_pos, :], dim=-1)
            bonus_token = torch.multinomial(last_target_prob, num_samples=1)
            accepted_tokens.append(bonus_token.item())

        # 更新统计
        self.total_draft_tokens += K
        self.total_accepted_tokens += len(accepted_tokens) - 1  # 不算 bonus

        return accepted_tokens, len(accepted_tokens)

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def speedup_estimate(self) -> float:
        """估算加速比"""
        alpha = self.acceptance_rate
        K = self.num_speculative_tokens
        # 平均每轮产出 = alpha*K + 1
        # 假设 draft 时间可忽略, verify 时间 ≈ 1 decode step
        return alpha * K + 1


def simulate_speculative_decoding():
    """
    模拟投机解码 (不需要实际模型)

    用随机概率分布模拟 draft 和 target 模型。
    """
    print("\n" + "=" * 70)
    print("  Speculative Decoding Simulation")
    print("=" * 70)

    vocab_size = 1000
    K = 5  # 每轮猜 5 个 token

    # 模拟不同的 draft-target 匹配度
    for similarity in [0.9, 0.7, 0.5, 0.3]:
        total_accepted = 0
        total_drafted = 0
        total_rounds = 0
        total_output = 0

        for _ in range(1000):  # 1000 轮模拟
            # 模拟 target 分布 (某个真实的 next token 分布)
            target_logits = torch.randn(vocab_size)
            target_prob = F.softmax(target_logits / 0.7, dim=-1)

            # 模拟 draft 分布 (与 target 有一定相似度)
            noise = torch.randn(vocab_size) * (1 - similarity)
            draft_logits = target_logits * similarity + noise
            draft_prob = F.softmax(draft_logits / 0.7, dim=-1)

            # 模拟一轮投机
            accepted = 0
            for i in range(K):
                # Draft 采样
                draft_token = torch.multinomial(draft_prob, 1).item()

                # Rejection sampling
                p = target_prob[draft_token].item()
                q = draft_prob[draft_token].item()

                if q > 0 and torch.rand(1).item() < min(1.0, p / q):
                    accepted += 1
                else:
                    accepted += 1  # 重采样也产出一个 token
                    break

            if accepted == K:
                accepted += 1  # bonus token

            total_accepted += accepted - 1  # 不算最后的 bonus/resample
            total_drafted += K
            total_rounds += 1
            total_output += accepted

        acc_rate = total_accepted / total_drafted
        avg_output = total_output / total_rounds
        speedup = avg_output  # 每轮产出 vs 正常的 1

        print(f"\n  Similarity={similarity:.1f}: acc_rate={acc_rate:.3f}, "
              f"avg_output/round={avg_output:.2f}, speedup={speedup:.2f}x")

    print(f"\n  Key insight:")
    print(f"  - Higher draft-target similarity → higher acceptance rate → more speedup")
    print(f"  - Even with low similarity, output quality is UNCHANGED (rejection sampling)")
    print(f"  - Only speedup is affected, never quality!")


if __name__ == "__main__":
    simulate_speculative_decoding()

    print("\n" + "=" * 70)
    print("  Using Speculative Decoding in vLLM:")
    print("  vllm serve meta-llama/Llama-2-70b-hf \\")
    print("    --speculative-model meta-llama/Llama-2-7b-hf \\")
    print("    --num-speculative-tokens 5 \\")
    print("    --tensor-parallel-size 8")
    print("=" * 70)
