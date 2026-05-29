"""
Watermark Embedder - Token distribution shift-based text watermarking.

Embeds a statistical watermark into LLM-generated text by biasing token
selection toward a pseudorandomly determined 'green list' partition.
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WatermarkConfig:
    """Configuration for watermark embedding."""
    gamma: float = 0.5          # Fraction of vocab in green list
    delta: float = 2.0          # Logit bias for green tokens
    seed_key: str = "wm-secret-key-2024"  # Secret key for hash
    vocab_size: int = 50257     # GPT-2 vocabulary size


class WatermarkEmbedder:
    """Embeds watermarks into text generation via logit manipulation."""

    def __init__(self, config: Optional[WatermarkConfig] = None):
        self.config = config or WatermarkConfig()

    def _get_green_list(self, prev_token_id: int) -> np.ndarray:
        """Compute green list using hash of previous token + secret key."""
        hash_input = f"{self.config.seed_key}:{prev_token_id}".encode()
        seed = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        vocab_indices = np.arange(self.config.vocab_size)
        rng.shuffle(vocab_indices)

        green_count = int(self.config.gamma * self.config.vocab_size)
        return set(vocab_indices[:green_count].tolist())

    def apply_watermark(self, logits: np.ndarray, prev_token_id: int) -> np.ndarray:
        """Apply watermark bias to logits before sampling.

        Args:
            logits: Raw logits from the model (shape: [vocab_size])
            prev_token_id: The previously generated token ID

        Returns:
            Modified logits with green list bias applied
        """
        green_list = self._get_green_list(prev_token_id)
        watermarked_logits = logits.copy()

        for token_id in green_list:
            watermarked_logits[token_id] += self.config.delta

        return watermarked_logits

    def generate_watermarked(
        self, model_fn, prompt_ids: List[int], max_tokens: int = 100
    ) -> List[int]:
        """Generate watermarked text token by token.

        Args:
            model_fn: Callable that takes token IDs and returns logits
            prompt_ids: Tokenized prompt
            max_tokens: Maximum tokens to generate

        Returns:
            List of generated token IDs (watermarked)
        """
        generated = list(prompt_ids)

        for _ in range(max_tokens):
            logits = model_fn(generated)
            prev_token = generated[-1]

            watermarked_logits = self.apply_watermark(logits, prev_token)

            # Temperature-based sampling
            probs = _softmax(watermarked_logits)
            next_token = np.random.choice(len(probs), p=probs)

            generated.append(next_token)
            if next_token == 50256:  # EOS token
                break

        return generated[len(prompt_ids):]

    def get_green_fraction(self, token_ids: List[int]) -> float:
        """Calculate fraction of tokens that fall in green list."""
        if len(token_ids) < 2:
            return 0.0

        green_count = 0
        for i in range(1, len(token_ids)):
            green_list = self._get_green_list(token_ids[i - 1])
            if token_ids[i] in green_list:
                green_count += 1

        return green_count / (len(token_ids) - 1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


if __name__ == "__main__":
    config = WatermarkConfig(gamma=0.5, delta=2.0)
    embedder = WatermarkEmbedder(config)

    # Demo: show green list bias effect
    fake_logits = np.zeros(config.vocab_size)
    modified = embedder.apply_watermark(fake_logits, prev_token_id=42)

    green_count = np.sum(modified > 0)
    print(f"Green list size: {green_count}/{config.vocab_size}")
    print(f"Expected: {int(config.gamma * config.vocab_size)}")
    print(f"Delta applied: {modified[modified > 0][0]:.1f}")
