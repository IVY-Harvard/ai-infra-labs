"""
Robustness Test - Evaluate watermark survival under various text attacks.

Tests whether the watermark can still be detected after paraphrasing,
truncation, word substitution, and other common text transformations.
"""

import random
import string
from dataclasses import dataclass
from typing import List, Dict, Callable

from watermark_embedder import WatermarkConfig, WatermarkEmbedder
from watermark_detector import WatermarkDetector, DetectionResult


@dataclass
class AttackResult:
    """Result of a single robustness attack."""
    attack_name: str
    original_z_score: float
    attacked_z_score: float
    survived: bool
    token_retention_rate: float


class RobustnessTest:
    """Test watermark robustness against text modifications."""

    def __init__(self, config: WatermarkConfig, z_threshold: float = 4.0):
        self.config = config
        self.detector = WatermarkDetector(config, z_threshold)
        self.z_threshold = z_threshold

        self.attacks: Dict[str, Callable] = {
            "truncate_head": self._truncate_head,
            "truncate_tail": self._truncate_tail,
            "random_swap": self._random_swap,
            "random_insert": self._random_insert,
            "random_delete": self._random_delete,
            "shuffle_sentences": self._shuffle_sentences,
        }

    def run_all_attacks(
        self, token_ids: List[int], intensity: float = 0.2
    ) -> List[AttackResult]:
        """Run all registered attacks and report survival.

        Args:
            token_ids: Original watermarked token sequence
            intensity: Attack strength (0.0 to 1.0)

        Returns:
            List of AttackResult for each attack type
        """
        original_result = self.detector.detect(token_ids)
        results = []

        for name, attack_fn in self.attacks.items():
            attacked_tokens = attack_fn(token_ids, intensity)
            attack_result = self.detector.detect(attacked_tokens)

            retention = len(attacked_tokens) / len(token_ids)
            results.append(AttackResult(
                attack_name=name,
                original_z_score=original_result.z_score,
                attacked_z_score=attack_result.z_score,
                survived=attack_result.is_watermarked,
                token_retention_rate=retention,
            ))

        return results

    def _truncate_head(self, tokens: List[int], intensity: float) -> List[int]:
        """Remove tokens from the beginning."""
        cut = int(len(tokens) * intensity)
        return tokens[cut:]

    def _truncate_tail(self, tokens: List[int], intensity: float) -> List[int]:
        """Remove tokens from the end."""
        cut = int(len(tokens) * intensity)
        return tokens[: len(tokens) - cut] if cut > 0 else tokens

    def _random_swap(self, tokens: List[int], intensity: float) -> List[int]:
        """Randomly swap adjacent token pairs."""
        result = tokens.copy()
        num_swaps = int(len(result) * intensity)
        for _ in range(num_swaps):
            idx = random.randint(0, len(result) - 2)
            result[idx], result[idx + 1] = result[idx + 1], result[idx]
        return result

    def _random_insert(self, tokens: List[int], intensity: float) -> List[int]:
        """Insert random tokens at random positions."""
        result = tokens.copy()
        num_inserts = int(len(result) * intensity)
        for _ in range(num_inserts):
            idx = random.randint(0, len(result))
            fake_token = random.randint(0, self.config.vocab_size - 1)
            result.insert(idx, fake_token)
        return result

    def _random_delete(self, tokens: List[int], intensity: float) -> List[int]:
        """Delete random tokens from the sequence."""
        result = tokens.copy()
        num_deletes = int(len(result) * intensity)
        for _ in range(min(num_deletes, len(result) - 2)):
            idx = random.randint(0, len(result) - 1)
            result.pop(idx)
        return result

    def _shuffle_sentences(self, tokens: List[int], intensity: float) -> List[int]:
        """Simulate sentence reordering by shuffling chunks."""
        chunk_size = max(10, int(len(tokens) * 0.1))
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        num_shuffles = max(1, int(len(chunks) * intensity))
        for _ in range(num_shuffles):
            i, j = random.sample(range(len(chunks)), 2)
            chunks[i], chunks[j] = chunks[j], chunks[i]
        return [t for chunk in chunks for t in chunk]

    def print_report(self, results: List[AttackResult]) -> None:
        """Print a formatted robustness report."""
        print(f"{'Attack':<20} {'Z-Score':<10} {'Survived':<10} {'Retention':<10}")
        print("-" * 50)
        for r in results:
            status = "PASS" if r.survived else "FAIL"
            print(f"{r.attack_name:<20} {r.attacked_z_score:<10.2f} {status:<10} "
                  f"{r.token_retention_rate:<10.2%}")

        survived = sum(1 for r in results if r.survived)
        print(f"\nSurvival rate: {survived}/{len(results)} attacks")


if __name__ == "__main__":
    config = WatermarkConfig(gamma=0.5, delta=2.0)
    tester = RobustnessTest(config)

    # Simulate a watermarked sequence (biased toward green tokens)
    random.seed(42)
    fake_tokens = list(range(200, 400))

    print("Running robustness tests (intensity=0.2)...")
    results = tester.run_all_attacks(fake_tokens, intensity=0.2)
    tester.print_report(results)
