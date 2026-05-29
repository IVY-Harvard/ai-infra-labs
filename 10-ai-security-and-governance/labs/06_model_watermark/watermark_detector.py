"""
Watermark Detector - Statistical detection of token distribution watermarks.

Uses a one-proportion z-test to determine whether text contains a watermark
by measuring deviation from expected green token frequency.
"""

import math
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from watermark_embedder import WatermarkConfig


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    z_score: float
    p_value: float
    green_fraction: float
    num_tokens_scored: int
    is_watermarked: bool
    confidence: str  # "high", "medium", "low", "none"


class WatermarkDetector:
    """Detects watermarks in text using statistical z-test."""

    def __init__(self, config: WatermarkConfig, z_threshold: float = 4.0):
        self.config = config
        self.z_threshold = z_threshold

    def _get_green_list(self, prev_token_id: int) -> set:
        """Reconstruct green list for a given previous token."""
        hash_input = f"{self.config.seed_key}:{prev_token_id}".encode()
        seed = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        vocab_indices = np.arange(self.config.vocab_size)
        rng.shuffle(vocab_indices)

        green_count = int(self.config.gamma * self.config.vocab_size)
        return set(vocab_indices[:green_count].tolist())

    def score_tokens(self, token_ids: List[int]) -> List[bool]:
        """Score each token as green (True) or red (False).

        Args:
            token_ids: Sequence of token IDs to analyze

        Returns:
            Boolean list indicating green membership for each token (except first)
        """
        green_flags = []
        for i in range(1, len(token_ids)):
            green_list = self._get_green_list(token_ids[i - 1])
            green_flags.append(token_ids[i] in green_list)
        return green_flags

    def detect(self, token_ids: List[int]) -> DetectionResult:
        """Run watermark detection on a token sequence.

        Uses one-proportion z-test:
            H0: green fraction = gamma (no watermark)
            H1: green fraction > gamma (watermark present)

        Args:
            token_ids: Tokenized text to test

        Returns:
            DetectionResult with z-score, p-value, and verdict
        """
        if len(token_ids) < 2:
            return DetectionResult(
                z_score=0.0, p_value=1.0, green_fraction=0.0,
                num_tokens_scored=0, is_watermarked=False, confidence="none"
            )

        green_flags = self.score_tokens(token_ids)
        n = len(green_flags)
        green_count = sum(green_flags)
        green_fraction = green_count / n

        # One-proportion z-test
        expected = self.config.gamma
        std_dev = math.sqrt(expected * (1 - expected) / n)
        z_score = (green_fraction - expected) / std_dev if std_dev > 0 else 0.0

        # One-tailed p-value
        p_value = 0.5 * math.erfc(z_score / math.sqrt(2))

        is_watermarked = z_score >= self.z_threshold
        confidence = self._classify_confidence(z_score)

        return DetectionResult(
            z_score=z_score,
            p_value=p_value,
            green_fraction=green_fraction,
            num_tokens_scored=n,
            is_watermarked=is_watermarked,
            confidence=confidence,
        )

    def _classify_confidence(self, z_score: float) -> str:
        """Map z-score to confidence level."""
        if z_score >= 6.0:
            return "high"
        elif z_score >= self.z_threshold:
            return "medium"
        elif z_score >= 2.0:
            return "low"
        return "none"

    def windowed_detect(
        self, token_ids: List[int], window_size: int = 50, stride: int = 25
    ) -> List[Tuple[int, int, float]]:
        """Sliding window detection for partial watermarks.

        Returns:
            List of (start, end, z_score) tuples for each window
        """
        results = []
        for start in range(0, len(token_ids) - window_size, stride):
            window = token_ids[start : start + window_size]
            result = self.detect(window)
            results.append((start, start + window_size, result.z_score))
        return results


if __name__ == "__main__":
    config = WatermarkConfig()
    detector = WatermarkDetector(config, z_threshold=4.0)

    # Simulate watermarked text (high green fraction)
    np.random.seed(42)
    fake_watermarked = list(range(100, 200))
    result = detector.detect(fake_watermarked)

    print(f"Z-score: {result.z_score:.2f}")
    print(f"Green fraction: {result.green_fraction:.3f} (expected ~{config.gamma})")
    print(f"P-value: {result.p_value:.6f}")
    print(f"Watermarked: {result.is_watermarked} ({result.confidence})")
    print(f"Tokens scored: {result.num_tokens_scored}")
