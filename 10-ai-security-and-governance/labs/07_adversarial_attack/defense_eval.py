"""
Defense Evaluation - Assess model robustness against adversarial attacks.

Implements multiple defense strategies and provides metrics to evaluate
their effectiveness against word-level perturbation attacks.
"""

import re
import time
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional
from adversarial_examples import TextFoolerAttack, AttackConfig, AdversarialResult


@dataclass
class DefenseMetrics:
    """Metrics for evaluating a defense strategy."""
    defense_name: str
    attack_success_rate: float    # Lower is better
    avg_perturbation_rate: float
    avg_latency_ms: float
    samples_tested: int
    blocked_attacks: int


class SpellCheckDefense:
    """Defense via input spell-checking and normalization."""

    # Common homoglyph characters to normalize
    NORMALIZE_MAP = {
        "а": "a", "е": "e", "о": "o", "р": "p",
        "с": "c", "х": "x", "і": "i", "ѕ": "s",
    }

    def __call__(self, text: str) -> str:
        """Normalize text by replacing homoglyphs and fixing common typos."""
        result = text
        for homoglyph, ascii_char in self.NORMALIZE_MAP.items():
            result = result.replace(homoglyph, ascii_char)
        # Remove zero-width characters
        result = re.sub(r"[​‌‍﻿]", "", result)
        return result


class EnsembleDefense:
    """Defense via ensemble voting across multiple model variants."""

    def __init__(self, model_fns: List[Callable[[str], Dict[str, float]]]):
        self.models = model_fns

    def predict(self, text: str) -> Dict[str, float]:
        """Aggregate predictions from all models via soft voting."""
        aggregated: Dict[str, float] = {}
        for model_fn in self.models:
            scores = model_fn(text)
            for label, conf in scores.items():
                aggregated[label] = aggregated.get(label, 0.0) + conf

        total = sum(aggregated.values())
        return {k: v / total for k, v in aggregated.items()}


class PerplexityFilter:
    """Defense by rejecting inputs with abnormally high perplexity."""

    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold

    def is_suspicious(self, text: str) -> bool:
        """Check if text has unusual character patterns suggesting attack."""
        words = text.split()
        if not words:
            return False

        # Heuristic: check for non-ASCII in mostly ASCII text
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / max(len(text), 1)

        # Check for excessive character repetition
        repeated_chars = sum(1 for i in range(1, len(text)) if text[i] == text[i-1])
        repeat_ratio = repeated_chars / max(len(text), 1)

        return non_ascii_ratio > 0.05 or repeat_ratio > 0.3


class DefenseEvaluator:
    """Evaluate defense effectiveness against adversarial attacks."""

    def __init__(self, attack_config: Optional[AttackConfig] = None):
        self.attacker = TextFoolerAttack(attack_config or AttackConfig())

    def evaluate_defense(
        self,
        model_fn: Callable[[str], Dict[str, float]],
        defended_fn: Callable[[str], Dict[str, float]],
        test_inputs: List[str],
        defense_name: str = "unknown",
    ) -> DefenseMetrics:
        """Evaluate a defense by attacking both raw and defended models.

        Args:
            model_fn: Original (undefended) model
            defended_fn: Model with defense applied
            test_inputs: List of test texts
            defense_name: Name for reporting

        Returns:
            DefenseMetrics summarizing effectiveness
        """
        successes = 0
        total_perturbation = 0.0
        total_latency = 0.0
        blocked = 0

        for text in test_inputs:
            start = time.time()
            result = self.attacker.attack(text, defended_fn)
            elapsed = (time.time() - start) * 1000

            total_latency += elapsed
            if result.success:
                successes += 1
                total_perturbation += result.perturbation_rate
            else:
                # Check if raw model was vulnerable
                raw_result = self.attacker.attack(text, model_fn)
                if raw_result.success:
                    blocked += 1

        n = len(test_inputs)
        return DefenseMetrics(
            defense_name=defense_name,
            attack_success_rate=successes / max(n, 1),
            avg_perturbation_rate=total_perturbation / max(successes, 1),
            avg_latency_ms=total_latency / max(n, 1),
            samples_tested=n,
            blocked_attacks=blocked,
        )

    def compare_defenses(
        self, results: List[DefenseMetrics]
    ) -> None:
        """Print comparison table of defense strategies."""
        print(f"\n{'Defense':<20} {'ASR':<8} {'Blocked':<10} {'Latency(ms)':<12}")
        print("-" * 50)
        for r in results:
            print(f"{r.defense_name:<20} {r.attack_success_rate:<8.2%} "
                  f"{r.blocked_attacks:<10} {r.avg_latency_ms:<12.1f}")

        best = min(results, key=lambda r: r.attack_success_rate)
        print(f"\nBest defense: {best.defense_name} "
              f"(ASR: {best.attack_success_rate:.2%})")


if __name__ == "__main__":
    # Mock model for demonstration
    def mock_model(text: str) -> Dict[str, float]:
        positive_words = {"good", "great", "excellent", "wonderful"}
        words = set(text.lower().split())
        pos = len(words & positive_words) * 0.25
        score = min(0.5 + pos, 0.95)
        return {"positive": score, "negative": 1 - score}

    # Test inputs
    test_texts = [
        "This product is great and wonderful",
        "The service was good and efficient",
        "An excellent experience overall",
    ]

    evaluator = DefenseEvaluator()

    # Evaluate spell-check defense
    spell_defense = SpellCheckDefense()
    defended_model = lambda text: mock_model(spell_defense(text))

    metrics = evaluator.evaluate_defense(
        mock_model, defended_model, test_texts, "spell-check"
    )
    print(f"Defense: {metrics.defense_name}")
    print(f"ASR: {metrics.attack_success_rate:.2%}")
    print(f"Blocked: {metrics.blocked_attacks}")
