"""
Adversarial Examples - TextFooler-style word perturbation attacks.

Implements word importance ranking and synonym-based substitution to
generate adversarial text that fools classifiers while remaining readable.
"""

import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional


@dataclass
class AttackConfig:
    """Configuration for adversarial attack."""
    max_perturbation_rate: float = 0.3   # Max fraction of words to change
    similarity_threshold: float = 0.7     # Min semantic similarity
    max_candidates: int = 10              # Synonym candidates per word
    use_homoglyphs: bool = True           # Enable character-level attacks


@dataclass
class AdversarialResult:
    """Result of an adversarial attack attempt."""
    original_text: str
    adversarial_text: str
    original_label: str
    adversarial_label: str
    success: bool
    perturbation_rate: float
    words_changed: List[Tuple[str, str]] = field(default_factory=list)


# Simple synonym dictionary (in production, use WordNet or embedding neighbors)
SYNONYM_TABLE: Dict[str, List[str]] = {
    "good": ["great", "fine", "decent", "solid", "positive"],
    "great": ["excellent", "wonderful", "fantastic", "superb", "outstanding"],
    "bad": ["poor", "terrible", "awful", "inferior", "lousy"],
    "happy": ["glad", "pleased", "content", "cheerful", "joyful"],
    "sad": ["unhappy", "gloomy", "depressed", "sorrowful", "melancholy"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["sluggish", "gradual", "leisurely", "unhurried", "delayed"],
    "important": ["crucial", "vital", "significant", "essential", "critical"],
    "small": ["tiny", "little", "minor", "compact", "modest"],
    "large": ["big", "huge", "enormous", "massive", "vast"],
}

# Homoglyph substitutions (visually similar characters)
HOMOGLYPHS: Dict[str, str] = {
    "a": "а", "e": "е", "o": "о", "p": "р",
    "c": "с", "x": "х", "i": "і", "s": "ѕ",
}


class TextFoolerAttack:
    """Word-level adversarial attack using importance ranking + synonyms."""

    def __init__(self, config: Optional[AttackConfig] = None):
        self.config = config or AttackConfig()

    def rank_word_importance(
        self, text: str, model_fn: Callable[[str], Dict[str, float]]
    ) -> List[Tuple[int, str, float]]:
        """Rank words by their importance to the model's prediction.

        Importance = drop in confidence when word is removed.
        """
        words = text.split()
        base_scores = model_fn(text)
        base_label = max(base_scores, key=base_scores.get)
        base_conf = base_scores[base_label]

        importance_scores = []
        for i, word in enumerate(words):
            reduced = words[:i] + words[i + 1:]
            reduced_text = " ".join(reduced)
            new_scores = model_fn(reduced_text)
            new_conf = new_scores.get(base_label, 0.0)
            importance = base_conf - new_conf
            importance_scores.append((i, word, importance))

        importance_scores.sort(key=lambda x: x[2], reverse=True)
        return importance_scores

    def get_candidates(self, word: str) -> List[str]:
        """Get replacement candidates for a word."""
        candidates = []

        # Synonym lookup
        lower = word.lower()
        if lower in SYNONYM_TABLE:
            candidates.extend(SYNONYM_TABLE[lower])

        # Character-level perturbations
        if self.config.use_homoglyphs and len(word) > 3:
            homoglyph_variant = self._apply_homoglyph(word)
            if homoglyph_variant != word:
                candidates.append(homoglyph_variant)

        # Typo insertion (swap adjacent chars)
        if len(word) > 4:
            idx = random.randint(1, len(word) - 2)
            typo = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
            candidates.append(typo)

        return candidates[: self.config.max_candidates]

    def _apply_homoglyph(self, word: str) -> str:
        """Replace one character with a visually similar homoglyph."""
        for i, char in enumerate(word.lower()):
            if char in HOMOGLYPHS:
                return word[:i] + HOMOGLYPHS[char] + word[i + 1:]
        return word

    def attack(
        self, text: str, model_fn: Callable[[str], Dict[str, float]]
    ) -> AdversarialResult:
        """Execute the adversarial attack.

        Args:
            text: Input text to perturb
            model_fn: Classification function returning {label: confidence}

        Returns:
            AdversarialResult with attack outcome
        """
        words = text.split()
        max_changes = int(len(words) * self.config.max_perturbation_rate)

        base_scores = model_fn(text)
        original_label = max(base_scores, key=base_scores.get)

        ranked_words = self.rank_word_importance(text, model_fn)
        modified_words = list(words)
        changes = []

        for idx, word, importance in ranked_words[:max_changes]:
            candidates = self.get_candidates(word)
            for candidate in candidates:
                modified_words[idx] = candidate
                new_text = " ".join(modified_words)
                new_scores = model_fn(new_text)
                new_label = max(new_scores, key=new_scores.get)

                if new_label != original_label:
                    changes.append((word, candidate))
                    return AdversarialResult(
                        original_text=text,
                        adversarial_text=new_text,
                        original_label=original_label,
                        adversarial_label=new_label,
                        success=True,
                        perturbation_rate=len(changes) / len(words),
                        words_changed=changes,
                    )
            modified_words[idx] = word  # Revert if no candidate works

        return AdversarialResult(
            original_text=text,
            adversarial_text=" ".join(modified_words),
            original_label=original_label,
            adversarial_label=original_label,
            success=False,
            perturbation_rate=0.0,
            words_changed=[],
        )


if __name__ == "__main__":
    # Mock classifier for demonstration
    def mock_classifier(text: str) -> Dict[str, float]:
        positive_words = {"good", "great", "excellent", "wonderful", "fantastic"}
        negative_words = {"bad", "terrible", "awful", "poor", "lousy"}
        words = set(text.lower().split())
        pos = len(words & positive_words) * 0.3
        neg = len(words & negative_words) * 0.3
        pos_score = 0.5 + pos - neg
        return {"positive": min(pos_score, 0.99), "negative": 1 - min(pos_score, 0.99)}

    attacker = TextFoolerAttack()
    result = attacker.attack("This movie is great and wonderful", mock_classifier)

    print(f"Original: {result.original_text} [{result.original_label}]")
    print(f"Adversarial: {result.adversarial_text} [{result.adversarial_label}]")
    print(f"Success: {result.success}")
    print(f"Perturbation rate: {result.perturbation_rate:.2%}")
