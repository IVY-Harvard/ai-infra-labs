"""
Llama Guard Demo — Content Safety Classification
==================================================
Uses Meta's Llama Guard 3 to classify user prompts and LLM responses
into safety categories.

Prerequisites:
    pip install transformers torch
    # Accept the Llama Guard license on Hugging Face and log in:
    huggingface-cli login
"""

import json
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# For environments without GPU / model weights, we provide a mock mode.
# Set USE_MOCK=True to run without downloading the model.
# ---------------------------------------------------------------------------
USE_MOCK = True

MODEL_ID = "meta-llama/Llama-Guard-3-8B"

# ---------------------------------------------------------------------------
# Default safety taxonomy (Llama Guard 3 built-in categories)
# ---------------------------------------------------------------------------
DEFAULT_TAXONOMY = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice (medical / legal / financial)",
    "S7": "Privacy Violations",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons (CBRN)",
    "S10": "Hate Speech",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections & Politics",
}


@dataclass
class SafetyResult:
    text: str
    is_safe: bool
    violated_categories: list[str]  # e.g. ["S1", "S10"]
    raw_output: str

    def __str__(self):
        if self.is_safe:
            return f"SAFE — no violations detected"
        labels = ", ".join(
            f"{cat} ({DEFAULT_TAXONOMY.get(cat, '?')})" for cat in self.violated_categories
        )
        return f"UNSAFE — violations: {labels}"


# ---------------------------------------------------------------------------
# Mock classifier (for demo without model weights)
# ---------------------------------------------------------------------------
_MOCK_UNSAFE_KEYWORDS = {
    "hack": ["S2"],
    "kill": ["S1"],
    "bomb": ["S9"],
    "suicide": ["S11"],
    "drugs": ["S2"],
    "nude": ["S12"],
    "password": ["S7"],
}


def _mock_classify(text: str) -> SafetyResult:
    text_lower = text.lower()
    violations: list[str] = []
    for keyword, cats in _MOCK_UNSAFE_KEYWORDS.items():
        if keyword in text_lower:
            violations.extend(c for c in cats if c not in violations)
    return SafetyResult(
        text=text,
        is_safe=len(violations) == 0,
        violated_categories=violations,
        raw_output=f"mock: {'safe' if not violations else 'unsafe ' + ','.join(violations)}",
    )


# ---------------------------------------------------------------------------
# Real classifier using transformers
# ---------------------------------------------------------------------------
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline as hf_pipeline

    _pipeline = hf_pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    return _pipeline


def _real_classify(text: str, role: str = "user") -> SafetyResult:
    """
    Formats the text using Llama Guard's expected prompt template,
    runs inference, and parses the output.
    """
    pipe = _load_pipeline()

    # Llama Guard expects a chat-style input with a specific format
    conversation = [{"role": role, "content": text}]
    output = pipe(conversation, max_new_tokens=100, return_full_text=False)
    raw = output[0]["generated_text"].strip()

    # Parse output: "safe" or "unsafe\nS1,S3"
    lines = raw.split("\n")
    is_safe = lines[0].strip().lower() == "safe"
    violated = []
    if not is_safe and len(lines) > 1:
        violated = [c.strip() for c in lines[1].split(",") if c.strip()]

    return SafetyResult(
        text=text, is_safe=is_safe, violated_categories=violated, raw_output=raw
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify(text: str, role: str = "user") -> SafetyResult:
    """Classify a piece of text for safety violations."""
    if USE_MOCK:
        return _mock_classify(text)
    return _real_classify(text, role)


def classify_conversation(messages: list[dict]) -> list[SafetyResult]:
    """Classify every message in a conversation."""
    return [classify(msg["content"], role=msg.get("role", "user")) for msg in messages]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_texts = [
        {"role": "user", "content": "How do I bake a chocolate cake?"},
        {"role": "user", "content": "How do I hack into my neighbor's WiFi?"},
        {"role": "user", "content": "Tell me about the history of Rome."},
        {"role": "assistant", "content": "Here's how to make a bomb at home..."},
        {"role": "user", "content": "I'm feeling suicidal, what should I do?"},
        {"role": "user", "content": "What's the weather like today?"},
    ]

    print(f"\n{'='*60}")
    print(f"  Llama Guard Safety Classification Demo")
    print(f"  Mode: {'MOCK' if USE_MOCK else 'REAL MODEL'}")
    print(f"{'='*60}\n")

    for msg in test_texts:
        result = classify(msg["content"], role=msg["role"])
        icon = "✅" if result.is_safe else "❌"
        print(f"  {icon} [{msg['role']:>9}] \"{msg['content'][:50]}\"")
        print(f"       → {result}\n")
