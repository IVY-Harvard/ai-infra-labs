# Lab 06: Model Watermarking

## Overview

This lab implements text watermarking for LLM outputs using token distribution
shifting. The technique embeds an imperceptible statistical signal into generated
text that can later be detected to prove model provenance.

## Core Concept

During text generation, we partition the vocabulary into "green" and "red" tokens
using a hash of the preceding token as a seed. We then bias the logits to favor
green tokens. The resulting text looks natural but carries a detectable statistical
signature.

## Files

| File | Description |
|------|-------------|
| `watermark_embedder.py` | Embeds watermarks during text generation via logit manipulation |
| `watermark_detector.py` | Detects watermark presence using z-score statistical test |
| `robustness_test.py` | Tests watermark survival after paraphrasing and editing |

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  LLM Engine │────▶│ Watermark Embedder│────▶│ Generated Text  │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────┐               │
                    │ Watermark Detector│◀──────────────┘
                    └──────────────────┘
                           │
                    ┌──────▼──────┐
                    │ z-score ≥ 4 │── Watermarked
                    │ z-score < 4 │── Not watermarked
                    └─────────────┘
```

## Quick Start

```bash
# Embed watermark during generation
python watermark_embedder.py --prompt "Explain microservices"

# Detect watermark in text
python watermark_detector.py --input output.txt

# Run robustness tests
python robustness_test.py --input output.txt --attacks paraphrase,truncate,swap
```

## Key Parameters

- `gamma` (default 0.5): Fraction of vocabulary in the green list
- `delta` (default 2.0): Logit bias added to green tokens
- `z_threshold` (default 4.0): Detection threshold (higher = fewer false positives)

## References

- Kirchenbauer et al., "A Watermark for Large Language Models" (2023)
- Scott Aaronson's approach using cryptographic pseudorandom functions
