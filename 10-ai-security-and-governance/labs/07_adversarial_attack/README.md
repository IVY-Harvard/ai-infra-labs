# Lab 07: Adversarial Attacks on Text Models

## Overview

This lab demonstrates adversarial attack techniques against NLP/LLM systems and
evaluates defense mechanisms. We implement character-level and word-level
perturbation methods inspired by TextFooler and related work.

## Threat Model

An attacker modifies input text to cause model misclassification while
maintaining semantic similarity and human readability. Common strategies:

1. **Character-level**: typos, homoglyphs, invisible characters
2. **Word-level**: synonym substitution, word insertion/deletion
3. **Sentence-level**: paraphrasing, syntactic restructuring

## Files

| File | Description |
|------|-------------|
| `adversarial_examples.py` | Implements TextFooler-style word perturbation attacks |
| `defense_eval.py` | Evaluates model robustness and defense effectiveness |

## Attack Pipeline

```
┌──────────┐    ┌────────────────┐    ┌──────────────┐    ┌──────────────┐
│  Input   │───▶│ Word Importance │───▶│  Candidate   │───▶│  Adversarial │
│  Text    │    │   Ranking      │    │  Generation  │    │   Example    │
└──────────┘    └────────────────┘    └──────────────┘    └──────────────┘
                       │                      │
                 Delete each word        Synonym lookup
                 Measure Δ confidence    Semantic filter
```

## Quick Start

```bash
# Generate adversarial examples
python adversarial_examples.py --input "This movie is great" --target negative

# Evaluate model defenses
python defense_eval.py --model bert-base --dataset samples.json
```

## Defense Strategies Evaluated

- **Input preprocessing**: spell-check, character normalization
- **Adversarial training**: augmenting training data with perturbations
- **Ensemble voting**: multiple models reduce attack transferability
- **Certified robustness**: randomized smoothing for provable guarantees

## Key Metrics

- Attack Success Rate (ASR): % of inputs successfully misclassified
- Perturbation Rate: % of words modified
- Semantic Similarity: cosine similarity of embeddings (pre/post attack)
- Query Efficiency: number of model queries needed per attack
