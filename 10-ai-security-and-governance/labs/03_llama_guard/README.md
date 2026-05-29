# Lab 03 — Llama Guard Content Safety

## Overview

**Llama Guard** is Meta's open-source content safety classifier built on top of
Llama. It classifies user prompts and LLM responses into safety categories
(violence, self-harm, criminal activity, etc.) so you can block or flag unsafe
content in real time.

## What You Will Learn

| Topic | File |
|---|---|
| Basic classification demo | `llama_guard_demo.py` |
| Custom safety taxonomy | `custom_taxonomy.py` |
| Performance benchmarking | `benchmark.py` |

## Key Concepts

- **Safety Taxonomy** — A set of categories (e.g., S1: Violence, S2: Sexual
  Content) that the model uses to classify text.
- **Prompt Classification** — Checking whether the *user's input* is safe.
- **Response Classification** — Checking whether the *model's output* is safe.
- **Custom Taxonomy** — Extending or replacing default categories with
  domain-specific safety rules.

## Prerequisites

- Python 3.10+
- PyTorch with CUDA (recommended) or CPU
- `transformers` library (`pip install transformers torch`)
- ~14 GB disk space for the Llama Guard 3 model weights

## Quick Start

```bash
pip install transformers torch
python llama_guard_demo.py
```

## Architecture

```
User / LLM Text
       │
       ▼
┌──────────────────┐
│  Llama Guard 3   │  ← Classifies text against safety taxonomy
└──────┬───────────┘
       │
       ▼
  "safe" / "unsafe S1, S3"   ← Category labels returned
```

## Further Reading

- [Llama Guard Paper (Meta)](https://arxiv.org/abs/2312.06674)
- [Llama Guard 3 on Hugging Face](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
