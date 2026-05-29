# Lab 02 — NeMo Guardrails

## Overview

NVIDIA NeMo Guardrails is an open-source toolkit that adds programmable safety
rails to LLM-based applications. Instead of relying solely on prompt engineering,
you define **rules** in a domain-specific language called **Colang** that the
runtime enforces at every turn of conversation.

## What You Will Learn

| Topic | File |
|---|---|
| Installation & configuration | `nemo_setup.md` |
| Colang rule syntax | `colang_rules.co` |
| Guardrails server integration | `guardrails_server.py` |

## Key Concepts

- **Colang** — A lightweight language for defining conversational flows and
  safety constraints (e.g., "if the user asks for harmful content, refuse").
- **Input Rails** — Rules that filter or transform **user** messages before
  they reach the LLM.
- **Output Rails** — Rules that filter or transform **LLM** responses before
  they are returned to the user.
- **Dialog Rails** — High-level conversational flows (e.g., always greet the
  user, never discuss competitors).

## Prerequisites

- Python 3.10+
- An OpenAI-compatible API key (set `OPENAI_API_KEY`)
- Basic familiarity with YAML configuration

## Quick Start

```bash
pip install nemoguardrails
cd 02_guardrails_nemo
nemoguardrails chat --config .
```

## Architecture

```
User Message
    │
    ▼
┌──────────────┐
│ Input Rails  │  ← Colang rules check / transform user input
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   LLM Call   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Output Rails │  ← Colang rules check / transform LLM output
└──────┬───────┘
       │
       ▼
  Final Response
```

## Further Reading

- [NeMo Guardrails GitHub](https://github.com/NVIDIA/NeMo-Guardrails)
- [Colang Language Reference](https://docs.nvidia.com/nemo/guardrails/colang/index.html)
