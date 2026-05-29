"""
Guardrails Server — HTTP wrapper around NeMo Guardrails
========================================================
Exposes a FastAPI endpoint that applies Colang safety rules to every request
before forwarding to the underlying LLM.

Usage:
    pip install nemoguardrails fastapi uvicorn
    python guardrails_server.py
    # POST http://localhost:8080/chat  {"message": "Hello"}
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# NeMo Guardrails import (install: pip install nemoguardrails)
# ---------------------------------------------------------------------------
try:
    from nemoguardrails import RailsConfig, LLMRails
except ImportError:
    raise SystemExit(
        "nemoguardrails is not installed. Run: pip install nemoguardrails"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))  # directory with config.yml & .co files
LISTEN_HOST = os.getenv("GUARD_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("GUARD_PORT", "8080"))

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    guardrail_triggered: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# In-memory conversation store (swap with Redis for production)
# ---------------------------------------------------------------------------
conversations: dict[str, list[dict]] = {}


def _get_or_create_history(conv_id: str) -> list[dict]:
    if conv_id not in conversations:
        conversations[conv_id] = []
    return conversations[conv_id]


# ---------------------------------------------------------------------------
# Application lifecycle — load rails once on startup
# ---------------------------------------------------------------------------
rails_engine: LLMRails | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rails_engine
    logger.info("Loading NeMo Guardrails config from %s", CONFIG_DIR)
    config = RailsConfig.from_path(CONFIG_DIR)
    rails_engine = LLMRails(config)
    logger.info("Guardrails engine ready")
    yield
    logger.info("Shutting down guardrails server")


app = FastAPI(
    title="NeMo Guardrails Server",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": rails_engine is not None}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if rails_engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    conv_id = req.conversation_id or f"conv-{int(time.time()*1000)}"
    history = _get_or_create_history(conv_id)

    # Append user message
    history.append({"role": "user", "content": req.message})

    start = time.perf_counter()
    try:
        result = await rails_engine.generate_async(messages=history)
    except Exception as exc:
        logger.exception("Guardrails engine error")
        raise HTTPException(status_code=500, detail=str(exc))
    latency = (time.perf_counter() - start) * 1000

    # NeMo returns a dict with "content" or a list of messages
    if isinstance(result, dict):
        reply = result.get("content", str(result))
    elif isinstance(result, list) and result:
        reply = result[-1].get("content", str(result[-1]))
    else:
        reply = str(result)

    # Simple heuristic: if the reply matches a refusal pattern, a rail was triggered
    refusal_markers = ["i'm not able to", "cannot", "i've detected", "please don't share"]
    guardrail_triggered = any(m in reply.lower() for m in refusal_markers)

    # Store assistant reply
    history.append({"role": "assistant", "content": reply})

    logger.info(
        "conv=%s | guardrail=%s | latency=%.0fms | user=%s",
        conv_id, guardrail_triggered, latency, req.message[:60],
    )

    return ChatResponse(
        reply=reply,
        conversation_id=conv_id,
        guardrail_triggered=guardrail_triggered,
        latency_ms=round(latency, 1),
    )


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------
@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if conv_id in conversations:
        del conversations[conv_id]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/conversations")
async def list_conversations():
    return {
        "active_conversations": len(conversations),
        "ids": list(conversations.keys()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "guardrails_server:app",
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        reload=True,
    )
