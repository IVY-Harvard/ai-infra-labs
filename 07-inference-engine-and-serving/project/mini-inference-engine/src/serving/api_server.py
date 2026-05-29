"""
FastAPI OpenAI 兼容 API Server

提供 /v1/chat/completions 和 /v1/completions 接口。
支持流式输出 (SSE)。
"""

import asyncio
import time
import uuid
import argparse
from typing import Optional, List, Dict, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from ..core.engine import LLMEngine, EngineConfig
from ..core.sequence import SamplingParams
from .streaming import create_sse_response


# ============ Request/Response Models ============

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


# ============ Server ============

app = FastAPI(title="Mini Inference Engine", version="0.1.0")
engine: Optional[LLMEngine] = None


@app.on_event("startup")
async def startup():
    """启动时加载模型"""
    global engine
    # Engine 由命令行参数初始化
    pass


@app.get("/health")
async def health():
    """健康检查"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    stats = engine.get_stats()
    return {"status": "healthy", "stats": stats}


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [{"id": engine.config.model_name if engine else "unknown", "object": "model"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的 Chat Completions API"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    # 构建 prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if request.stream:
        return StreamingResponse(
            generate_stream(request_id, prompt, sampling_params, request.model),
            media_type="text/event-stream",
        )
    else:
        return await generate_complete(request_id, prompt, sampling_params, request.model)


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI 兼容的 Completions API"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    return await generate_complete(request_id, request.prompt, sampling_params, request.model)


async def generate_complete(
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
) -> JSONResponse:
    """非流式生成"""
    engine.add_request(request_id, prompt, sampling_params)

    # 运行直到完成
    result = None
    while engine.has_unfinished_requests():
        results = engine.step()
        for r in results:
            if r["request_id"] == request_id:
                result = r
                break
        if result:
            break
        await asyncio.sleep(0.001)

    if result is None:
        raise HTTPException(status_code=500, detail="Generation failed")

    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["output_text"]},
            "finish_reason": result["finish_reason"],
        }],
        "usage": {
            "prompt_tokens": result["prompt_len"],
            "completion_tokens": result["output_len"],
            "total_tokens": result["prompt_len"] + result["output_len"],
        },
    })


async def generate_stream(
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """流式生成 (SSE)"""
    engine.add_request(request_id, prompt, sampling_params)

    prev_output_len = 0
    while engine.has_unfinished_requests():
        results = engine.step()

        # 检查是否有我们的请求完成
        for r in results:
            if r["request_id"] == request_id:
                # 发送最终 chunk
                yield create_sse_response(
                    request_id, model_name,
                    r["output_text"],
                    r["finish_reason"],
                )
                yield "data: [DONE]\n\n"
                return

        await asyncio.sleep(0.001)


def run_server(
    model: str = "gpt2",
    host: str = "0.0.0.0",
    port: int = 8000,
    num_gpu_blocks: int = 256,
    block_size: int = 16,
    max_model_len: int = 1024,
):
    """启动 API 服务"""
    import uvicorn

    global engine

    config = EngineConfig(
        model_name=model,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        max_model_len=max_model_len,
    )
    engine = LLMEngine(config)
    engine.load_model()

    print(f"\n{'='*60}")
    print(f"  Mini Inference Engine")
    print(f"  Model: {model}")
    print(f"  API: http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    run_server(
        model=args.model,
        host=args.host,
        port=args.port,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        max_model_len=args.max_model_len,
    )
