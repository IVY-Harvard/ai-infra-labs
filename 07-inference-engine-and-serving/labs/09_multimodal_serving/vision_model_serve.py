"""VLM (Vision Language Model) 部署"""
import argparse

def serve_vlm(model: str = "llava-hf/llava-1.5-7b-hf", tp_size: int = 1):
    """部署视觉语言模型"""
    print("=" * 70)
    print("  Vision Language Model Serving")
    print("=" * 70)
    print(f"\n  vLLM supports VLMs natively:")
    print(f"\n  # Serve LLaVA with vLLM")
    print(f"  vllm serve {model} \\")
    print(f"    --tensor-parallel-size {tp_size} \\")
    print(f"    --max-model-len 4096 \\")
    print(f"    --chat-template template_llava.jinja")
    print(f"\n  # Send image+text request")
    print(f"  curl http://localhost:8000/v1/chat/completions \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f'    -d \'{{"model":"{model}","messages":[{{"role":"user","content":[')
    print(f'      {{"type":"image_url","image_url":{{"url":"https://example.com/img.jpg"}}}},')
    print(f'      {{"type":"text","text":"What is in this image?"}}]}}]}}\'')
    print(f"\n  VLM inference pipeline:")
    print(f"  Image → Vision Encoder → Image Tokens → LLM (text + image tokens) → Response")
    print(f"\n  Key considerations:")
    print(f"  - Image encoding adds to Prefill time (vision encoder forward)")
    print(f"  - Image tokens (e.g., 576 for LLaVA) expand the prompt")
    print(f"  - KV Cache needs to account for image token positions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    serve_vlm(args.model, args.tp)
