"""降级策略"""

def demo_fallback():
    print("=" * 70)
    print("  Fallback Strategy")
    print("=" * 70)
    print(f"\n  Fallback strategies for LLM serving:")
    print(f"\n  1. Model Downgrade:")
    print(f"     70B timeout/overload → try 13B → try 7B")
    print(f"     Quality decreases but availability maintained")
    print(f"\n  2. Cache Fallback:")
    print(f"     Similar request → return cached response")
    print(f"     Fast but may be stale")
    print(f"\n  3. Queue with Backpressure:")
    print(f"     Queue full → return 429 (Too Many Requests)")
    print(f"     Client retries with backoff")
    print(f"\n  4. Degraded Response:")
    print(f"     Return shorter/simpler response")
    print(f"     Reduce max_tokens, increase temperature")
    print(f"\n  Implementation in API Gateway:")
    print(f"  if primary.latency > threshold or primary.error:")
    print(f"      response = fallback_model.generate(request)")
    print(f"      response.add_header('X-Fallback: true')")

if __name__ == "__main__":
    demo_fallback()
