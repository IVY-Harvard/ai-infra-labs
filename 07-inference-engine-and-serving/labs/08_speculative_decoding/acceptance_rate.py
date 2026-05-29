"""接受率分析: 不同场景下投机解码的效果"""

def analyze_acceptance_rate():
    print("=" * 70)
    print("  Acceptance Rate Analysis")
    print("=" * 70)
    print(f"\n  Factors affecting acceptance rate:")
    print(f"  1. Draft model quality (closer to target → higher)")
    print(f"  2. Temperature (lower → higher acceptance)")
    print(f"  3. Content type (deterministic → higher)")
    print(f"\n  Typical acceptance rates:")
    print(f"  {'Scenario':<35} {'Acc Rate':<10} {'K=5 Speedup'}")
    print(f"  {'-'*55}")
    print(f"  {'LLaMA-7B → LLaMA-70B (code)':<35} {'0.80':<10} {'5.0x'}")
    print(f"  {'LLaMA-7B → LLaMA-70B (chat)':<35} {'0.65':<10} {'4.25x'}")
    print(f"  {'LLaMA-1B → LLaMA-70B (chat)':<35} {'0.45':<10} {'3.25x'}")
    print(f"  {'LLaMA-7B → LLaMA-70B (creative)':<35} {'0.40':<10} {'3.0x'}")
    print(f"\n  When NOT to use speculative decoding:")
    print(f"  - Batch size > 32 (GPU already compute-bound)")
    print(f"  - High concurrency (throughput matters more than latency)")
    print(f"  - Very different draft/target (low acceptance)")

if __name__ == "__main__":
    analyze_acceptance_rate()
