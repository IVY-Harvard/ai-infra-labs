"""语音模型部署"""
def serve_audio_model():
    print("=" * 70)
    print("  Audio Model Serving")
    print("=" * 70)
    print(f"\n  Audio models have different serving patterns:")
    print(f"\n  1. ASR (Speech-to-Text): Whisper")
    print(f"     - Input: audio → spectrogram → encoder → text")
    print(f"     - Typically use separate serving (not LLM engine)")
    print(f"     - Faster Whisper: CTranslate2 backend")
    print(f"\n  2. TTS (Text-to-Speech): VITS, CosyVoice")
    print(f"     - Input: text → audio tokens → vocoder → audio")
    print(f"     - Streaming: chunk-by-chunk generation")
    print(f"\n  3. Speech LLM: Qwen-Audio, SALMONN")
    print(f"     - Input: audio + text → LLM → text response")
    print(f"     - Similar to VLM serving pattern")
    print(f"     - Audio encoder replaces vision encoder")
    print(f"\n  Deployment:")
    print(f"  # Qwen-Audio with vLLM")
    print(f"  vllm serve Qwen/Qwen-Audio-Chat --trust-remote-code")

if __name__ == "__main__":
    serve_audio_model()
