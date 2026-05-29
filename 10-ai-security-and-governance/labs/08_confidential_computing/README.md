# Lab 08: Confidential Computing for AI

## Overview

This lab explores Trusted Execution Environments (TEEs) and how they enable
secure model inference. We cover Intel SGX, Intel TDX, and AMD SEV design
patterns relevant to backend engineers deploying ML workloads.

## Threat Model

```
┌─────────────────────────────────────────────────────────┐
│  Untrusted Host / Cloud Provider                        │
│                                                         │
│   ┌───────────────────────────────────────────────┐     │
│   │  Trusted Execution Environment (Enclave)      │     │
│   │                                               │     │
│   │   ┌─────────┐   ┌──────────┐   ┌─────────┐  │     │
│   │   │  Model  │──▶│Inference │──▶│ Result  │  │     │
│   │   │ Weights │   │  Engine  │   │(encrypted)│ │     │
│   │   └─────────┘   └──────────┘   └─────────┘  │     │
│   │                                               │     │
│   │   Memory encrypted + integrity protected      │     │
│   └───────────────────────────────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `tee_overview.md` | TEE technologies: SGX, TDX, SEV comparison |
| `secure_inference_design.md` | Architecture for confidential ML inference |

## Key Concepts

- **Attestation**: Cryptographic proof that code runs inside a genuine TEE
- **Sealing**: Encrypting data so only the same enclave can decrypt it
- **Side-channel resistance**: Mitigating timing/cache-based leakage
- **Memory encryption**: Hardware-level encryption of enclave memory pages

## Deployment Considerations

1. **Performance overhead**: 5-30% latency increase depending on workload
2. **Memory limits**: SGX enclaves limited to EPC size (~256 MB typical)
3. **Attestation flow**: Remote verifier must validate enclave quote
4. **Key management**: Sealing keys tied to enclave identity (MRENCLAVE)

## References

- Intel SGX Developer Guide
- Intel TDX Module Architecture Specification
- AMD SEV-SNP Firmware ABI Specification
- Gramine Library OS (SGX shielding layer)
