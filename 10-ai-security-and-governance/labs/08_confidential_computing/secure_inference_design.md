# Secure Inference Architecture Design

## Goals

1. Protect model weights from extraction by the infrastructure operator
2. Protect user prompts/responses from observation
3. Provide cryptographic attestation to end users
4. Minimize performance overhead for production workloads

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Client                                                     │
│  ┌─────────────────────────────────────────────────┐        │
│  │ 1. Verify attestation quote                     │        │
│  │ 2. Establish TLS-to-enclave channel             │        │
│  │ 3. Send encrypted inference request             │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────┬───────────────────────────────────────┘
                      │ mTLS (pinned to enclave identity)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Confidential VM / Enclave                                  │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │  TLS     │──▶│  Request     │──▶│  Model Runtime    │   │
│  │ Terminator│   │  Validator   │   │  (ONNX/TensorRT) │   │
│  └──────────┘   └──────────────┘   └───────────────────┘   │
│                                             │               │
│  ┌──────────────────────────────────────────┘               │
│  │                                                          │
│  ▼                                                          │
│  ┌──────────────┐   ┌──────────────┐                        │
│  │  Sealed Key  │   │  Audit Log   │ (encrypted, append-only)│
│  │  Store       │   │  Buffer      │                        │
│  └──────────────┘   └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Attestation Service

- Generate enclave identity (MRENCLAVE / TD measurement)
- Expose `/v1/attestation` endpoint returning a signed quote
- Client verifies quote against Intel/AMD root of trust
- Quote includes: measurement, signer, security version, custom data

### 2. Key Management

```
Boot sequence:
  1. Enclave starts, generates ephemeral TLS keypair
  2. Derives sealing key from hardware (MRSIGNER + SVN)
  3. Unseals model decryption key from persistent storage
  4. Loads encrypted model weights into protected memory
  5. TLS public key hash embedded in attestation quote
```

### 3. Inference Pipeline

| Stage | Location | Protection |
|-------|----------|------------|
| Request decrypt | Enclave | Hardware memory encryption |
| Tokenization | Enclave | No data leaves boundary |
| Forward pass | Enclave | Weights never in plaintext outside |
| Response encrypt | Enclave | Client-specific session key |
| Audit logging | Enclave | Sealed + integrity protected |

### 4. Model Loading Strategy

For large models exceeding EPC/enclave memory:

- **Paging**: SGX supports EPC paging (high overhead, ~2-5x slowdown)
- **TDX approach**: Use full VM memory (tens of GB), no paging penalty
- **Model sharding**: Split model across multiple enclaves with secure RPC
- **Quantization**: INT8/INT4 reduces memory footprint 2-4x

## Security Properties

| Property | Mechanism |
|----------|-----------|
| Confidentiality | Memory encryption (AES-XTS / MEE) |
| Integrity | MAC on cache lines + integrity tree |
| Freshness | Monotonic counters for sealed data |
| Authenticity | Remote attestation with signed quotes |
| Forward secrecy | Ephemeral session keys per client |

## Deployment Pattern: Kubernetes + Confidential VMs

```yaml
# Example: Confidential container deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-inference
spec:
  template:
    spec:
      runtimeClassName: kata-cc  # Confidential Containers
      containers:
      - name: inference
        image: registry/model-server:latest
        resources:
          limits:
            sgx.intel.com/epc: "64Mi"  # SGX EPC allocation
        env:
        - name: ATTESTATION_MODE
          value: "dcap"
```

## Performance Budgeting

| Operation | Overhead vs. Plaintext |
|-----------|----------------------|
| TLS termination in enclave | +1-2 ms |
| Model loading (sealed) | +500 ms (one-time) |
| Per-token inference | +5-15% latency |
| Attestation generation | +50-100 ms (cached) |
| Memory encryption | ~2% throughput loss (TDX) |

## Threat Mitigations

1. **Model theft** → Weights sealed, never in host memory plaintext
2. **Prompt leakage** → End-to-end encryption, TLS inside enclave
3. **Tampering** → Integrity verification on every memory access
4. **Rollback** → Sealed data versioned with monotonic counters
5. **Side-channel** → Constant-time operations, ORAM for access patterns
