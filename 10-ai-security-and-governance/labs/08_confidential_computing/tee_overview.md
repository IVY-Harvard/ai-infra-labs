# Trusted Execution Environments (TEE) Overview

## What is a TEE?

A hardware-isolated execution context that protects code and data from
the host OS, hypervisor, and physical attackers. The CPU enforces
confidentiality and integrity of memory within the TEE boundary.

## Technology Comparison

| Feature | Intel SGX | Intel TDX | AMD SEV-SNP |
|---------|-----------|-----------|-------------|
| Granularity | Process (enclave) | VM (trust domain) | VM (guest) |
| Memory Encryption | MEE (128-bit) | MKTME (AES-XTS) | AES-128 SME |
| Max Protected Memory | EPC (~256 MB) | Full VM RAM | Full VM RAM |
| Attestation | EPID / DCAP | TD Quote | SNP Report |
| TCB | CPU + enclave code | CPU + TD module | CPU + firmware |
| OS Trust Required | No | No (host untrusted) | No (hypervisor untrusted) |
| GPU Support | Limited | Emerging (TDX Connect) | Emerging (SEV-TIO) |

## Intel SGX Architecture

```
User Space
┌────────────────────────────────────┐
│  Application (untrusted)           │
│       │                            │
│       ▼                            │
│  ┌──────────────────────┐          │
│  │  Enclave (trusted)   │  ECALL   │
│  │  - Model weights     │◀─────────│
│  │  - Inference logic   │          │
│  │  - Sealed keys       │─────────▶│
│  └──────────────────────┘  OCALL   │
└────────────────────────────────────┘

Hardware
┌────────────────────────────────────┐
│  EPC (Enclave Page Cache)          │
│  - Pages encrypted in DRAM         │
│  - Integrity tree (replay protect) │
│  - Access control via EPCM         │
└────────────────────────────────────┘
```

## Intel TDX Architecture

TDX operates at VM granularity, removing the host OS and VMM from the
trusted computing base:

- **TD Module**: Firmware component managing trust domain lifecycle
- **Secure EPT**: Hardware-enforced page tables isolating TD memory
- **TDCALL/TDVMCALL**: Guest-to-module and guest-to-VMM interfaces
- **TD Quote**: Attestation evidence signed by Intel's quoting enclave

## AMD SEV-SNP

Secure Nested Paging adds integrity protection to SEV's encryption:

- **RMP (Reverse Map Table)**: Tracks page ownership, prevents remapping
- **VMPL (VM Privilege Levels)**: 4 levels within a guest for defense-in-depth
- **SNP Report**: Signed attestation including launch measurement
- **Migration Agent**: Enables live migration without exposing secrets

## Attestation Flow

```
┌──────────┐         ┌───────────┐         ┌──────────────┐
│  Client  │         │  Enclave  │         │  Attestation │
│(verifier)│         │ (prover)  │         │   Service    │
└────┬─────┘         └─────┬─────┘         └──────┬───────┘
     │  1. Challenge (nonce) │                      │
     │─────────────────────▶│                      │
     │                      │  2. Generate quote    │
     │                      │  (measurement+nonce)  │
     │                      │─────────────────────▶│
     │                      │  3. Signed quote      │
     │                      │◀─────────────────────│
     │  4. Return quote      │                      │
     │◀─────────────────────│                      │
     │                                              │
     │  5. Verify signature + check measurements   │
     │─────────────────────────────────────────────▶│
     │  6. Verification result                      │
     │◀─────────────────────────────────────────────│
```

## Practical Limitations

1. **Side channels**: Cache timing, branch prediction, power analysis
2. **Rollback attacks**: Sealed data can be replayed (use monotonic counters)
3. **Iago attacks**: Untrusted OS returns crafted values on system calls
4. **Supply chain**: Must trust CPU vendor's attestation infrastructure
5. **Debugging**: Production enclaves disable debug mode (no inspection)

## When to Use Each Technology

- **SGX**: Single-service secrets (key management, small model inference)
- **TDX**: Full VM workloads (large model serving, multi-process pipelines)
- **SEV-SNP**: Multi-tenant cloud VMs (customer-owned encryption keys)
