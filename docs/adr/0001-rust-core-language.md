# ADR-0001: Rust as Core Protocol Language

**Status:** Accepted
**Date:** 2026-07-27

## Context

Synapse is a P2P protocol requiring: concurrent network I/O (thousands of simultaneous libp2p connections), cryptographic operations (Ed25519 signing, SHA256 hashing, Noise handshake), deterministic execution for auditable inference, and cross-platform support (Linux, macOS, Windows for consumer GPUs).

The runtime layer (vLLM) is Python — that's fixed by the ML ecosystem. But the core protocol, gateway, and swarm orchestration need a systems language.

## Decision

Use **Rust** as the core language. The `vLLM` runtime remains Python. Smart contracts are Solidity.

## Alternatives Considered

### Go
- **Pros:** Excellent concurrency model (goroutines), fast compile times, good P2P libraries (go-libp2p is the reference implementation)
- **Cons:** GC pauses at scale with thousands of connections, weaker FFI with CUDA/nvml libraries for GPU monitoring, runtime overhead for the gateway's zero-allocation hot path
- **Rejected:** The GC unpredictability is unacceptable for a protocol that routes inference requests with P99 latency targets under 500ms

### C++
- **Pros:** Maximum performance, direct CUDA integration, full control over memory layout
- **Cons:** Memory safety risks are critical in a P2P network (buffer overflows in crypto code = node compromise), longer development cycles, ecosystem fragmentation for async I/O
- **Rejected:** The security requirements of a decentralized financial protocol (staking, slashing) make memory safety non-negotiable

### Python (async) for the gateway
- **Pros:** Single language across the stack, fast prototyping
- **Cons:** GIL limits concurrency, no true parallelism for CPU-bound orchestration, runtime overhead for the gateway's per-request processing, no `ed25519-dalek`-grade crypto libraries
- **Rejected:** The gateway is the bottleneck — it needs maximum throughput per core
- **Where used:** ML runtime adapter layer only (vLLM subprocess)

### Rust (chosen)
- **Pros:** Memory safety without GC, zero-cost abstractions, tokio for production async, `ed25519-dalek` and `libp2p` ecosystem, compile-time enforcement of DDD purity boundary (Send + Sync traits), cross-compilation to consumer GPU hosts
- **Cons:** Steeper learning curve, slower compile times, FFI boundary for ML runtime (Unix socket + protobuf instead of in-process calls)

## Consequences

- **Positive:** Memory safety prevents the class of bugs most dangerous in a P2P financial protocol. The borrow checker enforces the domain/infrastructure boundary. Tokio handles thousands of concurrent libp2p connections.
- **Negative:** ML runtime integration requires an IPC boundary (protobuf over Unix socket) rather than in-process calls. Rust's async ecosystem has version churn risk (tokio, libp2p). Contributors need Rust proficiency.
- **Mitigation:** The Rust surface is a single crate (`synapse-core/`). The Python runtime is a separate package (`synapse-runtime/`). Boundary is the `InferencePort` trait + protobuf.
