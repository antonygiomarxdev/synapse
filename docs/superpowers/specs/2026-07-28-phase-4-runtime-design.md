# Phase 4: Inference Runtime — Runtime-Agnostic Adapter Design

**Date:** 2026-07-28
**Status:** Ready for implementation planning
**Issue:** #4

---

## 1. OVERVIEW

### 1.1 Goal

Build the inference runtime layer that bridges the Rust swarm core to any inference engine. Communication is via Unix socket + protobuf through the `InferencePort` trait. V1 backend: vLLM. V2+: llama.cpp, SGLang.

### 1.2 Design Principle: Clean Architecture + DDD

```
Domain (runtime/ports.rs)     ← pure traits, no I/O, no framework deps
    ↑
Infrastructure (infrastructure/) ← Unix socket bridge, protobuf serialization
    ↑
Python subprocess (synapse-runtime/) ← vLLM engine, weight loader, VRAM detection
```

Dependencies always point inward. The domain defines `InferencePort`; infrastructure implements it. The Rust core never imports Python.

### 1.3 Trait Separation

Two distinct traits with different responsibilities:

| Trait | Location | Methods | Consumer |
|---|---|---|---|
| `InferenceEngine` | `swarm/ports.rs` (existing) | `generate()` | Swarm coordinators |
| `InferencePort` | `runtime/ports.rs` (new) | `load()`, `generate()`, `verify()`, `detect_vram()` | Runtime infrastructure |

`InferenceEngine` stays minimal — the swarm only needs `generate()`. `InferencePort` is the adapter contract that any runtime (vLLM, llama.cpp, SGLang) implements.

---

## 2. ARCHITECTURE

### 2.1 Module Structure

```
synapse-core/src/
├── runtime/                              # NEW
│   ├── mod.rs                            #   re-exports
│   ├── ports.rs                          #   InferencePort trait
│   ├── protocol.rs                       #   Bridge value objects
│   └── infrastructure/
│       ├── mod.rs
│       ├── unix_socket_bridge.rs         #   Rust ↔ Python adapter
│       └── proto/
│           └── runtime.rs                #   Compiled from runtime.proto

synapse-core/proto/
├── synapse.proto                         #   UNCHANGED — P2P wire protocol
└── runtime.proto                         #   NEW — bridge messages

synapse-runtime/synapse_runtime/
├── __init__.py
├── protocol.py                           #   Request/Response dataclasses
├── server.py                             #   Unix socket server
├── engine.py                             #   vLLM wrapper
├── loader.py                             #   HF download + SHA256 + expert extraction
├── deterministic.py                      #   seed=0 enforcement
└── auto_assign.py                        #   VRAM detection + assignment
```

### 2.2 Data Flow

```
Swarm Core           InferenceEngine.generate()
    │
    ▼
InferencePort        .load() → .generate() → .verify()
    │
    ▼ (Unix socket + runtime.proto)
server.py            recv LoadModelRequest → engine.load_model()
                     recv GenerateRequest  → engine.generate()
                     recv VerifyHashRequest → loader.verify()
                     recv VramQueryRequest → auto_assign.detect_vram()
```

### 2.3 Protobuf Schema (runtime.proto)

Separate from `synapse.proto` (P2P wire protocol). Bridge-local messages:

```protobuf
syntax = "proto3";
package synapse.proto.runtime;

message LoadModelRequest {
  string model_id = 1;
  repeated uint32 expert_indices = 2;
}

message LoadModelResponse {
  bool success = 1;
  string error = 2;
  uint32 loaded_experts = 3;
}

message GenerateRequest {
  bytes request_id = 1;
  repeated uint32 token_ids = 2;
  uint32 seed = 3;
  uint32 max_tokens = 4;
}

message GenerateResponse {
  bytes request_id = 1;
  repeated uint32 token_ids = 2;
  repeated float log_probs = 3;
  bool finished = 4;
}

message VerifyHashRequest {
  string model_id = 1;
  string expected_sha256 = 2;
}

message VerifyHashResponse {
  bool matches = 1;
  string actual_sha256 = 2;
}

message VramQueryRequest {}

message VramQueryResponse {
  uint32 total_mb = 1;
  uint32 available_mb = 2;
}
```

### 2.4 InferencePort Trait (Rust)

```rust
/// Port for runtime adapters (vLLM, llama.cpp, SGLang).
///
/// Implementations communicate with the Python subprocess via
/// Unix socket + protobuf. The domain depends on this trait only;
/// infrastructure provides the concrete adapter.
pub trait InferencePort {
    fn load(&self, model: &ModelId, experts: &[ExpertId]) -> Result<(), DomainError>;
    fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError>;
    fn verify(&self, model: &ModelId, expected_hash: &str) -> Result<bool, DomainError>;
    fn detect_vram(&self) -> Result<u32, DomainError>;
}
```

---

## 3. COMPONENT DETAILS

### 3.1 Weight Loader (`loader.py`)

- Downloads model from HuggingFace Hub via `huggingface_hub.snapshot_download()`
- Computes SHA256 of downloaded weights
- Extracts expert-specific checkpoint shards from safetensors files
- Validates expert indices against model metadata

### 3.2 vLLM Engine (`engine.py`)

- Wraps `vllm.LLM` for model loading and generation
- `load_model(model_id, expert_indices)`: loads the model with specified experts active
- `generate(prompt_tokens, seed, max_tokens)`: runs inference, returns token IDs + logprobs
- Handles vLLM-specific configuration (tensor parallelism, GPU memory fraction)

### 3.3 Deterministic Mode (`deterministic.py`)

- Enforces `seed=0` for deterministic generation
- Locks precision settings (no mixed precision variance)
- Verification: two runs with same input → identical output tokens and logprobs

### 3.4 Auto-Assign (`auto_assign.py`)

- Detects VRAM via `torch.cuda.mem_get_info()` (or `nvidia-smi` fallback)
- Calculates optimal expert count: `floor((available_vram - shared_params) / expert_size)`
- Selects runtime: vLLM (GPU), llama.cpp (CPU/GPU hybrid, V2+)
- Returns recommended expert indices based on DHT co-activation heatmap (V2)

### 3.5 Unix Socket Server (`server.py`)

- Binds to Unix domain socket (`/tmp/synapse-runtime.sock`)
- Receives protobuf-encoded `LoadModelRequest` / `GenerateRequest` / `VerifyHashRequest` / `VramQueryRequest`
- Dispatches to engine/loader/deterministic/auto_assign
- Returns protobuf-encoded responses
- Handles connection drops, reconnection, and graceful shutdown

### 3.6 UnixSocketBridge (Rust adapter)

- Opens Unix socket connection to Python subprocess
- Serializes domain types (`ModelId`, `ExpertId`, `InferenceRequest`) → runtime.proto messages
- Deserializes runtime.proto responses → domain types (`InferenceOutput`)
- Implements `InferencePort` trait
- Handles socket errors, timeouts, and reconnection

---

## 4. ERROR HANDLING

| Error Scenario | Python Layer | Rust Adapter | Domain Error |
|---|---|---|---|
| Model not downloaded | `FileNotFoundError` → `LoadModelResponse.error` | Maps to `DomainError::ModelNotFound` | Propagated to caller |
| SHA256 mismatch | `VerifyHashResponse.matches=false` | Returns `Ok(false)` | Caller decides action |
| vLLM OOM | `torch.cuda.OutOfMemoryError` | Maps to `DomainError::StorageError` | Node marks unavailable |
| Socket disconnect | `BrokenPipeError` | Reconnection with exponential backoff | Transient; retry |
| Invalid protobuf | `DecodeError` | Maps to `DomainError::InvalidToken` | Request dropped |
| Timeout | `socket.timeout` | Maps to `DomainError::StorageError` | Retry or fail |

---

## 5. TESTING STRATEGY

### 5.1 Unit Tests (CI — always run)

- **Python:** Mock vLLM, test protocol serialization/deserialization, error paths, VRAM detection mock, seed enforcement logic
- **Rust:** Test trait contracts, protocol value object validation, socket message framing

### 5.2 Integration Tests (Local only — `@pytest.mark.gpu`)

- End-to-end: Rust UnixSocketBridge ↔ Python server ↔ vLLM with DeepSeek-V2 Lite (lightest model, 64 experts at ~0.15GB each)
- SHA256 verification against official HuggingFace hash
- Determinism: two runs with seed=0 → identical output
- Reconnection: kill Python process, verify Rust adapter recovers

### 5.3 What Does NOT Run in CI

- Any test requiring GPU (`@pytest.mark.gpu`)
- Model download tests (network + disk)
- Multi-node swarm integration (tested with mocks)

---

## 6. PHASES

| Phase | Name | Deliverable | Depends On |
|---|---|---|---|
| A | Contract | `InferencePort` trait + `runtime.proto` + Python protocol dataclasses + VRAM detection | Nothing |
| B | Engine | vLLM wrapper + weight loader + deterministic mode + auto-assign | Phase A |
| C | Bridge | Unix socket server (Python) + UnixSocketBridge adapter (Rust) + end-to-end integration | Phase B |

Each phase contains 3-6 bite-sized TDD tasks. Every task produces an independently testable unit.
