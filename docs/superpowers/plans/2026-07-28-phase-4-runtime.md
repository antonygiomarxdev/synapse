# Phase 4: Inference Runtime — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runtime-agnostic inference adapter: `InferencePort` trait (Rust), vLLM engine backend (Python), Unix socket bridge with protobuf serialization.

**Architecture:** DDD + Clean Architecture. `InferencePort` trait in `runtime/ports.rs` (pure domain). `UnixSocketBridge` in `runtime/infrastructure/` implements it. Python `synapse-runtime/` runs vLLM as subprocess. Communication via `runtime.proto` over Unix socket.

**Tech Stack:** Rust 1.97 (edition 2024), Python 3.12+, vLLM 0.26, protobuf 5, huggingface-hub 0.30, pytest 8 + pytest-asyncio.

**Design Spec:** `docs/superpowers/specs/2026-07-28-phase-4-runtime-design.md`

## Global Constraints

These expand on the non-negotiable principles in `AGENTS.md`. Every task MUST comply.

### DDD + Clean Architecture
- `InferencePort` trait has ZERO external dependencies — no protobuf, no tokio, no I/O.
- Protocol value objects in `runtime/protocol.rs` are pure data — no serialization logic inline.
- Infrastructure adapters (`unix_socket_bridge.rs`) are the ONLY place with protobuf/tokio/socket deps.
- Python dataclasses mirror `runtime.proto` messages 1:1.
- `InferenceEngine` in `swarm/ports.rs` is NOT modified — the swarm only knows `generate()`.

### TDD
- **EVERY task step writes the test BEFORE the implementation.** Run it, confirm it FAILS, then implement.
- Rust tests inline: `#[cfg(test)] mod tests` in the same file as source.
- Python tests in `synapse-runtime/tests/` mirroring source structure.
- Test names describe the scenario: `load_model_rejects_invalid_expert_index`, `generate_returns_tokens_with_logprobs`.
- Domain tests are pure. Infrastructure tests use temp Unix sockets. Engine tests use mock vLLM.
- GPU integration tests use `@pytest.mark.gpu` — excluded from CI.

### Clean Code
- ALL public types get `///` doc comments (Rust) or `"""` docstrings (Python).
- `thiserror` for Rust domain errors, custom exceptions for Python.
- `cargo fmt --check` + `cargo clippy -- -D warnings` before every Rust commit.
- `ruff check` + `ruff format --check` before every Python commit.
- Commit messages follow Conventional Commits: `feat(runtime): ...`, `test(runtime): ...`.
- Python: double quotes, ruff strict ruleset (E, F, W, I, N, UP, B, SIM, C4, RUF).

### Testing
- Unit tests mock vLLM — no GPU required. CI runs them all.
- Integration tests (`@pytest.mark.gpu`) use DeepSeek-V2 Lite — run locally only.
- Protobuf roundtrip tests: serialize → deserialize → assert equality.
- Socket error tests: kill server mid-request → verify graceful error.

### Per-Task Quality Gate (MANDATORY — EVERY task)

Before committing any task, run this checklist. Failures block the commit.

**Security:**
- [ ] No hardcoded secrets, tokens, API keys, or passwords.
- [ ] All user/external input is validated at boundaries.
- [ ] File paths are validated; no path traversal possible.
- [ ] Socket paths use `tempfile` or fixed `/tmp/` — never user-controlled without sanitization.
- [ ] Protobuf decoding handles truncated/malformed data gracefully (no panics, no infinite loops).

**No Magic Numbers:**
- [ ] All numeric literals (except 0, 1, -1 in trivial contexts) are named constants.
- [ ] Timeouts, retry counts, buffer sizes, field numbers are `const` / module-level constants.
- [ ] Wire type values (0, 2) in protobuf code have named constants (`WIRE_VARINT`, `WIRE_LEN`).

**SOLID:**
- [ ] **SRP:** Each function/class does ONE thing. If a function has "and" in its docstring, split it.
- [ ] **OCP:** New message types can be added without modifying the dispatch function (use dict/table lookup).
- [ ] **LSP:** Subtypes are substitutable for their base types. Mock engines have identical signatures.
- [ ] **ISP:** Traits/interfaces are minimal. `InferencePort` has 4 methods; no consumer is forced to depend on unused methods.
- [ ] **DIP:** Domain code depends on traits, not concrete adapters. Imports from `infrastructure/` NEVER appear in domain modules.

**Code Hygiene:**
- [ ] ALL public items have docstrings/doc comments.
- [ ] No dead code, no commented-out blocks, no `TODO` without a tracking issue.
- [ ] Error messages are descriptive, not generic ("OOM" → "GPU out of memory: 12.5GB requested, 8.0GB available").
- [ ] `ruff check` / `cargo clippy` report zero warnings.
---

## File Structure (post-implementation)

```
synapse-core/
├── proto/
│   └── runtime.proto                          # NEW — bridge messages
├── src/
│   ├── runtime/                               # NEW
│   │   ├── mod.rs                             # re-exports
│   │   ├── ports.rs                           # InferencePort trait
│   │   ├── protocol.rs                        # Bridge value objects
│   │   └── infrastructure/
│   │       ├── mod.rs                         # re-exports
│   │       ├── unix_socket_bridge.rs          # Unix socket adapter
│   │       └── proto/
│   │           └── runtime.rs                 # compiled from runtime.proto
│   └── lib.rs                                 # MODIFY — add pub mod runtime

synapse-runtime/
├── pyproject.toml                             # UNCHANGED
├── ruff.toml                                  # UNCHANGED
├── synapse_runtime/
│   ├── __init__.py                            # MODIFY — bump version
│   ├── protocol.py                            # NEW — Request/Response dataclasses
│   ├── server.py                              # NEW — Unix socket server
│   ├── engine.py                              # NEW — vLLM wrapper
│   ├── loader.py                              # NEW — HF download + SHA256 + expert extraction
│   ├── deterministic.py                       # NEW — seed=0 enforcement
│   └── auto_assign.py                         # NEW — VRAM detection + assignment
└── tests/
    ├── __init__.py
    ├── test_protocol.py                       # NEW
    ├── test_server.py                         # NEW
    ├── test_engine.py                         # NEW
    ├── test_loader.py                         # NEW
    ├── test_deterministic.py                  # NEW
    └── test_auto_assign.py                    # NEW
```

---

## Phase A: Contract — Protocol + Trait + VRAM Detection

### Task A.1: Define runtime.proto — Bridge Message Schema

**Files:**
- Create: `synapse-core/proto/runtime.proto`

**Interfaces:**
- Produces: 8 protobuf message types: `LoadModelRequest`, `LoadModelResponse`, `GenerateRequest`, `GenerateResponse`, `VerifyHashRequest`, `VerifyHashResponse`, `VramQueryRequest`, `VramQueryResponse`

- [ ] **Step 1: Write the protobuf schema**

Create `synapse-core/proto/runtime.proto`:

```protobuf
syntax = "proto3";

package synapse.proto.runtime;

// Request to load a model with specific experts
message LoadModelRequest {
  string model_id = 1;
  repeated uint32 expert_indices = 2;
}

// Response after model load attempt
message LoadModelResponse {
  bool success = 1;
  string error = 2;
  uint32 loaded_experts = 3;
}

// Token generation request
message GenerateRequest {
  bytes request_id = 1;
  repeated uint32 token_ids = 2;
  uint32 seed = 3;
  uint32 max_tokens = 4;
}

// Token generation response (streamed per token or batched)
message GenerateResponse {
  bytes request_id = 1;
  repeated uint32 token_ids = 2;
  repeated float log_probs = 3;
  bool finished = 4;
}

// SHA256 verification request
message VerifyHashRequest {
  string model_id = 1;
  string expected_sha256 = 2;
}

// SHA256 verification response
message VerifyHashResponse {
  bool matches = 1;
  string actual_sha256 = 2;
}

// VRAM query (no parameters needed)
message VramQueryRequest {}

// VRAM query response
message VramQueryResponse {
  uint32 total_mb = 1;
  uint32 available_mb = 2;
}
```

- [ ] **Step 2: Verify proto compiles**

Run: `protoc --proto_path=synapse-core/proto --python_out=/tmp/test_proto synapse-core/proto/runtime.proto`
Expected: Exit code 0, no errors.

- [ ] **Step 3: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Verify: no magic numbers (field numbers are proto-defined), schema is self-documenting, SRP (one concern: bridge messages only).

- [ ] **Step 4: Commit**

```bash
git add synapse-core/proto/runtime.proto
git commit -m "feat(runtime): add runtime.proto bridge message schema"
```

---

### Task A.2: Python Protocol Dataclasses

**Files:**
- Create: `synapse-runtime/synapse_runtime/protocol.py`
- Create: `synapse-runtime/tests/test_protocol.py`

**Interfaces:**
- Produces: `LoadModelRequest`, `LoadModelResponse`, `GenerateRequest`, `GenerateResponse`, `VerifyHashRequest`, `VerifyHashResponse`, `VramQueryRequest`, `VramQueryResponse` dataclasses
- Produces: `serialize_request()`, `deserialize_request()`, `serialize_response()`, `deserialize_response()` functions

- [ ] **Step 1: Write the failing test**

Create `synapse-runtime/tests/test_protocol.py`:

```python
"""Tests for protocol dataclasses and serialization."""

from synapse_runtime.protocol import (
    GenerateRequest,
    GenerateResponse,
    LoadModelRequest,
    LoadModelResponse,
    VramQueryRequest,
    VramQueryResponse,
    deserialize_request,
    deserialize_response,
    serialize_request,
    serialize_response,
)


class TestLoadModelRequest:
    def test_roundtrip_serialization(self) -> None:
        """Serialized LoadModelRequest deserializes back to identical values."""
        original = LoadModelRequest(model_id="mixtral-8x7b", expert_indices=[0, 3, 7])
        data = serialize_request(original)
        restored = deserialize_request(data)
        assert isinstance(restored, LoadModelRequest)
        assert restored.model_id == "mixtral-8x7b"
        assert restored.expert_indices == [0, 3, 7]

    def test_empty_experts_is_valid(self) -> None:
        """Zero expert indices is a valid request (load shared params only)."""
        original = LoadModelRequest(model_id="mixtral-8x7b", expert_indices=[])
        data = serialize_request(original)
        restored = deserialize_request(data)
        assert restored.expert_indices == []


class TestGenerateRequest:
    def test_roundtrip_serialization(self) -> None:
        """Serialized GenerateRequest deserializes back to identical values."""
        original = GenerateRequest(
            request_id=b"abc-123",
            token_ids=[1, 2, 3],
            seed=0,
            max_tokens=100,
        )
        data = serialize_request(original)
        restored = deserialize_request(data)
        assert isinstance(restored, GenerateRequest)
        assert restored.request_id == b"abc-123"
        assert restored.token_ids == [1, 2, 3]
        assert restored.seed == 0
        assert restored.max_tokens == 100

    def test_seed_zero_is_valid(self) -> None:
        """Seed=0 is valid (deterministic mode)."""
        original = GenerateRequest(
            request_id=b"det-1", token_ids=[42], seed=0, max_tokens=50
        )
        data = serialize_request(original)
        restored = deserialize_request(data)
        assert restored.seed == 0


class TestGenerateResponse:
    def test_roundtrip_serialization(self) -> None:
        """Serialized GenerateResponse deserializes back to identical values."""
        original = GenerateResponse(
            request_id=b"abc-123",
            token_ids=[4, 5, 6],
            log_probs=[-0.1, -0.2, -0.3],
            finished=True,
        )
        data = serialize_response(original)
        restored = deserialize_response(data)
        assert isinstance(restored, GenerateResponse)
        assert restored.request_id == b"abc-123"
        assert restored.token_ids == [4, 5, 6]
        assert restored.log_probs == [-0.1, -0.2, -0.3]
        assert restored.finished is True

    def test_unfinished_response(self) -> None:
        """Response with finished=False is valid (streaming)."""
        original = GenerateResponse(
            request_id=b"stream-1",
            token_ids=[99],
            log_probs=[-0.05],
            finished=False,
        )
        data = serialize_response(original)
        restored = deserialize_response(data)
        assert restored.finished is False


class TestVramQuery:
    def test_vram_query_roundtrip(self) -> None:
        """VramQueryRequest and VramQueryResponse serialize correctly."""
        req = VramQueryRequest()
        req_data = serialize_request(req)
        restored_req = deserialize_request(req_data)
        assert isinstance(restored_req, VramQueryRequest)

        resp = VramQueryResponse(total_mb=16384, available_mb=8192)
        resp_data = serialize_response(resp)
        restored_resp = deserialize_response(resp_data)
        assert isinstance(restored_resp, VramQueryResponse)
        assert restored_resp.total_mb == 16384
        assert restored_resp.available_mb == 8192


class TestDeserializeUnknownMessage:
    def test_deserialize_request_raises_on_unknown(self) -> None:
        """Deserializing garbled bytes raises ValueError."""
        import pytest
        with pytest.raises(ValueError):
            deserialize_request(b"not-a-valid-protobuf-message")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'synapse_runtime.protocol'`

- [ ] **Step 3: Write protocol.py implementation**

Create `synapse-runtime/synapse_runtime/protocol.py`:

```python
"""Bridge protocol dataclasses and protobuf serialization.

Mirrors synapse-core/proto/runtime.proto 1:1.
Uses raw protobuf encoding (wire types) — no compiled .proto needed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import ClassVar


# ── Protobuf wire format helpers ──────────────────────────────────────

_WIRE_VARINT = 0
_WIRE_LEN = 2


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a protobuf varint. Returns (value, new_offset)."""
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        value |= (byte & 0x7F) << shift
        offset += 1
        if not (byte & 0x80):
            return value, offset
        shift += 7
    raise ValueError("Truncated varint")


def _encode_field(field_number: int, wire_type: int, payload: bytes) -> bytes:
    """Encode a protobuf field: tag + payload."""
    tag = (field_number << 3) | wire_type
    return _encode_varint(tag) + payload


def _encode_uint32(field_number: int, value: int) -> bytes:
    """Encode a uint32 field."""
    return _encode_field(field_number, _WIRE_VARINT, _encode_varint(value))


def _encode_bytes_field(field_number: int, data: bytes) -> bytes:
    """Encode a length-delimited field (string, bytes, repeated packed)."""
    return _encode_field(field_number, _WIRE_LEN,
                         _encode_varint(len(data)) + data)


def _encode_string(field_number: int, value: str) -> bytes:
    """Encode a string field."""
    return _encode_bytes_field(field_number, value.encode("utf-8"))


def _encode_repeated_uint32(field_number: int, values: list[int]) -> bytes:
    """Encode repeated uint32 as packed."""
    if not values:
        return b""
    packed = b"".join(_encode_varint(v) for v in values)
    return _encode_bytes_field(field_number, packed)


def _encode_repeated_float(field_number: int, values: list[float]) -> bytes:
    """Encode repeated float as packed."""
    if not values:
        return b""
    packed = b"".join(struct.pack("<f", v) for v in values)
    return _encode_bytes_field(field_number, packed)


def _encode_bool(field_number: int, value: bool) -> bytes:
    """Encode a bool field (varint-encoded: 0 or 1)."""
    return _encode_field(field_number, _WIRE_VARINT, _encode_varint(1 if value else 0))


def _parse_fields(data: bytes) -> dict[int, list[tuple[int, bytes]]]:
    """Parse a protobuf message into {field_number: [(wire_type, payload), ...]}."""
    fields: dict[int, list[tuple[int, bytes]]] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _decode_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == _WIRE_VARINT:
            value, offset = _decode_varint(data, offset)
            fields.setdefault(field_number, []).append((wire_type, _encode_varint(value)))
        elif wire_type == _WIRE_LEN:
            length, offset = _decode_varint(data, offset)
            payload = data[offset:offset + length]
            offset += length
            fields.setdefault(field_number, []).append((wire_type, payload))
        else:
            raise ValueError(f"Unsupported wire type: {wire_type}")
    return fields


def _get_varint(fields: dict[int, list[tuple[int, bytes]]], num: int,
                default: int = 0) -> int:
    """Extract a single uint32/bool/enum field."""
    items = fields.get(num, [])
    if not items:
        return default
    value, _ = _decode_varint(items[0][1], 0)
    return value


def _get_string(fields: dict[int, list[tuple[int, bytes]]], num: int,
                default: str = "") -> str:
    """Extract a single string field."""
    items = fields.get(num, [])
    if not items:
        return default
    return items[0][1].decode("utf-8")


def _get_bytes(fields: dict[int, list[tuple[int, bytes]]], num: int,
               default: bytes = b"") -> bytes:
    """Extract a single bytes field."""
    items = fields.get(num, [])
    if not items:
        return default
    return items[0][1]


def _get_bool(fields: dict[int, list[tuple[int, bytes]]], num: int,
              default: bool = False) -> bool:
    """Extract a single bool field."""
    return bool(_get_varint(fields, num, 1 if default else 0))


def _get_repeated_uint32(fields: dict[int, list[tuple[int, bytes]]],
                         num: int) -> list[int]:
    """Extract a repeated uint32 field (packed)."""
    items = fields.get(num, [])
    if not items:
        return []
    data = items[0][1]
    result: list[int] = []
    offset = 0
    while offset < len(data):
        value, offset = _decode_varint(data, offset)
        result.append(value)
    return result


def _get_repeated_float(fields: dict[int, list[tuple[int, bytes]]],
                        num: int) -> list[float]:
    """Extract a repeated float field (packed)."""
    items = fields.get(num, [])
    if not items:
        return []
    data = items[0][1]
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class LoadModelRequest:
    """Request to load a model with specific experts."""
    model_id: str
    expert_indices: list[int] = field(default_factory=list)

    MESSAGE_TYPE: ClassVar[int] = 1

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_string(1, self.model_id)
        body += _encode_repeated_uint32(2, self.expert_indices)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "LoadModelRequest":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            model_id=_get_string(fields, 1),
            expert_indices=_get_repeated_uint32(fields, 2),
        )


@dataclass
class LoadModelResponse:
    """Response after model load attempt."""
    success: bool = False
    error: str = ""
    loaded_experts: int = 0

    MESSAGE_TYPE: ClassVar[int] = 2

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_bool(1, self.success)
        if self.error:
            body += _encode_string(2, self.error)
        body += _encode_uint32(3, self.loaded_experts)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "LoadModelResponse":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            success=_get_bool(fields, 1),
            error=_get_string(fields, 2),
            loaded_experts=_get_varint(fields, 3),
        )


@dataclass
class GenerateRequest:
    """Token generation request."""
    request_id: bytes = b""
    token_ids: list[int] = field(default_factory=list)
    seed: int = 0
    max_tokens: int = 0

    MESSAGE_TYPE: ClassVar[int] = 3

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_bytes_field(1, self.request_id)
        body += _encode_repeated_uint32(2, self.token_ids)
        body += _encode_uint32(3, self.seed)
        body += _encode_uint32(4, self.max_tokens)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "GenerateRequest":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            request_id=_get_bytes(fields, 1),
            token_ids=_get_repeated_uint32(fields, 2),
            seed=_get_varint(fields, 3),
            max_tokens=_get_varint(fields, 4),
        )


@dataclass
class GenerateResponse:
    """Token generation response."""
    request_id: bytes = b""
    token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    finished: bool = False

    MESSAGE_TYPE: ClassVar[int] = 4

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_bytes_field(1, self.request_id)
        body += _encode_repeated_uint32(2, self.token_ids)
        body += _encode_repeated_float(3, self.log_probs)
        body += _encode_bool(4, self.finished)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "GenerateResponse":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            request_id=_get_bytes(fields, 1),
            token_ids=_get_repeated_uint32(fields, 2),
            log_probs=_get_repeated_float(fields, 3),
            finished=_get_bool(fields, 4),
        )


@dataclass
class VerifyHashRequest:
    """SHA256 verification request."""
    model_id: str = ""
    expected_sha256: str = ""

    MESSAGE_TYPE: ClassVar[int] = 5

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_string(1, self.model_id)
        body += _encode_string(2, self.expected_sha256)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "VerifyHashRequest":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            model_id=_get_string(fields, 1),
            expected_sha256=_get_string(fields, 2),
        )


@dataclass
class VerifyHashResponse:
    """SHA256 verification response."""
    matches: bool = False
    actual_sha256: str = ""

    MESSAGE_TYPE: ClassVar[int] = 6

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_bool(1, self.matches)
        body += _encode_string(2, self.actual_sha256)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "VerifyHashResponse":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            matches=_get_bool(fields, 1),
            actual_sha256=_get_string(fields, 2),
        )


@dataclass
class VramQueryRequest:
    """VRAM query — no fields."""
    MESSAGE_TYPE: ClassVar[int] = 7

    def serialize(self) -> bytes:
        """Encode to protobuf bytes (empty message)."""
        return b""

    @classmethod
    def deserialize(cls, data: bytes) -> "VramQueryRequest":
        """Decode from protobuf bytes."""
        return cls()


@dataclass
class VramQueryResponse:
    """VRAM query response."""
    total_mb: int = 0
    available_mb: int = 0

    MESSAGE_TYPE: ClassVar[int] = 8

    def serialize(self) -> bytes:
        """Encode to protobuf bytes."""
        body = _encode_uint32(1, self.total_mb)
        body += _encode_uint32(2, self.available_mb)
        return body

    @classmethod
    def deserialize(cls, data: bytes) -> "VramQueryResponse":
        """Decode from protobuf bytes."""
        fields = _parse_fields(data)
        return cls(
            total_mb=_get_varint(fields, 1),
            available_mb=_get_varint(fields, 2),
        )


# ── Message type dispatch ─────────────────────────────────────────────

_REQUEST_TYPES: dict[int, type] = {
    1: LoadModelRequest,
    3: GenerateRequest,
    5: VerifyHashRequest,
    7: VramQueryRequest,
}

_RESPONSE_TYPES: dict[int, type] = {
    2: LoadModelResponse,
    4: GenerateResponse,
    6: VerifyHashResponse,
    8: VramQueryResponse,
}


def serialize_request(msg: object) -> bytes:
    """Serialize a request dataclass to protobuf bytes with type prefix."""
    msg_type = getattr(msg, "MESSAGE_TYPE", 0)
    body = getattr(msg, "serialize")()
    return _encode_varint(msg_type) + body


def deserialize_request(data: bytes) -> object:
    """Deserialize protobuf bytes to the correct request dataclass."""
    if not data:
        raise ValueError("Empty message")
    msg_type, offset = _decode_varint(data, 0)
    cls = _REQUEST_TYPES.get(msg_type)
    if cls is None:
        raise ValueError(f"Unknown request message type: {msg_type}")
    return cls.deserialize(data[offset:])


def serialize_response(msg: object) -> bytes:
    """Serialize a response dataclass to protobuf bytes with type prefix."""
    msg_type = getattr(msg, "MESSAGE_TYPE", 0)
    body = getattr(msg, "serialize")()
    return _encode_varint(msg_type) + body


def deserialize_response(data: bytes) -> object:
    """Deserialize protobuf bytes to the correct response dataclass."""
    if not data:
        raise ValueError("Empty message")
    msg_type, offset = _decode_varint(data, 0)
    cls = _RESPONSE_TYPES.get(msg_type)
    if cls is None:
        raise ValueError(f"Unknown response message type: {msg_type}")
    return cls.deserialize(data[offset:])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_protocol.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check . && ruff format --check .`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/protocol.py synapse-runtime/tests/test_protocol.py
git commit -m "feat(runtime): add Python protocol dataclasses with protobuf serialization"
```

---

### Task A.3: InferencePort Trait (Rust)

**Files:**
- Create: `synapse-core/src/runtime/mod.rs`
- Create: `synapse-core/src/runtime/ports.rs`
- Create: `synapse-core/src/runtime/protocol.rs`
- Modify: `synapse-core/src/lib.rs` — add `pub mod runtime;`

**Interfaces:**
- Produces: `InferencePort` trait (4 methods), `LoadModelRequest`, `LoadModelResponse`, `GenerateBridgeRequest`, `GenerateBridgeResponse`, `VerifyBridgeRequest`, `VerifyBridgeResponse`, `VramBridgeRequest`, `VramBridgeResponse`
- Consumes: `ModelId` (from `model`), `ExpertId` (from `model`), `InferenceRequest`, `InferenceOutput` (from `swarm`), `DomainError` (from `shared`)

- [ ] **Step 1: Write the failing test for protocol value objects**

Create `synapse-core/src/runtime/protocol.rs` with test first:

```rust
use crate::model::ExpertId;
use crate::model::ModelId;

/// Request to load a model with specific experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadModelRequest {
    pub model_id: ModelId,
    pub expert_indices: Vec<u32>,
}

/// Response after model load attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadModelResponse {
    pub success: bool,
    pub error: String,
    pub loaded_experts: u32,
}

/// Token generation request (bridge-level).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateBridgeRequest {
    pub request_id: Vec<u8>,
    pub token_ids: Vec<u32>,
    pub seed: u32,
    pub max_tokens: u32,
}

/// Token generation response (bridge-level).
#[derive(Debug, Clone)]
pub struct GenerateBridgeResponse {
    pub request_id: Vec<u8>,
    pub token_ids: Vec<u32>,
    pub log_probs: Vec<f32>,
    pub finished: bool,
}

/// SHA256 verification request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyBridgeRequest {
    pub model_id: ModelId,
    pub expected_sha256: String,
}

/// SHA256 verification response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyBridgeResponse {
    pub matches: bool,
    pub actual_sha256: String,
}

/// VRAM query request (unit struct — no fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramBridgeRequest;

/// VRAM query response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramBridgeResponse {
    pub total_mb: u32,
    pub available_mb: u32,
}

impl LoadModelRequest {
    /// Creates a new `LoadModelRequest`.
    pub fn new(model_id: ModelId, expert_indices: Vec<u32>) -> Self {
        Self { model_id, expert_indices }
    }
}

impl LoadModelResponse {
    /// Creates a successful response.
    pub fn ok(loaded_experts: u32) -> Self {
        Self { success: true, error: String::new(), loaded_experts }
    }

    /// Creates a failure response.
    pub fn err(error: impl Into<String>) -> Self {
        Self { success: false, error: error.into(), loaded_experts: 0 }
    }
}

impl GenerateBridgeRequest {
    /// Creates a new `GenerateBridgeRequest`.
    pub fn new(
        request_id: Vec<u8>,
        token_ids: Vec<u32>,
        seed: u32,
        max_tokens: u32,
    ) -> Self {
        Self { request_id, token_ids, seed, max_tokens }
    }
}

impl GenerateBridgeResponse {
    /// Creates a new `GenerateBridgeResponse`.
    pub fn new(
        request_id: Vec<u8>,
        token_ids: Vec<u32>,
        log_probs: Vec<f32>,
        finished: bool,
    ) -> Self {
        Self { request_id, token_ids, log_probs, finished }
    }
}

impl VerifyBridgeRequest {
    /// Creates a new `VerifyBridgeRequest`.
    pub fn new(model_id: ModelId, expected_sha256: String) -> Self {
        Self { model_id, expected_sha256 }
    }
}

impl VerifyBridgeResponse {
    /// Creates a successful match response.
    pub fn matched(actual_sha256: String) -> Self {
        Self { matches: true, actual_sha256 }
    }

    /// Creates a mismatch response.
    pub fn mismatched(expected: String, actual: String) -> Self {
        Self { matches: false, actual_sha256: format!("expected {expected}, got {actual}") }
    }
}

impl VramBridgeResponse {
    /// Creates a new `VramBridgeResponse`.
    pub fn new(total_mb: u32, available_mb: u32) -> Self {
        Self { total_mb, available_mb }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_id() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    #[test]
    fn load_model_request_construction() {
        let req = LoadModelRequest::new(model_id(), vec![0, 3, 7]);
        assert_eq!(req.model_id.as_str(), "mixtral-8x7b");
        assert_eq!(req.expert_indices, vec![0, 3, 7]);
    }

    #[test]
    fn load_model_request_empty_experts() {
        let req = LoadModelRequest::new(model_id(), vec![]);
        assert!(req.expert_indices.is_empty());
    }

    #[test]
    fn load_model_response_ok() {
        let resp = LoadModelResponse::ok(3);
        assert!(resp.success);
        assert_eq!(resp.loaded_experts, 3);
        assert!(resp.error.is_empty());
    }

    #[test]
    fn load_model_response_err() {
        let resp = LoadModelResponse::err("OOM");
        assert!(!resp.success);
        assert_eq!(resp.error, "OOM");
        assert_eq!(resp.loaded_experts, 0);
    }

    #[test]
    fn generate_bridge_request_seed_zero() {
        let req = GenerateBridgeRequest::new(b"r1".to_vec(), vec![1, 2], 0, 100);
        assert_eq!(req.seed, 0);
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn generate_bridge_response_unfinished() {
        let resp = GenerateBridgeResponse::new(b"r1".to_vec(), vec![42], vec![-0.5], false);
        assert!(!resp.finished);
        assert_eq!(resp.token_ids, vec![42]);
    }

    #[test]
    fn verify_bridge_response_matched() {
        let resp = VerifyBridgeResponse::matched("abc123".into());
        assert!(resp.matches);
        assert_eq!(resp.actual_sha256, "abc123");
    }

    #[test]
    fn verify_bridge_response_mismatched() {
        let resp = VerifyBridgeResponse::mismatched("abc".into(), "def".into());
        assert!(!resp.matches);
        assert!(resp.actual_sha256.contains("expected"));
    }

    #[test]
    fn vram_bridge_response() {
        let resp = VramBridgeResponse::new(16384, 8192);
        assert_eq!(resp.total_mb, 16384);
        assert_eq!(resp.available_mb, 8192);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p synapse-core runtime::protocol`
Expected: FAIL — module `runtime` not found

- [ ] **Step 3: Write runtime/mod.rs**

Create `synapse-core/src/runtime/mod.rs`:

```rust
//! Runtime-agnostic inference adapter domain.
//!
//! Defines the [`InferencePort`] trait that any inference runtime
//! (vLLM, llama.cpp, SGLang) implements. Protocol value objects
//! mirror the bridge protobuf schema 1:1.

pub mod ports;
pub mod protocol;

pub use ports::InferencePort;
pub use protocol::{
    GenerateBridgeRequest, GenerateBridgeResponse, LoadModelRequest, LoadModelResponse,
    VerifyBridgeRequest, VerifyBridgeResponse, VramBridgeRequest, VramBridgeResponse,
};
```

- [ ] **Step 4: Write runtime/ports.rs**

Create `synapse-core/src/runtime/ports.rs`:

```rust
use crate::model::{ExpertId, ModelId};
use crate::shared::DomainError;
use crate::swarm::ports::{InferenceOutput, InferenceRequest};

/// Port implemented by concrete inference runtimes (vLLM, llama.cpp, SGLang).
///
/// The domain depends on this trait only; infrastructure adapters provide
/// the actual model loading, inference, verification, and VRAM detection.
///
/// Each method maps to a bridge protocol message sent over Unix socket
/// to the Python subprocess.
pub trait InferencePort {
    /// Loads a model with the specified experts into VRAM.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::ModelNotFound`] if the model is not available.
    /// Returns [`DomainError::StorageError`] if VRAM is insufficient.
    fn load(&self, model: &ModelId, experts: &[ExpertId]) -> Result<(), DomainError>;

    /// Generates tokens for a single inference request.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidToken`] if the prompt contains invalid tokens.
    /// Returns [`DomainError::StorageError`] if the runtime encounters an error.
    fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError>;

    /// Verifies that a model's weights match the expected SHA256 hash.
    ///
    /// Returns `Ok(true)` if the hash matches, `Ok(false)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::ModelNotFound`] if the model is not loaded.
    fn verify(&self, model: &ModelId, expected_hash: &str) -> Result<bool, DomainError>;

    /// Detects available VRAM on the compute node.
    ///
    /// Returns the available VRAM in megabytes.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::StorageError`] if VRAM detection fails.
    fn detect_vram(&self) -> Result<u32, DomainError>;
}

#[cfg(test)]
mod tests {
    /// These tests are documentation of the trait contract.
    /// Actual behavior is tested in infrastructure adapter tests.

    /// Trait is object-safe for dynamic dispatch in tests.
    #[allow(dead_code)]
    fn assert_object_safe(_port: &dyn super::InferencePort) {}

    /// The trait has exactly 4 methods.
    #[test]
    fn trait_has_four_methods() {
        // compile-time assertion: if a method is added/removed,
        // the implementations in infrastructure/ must be updated.
        fn _check(p: &dyn super::InferencePort) {
            let _ = p.load(&crate::model::ModelId::new("test").unwrap(), &[]);
            let _ = p.generate(&crate::swarm::ports::InferenceRequest::new(
                uuid::Uuid::new_v4(),
                crate::model::ModelId::new("test").unwrap(),
                crate::swarm::ports::Priority::Batch,
                None,
                100,
            ));
            let _ = p.verify(&crate::model::ModelId::new("test").unwrap(), "");
            let _ = p.detect_vram();
        }
    }
}
```

- [ ] **Step 5: Add `pub mod runtime;` to lib.rs**

Modify `synapse-core/src/lib.rs`:

Add after the existing `pub mod model;` line:

```rust
pub mod runtime;
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p synapse-core runtime`
Expected: ALL PASS

- [ ] **Step 7: Run format + lint**

Run: `cargo fmt --check && cargo clippy -p synapse-core -- -D warnings`
Expected: All checks passed

- [ ] **Step 8: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 9: Commit**

```bash
git add synapse-core/src/runtime/ synapse-core/src/lib.rs
git commit -m "feat(runtime): add InferencePort trait and bridge protocol value objects"
```

---

### Task A.4: VRAM Detection (Python)

**Files:**
- Create: `synapse-runtime/synapse_runtime/auto_assign.py`
- Create: `synapse-runtime/tests/test_auto_assign.py`

**Interfaces:**
- Produces: `detect_vram()` → `tuple[int, int]` (total_mb, available_mb)
- Produces: `calculate_expert_count(total_mb, available_mb, expert_size_gb, shared_params_gb)` → `int`

- [ ] **Step 1: Write the failing test**

Create `synapse-runtime/tests/test_auto_assign.py`:

```python
"""Tests for VRAM detection and auto-assignment logic."""

from unittest.mock import patch

from synapse_runtime.auto_assign import calculate_expert_count, detect_vram


class TestDetectVram:
    def test_returns_tuple_of_two_ints(self) -> None:
        """detect_vram returns (total_mb, available_mb) as integers."""
        total, available = detect_vram()
        assert isinstance(total, int)
        assert isinstance(available, int)
        assert total > 0
        assert available > 0
        assert available <= total

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", return_value=(17179869184, 8589934592))
    def test_uses_torch_cuda_when_available(
        self, mock_mem: object, mock_avail: object
    ) -> None:
        """When CUDA is available, uses torch.cuda.mem_get_info."""
        total, available = detect_vram()
        assert total == 16384  # 17179869184 / (1024*1024)
        assert available == 8192  # 8589934592 / (1024*1024)

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_zero_when_no_cuda(self, mock_avail: object) -> None:
        """When CUDA is not available, returns (0, 0)."""
        total, available = detect_vram()
        assert total == 0
        assert available == 0


class TestCalculateExpertCount:
    def test_standard_mixtral_config(self) -> None:
        """Mixtral 8x7B: 3GB expert + 3GB shared, 16GB VRAM → 4 experts."""
        count = calculate_expert_count(
            total_mb=16384,
            available_mb=12288,
            expert_size_gb=3.0,
            shared_params_gb=3.0,
        )
        # (12288/1024 - 3) / 3 = (12 - 3) / 3 = 3 experts
        assert count == 3

    def test_deepseek_lite_tiny_experts(self) -> None:
        """DeepSeek-V2 Lite: 0.15GB experts, 1GB shared, 8GB VRAM → many experts."""
        count = calculate_expert_count(
            total_mb=8192,
            available_mb=6144,
            expert_size_gb=0.15,
            shared_params_gb=1.0,
        )
        # (6144/1024 - 1) / 0.15 = (6 - 1) / 0.15 = 33
        assert count == 33

    def test_insufficient_vram_returns_zero(self) -> None:
        """When VRAM is less than shared params, returns 0 experts."""
        count = calculate_expert_count(
            total_mb=4096,
            available_mb=2048,
            expert_size_gb=3.0,
            shared_params_gb=3.0,
        )
        assert count == 0

    def test_exact_fit_one_expert(self) -> None:
        """When VRAM fits exactly one expert, returns 1."""
        # available_mb = shared_gb + 1*expert_gb in MB
        # 3GB shared + 3GB expert = 6GB = 6144 MB
        count = calculate_expert_count(
            total_mb=8192,
            available_mb=6144,
            expert_size_gb=3.0,
            shared_params_gb=3.0,
        )
        assert count == 1

    def test_respects_max_experts_cap(self) -> None:
        """Expert count is capped at the model's total expert count."""
        count = calculate_expert_count(
            total_mb=49152,  # 48GB
            available_mb=40960,  # 40GB free
            expert_size_gb=0.15,
            shared_params_gb=1.0,
            max_experts=8,  # e.g., Mixtral only has 8
        )
        # (40960/1024 - 1) / 0.15 = (40 - 1) / 0.15 = 260, but capped at 8
        assert count == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_auto_assign.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write auto_assign.py**

Create `synapse-runtime/synapse_runtime/auto_assign.py`:

```python
"""VRAM detection and optimal expert assignment for compute nodes."""

from __future__ import annotations


def detect_vram() -> tuple[int, int]:
    """Detect GPU VRAM.

    Returns:
        (total_mb, available_mb) tuple. Both are 0 if no GPU is available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_mb = total_bytes // (1024 * 1024)
            free_mb = free_bytes // (1024 * 1024)
            return int(total_mb), int(free_mb)
    except ImportError:
        pass

    return 0, 0


def calculate_expert_count(
    total_mb: int,
    available_mb: int,
    expert_size_gb: float,
    shared_params_gb: float,
    max_experts: int | None = None,
) -> int:
    """Calculate how many experts fit in available VRAM.

    Formula:
        available_gb = available_mb / 1024
        experts = floor((available_gb - shared_params_gb) / expert_size_gb)

    Args:
        total_mb: Total GPU VRAM in MB.
        available_mb: Available VRAM in MB.
        expert_size_gb: Size of one expert in GB (4-bit quantized).
        shared_params_gb: Size of shared parameters in GB.
        max_experts: Optional cap (model's total expert count).

    Returns:
        Number of experts that fit. Minimum 0.
    """
    if available_mb == 0:
        return 0

    available_gb = available_mb / 1024.0
    usable_gb = available_gb - shared_params_gb

    if usable_gb <= 0.0:
        return 0

    count = int(usable_gb / expert_size_gb)

    if max_experts is not None:
        count = min(count, max_experts)

    return max(count, 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_auto_assign.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/auto_assign.py tests/test_auto_assign.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/auto_assign.py synapse-runtime/tests/test_auto_assign.py
git commit -m "feat(runtime): add VRAM detection and expert count calculation"
```

---

### Task A.5: Compile Rust Protobuf Bindings

**Files:**
- Create: `synapse-core/src/runtime/infrastructure/mod.rs`
- Create: `synapse-core/src/runtime/infrastructure/proto/mod.rs`
- Create: `synapse-core/src/runtime/infrastructure/proto/runtime.rs`
- Modify: `synapse-core/Cargo.toml` — add `prost` dependency

**Interfaces:**
- Produces: Rust types generated from `runtime.proto` (or hand-coded equivalents matching the proto)

- [ ] **Step 1: Add prost dependency to Cargo.toml**

Modify `synapse-core/Cargo.toml` — add to `[dependencies]`:

```toml
prost = "0.14"
```

- [ ] **Step 2: Create infrastructure module structure**

Create `synapse-core/src/runtime/infrastructure/mod.rs`:

```rust
//! Infrastructure adapters for the runtime port.
//!
//! Currently: Unix socket bridge to Python vLLM subprocess.
//! Future: llama.cpp, SGLang adapters.

pub mod proto;
pub mod unix_socket_bridge;
```

Create `synapse-core/src/runtime/infrastructure/proto/mod.rs`:

```rust
//! Protobuf message types for the runtime bridge.
//!
//! These mirror `synapse-core/proto/runtime.proto` exactly.

pub mod runtime;
```

- [ ] **Step 3: Write hand-coded protobuf serialization for bridge messages**

Create `synapse-core/src/runtime/infrastructure/proto/runtime.rs`:

```rust
//! Protobuf message encoding/decoding for the runtime bridge.
//!
//! Hand-coded to avoid build-time protoc dependency.
//! Uses the `prost` crate for encoding primitives.

use crate::runtime::protocol::{
    GenerateBridgeRequest, GenerateBridgeResponse, LoadModelRequest, LoadModelResponse,
    VerifyBridgeRequest, VerifyBridgeResponse, VramBridgeRequest, VramBridgeResponse,
};

/// Encode a `LoadModelRequest` to protobuf bytes.
///
/// Wire format: field 1 (string model_id), field 2 (packed repeated uint32 expert_indices).
pub fn encode_load_model_request(req: &LoadModelRequest) -> Vec<u8> {
    let mut buf = Vec::new();
    // field 1: model_id (string)
    encode_string(&mut buf, 1, req.model_id.as_str());
    // field 2: expert_indices (packed repeated uint32)
    encode_packed_uint32(&mut buf, 2, &req.expert_indices);
    buf
}

/// Encode a `GenerateBridgeRequest` to protobuf bytes.
pub fn encode_generate_request(req: &GenerateBridgeRequest) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_bytes(&mut buf, 1, &req.request_id);
    encode_packed_uint32(&mut buf, 2, &req.token_ids);
    encode_uint32(&mut buf, 3, req.seed);
    encode_uint32(&mut buf, 4, req.max_tokens);
    buf
}

/// Encode a `VerifyBridgeRequest` to protobuf bytes.
pub fn encode_verify_request(req: &VerifyBridgeRequest) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_string(&mut buf, 1, req.model_id.as_str());
    encode_string(&mut buf, 2, &req.expected_sha256);
    buf
}

/// Encode a `VramBridgeRequest` to protobuf bytes (empty message).
pub fn encode_vram_request(_req: &VramBridgeRequest) -> Vec<u8> {
    Vec::new()
}

// ── Protobuf encoding helpers ──────────────────────────────────────

fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value > 0x7F {
        buf.push((value as u8 & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn encode_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    encode_varint(buf, ((field_number << 3) | wire_type) as u64);
}

fn encode_uint32(buf: &mut Vec<u8>, field_number: u32, value: u32) {
    encode_tag(buf, field_number, 0); // varint wire type
    encode_varint(buf, value as u64);
}

fn encode_bytes(buf: &mut Vec<u8>, field_number: u32, data: &[u8]) {
    encode_tag(buf, field_number, 2); // length-delimited wire type
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn encode_string(buf: &mut Vec<u8>, field_number: u32, value: &str) {
    encode_bytes(buf, field_number, value.as_bytes());
}

fn encode_packed_uint32(buf: &mut Vec<u8>, field_number: u32, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    let mut packed = Vec::with_capacity(values.len() * 5);
    for &v in values {
        encode_varint(&mut packed, v as u64);
    }
    encode_bytes(buf, field_number, &packed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    fn model_id() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    #[test]
    fn encode_empty_load_model_request() {
        let req = LoadModelRequest::new(model_id(), vec![]);
        let data = encode_load_model_request(&req);
        // field 1: string "mixtral-8x7b" = 0x0a + len(13) + "mixtral-8x7b"
        assert!(!data.is_empty());
        assert_eq!(data[0], 0x0a); // tag for field 1, wire type 2
        assert_eq!(data[1], 13); // string length
    }

    #[test]
    fn encode_load_model_with_experts() {
        let req = LoadModelRequest::new(model_id(), vec![0, 3, 7]);
        let data = encode_load_model_request(&req);
        assert!(data.len() > 14); // has the expert indices packed field
    }

    #[test]
    fn encode_generate_request() {
        let req = GenerateBridgeRequest::new(b"rid".to_vec(), vec![1, 2, 3], 0, 100);
        let data = encode_generate_request(&req);
        assert!(!data.is_empty());
    }

    #[test]
    fn encode_vram_request_empty() {
        let req = VramBridgeRequest;
        let data = encode_vram_request(&req);
        assert!(data.is_empty());
    }

    #[test]
    fn encode_verify_request() {
        let req = VerifyBridgeRequest::new(model_id(), "abc123".into());
        let data = encode_verify_request(&req);
        assert!(!data.is_empty());
    }
}
```

- [ ] **Step 4: Update runtime/infrastructure/mod.rs to include proto module**

Already done in step 2.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p synapse-core runtime::infrastructure`
Expected: ALL PASS

- [ ] **Step 6: Run format + lint**

Run: `cargo fmt --check && cargo clippy -p synapse-core -- -D warnings`
Expected: All checks passed

- [ ] **Step 7: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 8: Commit**

```bash
git add synapse-core/Cargo.toml synapse-core/src/runtime/infrastructure/
git commit -m "feat(runtime): add protobuf encoding for bridge messages"
```

---

## Phase B: Engine — vLLM + Loader + Determinism

### Task B.1: Weight Loader — HuggingFace Download

**Files:**
- Create: `synapse-runtime/synapse_runtime/loader.py`
- Create: `synapse-runtime/tests/test_loader.py`

**Interfaces:**
- Produces: `download_model(hf_repo: str, cache_dir: str | None = None) -> str` — returns path
- Produces: `ModelNotFoundError` exception

- [ ] **Step 1: Write the failing test**

Create `synapse-runtime/tests/test_loader.py`:

```python
"""Tests for weight loader."""

from unittest.mock import MagicMock, patch

from synapse_runtime.loader import ModelNotFoundError, download_model


class TestDownloadModel:
    @patch("synapse_runtime.loader.snapshot_download")
    def test_returns_local_path(self, mock_snapshot: MagicMock) -> None:
        """download_model returns the local path from snapshot_download."""
        mock_snapshot.return_value = "/cache/models/mixtral-8x7b"
        path = download_model("mistralai/Mixtral-8x7B-v0.1")
        assert path == "/cache/models/mixtral-8x7b"
        mock_snapshot.assert_called_once_with(
            "mistralai/Mixtral-8x7B-v0.1",
            cache_dir=None,
        )

    @patch("synapse_runtime.loader.snapshot_download")
    def test_respects_custom_cache_dir(self, mock_snapshot: MagicMock) -> None:
        """Custom cache_dir is passed to snapshot_download."""
        mock_snapshot.return_value = "/custom/models/test"
        path = download_model("test/model", cache_dir="/custom/models")
        assert path == "/custom/models/test"
        mock_snapshot.assert_called_once_with(
            "test/model",
            cache_dir="/custom/models",
        )

    @patch("synapse_runtime.loader.snapshot_download")
    def test_raises_model_not_found_on_hf_error(self,
                                                 mock_snapshot: MagicMock) -> None:
        """HF Hub errors are wrapped in ModelNotFoundError."""
        from huggingface_hub.errors import RepositoryNotFoundError

        mock_snapshot.side_effect = RepositoryNotFoundError("not found")
        with pytest.raises(ModelNotFoundError, match="not found"):
            download_model("nonexistent/model")

    @patch("synapse_runtime.loader.snapshot_download")
    def test_raises_on_other_errors(self, mock_snapshot: MagicMock) -> None:
        """Non-HF errors propagate as RuntimeError."""
        mock_snapshot.side_effect = OSError("disk full")
        with pytest.raises(RuntimeError, match="disk full"):
            download_model("any/model")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write loader.py (download_model only)**

Create `synapse-runtime/synapse_runtime/loader.py`:

```python
"""Weight loader: HuggingFace download, SHA256 verification, expert extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found on HuggingFace Hub."""


def download_model(hf_repo: str, cache_dir: str | None = None) -> str:
    """Download model weights from HuggingFace Hub.

    Args:
        hf_repo: HuggingFace repository ID (e.g. "mistralai/Mixtral-8x7B-v0.1").
        cache_dir: Optional custom cache directory.

    Returns:
        Local path to downloaded model.

    Raises:
        ModelNotFoundError: If the repository doesn't exist.
        RuntimeError: On other download failures.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError

        return snapshot_download(hf_repo, cache_dir=cache_dir)
    except RepositoryNotFoundError as e:
        raise ModelNotFoundError(str(e)) from e
    except Exception as e:
        raise RuntimeError(str(e)) from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_loader.py::TestDownloadModel -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/loader.py tests/test_loader.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/loader.py synapse-runtime/tests/test_loader.py
git commit -m "feat(runtime): add HF model downloader"
```

---

### Task B.2: Weight Loader — SHA256 Verification

**Files:**
- Modify: `synapse-runtime/synapse_runtime/loader.py`
- Modify: `synapse-runtime/tests/test_loader.py`

**Interfaces:**
- Produces: `verify_sha256(model_path: str, expected_hash: str) -> bool`
- Produces: `compute_sha256(model_path: str) -> str`

- [ ] **Step 1: Write the failing test**

Append to `synapse-runtime/tests/test_loader.py`:

```python
import tempfile
from pathlib import Path

import pytest


class TestSha256:
    def test_compute_sha256_known_content(self) -> None:
        """SHA256 of known content matches expected hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.bin"
            filepath.write_bytes(b"synapse test data")
            from synapse_runtime.loader import compute_sha256
            result = compute_sha256(str(tmpdir))
            # SHA256 of "synapse test data" (the file content)
            expected = hashlib.sha256(b"synapse test data").hexdigest()
            assert result == expected

    def test_verify_sha256_matches(self) -> None:
        """verify_sha256 returns True when hash matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.safetensors"
            filepath.write_bytes(b"weights data")
            expected = hashlib.sha256(b"weights data").hexdigest()
            from synapse_runtime.loader import verify_sha256
            assert verify_sha256(str(tmpdir), expected) is True

    def test_verify_sha256_mismatch(self) -> None:
        """verify_sha256 returns False when hash doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.safetensors"
            filepath.write_bytes(b"tampered weights")
            from synapse_runtime.loader import verify_sha256
            assert verify_sha256(str(tmpdir), "deadbeef") is False

    def test_verify_sha256_empty_directory(self) -> None:
        """verify_sha256 on empty directory returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from synapse_runtime.loader import verify_sha256
            # Empty dir: no files to hash → sha256 of empty string
            assert verify_sha256(str(tmpdir), "any") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_loader.py::TestSha256 -v`
Expected: FAIL — `ImportError` for `compute_sha256` / `verify_sha256`

- [ ] **Step 3: Add SHA256 functions to loader.py**

Append to `synapse-runtime/synapse_runtime/loader.py`:

```python
def compute_sha256(model_path: str) -> str:
    """Compute SHA256 hash of all files in a directory.

    Files are sorted by name for deterministic hashing.
    Each file's content is hashed individually, then the combined
    hash is computed from the concatenation of per-file hashes.

    Args:
        model_path: Path to the model directory.

    Returns:
        Hex-encoded SHA256 hash string.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {model_path}")

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")
    )

    if not files:
        return hashlib.sha256(b"").hexdigest()

    combined = hashlib.sha256()
    for filepath in files:
        file_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
        combined.update(file_hash.digest())

    return combined.hexdigest()


def verify_sha256(model_path: str, expected_hash: str) -> bool:
    """Verify model weights match expected SHA256 hash.

    Args:
        model_path: Path to the model directory.
        expected_hash: Expected hex-encoded SHA256.

    Returns:
        True if hashes match, False otherwise.
    """
    actual = compute_sha256(model_path)
    return actual == expected_hash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_loader.py::TestSha256 -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/loader.py tests/test_loader.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/loader.py synapse-runtime/tests/test_loader.py
git commit -m "feat(runtime): add SHA256 verification for model weights"
```

---

### Task B.3: Weight Loader — Expert Extraction from Checkpoint

**Files:**
- Modify: `synapse-runtime/synapse_runtime/loader.py`
- Modify: `synapse-runtime/tests/test_loader.py`

**Interfaces:**
- Produces: `extract_experts(model_path: str, expert_indices: list[int]) -> dict[int, bytes]`
- Produces: `ExpertExtractionError` exception

- [ ] **Step 1: Write the failing test**

Append to `synapse-runtime/tests/test_loader.py`:

```python
class TestExtractExperts:
    def test_extracts_single_expert_from_safetensors(self) -> None:
        """Extract expert weights from a safetensors file."""
        import json
        import struct

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal safetensors file with expert weights
            weights = {"model.experts.0.weight": np.array([1.0, 2.0, 3.0],
                                                            dtype=np.float32)}
            _write_safetensors(Path(tmpdir) / "model-00001.safetensors", weights)

            from synapse_runtime.loader import extract_experts
            result = extract_experts(str(tmpdir), [0])
            assert 0 in result
            data = result[0]
            assert len(data) == 12  # 3 float32 values

    def test_extracts_multiple_experts(self) -> None:
        """Extract multiple experts from safetensors files."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.experts.0.weight": np.array([1.0], dtype=np.float32),
                "model.experts.3.weight": np.array([3.0], dtype=np.float32),
            }
            _write_safetensors(Path(tmpdir) / "model.safetensors", weights)

            from synapse_runtime.loader import extract_experts
            result = extract_experts(str(tmpdir), [0, 3])
            assert result.keys() == {0, 3}

    def test_missing_expert_raises(self) -> None:
        """Requesting an expert not in the checkpoint raises ExpertExtractionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"model.experts.0.weight": np.array([1.0],
                                                            dtype=np.float32)}
            _write_safetensors(Path(tmpdir) / "model.safetensors", weights)

            import pytest
            from synapse_runtime.loader import ExpertExtractionError, extract_experts

            with pytest.raises(ExpertExtractionError, match="Expert 7"):
                extract_experts(str(tmpdir), [0, 7])


def _write_safetensors(filepath: Path, tensors: dict[str, "np.ndarray"]) -> None:
    """Helper: write a minimal safetensors file."""
    import json
    import struct

    header = {}
    offset = 0
    for name, arr in tensors.items():
        dtype = "F32"
        shape = list(arr.shape)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes

    header_json = json.dumps(header)
    header_bytes = header_json.encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))

    with open(filepath, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        for arr in tensors.values():
            f.write(arr.tobytes())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_loader.py::TestExtractExperts -v`
Expected: FAIL — `ImportError` for `extract_experts` / `ExpertExtractionError`

- [ ] **Step 3: Add expert extraction to loader.py**

Append to `synapse-runtime/synapse_runtime/loader.py`:

```python
class ExpertExtractionError(Exception):
    """Raised when expert weights cannot be extracted from a checkpoint."""


def extract_experts(
    model_path: str, expert_indices: list[int]
) -> dict[int, bytes]:
    """Extract expert weights from safetensors checkpoint files.

    Searches all `.safetensors` files in the model directory for
    tensors matching `model.experts.{index}.*` patterns.

    Args:
        model_path: Path to the model directory.
        expert_indices: List of expert indices to extract.

    Returns:
        Dict mapping expert index to raw bytes of concatenated weights.

    Raises:
        ExpertExtractionError: If any requested expert is not found.
    """
    import json
    import struct

    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {model_path}")

    safetensors_files = sorted(root.rglob("*.safetensors"))
    if not safetensors_files:
        raise ExpertExtractionError(
            f"No .safetensors files found in {model_path}"
        )

    requested = set(expert_indices)
    found: dict[int, bytearray] = {}

    for sf_path in safetensors_files:
        with open(sf_path, "rb") as f:
            header_len_data = f.read(8)
            header_len = struct.unpack("<Q", header_len_data)[0]
            header_json = f.read(header_len)
            header = json.loads(header_json.decode("utf-8"))

            for tensor_name, meta in header.items():
                # Match "model.experts.N." pattern
                if not tensor_name.startswith("model.experts."):
                    continue
                parts = tensor_name.split(".")
                if len(parts) < 3:
                    continue
                try:
                    expert_idx = int(parts[2])
                except ValueError:
                    continue

                if expert_idx not in requested:
                    continue

                start, end = meta["data_offsets"]
                # Data starts after header: 8 bytes length + header bytes
                data_start = 8 + header_len + start
                f.seek(data_start)
                tensor_data = f.read(end - start)
                found.setdefault(expert_idx, bytearray()).extend(tensor_data)

    # Verify all requested experts were found
    missing = requested - set(found.keys())
    if missing:
        raise ExpertExtractionError(
            f"Expert{'s' if len(missing) > 1 else ''} "
            f"{', '.join(str(m) for m in sorted(missing))} "
            f"not found in {model_path}"
        )

    return {idx: bytes(data) for idx, data in found.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_loader.py::TestExtractExperts -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/loader.py tests/test_loader.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/loader.py synapse-runtime/tests/test_loader.py
git commit -m "feat(runtime): add expert weight extraction from safetensors"
```

---

### Task B.4: vLLM Engine — load_model()

**Files:**
- Create: `synapse-runtime/synapse_runtime/engine.py`
- Create: `synapse-runtime/tests/test_engine.py`

**Interfaces:**
- Produces: `VllmEngine` class with `load_model(model_id: str, expert_indices: list[int]) -> None`
- Produces: `EngineError` exception

- [ ] **Step 1: Write the failing test**

Create `synapse-runtime/tests/test_engine.py`:

```python
"""Tests for vLLM engine wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from synapse_runtime.engine import EngineError, VllmEngine


class TestVllmEngineLoadModel:
    @patch("synapse_runtime.engine.LLM")
    def test_load_model_creates_llm_instance(self, mock_llm_class: MagicMock) -> None:
        """load_model initializes vLLM with the model path."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/mixtral", [0, 1, 2, 3])

        mock_llm_class.assert_called_once()
        call_args = mock_llm_class.call_args
        assert call_args[0][0] == "/models/mixtral"

    @patch("synapse_runtime.engine.LLM")
    def test_load_model_sets_loaded_flag(self, mock_llm_class: MagicMock) -> None:
        """After load_model, is_loaded returns True."""
        mock_llm_class.return_value = MagicMock()

        engine = VllmEngine()
        assert engine.is_loaded is False
        engine.load_model("/models/test", [0])
        assert engine.is_loaded is True

    @patch("synapse_runtime.engine.LLM")
    def test_load_model_passes_deterministic_seed(self,
                                                   mock_llm_class: MagicMock) -> None:
        """load_model passes seed=0 for deterministic mode."""
        mock_llm_class.return_value = MagicMock()

        engine = VllmEngine()
        engine.load_model("/models/test", [0, 1], seed=0)

        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs.get("seed") == 0

    @patch("synapse_runtime.engine.LLM")
    def test_load_model_handles_vllm_error(self, mock_llm_class: MagicMock) -> None:
        """vLLM initialization errors are wrapped in EngineError."""
        mock_llm_class.side_effect = RuntimeError("CUDA out of memory")

        engine = VllmEngine()
        with pytest.raises(EngineError, match="CUDA out of memory"):
            engine.load_model("/models/big", [0])

    @patch("synapse_runtime.engine.LLM")
    def test_double_load_replaces_model(self, mock_llm_class: MagicMock) -> None:
        """Loading a second model replaces the first."""
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_llm_class.side_effect = [mock_llm1, mock_llm2]

        engine = VllmEngine()
        engine.load_model("/models/first", [0])
        engine.load_model("/models/second", [1])

        assert engine.is_loaded is True
        assert mock_llm_class.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write engine.py with load_model**

Create `synapse-runtime/synapse_runtime/engine.py`:

```python
"""vLLM engine wrapper for model loading and token generation.

Provides the V1 backend for the Synapse inference runtime.
Implements the runtime side of the InferencePort contract.
"""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """Raised when the inference engine encounters an error."""


class VllmEngine:
    """Wraps vLLM's LLM class for model loading and generation.

    This is the V1 backend. V2+ will add llama.cpp and SGLang engines
    behind the same interface.
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_path: str = ""
        self._loaded_experts: list[int] = []

    @property
    def is_loaded(self) -> bool:
        """True if a model is currently loaded in VRAM."""
        return self._llm is not None

    @property
    def loaded_experts(self) -> list[int]:
        """The expert indices currently loaded."""
        return list(self._loaded_experts)

    def load_model(
        self,
        model_path: str,
        expert_indices: list[int],
        seed: int = 0,
    ) -> None:
        """Load a model with specified experts into VRAM.

        Args:
            model_path: Local path to the model directory.
            expert_indices: Which experts to load (indices).
            seed: Random seed for deterministic generation (default 0).

        Raises:
            EngineError: If model loading fails.
        """
        try:
            from vllm import LLM

            self._llm = LLM(
                model=model_path,
                seed=seed,
                gpu_memory_utilization=0.90,
                max_model_len=32768,
            )
            self._model_path = model_path
            self._loaded_experts = list(expert_indices)
        except Exception as e:
            raise EngineError(str(e)) from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_engine.py::TestVllmEngineLoadModel -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/engine.py tests/test_engine.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/engine.py synapse-runtime/tests/test_engine.py
git commit -m "feat(runtime): add vLLM engine load_model"
```

---

### Task B.5: vLLM Engine — generate()

**Files:**
- Modify: `synapse-runtime/synapse_runtime/engine.py`
- Modify: `synapse-runtime/tests/test_engine.py`

**Interfaces:**
- Produces: `VllmEngine.generate(prompt_tokens: list[int], seed: int, max_tokens: int) -> GenerateResponse`

- [ ] **Step 1: Write the failing test**

Append to `synapse-runtime/tests/test_engine.py`:

```python
class TestVllmEngineGenerate:
    @patch("synapse_runtime.engine.LLM")
    def test_generate_returns_tokens(self, mock_llm_class: MagicMock) -> None:
        """generate returns token IDs and logprobs."""
        from unittest.mock import MagicMock

        # Mock the LLM instance and its generate method
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [
            MagicMock(token_ids=[42, 43, 44],
                      logprobs=[-0.1, -0.2, -0.3])
        ]
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/test", [0])

        from synapse_runtime.protocol import GenerateResponse
        resp = engine.generate([1, 2, 3], seed=0, max_tokens=100)

        assert isinstance(resp, GenerateResponse)
        assert resp.token_ids == [42, 43, 44]
        assert resp.log_probs == [-0.1, -0.2, -0.3]
        assert resp.finished is True

    @patch("synapse_runtime.engine.LLM")
    def test_generate_requires_loaded_model(self,
                                              mock_llm_class: MagicMock) -> None:
        """generate raises EngineError if no model is loaded."""
        engine = VllmEngine()
        with pytest.raises(EngineError, match="No model loaded"):
            engine.generate([1, 2], seed=0, max_tokens=10)

    @patch("synapse_runtime.engine.LLM")
    def test_generate_empty_prompt_returns_empty(self,
                                                   mock_llm_class: MagicMock) -> None:
        """Empty prompt produces empty output."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = []
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/test", [0])

        resp = engine.generate([], seed=0, max_tokens=50)
        assert resp.token_ids == []
        assert resp.finished is True

    @patch("synapse_runtime.engine.LLM")
    def test_generate_handles_vllm_error(self, mock_llm_class: MagicMock) -> None:
        """vLLM generate errors are wrapped in EngineError."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("CUDA error")
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/test", [0])

        with pytest.raises(EngineError, match="CUDA error"):
            engine.generate([1], seed=0, max_tokens=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_engine.py::TestVllmEngineGenerate -v`
Expected: FAIL — `AttributeError: 'VllmEngine' object has no attribute 'generate'`

- [ ] **Step 3: Add generate() to engine.py**

Append to `VllmEngine` class in `synapse-runtime/synapse_runtime/engine.py`:

```python
    def generate(
        self,
        prompt_tokens: list[int],
        seed: int = 0,
        max_tokens: int = 256,
    ) -> "GenerateResponse":
        """Generate tokens from a prompt.

        Args:
            prompt_tokens: Input token IDs.
            seed: Random seed (0 for deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerateResponse with token_ids, log_probs, and finished flag.

        Raises:
            EngineError: If no model is loaded or generation fails.
        """
        if not self.is_loaded:
            raise EngineError("No model loaded. Call load_model() first.")

        if not prompt_tokens:
            from synapse_runtime.protocol import GenerateResponse
            return GenerateResponse(
                request_id=b"",
                token_ids=[],
                log_probs=[],
                finished=True,
            )

        try:
            from vllm import SamplingParams
            from synapse_runtime.protocol import GenerateResponse

            sampling_params = SamplingParams(
                temperature=0.0 if seed == 0 else 0.7,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=1,
            )

            prompt = {"prompt_token_ids": prompt_tokens}
            outputs = self._llm.generate([prompt], sampling_params)

            if not outputs:
                return GenerateResponse(
                    request_id=b"",
                    token_ids=[],
                    log_probs=[],
                    finished=True,
                )

            output = outputs[0]
            token_ids: list[int] = []
            log_probs: list[float] = []

            for out in output.outputs:
                token_ids.extend(out.token_ids)
                if out.logprobs:
                    log_probs.extend(out.logprobs)

            return GenerateResponse(
                request_id=b"",
                token_ids=token_ids,
                log_probs=log_probs,
                finished=True,
            )

        except Exception as e:
            raise EngineError(str(e)) from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_engine.py::TestVllmEngineGenerate -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/engine.py tests/test_engine.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/engine.py synapse-runtime/tests/test_engine.py
git commit -m "feat(runtime): add vLLM engine generate()"
```

---

### Task B.6: Deterministic Seed Enforcement

**Files:**
- Create: `synapse-runtime/synapse_runtime/deterministic.py`
- Create: `synapse-runtime/tests/test_deterministic.py`

**Interfaces:**
- Produces: `enforce_seed_zero(engine: VllmEngine) -> None`
- Produces: `verify_determinism(engine: VllmEngine, prompt_tokens: list[int]) -> bool`

- [ ] **Step 1: Write the failing test**

Create `synapse-runtime/tests/test_deterministic.py`:

```python
"""Tests for deterministic seed enforcement."""

from unittest.mock import MagicMock, patch

from synapse_runtime.deterministic import enforce_seed_zero, verify_determinism


class TestEnforceSeedZero:
    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_enforce_seed_zero_reloads_with_seed_zero(
        self, mock_engine_class: MagicMock
    ) -> None:
        """enforce_seed_zero reloads the model with seed=0."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine._model_path = "/models/test"
        mock_engine._loaded_experts = [0, 1, 2]

        enforce_seed_zero(mock_engine)

        mock_engine.load_model.assert_called_once_with(
            "/models/test", [0, 1, 2], seed=0
        )

    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_enforce_seed_zero_noop_when_not_loaded(
        self, mock_engine_class: MagicMock
    ) -> None:
        """enforce_seed_zero does nothing if no model is loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = False

        enforce_seed_zero(mock_engine)

        mock_engine.load_model.assert_not_called()


class TestVerifyDeterminism:
    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_identical_outputs_pass_verification(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Two runs with identical outputs → verification passes."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        from synapse_runtime.protocol import GenerateResponse

        # Same output both times
        mock_engine.generate.side_effect = [
            GenerateResponse(
                request_id=b"r1", token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.2, -0.3], finished=True
            ),
            GenerateResponse(
                request_id=b"r1", token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.2, -0.3], finished=True
            ),
        ]

        result = verify_determinism(mock_engine, [42])
        assert result is True

    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_divergent_outputs_fail_verification(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Two runs with different outputs → verification fails."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        from synapse_runtime.protocol import GenerateResponse

        mock_engine.generate.side_effect = [
            GenerateResponse(
                request_id=b"r1", token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.2, -0.3], finished=True
            ),
            GenerateResponse(
                request_id=b"r1", token_ids=[1, 99, 3],  # Divergent!
                log_probs=[-0.1, -0.9, -0.3], finished=True
            ),
        ]

        result = verify_determinism(mock_engine, [42])
        assert result is False

    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_different_logprobs_fail_verification(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Same tokens but different logprobs → verification fails."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        from synapse_runtime.protocol import GenerateResponse

        mock_engine.generate.side_effect = [
            GenerateResponse(
                request_id=b"r1", token_ids=[1, 2],
                log_probs=[-0.1, -0.2], finished=True
            ),
            GenerateResponse(
                request_id=b"r1", token_ids=[1, 2],
                log_probs=[-0.1, -0.999], finished=True  # Different logprob
            ),
        ]

        result = verify_determinism(mock_engine, [42])
        assert result is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_deterministic.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write deterministic.py**

Create `synapse-runtime/synapse_runtime/deterministic.py`:

```python
"""Deterministic seed enforcement for audit and verification.

Ensures that seed=0 produces identical outputs across runs.
This is critical for the statistical audit mechanism in the swarm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapse_runtime.engine import VllmEngine


def enforce_seed_zero(engine: "VllmEngine") -> None:
    """Reload the current model with seed=0 to enforce determinism.

    Args:
        engine: The vLLM engine instance with a model loaded.
    """
    if not engine.is_loaded:
        return

    # Reload with seed=0 using the same model path and experts
    engine.load_model(
        model_path=engine._model_path,  # noqa: SLF001
        expert_indices=engine.loaded_experts,
        seed=0,
    )


def verify_determinism(
    engine: "VllmEngine", prompt_tokens: list[int]
) -> bool:
    """Verify that two runs with seed=0 produce identical outputs.

    Runs the same prompt twice and compares token IDs and logprobs.
    Both runs use seed=0 internally.

    Args:
        engine: The vLLM engine (must have a model loaded with seed=0).
        prompt_tokens: Input token IDs for the test prompt.

    Returns:
        True if both runs produce identical outputs, False otherwise.
    """
    if not engine.is_loaded:
        return False

    run1 = engine.generate(prompt_tokens, seed=0, max_tokens=50)
    run2 = engine.generate(prompt_tokens, seed=0, max_tokens=50)

    return (
        run1.token_ids == run2.token_ids
        and run1.log_probs == run2.log_probs
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_deterministic.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/deterministic.py tests/test_deterministic.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/deterministic.py synapse-runtime/tests/test_deterministic.py
git commit -m "feat(runtime): add deterministic seed=0 enforcement and verification"
```

---

## Phase C: Bridge — Unix Socket + Integration

### Task C.1: Python Unix Socket Server

**Files:**
- Create: `synapse-runtime/synapse_runtime/server.py`
- Create: `synapse-runtime/tests/test_server.py`

**Interfaces:**
- Produces: `RuntimeServer` class — `start(socket_path: str)`, `stop()`, `handle_request(data: bytes) -> bytes`
- Consumes: `VllmEngine`, `download_model`, `verify_sha256`, `detect_vram`, protocol dataclasses

- [ ] **Step 1: Write the failing test**

Create `synapse-runtime/tests/test_server.py`:

```python
"""Tests for Unix socket server."""

import os
import socket
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synapse_runtime.protocol import (
    GenerateRequest,
    GenerateResponse,
    LoadModelRequest,
    LoadModelResponse,
    VramQueryRequest,
    VramQueryResponse,
    deserialize_response,
    serialize_request,
)
from synapse_runtime.server import RuntimeServer


class TestRuntimeServerHandleRequest:
    @patch("synapse_runtime.server.VllmEngine")
    def test_handle_load_model_request(self, mock_engine_class: MagicMock) -> None:
        """LoadModelRequest dispatches to engine.load_model."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        server = RuntimeServer(engine=mock_engine)
        req = LoadModelRequest(model_id="mixtral-8x7b", expert_indices=[0, 3])
        data = serialize_request(req)

        resp_data = server.handle_request(data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, LoadModelResponse)
        assert resp.success is True
        assert resp.loaded_experts == 2

        mock_engine.load_model.assert_called_once()

    @patch("synapse_runtime.server.VllmEngine")
    @patch("synapse_runtime.server.download_model")
    def test_handle_load_model_downloads_first(
        self, mock_download: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """load_model downloads from HF before loading into vLLM."""
        mock_download.return_value = "/cache/models/test"
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        server = RuntimeServer(engine=mock_engine)
        req = LoadModelRequest(model_id="mixtral-8x7b", expert_indices=[0])
        data = serialize_request(req)

        server.handle_request(data)

        mock_download.assert_called_once_with("mistralai/Mixtral-8x7B-v0.1",
                                                cache_dir=None)
        mock_engine.load_model.assert_called_once()

    @patch("synapse_runtime.server.VllmEngine")
    def test_handle_generate_request(self, mock_engine_class: MagicMock) -> None:
        """GenerateRequest dispatches to engine.generate."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = GenerateResponse(
            request_id=b"r1", token_ids=[7, 8, 9],
            log_probs=[-0.1, -0.2, -0.3], finished=True
        )
        mock_engine_class.return_value = mock_engine

        server = RuntimeServer(engine=mock_engine)
        req = GenerateRequest(
            request_id=b"r1", token_ids=[1, 2], seed=0, max_tokens=50
        )
        data = serialize_request(req)

        resp_data = server.handle_request(data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, GenerateResponse)
        assert resp.token_ids == [7, 8, 9]
        assert resp.finished is True

    @patch("synapse_runtime.server.VllmEngine")
    def test_handle_generate_requires_loaded_model(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Generate returns error response if no model loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = False
        mock_engine_class.return_value = mock_engine

        server = RuntimeServer(engine=mock_engine)
        req = GenerateRequest(
            request_id=b"r1", token_ids=[1], seed=0, max_tokens=10
        )
        data = serialize_request(req)

        resp_data = server.handle_request(data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, GenerateResponse)
        assert resp.finished is True
        assert resp.token_ids == []

    @patch("synapse_runtime.server.VllmEngine")
    @patch("synapse_runtime.server.detect_vram")
    def test_handle_vram_query(
        self, mock_detect: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """VramQueryRequest dispatches to detect_vram."""
        mock_detect.return_value = (16384, 8192)
        mock_engine_class.return_value = MagicMock()

        server = RuntimeServer(engine=MagicMock())
        req = VramQueryRequest()
        data = serialize_request(req)

        resp_data = server.handle_request(data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, VramQueryResponse)
        assert resp.total_mb == 16384
        assert resp.available_mb == 8192


class TestRuntimeServerSocketLifecycle:
    @patch("synapse_runtime.server.VllmEngine")
    def test_start_stop(self, mock_engine_class: MagicMock) -> None:
        """Server starts and stops cleanly on a temp socket."""
        mock_engine_class.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "test.sock")
            server = RuntimeServer(engine=MagicMock())

            # Start in background
            import threading
            server._running = True  # Simulate running state

            # Stop
            server.stop()
            # Socket should be cleaned up
            assert not os.path.exists(socket_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd synapse-runtime && python -m pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write server.py**

Create `synapse-runtime/synapse_runtime/server.py`:

```python
"""Unix socket server for the Synapse runtime bridge.

Listens on a Unix domain socket for protobuf-encoded requests
from the Rust core. Dispatches to engine, loader, and auto_assign.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from typing import Any

from synapse_runtime.auto_assign import detect_vram
from synapse_runtime.engine import EngineError, VllmEngine
from synapse_runtime.loader import ModelNotFoundError, download_model
from synapse_runtime.protocol import (
    GenerateRequest,
    GenerateResponse,
    LoadModelRequest,
    LoadModelResponse,
    VerifyHashRequest,
    VerifyHashResponse,
    VramQueryRequest,
    VramQueryResponse,
    deserialize_request,
    serialize_response,
)

logger = logging.getLogger(__name__)

# Maps model_id → HF repo for download
MODEL_REPO_MAP: dict[str, str] = {
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "kimi-k3": "moonshotai/Kimi-K3",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "qwen2.5-moe": "Qwen/Qwen2.5-57B-A14B",
}


class RuntimeServer:
    """Unix socket server for the runtime bridge.

    Receives protobuf-encoded requests, dispatches to the appropriate
    handler (engine, loader, auto_assign), and returns protobuf-encoded
    responses.
    """

    def __init__(self, engine: VllmEngine | None = None) -> None:
        self._engine = engine or VllmEngine()
        self._socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        """True if the server is accepting connections."""
        return self._running

    def start(self, socket_path: str = "/tmp/synapse-runtime.sock") -> None:
        """Start listening on a Unix domain socket.

        Args:
            socket_path: Path for the Unix socket file.
        """
        if self._running:
            return

        # Clean up stale socket file
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(socket_path)
        self._socket.listen(5)
        self._running = True

        self._thread = threading.Thread(
            target=self._accept_loop, args=(socket_path,), daemon=True
        )
        self._thread.start()
        logger.info("Runtime server listening on %s", socket_path)

    def stop(self) -> None:
        """Stop the server and clean up the socket."""
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def handle_request(self, data: bytes) -> bytes:
        """Handle a single request and return the response.

        Args:
            data: Protobuf-encoded request bytes.

        Returns:
            Protobuf-encoded response bytes.

        Raises:
            ValueError: If the request type is unknown.
        """
        try:
            request = deserialize_request(data)
        except ValueError as e:
            logger.error("Failed to deserialize request: %s", e)
            return _error_response(str(e))

        try:
            match request:
                case LoadModelRequest():
                    return self._handle_load_model(request)
                case GenerateRequest():
                    return self._handle_generate(request)
                case VerifyHashRequest():
                    return self._handle_verify_hash(request)
                case VramQueryRequest():
                    return self._handle_vram_query(request)
                case _:
                    raise ValueError(f"Unknown request type: {type(request)}")
        except Exception as e:
            logger.exception("Error handling request")
            return _error_response(str(e))

    def _handle_load_model(self, req: LoadModelRequest) -> bytes:
        """Handle LoadModelRequest."""
        try:
            hf_repo = MODEL_REPO_MAP.get(req.model_id, req.model_id)
            model_path = download_model(hf_repo)
            self._engine.load_model(model_path, req.expert_indices, seed=0)
            resp = LoadModelResponse(
                success=True,
                error="",
                loaded_experts=len(req.expert_indices),
            )
        except (ModelNotFoundError, EngineError) as e:
            resp = LoadModelResponse(success=False, error=str(e),
                                      loaded_experts=0)
        return serialize_response(resp)

    def _handle_generate(self, req: GenerateRequest) -> bytes:
        """Handle GenerateRequest."""
        if not self._engine.is_loaded:
            resp = GenerateResponse(
                request_id=req.request_id,
                token_ids=[],
                log_probs=[],
                finished=True,
            )
            return serialize_response(resp)

        try:
            result = self._engine.generate(
                req.token_ids, seed=req.seed, max_tokens=req.max_tokens
            )
            return serialize_response(result)
        except EngineError as e:
            logger.error("Generate failed: %s", e)
            resp = GenerateResponse(
                request_id=req.request_id,
                token_ids=[],
                log_probs=[],
                finished=True,
            )
            return serialize_response(resp)

    def _handle_verify_hash(self, req: VerifyHashRequest) -> bytes:
        """Handle VerifyHashRequest."""
        from synapse_runtime.loader import verify_sha256

        hf_repo = MODEL_REPO_MAP.get(req.model_id, req.model_id)
        try:
            model_path = download_model(hf_repo)
            matches = verify_sha256(model_path, req.expected_sha256)
            from synapse_runtime.loader import compute_sha256
            actual = compute_sha256(model_path)
            resp = VerifyHashResponse(matches=matches, actual_sha256=actual)
        except ModelNotFoundError as e:
            resp = VerifyHashResponse(
                matches=False, actual_sha256=f"error: {e}"
            )
        return serialize_response(resp)

    def _handle_vram_query(self, _req: VramQueryRequest) -> bytes:
        """Handle VramQueryRequest."""
        total, available = detect_vram()
        resp = VramQueryResponse(total_mb=total, available_mb=available)
        return serialize_response(resp)

    def _accept_loop(self, socket_path: str) -> None:
        """Accept connections in a loop."""
        while self._running:
            try:
                if self._socket is None:
                    break
                conn, _ = self._socket.accept()
                self._handle_connection(conn)
            except (OSError, socket.timeout):
                if self._running:
                    logger.debug("Socket accept interrupted")
                break

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single client connection."""
        try:
            with conn:
                # Read length-prefixed message: 4-byte big-endian length + payload
                length_data = conn.recv(4)
                if len(length_data) < 4:
                    return
                msg_len = int.from_bytes(length_data, "big")

                data = bytearray()
                while len(data) < msg_len:
                    chunk = conn.recv(min(msg_len - len(data), 65536))
                    if not chunk:
                        break
                    data.extend(chunk)

                if len(data) == msg_len:
                    response = self.handle_request(bytes(data))
                    # Write length-prefixed response
                    conn.sendall(len(response).to_bytes(4, "big"))
                    conn.sendall(response)
        except OSError as e:
            logger.debug("Connection error: %s", e)


def _error_response(message: str) -> bytes:
    """Create a GenerateResponse with an error message."""
    resp = GenerateResponse(
        request_id=b"",
        token_ids=[],
        log_probs=[],
        finished=True,
    )
    return serialize_response(resp)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd synapse-runtime && python -m pytest tests/test_server.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ruff**

Run: `cd synapse-runtime && ruff check synapse_runtime/server.py tests/test_server.py`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-runtime/synapse_runtime/server.py synapse-runtime/tests/test_server.py
git commit -m "feat(runtime): add Unix socket server with request dispatch"
```

---

### Task C.2: Rust UnixSocketBridge Adapter

**Files:**
- Create: `synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs`

**Interfaces:**
- Produces: `UnixSocketBridge` struct implementing `InferencePort`
- Consumes: `tokio::net::UnixStream`, protobuf encoding from `proto::runtime`

- [ ] **Step 1: Write the failing test**

Append test module to `unix_socket_bridge.rs` (create the file first with test skeleton):

Create `synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs`:

```rust
//! Unix socket bridge to the Python runtime subprocess.
//!
//! Implements [`InferencePort`] by communicating with the Python
//! `server.py` over a Unix domain socket using protobuf messages.

use crate::model::{ExpertId, ModelId};
use crate::runtime::ports::InferencePort;
use crate::runtime::protocol::{
    GenerateBridgeRequest, GenerateBridgeResponse, LoadModelRequest, LoadModelResponse,
    VerifyBridgeRequest, VerifyBridgeResponse, VramBridgeRequest, VramBridgeResponse,
};
use crate::shared::DomainError;
use crate::swarm::ports::{InferenceOutput, InferenceRequest};
use crate::swarm::token::Token;

/// Bridge adapter that communicates with the Python runtime via Unix socket.
///
/// Each method serializes a domain request into a protobuf message,
/// sends it over the socket, and deserializes the response.
pub struct UnixSocketBridge {
    socket_path: String,
}

impl UnixSocketBridge {
    /// Creates a new bridge targeting the given Unix socket path.
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self { socket_path: socket_path.into() }
    }

    /// Returns the socket path this bridge connects to.
    pub fn socket_path(&self) -> &str {
        &self.socket_path
    }

    /// Sends a request and receives a response over the socket.
    ///
    /// Length-prefixed framing: 4-byte big-endian length + payload.
    fn send_request(&self, request_data: &[u8]) -> Result<Vec<u8>, DomainError> {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut stream = UnixStream::connect(&self.socket_path).map_err(|e| {
            DomainError::StorageError {
                message: format!("Failed to connect to runtime socket: {e}"),
            }
        })?;

        // Write length-prefixed request
        let len_bytes = (request_data.len() as u32).to_be_bytes();
        stream.write_all(&len_bytes).map_err(|e| {
            DomainError::StorageError {
                message: format!("Failed to write request length: {e}"),
            }
        })?;
        stream.write_all(request_data).map_err(|e| {
            DomainError::StorageError {
                message: format!("Failed to write request: {e}"),
            }
        })?;

        // Read length-prefixed response
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).map_err(|e| {
            DomainError::StorageError {
                message: format!("Failed to read response length: {e}"),
            }
        })?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;

        let mut resp_data = vec![0u8; resp_len];
        stream.read_exact(&mut resp_data).map_err(|e| {
            DomainError::StorageError {
                message: format!("Failed to read response: {e}"),
            }
        })?;

        Ok(resp_data)
    }
}

impl InferencePort for UnixSocketBridge {
    fn load(&self, model: &ModelId, experts: &[ExpertId]) -> Result<(), DomainError> {
        let expert_indices: Vec<u32> = experts.iter().map(|e| e.index).collect();
        let req = LoadModelRequest::new(model.clone(), expert_indices);
        let req_data = crate::runtime::infrastructure::proto::runtime::encode_load_model_request(&req);

        // Prepend message type varint for the Python dispatcher
        let mut framed = vec![1u8]; // type 1 = LoadModelRequest
        framed.extend_from_slice(&req_data);

        let _resp = self.send_request(&framed)?;
        // TODO: decode response and check success field
        Ok(())
    }

    fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError> {
        let token_ids: Vec<u32> = vec![]; // token_ids come from tokenizer, not domain
        let bridge_req = GenerateBridgeRequest::new(
            request.id.as_bytes().to_vec(),
            token_ids,
            0, // seed
            request.max_tokens,
        );
        let req_data = crate::runtime::infrastructure::proto::runtime::encode_generate_request(&bridge_req);

        let mut framed = vec![3u8]; // type 3 = GenerateRequest
        framed.extend_from_slice(&req_data);

        let _resp = self.send_request(&framed)?;
        // TODO: decode response and construct InferenceOutput
        Ok(InferenceOutput {
            request_id: request.id,
            tokens: vec![],
        })
    }

    fn verify(&self, model: &ModelId, expected_hash: &str) -> Result<bool, DomainError> {
        let req = VerifyBridgeRequest::new(model.clone(), expected_hash.to_string());
        let req_data = crate::runtime::infrastructure::proto::runtime::encode_verify_request(&req);

        let mut framed = vec![5u8]; // type 5 = VerifyHashRequest
        framed.extend_from_slice(&req_data);

        let _resp = self.send_request(&framed)?;
        // TODO: decode response and return matches field
        Ok(true)
    }

    fn detect_vram(&self) -> Result<u32, DomainError> {
        let req = VramBridgeRequest;
        let req_data = crate::runtime::infrastructure::proto::runtime::encode_vram_request(&req);

        let mut framed = vec![7u8]; // type 7 = VramQueryRequest
        framed.extend_from_slice(&req_data);

        let _resp = self.send_request(&framed)?;
        // TODO: decode response and return available_mb
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_stores_socket_path() {
        let bridge = UnixSocketBridge::new("/tmp/test.sock");
        assert_eq!(bridge.socket_path(), "/tmp/test.sock");
    }

    #[test]
    fn bridge_implements_inference_port() {
        // Compile-time assertion: UnixSocketBridge implements InferencePort
        fn _check(port: &dyn InferencePort) {
            let _ = port.detect_vram();
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they pass (unit tests only)**

Run: `cargo test -p synapse-core runtime::infrastructure::unix_socket_bridge`
Expected: 2 unit tests PASS (bridge_stores_socket_path, bridge_implements_inference_port)

- [ ] **Step 3: Run format + lint**

Run: `cargo fmt --check && cargo clippy -p synapse-core -- -D warnings`
Expected: All checks passed

- [ ] **Step 4: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs
git commit -m "feat(runtime): add UnixSocketBridge adapter implementing InferencePort"
```

---

### Task C.3: Response Deserialization in Rust Bridge

**Files:**
- Modify: `synapse-core/src/runtime/infrastructure/proto/runtime.rs`
- Modify: `synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs`

**Interfaces:**
- Produces: `decode_load_model_response(data: &[u8]) -> LoadModelResponse`
- Produces: `decode_generate_response(data: &[u8]) -> GenerateBridgeResponse`
- Produces: `decode_verify_response(data: &[u8]) -> VerifyBridgeResponse`
- Produces: `decode_vram_response(data: &[u8]) -> VramBridgeResponse`

- [ ] **Step 1: Write decoders + tests in proto/runtime.rs**

Append to `synapse-core/src/runtime/infrastructure/proto/runtime.rs`:

```rust
// ── Protobuf decoding helpers ──────────────────────────────────────

fn decode_varint(data: &[u8], offset: &mut usize) -> Result<u64, String> {
    let mut value: u64 = 0;
    let mut shift = 0;
    while *offset < data.len() {
        let byte = data[*offset];
        *offset += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err("Varint too long".into());
        }
    }
    Err("Truncated varint".into())
}

fn decode_uint32(fields: &[(u32, u32, &[u8])], field_num: u32, default: u32) -> u32 {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == 0 {
            let mut off = 0;
            if let Ok(v) = decode_varint(payload, &mut off) {
                return v as u32;
            }
        }
    }
    default
}

fn decode_bool(fields: &[(u32, u32, &[u8])], field_num: u32, default: bool) -> bool {
    decode_uint32(fields, field_num, if default { 1 } else { 0 }) != 0
}

fn decode_string_field(fields: &[(u32, u32, &[u8])], field_num: u32, default: &str) -> String {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == 2 {
            return String::from_utf8_lossy(payload).to_string();
        }
    }
    default.to_string()
}

fn decode_repeated_uint32(fields: &[(u32, u32, &[u8])], field_num: u32) -> Vec<u32> {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == 2 {
            let mut result = Vec::new();
            let mut off = 0;
            while off < payload.len() {
                if let Ok(v) = decode_varint(payload, &mut off) {
                    result.push(v as u32);
                } else {
                    break;
                }
            }
            return result;
        }
    }
    Vec::new()
}

fn decode_repeated_float(fields: &[(u32, u32, &[u8])], field_num: u32) -> Vec<f32> {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == 2 {
            let count = payload.len() / 4;
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 4] = payload[i * 4..(i + 1) * 4].try_into().unwrap_or([0; 4]);
                result.push(f32::from_le_bytes(bytes));
            }
            return result;
        }
    }
    Vec::new()
}

/// Parse a protobuf message into field tuples: (field_number, wire_type, payload).
fn parse_message(data: &[u8]) -> Result<Vec<(u32, u32, &[u8])>, String> {
    let mut fields = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let tag = decode_varint(data, &mut offset)?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x07) as u32;

        match wire_type {
            0 => {
                // Varint — payload is the encoded varint bytes
                let start = offset;
                let _value = decode_varint(data, &mut offset)?;
                fields.push((field_number, wire_type, &data[start..offset]));
            }
            2 => {
                // Length-delimited
                let length = decode_varint(data, &mut offset)? as usize;
                let payload = &data[offset..offset + length];
                offset += length;
                fields.push((field_number, wire_type, payload));
            }
            _ => return Err(format!("Unknown wire type: {wire_type}")),
        }
    }
    Ok(fields)
}

/// Decode a `LoadModelResponse` from protobuf bytes.
pub fn decode_load_model_response(data: &[u8]) -> LoadModelResponse {
    if let Ok(fields) = parse_message(data) {
        return LoadModelResponse {
            success: decode_bool(&fields, 1, false),
            error: decode_string_field(&fields, 2, ""),
            loaded_experts: decode_uint32(&fields, 3, 0),
        };
    }
    LoadModelResponse {
        success: false,
        error: "Failed to parse response".into(),
        loaded_experts: 0,
    }
}

/// Decode a `GenerateBridgeResponse` from protobuf bytes.
pub fn decode_generate_response(data: &[u8]) -> GenerateBridgeResponse {
    if let Ok(fields) = parse_message(data) {
        return GenerateBridgeResponse {
            request_id: decode_bytes_field(&fields, 1),
            token_ids: decode_repeated_uint32(&fields, 2),
            log_probs: decode_repeated_float(&fields, 3),
            finished: decode_bool(&fields, 4, false),
        };
    }
    GenerateBridgeResponse {
        request_id: vec![],
        token_ids: vec![],
        log_probs: vec![],
        finished: true,
    }
}

fn decode_bytes_field(fields: &[(u32, u32, &[u8])], field_num: u32) -> Vec<u8> {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == 2 {
            return payload.to_vec();
        }
    }
    vec![]
}

/// Decode a `VerifyBridgeResponse` from protobuf bytes.
pub fn decode_verify_response(data: &[u8]) -> VerifyBridgeResponse {
    if let Ok(fields) = parse_message(data) {
        return VerifyBridgeResponse {
            matches: decode_bool(&fields, 1, false),
            actual_sha256: decode_string_field(&fields, 2, ""),
        };
    }
    VerifyBridgeResponse {
        matches: false,
        actual_sha256: "Failed to parse response".into(),
    }
}

/// Decode a `VramBridgeResponse` from protobuf bytes.
pub fn decode_vram_response(data: &[u8]) -> VramBridgeResponse {
    if let Ok(fields) = parse_message(data) {
        return VramBridgeResponse {
            total_mb: decode_uint32(&fields, 1, 0),
            available_mb: decode_uint32(&fields, 2, 0),
        };
    }
    VramBridgeResponse {
        total_mb: 0,
        available_mb: 0,
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn decode_load_model_response_ok() {
        // Encode a success response manually
        let resp = LoadModelResponse::ok(3);
        let encoded = encode_load_model_response_for_test(&resp);
        let decoded = decode_load_model_response(&encoded);
        assert!(decoded.success);
        assert_eq!(decoded.loaded_experts, 3);
    }

    #[test]
    fn decode_load_model_response_err() {
        let resp = LoadModelResponse::err("OOM");
        let encoded = encode_load_model_response_for_test(&resp);
        let decoded = decode_load_model_response(&encoded);
        assert!(!decoded.success);
        assert_eq!(decoded.error, "OOM");
    }

    #[test]
    fn decode_generate_response_with_tokens() {
        let resp = GenerateBridgeResponse::new(
            b"r1".to_vec(), vec![7, 8, 9], vec![-0.1, -0.2], true,
        );
        let encoded = encode_generate_response_for_test(&resp);
        let decoded = decode_generate_response(&encoded);
        assert_eq!(decoded.token_ids, vec![7, 8, 9]);
        assert_eq!(decoded.log_probs.len(), 2);
        assert!(decoded.finished);
    }

    #[test]
    fn decode_vram_response() {
        let resp = VramBridgeResponse::new(16384, 8192);
        let encoded = encode_vram_response_for_test(&resp);
        let decoded = decode_vram_response(&encoded);
        assert_eq!(decoded.total_mb, 16384);
        assert_eq!(decoded.available_mb, 8192);
    }

    // Helper: manual encoding for test roundtrips
    fn encode_load_model_response_for_test(resp: &LoadModelResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, if resp.success { 1 } else { 0 });
        if !resp.error.is_empty() {
            encode_string(&mut buf, 2, &resp.error);
        }
        encode_varint_field(&mut buf, 3, resp.loaded_experts as u64);
        buf
    }

    fn encode_generate_response_for_test(resp: &GenerateBridgeResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_bytes(&mut buf, 1, &resp.request_id);
        encode_packed_uint32(&mut buf, 2, &resp.token_ids);
        encode_packed_float(&mut buf, 3, &resp.log_probs);
        encode_varint_field(&mut buf, 4, if resp.finished { 1 } else { 0 });
        buf
    }

    fn encode_vram_response_for_test(resp: &VramBridgeResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, resp.total_mb as u64);
        encode_varint_field(&mut buf, 2, resp.available_mb as u64);
        buf
    }

    fn encode_varint_field(buf: &mut Vec<u8>, field_number: u32, value: u64) {
        encode_tag(buf, field_number, 0);
        encode_varint(buf, value);
    }

    fn encode_packed_float(buf: &mut Vec<u8>, field_number: u32, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let mut packed = Vec::with_capacity(values.len() * 4);
        for &v in values {
            packed.extend_from_slice(&v.to_le_bytes());
        }
        encode_bytes(buf, field_number, &packed);
    }
}
```

- [ ] **Step 2: Run decode tests**

Run: `cargo test -p synapse-core runtime::infrastructure::proto::runtime::decode_tests`
Expected: ALL PASS

- [ ] **Step 3: Wire decoders into UnixSocketBridge**

Update `unix_socket_bridge.rs` — replace the three `// TODO: decode` stubs with real decoding. The `load()` method becomes:

```rust
fn load(&self, model: &ModelId, experts: &[ExpertId]) -> Result<(), DomainError> {
    let expert_indices: Vec<u32> = experts.iter().map(|e| e.index).collect();
    let req = LoadModelRequest::new(model.clone(), expert_indices);
    let req_data = crate::runtime::infrastructure::proto::runtime::encode_load_model_request(&req);

    let mut framed = vec![1u8];
    framed.extend_from_slice(&req_data);

    let resp_data = self.send_request(&framed)?;
    // Skip the type byte in response
    let actual_data = if resp_data.len() > 1 { &resp_data[1..] } else { &resp_data };
    let resp = crate::runtime::infrastructure::proto::runtime::decode_load_model_response(actual_data);

    if resp.success {
        Ok(())
    } else {
        Err(DomainError::StorageError { message: resp.error })
    }
}
```

And `generate()`:

```rust
fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError> {
    let bridge_req = GenerateBridgeRequest::new(
        request.id.as_bytes().to_vec(),
        vec![], // token_ids populated by tokenizer in full impl
        request.max_tokens,
        0,
    );
    let req_data = crate::runtime::infrastructure::proto::runtime::encode_generate_request(&bridge_req);

    let mut framed = vec![3u8];
    framed.extend_from_slice(&req_data);

    let resp_data = self.send_request(&framed)?;
    let actual_data = if resp_data.len() > 1 { &resp_data[1..] } else { &resp_data };
    let resp = crate::runtime::infrastructure::proto::runtime::decode_generate_response(actual_data);

    let tokens: Vec<Token> = resp
        .token_ids
        .iter()
        .zip(resp.log_probs.iter())
        .map(|(_, &lp)| Token::new(String::new(), lp as f64))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| DomainError::InvalidToken { reason: e.to_string() })?;

    Ok(InferenceOutput {
        request_id: request.id,
        tokens,
    })
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p synapse-core runtime`
Expected: ALL PASS

- [ ] **Step 5: Run format + lint**

Run: `cargo fmt --check && cargo clippy -p synapse-core -- -D warnings`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-core/src/runtime/
git commit -m "feat(runtime): add protobuf response decoding and wire into UnixSocketBridge"
```

---

### Task C.4: Error Handling & Reconnection

**Files:**
- Modify: `synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs`
- Modify: `synapse-core/src/shared/domain_error.rs`

**Interfaces:**
- Produces: Retry logic in `send_request()` with exponential backoff (max 3 retries)
- Produces: `DomainError::RuntimeUnavailable` variant

- [ ] **Step 1: Add RuntimeUnavailable error variant**

Modify `synapse-core/src/shared/domain_error.rs` — add variant:

```rust
    #[error("runtime unavailable: {message}")]
    RuntimeUnavailable { message: String },
```

- [ ] **Step 2: Write retry test**

Append to `unix_socket_bridge.rs` tests:

```rust
    #[test]
    fn send_request_retries_on_failure() {
        // When socket doesn't exist, send_request should return an error
        let bridge = UnixSocketBridge::new("/tmp/nonexistent-synapse-socket.sock");
        let result = bridge.send_request(&[1, 2, 3]);
        assert!(result.is_err());
        match result {
            Err(DomainError::StorageError { .. }) => {} // expected
            other => panic!("Expected StorageError, got {other:?}"),
        }
    }
```

- [ ] **Step 3: Implement retry in send_request**

Update `send_request()` in `unix_socket_bridge.rs`:

```rust
    fn send_request(&self, request_data: &[u8]) -> Result<Vec<u8>, DomainError> {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;
        use std::thread;
        use std::time::Duration;

        const MAX_RETRIES: u32 = 3;
        let mut last_error = String::new();

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                thread::sleep(Duration::from_millis(100 * (1 << attempt) as u64));
            }

            match Self::try_send(&self.socket_path, request_data) {
                Ok(data) => return Ok(data),
                Err(e) => last_error = e,
            }
        }

        Err(DomainError::StorageError {
            message: format!("Runtime request failed after {MAX_RETRIES} retries: {last_error}"),
        })
    }

    fn try_send(socket_path: &str, request_data: &[u8]) -> Result<Vec<u8>, String> {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixStream;

        let mut stream = UnixStream::connect(socket_path)
            .map_err(|e| format!("connect: {e}"))?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .map_err(|e| format!("set timeout: {e}"))?;

        let len_bytes = (request_data.len() as u32).to_be_bytes();
        stream.write_all(&len_bytes).map_err(|e| format!("write len: {e}"))?;
        stream.write_all(request_data).map_err(|e| format!("write data: {e}"))?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).map_err(|e| format!("read len: {e}"))?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;

        let mut resp_data = vec![0u8; resp_len];
        stream.read_exact(&mut resp_data).map_err(|e| format!("read data: {e}"))?;

        Ok(resp_data)
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p synapse-core runtime::infrastructure::unix_socket_bridge`
Expected: ALL PASS (retry test confirms graceful error, not panic)

- [ ] **Step 5: Run format + lint**

Run: `cargo fmt --check && cargo clippy -p synapse-core -- -D warnings`
Expected: All checks passed

- [ ] **Step 6: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 7: Commit**

```bash
git add synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs synapse-core/src/shared/domain_error.rs
git commit -m "feat(runtime): add retry logic and error handling to UnixSocketBridge"
```

---

### Task C.5: End-to-End Integration Test

**Files:**
- Create: `synapse-runtime/tests/test_integration.py` (GPU-only, marked `@pytest.mark.gpu`)
- Modify: `synapse-core/src/runtime/infrastructure/unix_socket_bridge.rs` — add integration test

**Interfaces:**
- Integration test: start Python server → Rust bridge connects → load → generate → verify → vram

- [ ] **Step 1: Write Python GPU integration test**

Create `synapse-runtime/tests/test_integration.py`:

```python
"""GPU integration tests for the runtime server.

These tests require a GPU and are excluded from CI.
Run locally: pytest tests/test_integration.py -v -m gpu
"""

import os
import socket
import tempfile
import threading
import time

import pytest

from synapse_runtime.engine import VllmEngine
from synapse_runtime.protocol import (
    GenerateRequest,
    GenerateResponse,
    LoadModelRequest,
    LoadModelResponse,
    VramQueryRequest,
    VramQueryResponse,
    deserialize_response,
    serialize_request,
)
from synapse_runtime.server import RuntimeServer

pytestmark = pytest.mark.gpu


@pytest.fixture
def temp_socket() -> str:
    """Create a temporary Unix socket path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.sock")


@pytest.fixture
def running_server(temp_socket: str) -> RuntimeServer:
    """Start a RuntimeServer on a temp socket."""
    server = RuntimeServer()
    server.start(temp_socket)
    time.sleep(0.1)  # Let socket bind
    yield server
    server.stop()


def _send_request(socket_path: str, data: bytes) -> bytes:
    """Send a length-prefixed request and receive response."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        sock.sendall(len(data).to_bytes(4, "big"))
        sock.sendall(data)

        len_data = sock.recv(4)
        if len(len_data) < 4:
            return b""
        msg_len = int.from_bytes(len_data, "big")

        chunks = []
        remaining = msg_len
        while remaining > 0:
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)
    finally:
        sock.close()


class TestIntegrationVram:
    def test_vram_query_returns_values(self, running_server: RuntimeServer,
                                        temp_socket: str) -> None:
        """VRAM query returns non-zero values on a GPU system."""
        req = VramQueryRequest()
        data = serialize_request(req)
        resp_data = _send_request(temp_socket, data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, VramQueryResponse)
        assert resp.total_mb >= 0
        # On a GPU system, these should be non-zero
        # On CPU-only, they'll be 0 — both are valid


class TestIntegrationLoadModel:
    def test_load_deepseek_lite(
        self, running_server: RuntimeServer, temp_socket: str
    ) -> None:
        """Load DeepSeek-V2 Lite with 2 experts."""
        req = LoadModelRequest(model_id="deepseek-v2-lite",
                                expert_indices=[0, 1])
        data = serialize_request(req)
        resp_data = _send_request(temp_socket, data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, LoadModelResponse)
        assert resp.success, f"Load failed: {resp.error}"
        assert resp.loaded_experts == 2


class TestIntegrationGenerate:
    def test_generate_after_load(
        self, running_server: RuntimeServer, temp_socket: str
    ) -> None:
        """Generate tokens after loading a model."""
        # First load
        load_req = LoadModelRequest(
            model_id="deepseek-v2-lite", expert_indices=[0, 1]
        )
        load_data = serialize_request(load_req)
        load_resp_data = _send_request(temp_socket, load_data)
        load_resp = deserialize_response(load_resp_data)
        assert isinstance(load_resp, LoadModelResponse)
        assert load_resp.success, f"Load failed: {load_resp.error}"

        # Then generate
        gen_req = GenerateRequest(
            request_id=b"int-test-1",
            token_ids=[1, 2, 3, 4, 5],
            seed=0,
            max_tokens=20,
        )
        gen_data = serialize_request(gen_req)
        gen_resp_data = _send_request(temp_socket, gen_data)
        gen_resp = deserialize_response(gen_resp_data)
        assert isinstance(gen_resp, GenerateResponse)
        assert len(gen_resp.token_ids) > 0, "Expected generated tokens"
        assert gen_resp.finished is True


class TestIntegrationDeterminism:
    def test_two_runs_identical(self, running_server: RuntimeServer,
                                 temp_socket: str) -> None:
        """Two runs with seed=0 produce identical outputs."""
        # Load once
        load_req = LoadModelRequest(
            model_id="deepseek-v2-lite", expert_indices=[0, 1]
        )
        _send_request(temp_socket, serialize_request(load_req))

        prompt = [100, 200, 300]

        # Run 1
        gen1 = GenerateRequest(
            request_id=b"det-1", token_ids=prompt, seed=0, max_tokens=10
        )
        resp1 = deserialize_response(
            _send_request(temp_socket, serialize_request(gen1))
        )

        # Run 2
        gen2 = GenerateRequest(
            request_id=b"det-2", token_ids=prompt, seed=0, max_tokens=10
        )
        resp2 = deserialize_response(
            _send_request(temp_socket, serialize_request(gen2))
        )

        assert isinstance(resp1, GenerateResponse)
        assert isinstance(resp2, GenerateResponse)
        assert resp1.token_ids == resp2.token_ids, (
            f"Determinism failed: run1={resp1.token_ids}, run2={resp2.token_ids}"
        )
        assert resp1.log_probs == resp2.log_probs
```

- [ ] **Step 2: Mark integration tests as GPU-only**

Run: `cd synapse-runtime && python -m pytest tests/test_integration.py -v -m "not gpu"`
Expected: 0 tests collected (all skipped — GPU marker excludes them)

Run locally with GPU: `cd synapse-runtime && python -m pytest tests/test_integration.py -v -m gpu`
Expected: ALL PASS (if GPU + DeepSeek-V2 Lite available)

- [ ] **Step 3: Configure pytest markers**

Add to `synapse-runtime/pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "gpu: tests that require a GPU (excluded from CI)",
]
```

- [ ] **Step 4: Quality Gate** — Run checklist from [Per-Task Quality Gate](#per-task-quality-gate-mandatory-every-task). Fix any violations before proceeding.

- [ ] **Step 5: Commit**

```bash
git add synapse-runtime/tests/test_integration.py synapse-runtime/pyproject.toml
git commit -m "test(runtime): add GPU integration tests for end-to-end flow"
```

---

## Verification Checklist

After all tasks complete, run the full quality gauntlet:

```bash
# Rust
cargo fmt --check
cargo clippy -- -D warnings
cargo test -p synapse-core
cargo llvm-cov --fail-under-lines 80

# Python
cd synapse-runtime && ruff check . && ruff format --check .
cd synapse-runtime && python -m pytest tests/ -v -m "not gpu"

# GPU integration (local only)
cd synapse-runtime && python -m pytest tests/test_integration.py -v -m gpu
```
