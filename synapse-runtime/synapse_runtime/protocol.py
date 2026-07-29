"""Bridge protocol dataclasses and protobuf serialization.

Mirrors synapse-core/proto/runtime.proto 1:1.
Uses raw protobuf encoding (wire types) — no compiled .proto needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from synapse_runtime._wire import (
    _decode_varint,
    _encode_bool,
    _encode_bytes_field,
    _encode_repeated_float,
    _encode_repeated_uint32,
    _encode_string,
    _encode_uint32,
    _encode_varint,
    _get_bool,
    _get_bytes,
    _get_repeated_float,
    _get_repeated_uint32,
    _get_string,
    _get_varint,
    _parse_fields,
)

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
    def deserialize(cls, data: bytes) -> LoadModelRequest:
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
    def deserialize(cls, data: bytes) -> LoadModelResponse:
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
    def deserialize(cls, data: bytes) -> GenerateRequest:
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
    def deserialize(cls, data: bytes) -> GenerateResponse:
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
    def deserialize(cls, data: bytes) -> VerifyHashRequest:
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
    def deserialize(cls, data: bytes) -> VerifyHashResponse:
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
    def deserialize(cls, data: bytes) -> VramQueryRequest:
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
    def deserialize(cls, data: bytes) -> VramQueryResponse:
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
    body = msg.serialize()
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
    body = msg.serialize()
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
