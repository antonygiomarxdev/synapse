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
    return _encode_field(field_number, _WIRE_LEN, _encode_varint(len(data)) + data)


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
            lst = fields.setdefault(field_number, [])
            lst.append((wire_type, _encode_varint(value)))
        elif wire_type == _WIRE_LEN:
            length, offset = _decode_varint(data, offset)
            payload = data[offset : offset + length]
            offset += length
            fields.setdefault(field_number, []).append((wire_type, payload))
        else:
            raise ValueError(f"Unsupported wire type: {wire_type}")
    return fields


def _get_varint(
    fields: dict[int, list[tuple[int, bytes]]], num: int, default: int = 0
) -> int:
    """Extract a single uint32/bool/enum field."""
    items = fields.get(num, [])
    if not items:
        return default
    value, _ = _decode_varint(items[0][1], 0)
    return value


def _get_string(
    fields: dict[int, list[tuple[int, bytes]]], num: int, default: str = ""
) -> str:
    """Extract a single string field."""
    items = fields.get(num, [])
    if not items:
        return default
    return items[0][1].decode("utf-8")


def _get_bytes(
    fields: dict[int, list[tuple[int, bytes]]], num: int, default: bytes = b""
) -> bytes:
    """Extract a single bytes field."""
    items = fields.get(num, [])
    if not items:
        return default
    return items[0][1]


def _get_bool(
    fields: dict[int, list[tuple[int, bytes]]], num: int, default: bool = False
) -> bool:
    """Extract a single bool field."""
    return bool(_get_varint(fields, num, 1 if default else 0))


def _get_repeated_uint32(
    fields: dict[int, list[tuple[int, bytes]]], num: int
) -> list[int]:
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


def _get_repeated_float(
    fields: dict[int, list[tuple[int, bytes]]], num: int
) -> list[float]:
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
