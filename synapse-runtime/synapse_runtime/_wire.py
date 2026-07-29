"""Protobuf wire-format encoding/decoding helpers.

Used by protocol.py to serialize/deserialize bridge message dataclasses.
Low-level varint, length-delimited, and packed repeated field routines.
"""

from __future__ import annotations

import struct

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
            if offset + length > len(data):
                raise ValueError(
                    f"Truncated field {field_number}: declared {length} bytes, "
                    f"available {len(data) - offset}"
                )
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
