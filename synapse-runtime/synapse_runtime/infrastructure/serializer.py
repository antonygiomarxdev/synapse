"""Protobuf serialization/deserialization adapter.

Infrastructure layer — converts between domain dataclasses and wire bytes.
"""

from __future__ import annotations

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
from synapse_runtime.protocol import (
    GenerateRequest,
    GenerateResponse,
    LoadModelRequest,
    LoadModelResponse,
    VerifyHashRequest,
    VerifyHashResponse,
    VramQueryRequest,
    VramQueryResponse,
)

# ── Per-type serializers ─────────────────────────────────────────────


def serialize_load_model_request(req: LoadModelRequest) -> bytes:
    """Encode LoadModelRequest to protobuf bytes."""
    body = _encode_string(1, req.model_id)
    body += _encode_repeated_uint32(2, req.expert_indices)
    return body


def deserialize_load_model_request(data: bytes) -> LoadModelRequest:
    """Decode protobuf bytes to LoadModelRequest."""
    fields = _parse_fields(data)
    return LoadModelRequest(
        model_id=_get_string(fields, 1),
        expert_indices=_get_repeated_uint32(fields, 2),
    )


def serialize_load_model_response(resp: LoadModelResponse) -> bytes:
    """Encode LoadModelResponse to protobuf bytes."""
    body = _encode_bool(1, resp.success)
    if resp.error:
        body += _encode_string(2, resp.error)
    body += _encode_uint32(3, resp.loaded_experts)
    return body


def deserialize_load_model_response(data: bytes) -> LoadModelResponse:
    """Decode protobuf bytes to LoadModelResponse."""
    fields = _parse_fields(data)
    return LoadModelResponse(
        success=_get_bool(fields, 1),
        error=_get_string(fields, 2),
        loaded_experts=_get_varint(fields, 3),
    )


def serialize_generate_request(req: GenerateRequest) -> bytes:
    """Encode GenerateRequest to protobuf bytes."""
    body = _encode_bytes_field(1, req.request_id)
    body += _encode_repeated_uint32(2, req.token_ids)
    body += _encode_uint32(3, req.seed)
    body += _encode_uint32(4, req.max_tokens)
    return body


def deserialize_generate_request(data: bytes) -> GenerateRequest:
    """Decode protobuf bytes to GenerateRequest."""
    fields = _parse_fields(data)
    return GenerateRequest(
        request_id=_get_bytes(fields, 1),
        token_ids=_get_repeated_uint32(fields, 2),
        seed=_get_varint(fields, 3),
        max_tokens=_get_varint(fields, 4),
    )


def serialize_generate_response(resp: GenerateResponse) -> bytes:
    """Encode GenerateResponse to protobuf bytes."""
    body = _encode_bytes_field(1, resp.request_id)
    body += _encode_repeated_uint32(2, resp.token_ids)
    body += _encode_repeated_float(3, resp.log_probs)
    body += _encode_bool(4, resp.finished)
    return body


def deserialize_generate_response(data: bytes) -> GenerateResponse:
    """Decode protobuf bytes to GenerateResponse."""
    fields = _parse_fields(data)
    return GenerateResponse(
        request_id=_get_bytes(fields, 1),
        token_ids=_get_repeated_uint32(fields, 2),
        log_probs=_get_repeated_float(fields, 3),
        finished=_get_bool(fields, 4),
    )


def serialize_verify_hash_request(req: VerifyHashRequest) -> bytes:
    """Encode VerifyHashRequest to protobuf bytes."""
    body = _encode_string(1, req.model_id)
    body += _encode_string(2, req.expected_sha256)
    return body


def deserialize_verify_hash_request(data: bytes) -> VerifyHashRequest:
    """Decode protobuf bytes to VerifyHashRequest."""
    fields = _parse_fields(data)
    return VerifyHashRequest(
        model_id=_get_string(fields, 1),
        expected_sha256=_get_string(fields, 2),
    )


def serialize_verify_hash_response(resp: VerifyHashResponse) -> bytes:
    """Encode VerifyHashResponse to protobuf bytes."""
    body = _encode_bool(1, resp.matches)
    body += _encode_string(2, resp.actual_sha256)
    return body


def deserialize_verify_hash_response(data: bytes) -> VerifyHashResponse:
    """Decode protobuf bytes to VerifyHashResponse."""
    fields = _parse_fields(data)
    return VerifyHashResponse(
        matches=_get_bool(fields, 1),
        actual_sha256=_get_string(fields, 2),
    )


def serialize_vram_query_request(req: VramQueryRequest) -> bytes:
    """Encode VramQueryRequest to protobuf bytes (empty message)."""
    return b""


def deserialize_vram_query_request(data: bytes) -> VramQueryRequest:
    """Decode protobuf bytes to VramQueryRequest."""
    return VramQueryRequest()


def serialize_vram_query_response(resp: VramQueryResponse) -> bytes:
    """Encode VramQueryResponse to protobuf bytes."""
    body = _encode_uint32(1, resp.total_mb)
    body += _encode_uint32(2, resp.available_mb)
    return body


def deserialize_vram_query_response(data: bytes) -> VramQueryResponse:
    """Decode protobuf bytes to VramQueryResponse."""
    fields = _parse_fields(data)
    return VramQueryResponse(
        total_mb=_get_varint(fields, 1),
        available_mb=_get_varint(fields, 2),
    )


# ── Dispatch tables ──────────────────────────────────────────────────

_REQUEST_SERIALIZERS = {
    1: serialize_load_model_request,
    3: serialize_generate_request,
    5: serialize_verify_hash_request,
    7: serialize_vram_query_request,
}

_REQUEST_DESERIALIZERS = {
    1: deserialize_load_model_request,
    3: deserialize_generate_request,
    5: deserialize_verify_hash_request,
    7: deserialize_vram_query_request,
}

_RESPONSE_SERIALIZERS = {
    2: serialize_load_model_response,
    4: serialize_generate_response,
    6: serialize_verify_hash_response,
    8: serialize_vram_query_response,
}

_RESPONSE_DESERIALIZERS = {
    2: deserialize_load_model_response,
    4: deserialize_generate_response,
    6: deserialize_verify_hash_response,
    8: deserialize_vram_query_response,
}


# ── Message type dispatch ─────────────────────────────────────────────


def serialize_request(msg: object) -> bytes:
    """Serialize a request dataclass to protobuf bytes with type prefix."""
    msg_type = getattr(msg, "MESSAGE_TYPE", 0)
    serializer = _REQUEST_SERIALIZERS.get(msg_type)
    if serializer is None:
        raise ValueError(f"Unknown request message type: {msg_type}")
    body = serializer(msg)
    return _encode_varint(msg_type) + body


def deserialize_request(data: bytes) -> object:
    """Deserialize protobuf bytes to the correct request dataclass."""
    if not data:
        raise ValueError("Empty message")
    msg_type, offset = _decode_varint(data, 0)
    deserializer = _REQUEST_DESERIALIZERS.get(msg_type)
    if deserializer is None:
        raise ValueError(f"Unknown request message type: {msg_type}")
    return deserializer(data[offset:])


def serialize_response(msg: object) -> bytes:
    """Serialize a response dataclass to protobuf bytes with type prefix."""
    msg_type = getattr(msg, "MESSAGE_TYPE", 0)
    serializer = _RESPONSE_SERIALIZERS.get(msg_type)
    if serializer is None:
        raise ValueError(f"Unknown response message type: {msg_type}")
    body = serializer(msg)
    return _encode_varint(msg_type) + body


def deserialize_response(data: bytes) -> object:
    """Deserialize protobuf bytes to the correct response dataclass."""
    if not data:
        raise ValueError("Empty message")
    msg_type, offset = _decode_varint(data, 0)
    deserializer = _RESPONSE_DESERIALIZERS.get(msg_type)
    if deserializer is None:
        raise ValueError(f"Unknown response message type: {msg_type}")
    return deserializer(data[offset:])
