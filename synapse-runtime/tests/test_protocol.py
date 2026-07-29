"""Tests for protocol dataclasses and serialization."""

import pytest

from synapse_runtime.infrastructure.serializer import (
    deserialize_request,
    deserialize_response,
    serialize_request,
    serialize_response,
)
from synapse_runtime.protocol import (
    GenerateRequest,
    GenerateResponse,
    LoadModelRequest,
    VramQueryRequest,
    VramQueryResponse,
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
        assert restored.log_probs == pytest.approx([-0.1, -0.2, -0.3], rel=1e-6)
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
        with pytest.raises(ValueError):
            deserialize_request(b"not-a-valid-protobuf-message")
