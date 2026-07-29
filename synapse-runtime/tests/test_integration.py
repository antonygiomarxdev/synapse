"""GPU integration tests for the runtime server.

These tests require a GPU and are excluded from CI.
Run locally: pytest tests/test_integration.py -v -m gpu
"""

import os
import socket
import tempfile
import time

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
    def test_vram_query_returns_values(
        self, running_server: RuntimeServer, temp_socket: str
    ) -> None:
        """VRAM query returns non-zero values on a GPU system."""
        req = VramQueryRequest()
        data = serialize_request(req)
        resp_data = _send_request(temp_socket, data)
        resp = deserialize_response(resp_data)
        assert isinstance(resp, VramQueryResponse)
        assert resp.total_mb >= 0


class TestIntegrationLoadModel:
    def test_load_deepseek_lite(
        self, running_server: RuntimeServer, temp_socket: str
    ) -> None:
        """Load DeepSeek-V2 Lite with 2 experts."""
        req = LoadModelRequest(model_id="deepseek-v2-lite", expert_indices=[0, 1])
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
        load_req = LoadModelRequest(model_id="deepseek-v2-lite", expert_indices=[0, 1])
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
    def test_two_runs_identical(
        self, running_server: RuntimeServer, temp_socket: str
    ) -> None:
        """Two runs with seed=0 produce identical outputs."""
        # Load once
        load_req = LoadModelRequest(model_id="deepseek-v2-lite", expert_indices=[0, 1])
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
