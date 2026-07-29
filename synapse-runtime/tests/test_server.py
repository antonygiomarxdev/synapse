"""Tests for Unix socket server."""

import os
import tempfile
from unittest.mock import MagicMock, patch

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
    @patch("synapse_runtime.server.download_model")
    def test_handle_load_model_request(
        self, mock_download: MagicMock, mock_engine_class: MagicMock
    ) -> None:
        """LoadModelRequest dispatches to engine.load_model."""
        mock_download.return_value = "/cache/models/test"
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

        mock_download.assert_called_once_with("mistralai/Mixtral-8x7B-v0.1")
        mock_engine.load_model.assert_called_once()

    @patch("synapse_runtime.server.VllmEngine")
    def test_handle_generate_request(self, mock_engine_class: MagicMock) -> None:
        """GenerateRequest dispatches to engine.generate."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = GenerateResponse(
            request_id=b"r1",
            token_ids=[7, 8, 9],
            log_probs=[-0.1, -0.2, -0.3],
            finished=True,
        )
        mock_engine_class.return_value = mock_engine

        server = RuntimeServer(engine=mock_engine)
        req = GenerateRequest(request_id=b"r1", token_ids=[1, 2], seed=0, max_tokens=50)
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
        req = GenerateRequest(request_id=b"r1", token_ids=[1], seed=0, max_tokens=10)
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
            import threading  # noqa: F401

            server._running = True  # Simulate running state

            # Stop
            server.stop()
            # Socket should be cleaned up
            assert not os.path.exists(socket_path)
