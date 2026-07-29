"""Unix socket server for the Synapse runtime bridge.

Listens on a Unix domain socket for protobuf-encoded requests
from the Rust core. Dispatches to engine, loader, and auto_assign.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading

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

# Maps model_id to HuggingFace repository for download.
MODEL_REPO_MAP: dict[str, str] = {
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "kimi-k3": "moonshotai/Kimi-K3",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "qwen2.5-moe": "Qwen/Qwen2.5-57B-A14B",
}


def _resolve_socket_path(socket_path: str) -> str:
    """Resolve and validate a socket path, preventing path traversal.

    Args:
        socket_path: Raw socket path from caller.

    Returns:
        Resolved absolute path.

    Raises:
        ValueError: If the path contains traversal components.
    """
    resolved = os.path.realpath(socket_path)
    if ".." in resolved.split(os.sep):
        raise ValueError(f"Path traversal denied in socket path: {socket_path}")
    return resolved


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
        self._socket_path: str | None = None

    @property
    def is_running(self) -> bool:
        """True if the server is accepting connections."""
        return self._running

    def start(self, socket_path: str = "/tmp/synapse-runtime.sock") -> None:
        """Start listening on a Unix domain socket.

        Args:
            socket_path: Path for the Unix socket file.

        Raises:
            ValueError: If the socket path contains path traversal.
            OSError: If the socket cannot be created.
        """
        if self._running:
            return

        resolved = _resolve_socket_path(socket_path)

        # Clean up stale socket file
        if os.path.exists(resolved):
            os.unlink(resolved)
        self._socket_path = resolved

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._socket.bind(resolved)
            self._socket.listen(5)
            self._socket.settimeout(0.5)
            self._running = True

            self._thread = threading.Thread(
                target=self._accept_loop,
                args=(resolved,),
                daemon=True,
            )
            self._thread.start()
            logger.info("Runtime server listening on %s", resolved)
        except OSError:
            self._socket.close()
            self._socket = None
            raise

    def stop(self) -> None:
        """Stop the server and clean up the socket."""
        self._running = False
        if self._socket:
            with contextlib.suppress(OSError):
                self._socket.close()
        if self._thread:
            self._thread.join()
        if self._socket_path is not None and os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

    def handle_request(self, data: bytes) -> bytes:
        """Handle a single request and return the response.

        Args:
            data: Protobuf-encoded request bytes.

        Returns:
            Protobuf-encoded response bytes.

        Raises:
            ValueError: If the request cannot be deserialized or is unknown.
        """
        try:
            request = deserialize_request(data)
        except ValueError as e:
            logger.error("Failed to deserialize request: %s", e)
            raise

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
            resp = LoadModelResponse(success=False, error=str(e), loaded_experts=0)
        except Exception as e:
            logger.exception("Load model failed unexpectedly")
            resp = LoadModelResponse(success=False, error=str(e), loaded_experts=0)
        return serialize_response(resp)

    def _handle_generate(self, req: GenerateRequest) -> bytes:
        """Handle GenerateRequest."""
        if not self._engine.is_loaded:
            logger.error("Generate request received but engine is not loaded")
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
        except Exception:
            logger.exception("Generate failed unexpectedly")
            resp = GenerateResponse(
                request_id=req.request_id,
                token_ids=[],
                log_probs=[],
                finished=True,
            )
            return serialize_response(resp)

    def _handle_verify_hash(self, req: VerifyHashRequest) -> bytes:
        """Handle VerifyHashRequest."""
        from synapse_runtime.loader import compute_sha256, verify_sha256

        hf_repo = MODEL_REPO_MAP.get(req.model_id, req.model_id)
        try:
            model_path = download_model(hf_repo)
            matches = verify_sha256(model_path, req.expected_sha256)
            actual = compute_sha256(model_path)
            resp = VerifyHashResponse(matches=matches, actual_sha256=actual)
        except ModelNotFoundError as e:
            resp = VerifyHashResponse(matches=False, actual_sha256=f"error: {e}")
        except Exception as e:
            logger.exception("Verify hash failed unexpectedly")
            resp = VerifyHashResponse(matches=False, actual_sha256=f"error: {e}")
        return serialize_response(resp)

    def _handle_vram_query(self, _req: VramQueryRequest) -> bytes:
        """Handle VramQueryRequest."""
        try:
            total, available = detect_vram()
            resp = VramQueryResponse(total_mb=total, available_mb=available)
        except Exception as e:
            logger.error("VRAM query failed: %s", e)
            resp = VramQueryResponse(total_mb=0, available_mb=0)
        return serialize_response(resp)

    def _accept_loop(self, socket_path: str) -> None:
        """Accept connections in a loop."""
        while self._running:
            try:
                if self._socket is None:
                    break
                conn, _ = self._socket.accept()
                self._handle_connection(conn)
            except TimeoutError:
                continue
            except OSError:
                if self._running:
                    logger.debug("Socket accept interrupted")
                break

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single client connection."""
        try:
            with conn:
                # Read length-prefixed message:
                # 4-byte big-endian length + payload
                length_data = bytearray()
                while len(length_data) < 4:
                    chunk = conn.recv(4 - len(length_data))
                    if not chunk:
                        return
                    length_data.extend(chunk)
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
        except (OSError, ValueError) as e:
            logger.debug("Connection error: %s", e)
