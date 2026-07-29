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
import tomllib
from pathlib import Path

from synapse_runtime.auto_assign import detect_vram
from synapse_runtime.config import get_config
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


def _load_model_repo_map() -> dict[str, str]:
    """Load model_id -> HF repo mapping from models.toml."""
    mapping: dict[str, str] = {}
    try:
        base = Path(__file__).resolve().parent.parent.parent
        with open(base / "config" / "models.toml", "rb") as f:
            data = tomllib.load(f)
        for m in data.get("models", []):
            mid = m.get("id")
            repo = m.get("hf_repo")
            if mid and repo:
                mapping[mid] = repo
    except (FileNotFoundError, PermissionError):
        pass
    return mapping


MODEL_REPO_MAP = _load_model_repo_map()


def _resolve_socket_path(socket_path: str) -> str:
    """Resolve and validate a socket path, preventing path traversal.

    Args:
        socket_path: Raw socket path from caller.

    Returns:
        Resolved absolute path.

    Raises:
        ValueError: If the path contains traversal components.
    """
    resolved = os.path.normpath(socket_path)
    if ".." in resolved.split(os.sep):
        raise ValueError(f"Path traversal denied in socket path: {socket_path}")
    resolved = os.path.realpath(resolved)
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
        self._cfg = get_config()

    @property
    def is_running(self) -> bool:
        """True if the server is accepting connections."""
        return self._running

    def start(self, socket_path: str | None = None) -> None:
        """Start listening on a Unix domain socket.

        Args:
            socket_path: Path for the Unix socket file. Defaults to config value.

        Raises:
            ValueError: If the socket path contains path traversal.
            OSError: If the socket cannot be created.
        """
        if self._running:
            return

        resolved = _resolve_socket_path(socket_path or self._cfg.runtime.socket_path)

        # Clean up stale socket file
        if os.path.exists(resolved):
            os.unlink(resolved)
        self._socket_path = resolved

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._socket.bind(resolved)
            self._socket.listen(self._cfg.runtime.socket_backlog)
            self._socket.settimeout(self._cfg.runtime.socket_timeout)
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
            ValueError: If the request type is unknown.
        """
        try:
            request = deserialize_request(data)
        except ValueError as e:
            logger.error("Failed to deserialize request: %s", e)
            resp = GenerateResponse(
                request_id=f"ERROR:deserialize:{e}".encode(),
                token_ids=[],
                log_probs=[],
                finished=True,
            )
            return serialize_response(resp)

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
                request_id=b"ERROR:engine_not_loaded",
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
                request_id=f"ERROR:{e}".encode(),
                token_ids=[],
                log_probs=[],
                finished=True,
            )
            return serialize_response(resp)
        except Exception:
            logger.exception("Generate failed unexpectedly")
            resp = GenerateResponse(
                request_id=b"ERROR:unexpected_exception",
                token_ids=[],
                log_probs=[],
                finished=True,
            )
            return serialize_response(resp)

    def _handle_verify_hash(self, req: VerifyHashRequest) -> bytes:
        """Handle VerifyHashRequest."""
        from synapse_runtime.loader import compute_sha256

        hf_repo = MODEL_REPO_MAP.get(req.model_id, req.model_id)
        try:
            model_path = download_model(hf_repo)
            actual = compute_sha256(model_path)
            matches = actual == req.expected_sha256
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
                chunk = conn.recv(
                    min(msg_len - len(data), self._cfg.runtime.recv_chunk_size)
                )
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
            # Send a minimal error frame so the client doesn't block
            error_resp = (
                b"\x04" + b"\x00\x00\x00\x00"  # type 4 = GenerateResponse, no fields
            )
            try:
                conn.sendall(len(error_resp).to_bytes(4, "big"))
                conn.sendall(error_resp)
            except OSError:
                pass
        finally:
            conn.close()
