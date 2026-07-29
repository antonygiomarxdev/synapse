"""Synapse Spike Worker — inference adapter over Unix socket + protobuf.

Supports multiple backends:
  vllm   — vLLM offline engine (needs GPU + model download)
  ollama — Ollama HTTP API (needs ollama serve running)
  mock   — deterministic mock for protocol testing (no GPU)

Usage:
    python -m synapse_runtime.worker --socket /tmp/syn-0.sock --model ollama:qwen3:8b

Protocol (per message):
    [4 bytes BE length] [protobuf-encoded SpikeRequest or SpikeResponse]

Lifecycle:
    1. Load model / connect to backend
    2. Create Unix socket and listen
    3. Print "READY" to stdout -> coordinator proceeds
    4. Accept one connection, process requests until EOF
    5. Exit
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.request

from synapse_runtime import spike_pb2


def load_model(model_name: str):
    """Load the inference engine based on model name prefix.

    Prefixes:
        ollama:<name>  -> Ollama HTTP API (e.g. ollama:qwen3:8b)
        mock:<name>    -> MockEngine (protocol testing, no GPU)
        anything else  -> vLLM (original behavior)
    """
    if model_name.startswith("ollama:"):
        ollama_model = model_name.removeprefix("ollama:")
        return OllamaEngine(ollama_model)

    if model_name.startswith("mock:"):
        mock_name = model_name.removeprefix("mock:")
        return MockEngine(mock_name)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Fallback to mock engine.", file=sys.stderr)
        return MockEngine(model_name)

    print(f"Loading model: {model_name}", file=sys.stderr)
    start = time.monotonic()
    try:
        engine = LLM(
            model=model_name,
            gpu_memory_utilization=0.40,
            max_model_len=2048,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"vLLM failed to load model: {exc}", file=sys.stderr)
        print("Fallback to mock engine.", file=sys.stderr)
        return MockEngine(model_name)

    elapsed = time.monotonic() - start
    print(f"Model loaded in {elapsed:.1f}s", file=sys.stderr)
    return VllmEngine(engine)


class VllmEngine:
    """Thin wrapper around vLLM's offline LLM for the spike protocol."""

    def __init__(self, llm):
        self._llm = llm
        from vllm import SamplingParams
        self._SamplingParams = SamplingParams

    def generate(self, prompt: str, max_tokens: int, seed: int) -> tuple[str, int, float]:
        """Run generation. Returns (text, num_tokens, elapsed_ms)."""
        start = time.monotonic()
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0 if seed != 0 else 0.7,
            seed=seed if seed != 0 else None,
        )
        outputs = self._llm.generate([prompt], params)
        elapsed = (time.monotonic() - start) * 1000.0

        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)
        else:
            text = ""
            tokens = 0

        return text, tokens, elapsed


class OllamaEngine:
    """Inference via Ollama's HTTP API (localhost:11434).

    Uses the /api/generate endpoint with stream=false.
    The model must already be pulled (ollama pull <name>).
    """

    def __init__(self, model_name: str):
        self._model = model_name
        self._url = "http://localhost:11434/api/generate"
        # Verify the model exists
        try:
            tags_url = "http://localhost:11434/api/tags"
            with urllib.request.urlopen(tags_url, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if model_name not in models:
                    print(f"WARNING: model '{model_name}' not found in Ollama. "
                          f"Available: {models}", file=sys.stderr)
                else:
                    print(f"Ollama backend ready: {model_name} (local)", file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: cannot reach Ollama at localhost:11434 ({exc})", file=sys.stderr)

    def generate(self, prompt: str, max_tokens: int, seed: int) -> tuple[str, int, float]:
        """Run generation via Ollama HTTP API."""
        start = time.monotonic()
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "seed": seed if seed != 0 else 42,
                "temperature": 0.0 if seed != 0 else 0.7,
            },
        }).encode()
        req = urllib.request.Request(self._url, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000.0
            return f"[Ollama error: {exc}]", 0, elapsed

        elapsed = (time.monotonic() - start) * 1000.0
        text = data.get("response", "") or data.get("thinking", "")
        tokens = data.get("eval_count", len(text.split()))
        return text.strip(), tokens, elapsed


class MockEngine:
    """Deterministic mock engine for testing the protocol without a GPU.

    Returns echo responses with the model name and prompt length.
    This is only for protocol validation - real inference needs a backend.
    """

    def __init__(self, model_name: str):
        self._model = model_name

    def generate(self, prompt: str, max_tokens: int, seed: int) -> tuple[str, int, float]:
        """Return a deterministic mock response."""
        import hashlib

        h = hashlib.sha256(f"{prompt}:{seed}".encode()).hexdigest()[:16]
        text = (
            f"[MOCK:{self._model}] Received prompt ({len(prompt)} chars, "
            f"seed={seed}). Echo: {prompt[:80]}... Hash: {h}"
        )
        tokens = len(text.split())
        elapsed_ms = 5.0  # mock latency
        return text, tokens, elapsed_ms


def recv_message(sock) -> bytes:
    """Read a length-prefixed protobuf message from a socket.

    Returns the raw payload bytes.
    Raises EOFError on clean shutdown, OSError on other errors.
    """
    len_bytes = b""
    while len(len_bytes) < 4:
        chunk = sock.recv(4 - len(len_bytes))
        if not chunk:
            raise EOFError("Client disconnected")
        len_bytes += chunk
    msg_len = struct.unpack(">I", len_bytes)[0]

    if msg_len > 16 * 1024 * 1024:
        raise ValueError(f"Message too large: {msg_len} bytes")

    payload = b""
    while len(payload) < msg_len:
        chunk = sock.recv(msg_len - len(payload))
        if not chunk:
            raise EOFError("Client disconnected mid-message")
        payload += chunk

    return payload


def send_message(sock, payload: bytes) -> None:
    """Send a length-prefixed protobuf message over a socket."""
    prefix = struct.pack(">I", len(payload))
    sock.sendall(prefix + payload)


def handle_connection(sock, engine) -> None:
    """Process requests on a single connection until EOF or error."""
    request_count = 0
    while True:
        try:
            payload = recv_message(sock)
        except EOFError:
            print(f"Connection closed after {request_count} requests", file=sys.stderr)
            break
        except (OSError, ValueError) as exc:
            print(f"Connection error: {exc}", file=sys.stderr)
            break

        request = spike_pb2.SpikeRequest()
        try:
            request.ParseFromString(payload)
        except Exception as exc:
            print(f"Protobuf decode error: {exc}", file=sys.stderr)
            resp = spike_pb2.SpikeResponse()
            resp.error = f"Decode error: {exc}"
            send_message(sock, resp.SerializeToString())
            continue

        try:
            text, tokens, elapsed_ms = engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens or 64,
                seed=request.seed,
            )
        except Exception as exc:
            print(f"Generation error: {exc}", file=sys.stderr)
            resp = spike_pb2.SpikeResponse()
            resp.error = f"Generation error: {exc}"
            send_message(sock, resp.SerializeToString())
            request_count += 1
            continue

        resp = spike_pb2.SpikeResponse()
        resp.text = text
        resp.tokens_generated = tokens
        resp.elapsed_ms = int(elapsed_ms)
        resp.finished = tokens < (request.max_tokens or 64)

        send_message(sock, resp.SerializeToString())
        request_count += 1


def main():
    parser = argparse.ArgumentParser(description="Synapse Spike Worker")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    parser.add_argument("--model", required=True,
                        help="Model ID. Prefix ollama: for Ollama, mock: for mock")
    args = parser.parse_args()

    socket_path = args.socket
    model_name = args.model

    try:
        os.unlink(socket_path)
    except OSError:
        pass

    engine = load_model(model_name)

    import socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    os.chmod(socket_path, 0o600)

    print("READY", flush=True)

    try:
        conn, _addr = server.accept()
        print("Coordinator connected", file=sys.stderr)
        handle_connection(conn, engine)
    except KeyboardInterrupt:
        print("Worker interrupted", file=sys.stderr)
    finally:
        conn.close()
        server.close()
        try:
            os.unlink(socket_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
