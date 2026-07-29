"""Runtime configuration dataclasses.

Pure domain — no I/O, no file system access.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Runtime bridge configuration."""

    socket_path: str = "/tmp/synapse-runtime.sock"
    socket_backlog: int = 5
    socket_timeout: float = 0.5
    recv_chunk_size: int = 65536
    max_retries: int = 3
    read_timeout_secs: int = 30


@dataclass
class VllmConfig:
    """vLLM engine configuration."""

    gpu_memory_utilization: float = 0.90
    max_model_len: int = 32768
    default_max_tokens: int = 256
    temperature_deterministic: float = 0.0
    temperature_default: float = 0.7
    logprobs_count: int = 1


@dataclass
class AppConfig:
    """Top-level application config."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
