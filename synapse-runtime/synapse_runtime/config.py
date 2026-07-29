"""Runtime configuration loaded from config/default.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
    """Top-level application config loaded from config/default.toml."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)


# Module-level singleton
_config: AppConfig | None = None


def load_config(path: str | None = None) -> AppConfig:
    """Load config from default.toml. Falls back to defaults if file missing."""
    global _config
    if _config is not None:
        return _config

    cfg = AppConfig()

    if path is None:
        # Look relative to this file's location
        base = Path(__file__).resolve().parent.parent.parent
        path = str(base / "config" / "default.toml")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        if "runtime" in data:
            for key, value in data["runtime"].items():
                if hasattr(cfg.runtime, key):
                    setattr(cfg.runtime, key, value)
        if "vllm" in data:
            for key, value in data["vllm"].items():
                if hasattr(cfg.vllm, key):
                    setattr(cfg.vllm, key, value)
    except (FileNotFoundError, PermissionError):
        pass  # Use defaults

    _config = cfg
    return cfg


def get_config() -> AppConfig:
    """Get the loaded config, loading if needed."""
    if _config is None:
        return load_config()
    return _config
