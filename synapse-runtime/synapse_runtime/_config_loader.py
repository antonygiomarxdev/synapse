"""Loads configuration from TOML files at startup.

Infrastructure adapter — reads filesystem, produces AppConfig.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from synapse_runtime.config import AppConfig

# Module-level singleton
_config: AppConfig | None = None


def load_config(path: str | None = None) -> AppConfig:
    """Load config from default.toml. Falls back to defaults if file missing."""
    global _config
    if _config is not None:
        return _config

    cfg = AppConfig()

    if path is None:
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
