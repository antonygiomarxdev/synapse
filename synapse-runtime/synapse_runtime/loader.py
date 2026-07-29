"""Weight loader: HuggingFace download, SHA256 verification, expert extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found on HuggingFace Hub."""


def download_model(hf_repo: str, cache_dir: str | None = None) -> str:
    """Download model weights from HuggingFace Hub.

    Args:
        hf_repo: HuggingFace repository ID (e.g. "mistralai/Mixtral-8x7B-v0.1").
        cache_dir: Optional custom cache directory.

    Returns:
        Local path to downloaded model.

    Raises:
        ModelNotFoundError: If the repository doesn't exist.
        RuntimeError: On other download failures.
    """
    try:
        return snapshot_download(hf_repo, cache_dir=cache_dir)
    except RepositoryNotFoundError as e:
        raise ModelNotFoundError(str(e)) from e
    except Exception as e:
        raise RuntimeError(str(e)) from e


# Read buffer size for file hashing (8 KiB).
_SHA256_BLOCK_SIZE = 8192


def compute_sha256(model_path: str) -> str:
    """Compute SHA256 hash of all files in a directory.

    Files are sorted by name for deterministic hashing.
    Each file's content is hashed individually, then the combined
    hash is computed from the concatenation of per-file hashes.

    Args:
        model_path: Path to the model directory.

    Returns:
        Hex-encoded SHA256 hash string.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {model_path}")

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")
    )

    if not files:
        return hashlib.sha256(b"").hexdigest()

    combined = hashlib.sha256()
    for filepath in files:
        file_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(_SHA256_BLOCK_SIZE), b""):
                file_hash.update(chunk)
        combined.update(file_hash.digest())

    return combined.hexdigest()


def verify_sha256(model_path: str, expected_hash: str) -> bool:
    """Verify model weights match expected SHA256 hash.

    Args:
        model_path: Path to the model directory.
        expected_hash: Expected hex-encoded SHA256.

    Returns:
        True if hashes match, False otherwise.
    """
    actual = compute_sha256(model_path)
    return actual == expected_hash
