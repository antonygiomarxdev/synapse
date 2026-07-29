"""Weight loader: HuggingFace download, SHA256 verification, expert extraction."""

from __future__ import annotations

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

