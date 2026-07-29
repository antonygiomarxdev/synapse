"""Tests for weight loader."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synapse_runtime.loader import ModelNotFoundError, download_model


class TestDownloadModel:
    @patch("synapse_runtime.loader.snapshot_download")
    def test_returns_local_path(self, mock_snapshot: MagicMock) -> None:
        """download_model returns the local path from snapshot_download."""
        mock_snapshot.return_value = "/cache/models/mixtral-8x7b"
        path = download_model("mistralai/Mixtral-8x7B-v0.1")
        assert path == "/cache/models/mixtral-8x7b"
        mock_snapshot.assert_called_once_with(
            "mistralai/Mixtral-8x7B-v0.1",
            cache_dir=None,
        )

    @patch("synapse_runtime.loader.snapshot_download")
    def test_respects_custom_cache_dir(self, mock_snapshot: MagicMock) -> None:
        """Custom cache_dir is passed to snapshot_download."""
        mock_snapshot.return_value = "/custom/models/test"
        path = download_model("test/model", cache_dir="/custom/models")
        assert path == "/custom/models/test"
        mock_snapshot.assert_called_once_with(
            "test/model",
            cache_dir="/custom/models",
        )

    @patch("synapse_runtime.loader.snapshot_download")
    def test_raises_model_not_found_on_hf_error(
        self,
        mock_snapshot: MagicMock,
    ) -> None:
        """HF Hub errors are wrapped in ModelNotFoundError."""
        from huggingface_hub.errors import RepositoryNotFoundError

        mock_snapshot.side_effect = RepositoryNotFoundError(
            "not found",
            response=MagicMock(
                headers={},
                request=MagicMock(),
            ),
        )
        with pytest.raises(ModelNotFoundError, match="not found"):
            download_model("nonexistent/model")

    @patch("synapse_runtime.loader.snapshot_download")
    def test_raises_on_other_errors(self, mock_snapshot: MagicMock) -> None:
        """Non-HF errors propagate as RuntimeError."""
        mock_snapshot.side_effect = OSError("disk full")
        with pytest.raises(RuntimeError, match="disk full"):
            download_model("any/model")


class TestSha256:
    def test_compute_sha256_known_content(self) -> None:
        """SHA256 of known content matches expected hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.bin"
            filepath.write_bytes(b"synapse test data")
            from synapse_runtime.loader import compute_sha256

            result = compute_sha256(str(tmpdir))
            # compute_sha256 hashes each file individually, then feeds
            # the raw 32-byte SHA256 digest into a combined SHA256.
            expected = hashlib.sha256(
                hashlib.sha256(b"synapse test data").digest()
            ).hexdigest()
            assert result == expected

    def test_verify_sha256_matches(self) -> None:
        """verify_sha256 returns True when hash matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.safetensors"
            filepath.write_bytes(b"weights data")
            # compute_sha256 uses per-file SHA256 digests fed into
            # a combined SHA256, so the expected hash must match.
            expected = hashlib.sha256(
                hashlib.sha256(b"weights data").digest()
            ).hexdigest()
            from synapse_runtime.loader import verify_sha256

            assert verify_sha256(str(tmpdir), expected) is True

    def test_verify_sha256_mismatch(self) -> None:
        """verify_sha256 returns False when hash doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.safetensors"
            filepath.write_bytes(b"tampered weights")
            from synapse_runtime.loader import verify_sha256

            assert verify_sha256(str(tmpdir), "deadbeef") is False

    def test_verify_sha256_empty_directory(self) -> None:
        """verify_sha256 on empty directory returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from synapse_runtime.loader import verify_sha256

            # Empty dir: no files to hash -> sha256 of empty string
            assert verify_sha256(str(tmpdir), "any") is False
