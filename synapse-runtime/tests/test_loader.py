"""Tests for weight loader."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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


class TestExtractExperts:
    def test_extracts_single_expert_from_safetensors(self) -> None:
        """Extract expert weights from a safetensors file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal safetensors file with expert weights
            weights = {
                "model.experts.0.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32)
            }
            _write_safetensors(Path(tmpdir) / "model-00001.safetensors", weights)

            from synapse_runtime.loader import extract_experts

            result = extract_experts(str(tmpdir), [0])
            assert 0 in result
            data = result[0]
            assert len(data) == 12  # 3 float32 values

    def test_extracts_multiple_experts(self) -> None:
        """Extract multiple experts from safetensors files."""

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.experts.0.weight": np.array([1.0], dtype=np.float32),
                "model.experts.3.weight": np.array([3.0], dtype=np.float32),
            }
            _write_safetensors(Path(tmpdir) / "model.safetensors", weights)

            from synapse_runtime.loader import extract_experts

            result = extract_experts(str(tmpdir), [0, 3])
            assert result.keys() == {0, 3}

    def test_missing_expert_raises(self) -> None:
        """Requesting an expert not in the checkpoint raises ExpertExtractionError."""

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {"model.experts.0.weight": np.array([1.0], dtype=np.float32)}
            _write_safetensors(Path(tmpdir) / "model.safetensors", weights)

            from synapse_runtime.loader import ExpertExtractionError, extract_experts

            with pytest.raises(ExpertExtractionError, match="Expert 7"):
                extract_experts(str(tmpdir), [0, 7])


def _write_safetensors(filepath: Path, tensors: dict[str, np.ndarray]) -> None:
    """Helper: write a minimal safetensors file."""
    import json
    import struct

    header = {}
    offset = 0
    for name, arr in tensors.items():
        dtype = "F32"
        shape = list(arr.shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes

    header_json = json.dumps(header)
    header_bytes = header_json.encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))

    with open(filepath, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        for arr in tensors.values():
            f.write(arr.tobytes())
