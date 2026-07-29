"""Tests for weight loader."""

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
