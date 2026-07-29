"""Tests for vLLM engine wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from synapse_runtime.engine import EngineError, VllmEngine


class TestVllmEngineLoadModel:
    @patch("vllm.LLM")
    def test_load_model_creates_llm_instance(self, mock_llm_class: MagicMock) -> None:
        """load_model initializes vLLM with the model path."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/mixtral", [0, 1, 2, 3])

        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model"] == "/models/mixtral"

    @patch("vllm.LLM")
    def test_load_model_sets_loaded_flag(self, mock_llm_class: MagicMock) -> None:
        """After load_model, is_loaded returns True."""
        mock_llm_class.return_value = MagicMock()

        engine = VllmEngine()
        assert engine.is_loaded is False
        engine.load_model("/models/test", [0])
        assert engine.is_loaded is True

    @patch("vllm.LLM")
    def test_load_model_passes_deterministic_seed(
        self, mock_llm_class: MagicMock
    ) -> None:
        """load_model passes seed=0 for deterministic mode."""
        mock_llm_class.return_value = MagicMock()

        engine = VllmEngine()
        engine.load_model("/models/test", [0, 1], seed=0)

        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs.get("seed") == 0

    @patch("vllm.LLM")
    def test_load_model_handles_vllm_error(self, mock_llm_class: MagicMock) -> None:
        """vLLM initialization errors are wrapped in EngineError."""
        mock_llm_class.side_effect = RuntimeError("CUDA out of memory")

        engine = VllmEngine()
        with pytest.raises(EngineError, match="CUDA out of memory"):
            engine.load_model("/models/big", [0])

    @patch("vllm.LLM")
    def test_double_load_replaces_model(self, mock_llm_class: MagicMock) -> None:
        """Loading a second model replaces the first."""
        mock_llm1 = MagicMock()
        mock_llm2 = MagicMock()
        mock_llm_class.side_effect = [mock_llm1, mock_llm2]

        engine = VllmEngine()
        engine.load_model("/models/first", [0])
        engine.load_model("/models/second", [1])

        assert engine.is_loaded is True
        assert mock_llm_class.call_count == 2
