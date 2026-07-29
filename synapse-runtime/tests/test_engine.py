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


class TestVllmEngineGenerate:
    @patch("vllm.LLM")
    def test_generate_returns_tokens(self, mock_llm_class: MagicMock) -> None:
        """generate returns token IDs and logprobs."""
        from unittest.mock import MagicMock

        # Mock the LLM instance and its generate method
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [
            MagicMock(token_ids=[42, 43, 44],
                      logprobs=[-0.1, -0.2, -0.3])
        ]
        mock_llm.generate.return_value = [mock_output]
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/test", [0])

        from synapse_runtime.protocol import GenerateResponse
        resp = engine.generate([1, 2, 3], seed=0, max_tokens=100)

        assert isinstance(resp, GenerateResponse)
        assert resp.token_ids == [42, 43, 44]
        assert resp.log_probs == [-0.1, -0.2, -0.3]
        assert resp.finished is True

    @patch("vllm.LLM")
    def test_generate_requires_loaded_model(self,
                                              mock_llm_class: MagicMock) -> None:
        """generate raises EngineError if no model is loaded."""
        engine = VllmEngine()
        with pytest.raises(EngineError, match="No model loaded"):
            engine.generate([1, 2], seed=0, max_tokens=10)

    @patch("vllm.LLM")
    def test_generate_empty_prompt_returns_empty(self,
                                                   mock_llm_class: MagicMock) -> None:
        """Empty prompt produces empty output."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = []
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/test", [0])

        resp = engine.generate([], seed=0, max_tokens=50)
        assert resp.token_ids == []
        assert resp.finished is True

    @patch("vllm.LLM")
    def test_generate_handles_vllm_error(self, mock_llm_class: MagicMock) -> None:
        """vLLM generate errors are wrapped in EngineError."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("CUDA error")
        mock_llm_class.return_value = mock_llm

        engine = VllmEngine()
        engine.load_model("/models/test", [0])

        with pytest.raises(EngineError, match="CUDA error"):
            engine.generate([1], seed=0, max_tokens=10)
