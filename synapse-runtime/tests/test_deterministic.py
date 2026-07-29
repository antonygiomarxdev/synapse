"""Tests for deterministic seed enforcement."""

from unittest.mock import MagicMock, patch

from synapse_runtime.deterministic import enforce_seed_zero, verify_determinism


class TestEnforceSeedZero:
    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_enforce_seed_zero_reloads_with_seed_zero(
        self, mock_engine_class: MagicMock
    ) -> None:
        """enforce_seed_zero reloads the model with seed=0."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine._model_path = "/models/test"
        mock_engine.loaded_experts = [0, 1, 2]
        enforce_seed_zero(mock_engine)
        mock_engine.load_model.assert_called_once_with(
            model_path="/models/test", expert_indices=[0, 1, 2], seed=0
        )

    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_enforce_seed_zero_noop_when_not_loaded(
        self, mock_engine_class: MagicMock
    ) -> None:
        """enforce_seed_zero does nothing if no model is loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = False

        enforce_seed_zero(mock_engine)

        mock_engine.load_model.assert_not_called()


class TestVerifyDeterminism:
    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_identical_outputs_pass_verification(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Two runs with identical outputs -> verification passes."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        from synapse_runtime.protocol import GenerateResponse

        # Same output both times
        mock_engine.generate.side_effect = [
            GenerateResponse(
                request_id=b"r1",
                token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.2, -0.3],
                finished=True,
            ),
            GenerateResponse(
                request_id=b"r1",
                token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.2, -0.3],
                finished=True,
            ),
        ]

        result = verify_determinism(mock_engine, [42])
        assert result is True

    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_divergent_outputs_fail_verification(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Two runs with different outputs -> verification fails."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        from synapse_runtime.protocol import GenerateResponse

        mock_engine.generate.side_effect = [
            GenerateResponse(
                request_id=b"r1",
                token_ids=[1, 2, 3],
                log_probs=[-0.1, -0.2, -0.3],
                finished=True,
            ),
            GenerateResponse(
                request_id=b"r1",
                token_ids=[1, 99, 3],  # Divergent!
                log_probs=[-0.1, -0.9, -0.3],
                finished=True,
            ),
        ]

        result = verify_determinism(mock_engine, [42])
        assert result is False

    @patch("synapse_runtime.deterministic.VllmEngine")
    def test_different_logprobs_fail_verification(
        self, mock_engine_class: MagicMock
    ) -> None:
        """Same tokens but different logprobs -> verification fails."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True

        from synapse_runtime.protocol import GenerateResponse

        mock_engine.generate.side_effect = [
            GenerateResponse(
                request_id=b"r1",
                token_ids=[1, 2],
                log_probs=[-0.1, -0.2],
                finished=True,
            ),
            GenerateResponse(
                request_id=b"r1",
                token_ids=[1, 2],
                log_probs=[-0.1, -0.999],
                finished=True,  # Different logprob
            ),
        ]

        result = verify_determinism(mock_engine, [42])
        assert result is False
