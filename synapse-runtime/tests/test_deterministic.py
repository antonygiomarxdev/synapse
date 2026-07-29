"""Tests for deterministic seed enforcement."""

from unittest.mock import MagicMock

from synapse_runtime.deterministic import enforce_seed_zero, verify_determinism


class TestEnforceSeedZero:
    def test_enforce_seed_zero_reloads_with_seed_zero(self) -> None:
        """enforce_seed_zero reloads the model with seed=0."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine._model_path = "/models/test"
        mock_engine.loaded_experts = [0, 1, 2]
        enforce_seed_zero(mock_engine)
        mock_engine.load_model.assert_called_once_with(
            model_path="/models/test", expert_indices=[0, 1, 2], seed=0
        )

    def test_enforce_seed_zero_noop_when_not_loaded(self) -> None:
        """enforce_seed_zero does nothing if no model is loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = False
        enforce_seed_zero(mock_engine)
        mock_engine.load_model.assert_not_called()


class TestVerifyDeterminism:
    def test_identical_outputs_pass_verification(self) -> None:
        """Two runs with identical outputs -> verification passes."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.side_effect = [
            MagicMock(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3]),
            MagicMock(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3]),
        ]
        result = verify_determinism(mock_engine, [100, 200])
        assert result is True

    def test_divergent_outputs_fail_verification(self) -> None:
        """Two runs with different outputs -> verification fails."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.side_effect = [
            MagicMock(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3]),
            MagicMock(token_ids=[4, 5, 6], log_probs=[-0.4, -0.5, -0.6]),
        ]
        result = verify_determinism(mock_engine, [100, 200])
        assert result is False

    def test_different_logprobs_fail_verification(self) -> None:
        """Same tokens but different logprobs -> verification fails."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.side_effect = [
            MagicMock(token_ids=[1, 2, 3], log_probs=[-0.1, -0.2, -0.3]),
            MagicMock(token_ids=[1, 2, 3], log_probs=[-0.4, -0.5, -0.6]),
        ]
        result = verify_determinism(mock_engine, [100, 200])
        assert result is False

    def test_verification_returns_false_if_not_loaded(self) -> None:
        """verify_determinism returns False if no model is loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = False
        result = verify_determinism(mock_engine, [100, 200])
        assert result is False
