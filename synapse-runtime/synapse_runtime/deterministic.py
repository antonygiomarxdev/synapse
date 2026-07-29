"""Deterministic seed enforcement for audit and verification.

Ensures that seed=0 produces identical outputs across runs.
This is critical for the statistical audit mechanism in the swarm.
"""

from __future__ import annotations

from synapse_runtime.engine import VllmEngine

_VERIFY_MAX_TOKENS: int = 50


def enforce_seed_zero(engine: VllmEngine) -> None:
    """Reload the current model with seed=0 to enforce determinism.

    Args:
        engine: The vLLM engine instance with a model loaded.
    """
    if not engine.is_loaded:
        return

    # Reload with seed=0 using the same model path and experts
    engine.load_model(
        model_path=engine._model_path,
        expert_indices=engine.loaded_experts,
        seed=0,
    )


def verify_determinism(
    engine: VllmEngine, prompt_tokens: list[int]
) -> bool:
    """Verify that two runs with seed=0 produce identical outputs.

    Runs the same prompt twice and compares token IDs and logprobs.
    Both runs use seed=0 internally.

    Args:
        engine: The vLLM engine (must have a model loaded with seed=0).
        prompt_tokens: Input token IDs for the test prompt.

    Returns:
        True if both runs produce identical outputs, False otherwise.
    """
    if not engine.is_loaded:
        return False

    run1 = engine.generate(prompt_tokens, seed=0, max_tokens=_VERIFY_MAX_TOKENS)
    run2 = engine.generate(prompt_tokens, seed=0, max_tokens=_VERIFY_MAX_TOKENS)

    return (
        run1.token_ids == run2.token_ids
        and run1.log_probs == run2.log_probs
    )
