"""vLLM engine wrapper for model loading and token generation.

Provides the V1 backend for the Synapse inference runtime.
Implements the runtime side of the InferencePort contract.
"""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """Raised when the inference engine encounters an error."""


class VllmEngine:
    """Wraps vLLM's LLM class for model loading and generation.

    This is the V1 backend. V2+ will add llama.cpp and SGLang engines
    behind the same interface.
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._model_path: str = ""
        self._loaded_experts: list[int] = []

    @property
    def is_loaded(self) -> bool:
        """True if a model is currently loaded in VRAM."""
        return self._llm is not None

    @property
    def loaded_experts(self) -> list[int]:
        """The expert indices currently loaded."""
        return list(self._loaded_experts)

    def load_model(
        self,
        model_path: str,
        expert_indices: list[int],
        seed: int = 0,
    ) -> None:
        """Load a model with specified experts into VRAM.

        Args:
            model_path: Local path to the model directory.
            expert_indices: Which experts to load (indices).
            seed: Random seed for deterministic generation (default 0).

        Raises:
            EngineError: If model loading fails.
        """
        try:
            from vllm import LLM

            self._llm = LLM(
                model=model_path,
                seed=seed,
                gpu_memory_utilization=0.90,
                max_model_len=32768,
            )
            self._model_path = model_path
            self._loaded_experts = list(expert_indices)
        except Exception as e:
            raise EngineError(str(e)) from e
