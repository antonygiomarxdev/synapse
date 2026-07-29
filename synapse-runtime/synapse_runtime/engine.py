"""vLLM engine wrapper for model loading and token generation.

Provides the V1 backend for the Synapse inference runtime.
Implements the runtime side of the InferencePort contract.
"""

from __future__ import annotations

from typing import Any

from synapse_runtime.config import get_config
from synapse_runtime.protocol import GenerateResponse


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
        self._cfg = get_config()

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
                gpu_memory_utilization=self._cfg.vllm.gpu_memory_utilization,
                max_model_len=self._cfg.vllm.max_model_len,
            )
            self._model_path = model_path
            self._loaded_experts = list(expert_indices)
        except (RuntimeError, ValueError) as e:
            raise EngineError(str(e)) from e

    def generate(
        self,
        prompt_tokens: list[int],
        seed: int = 0,
        max_tokens: int | None = None,
    ) -> GenerateResponse:
        """Generate tokens from a prompt.

        Args:
            prompt_tokens: Input token IDs.
            seed: Random seed (0 for deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerateResponse with token_ids, log_probs, and finished flag.

        Raises:
            EngineError: If no model is loaded or generation fails.
        """
        if not self.is_loaded:
            raise EngineError("No model loaded. Call load_model() first.")

        if not prompt_tokens:
            return GenerateResponse(
                request_id=b"",
                token_ids=[],
                log_probs=[],
                finished=True,
            )

        try:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=(
                    self._cfg.vllm.temperature_deterministic
                    if seed == 0
                    else self._cfg.vllm.temperature_default
                ),
                seed=seed,
                max_tokens=max_tokens or self._cfg.vllm.default_max_tokens,
                logprobs=self._cfg.vllm.logprobs_count,
            )

            prompt = {"prompt_token_ids": prompt_tokens}
            outputs = self._llm.generate([prompt], sampling_params)

            if not outputs:
                return GenerateResponse(
                    request_id=b"",
                    token_ids=[],
                    log_probs=[],
                    finished=True,
                )

            output = outputs[0]
            token_ids: list[int] = []
            log_probs: list[float] = []

            for out in output.outputs:
                token_ids.extend(out.token_ids)
                if out.logprobs:
                    for pos_logprobs in out.logprobs:
                        if isinstance(pos_logprobs, dict):
                            # vLLM returns [{token_id: Logprob(...)}, ...]
                            # Logprob is a namedtuple with .logprob, .rank
                            best = max(
                                pos_logprobs,
                                key=lambda k: pos_logprobs[k].logprob,
                            )
                            log_probs.append(float(pos_logprobs[best].logprob))
                        elif isinstance(pos_logprobs, (int, float)):
                            log_probs.append(float(pos_logprobs))

            return GenerateResponse(
                request_id=b"",
                token_ids=token_ids,
                log_probs=log_probs,
                finished=True,
            )

        except (RuntimeError, ValueError, TypeError) as e:
            raise EngineError(str(e)) from e
