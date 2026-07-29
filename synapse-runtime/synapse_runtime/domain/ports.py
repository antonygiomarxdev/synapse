"""Domain ports for the runtime.

Mirrors the Rust `InferencePort` trait. Pure domain — no I/O, no framework imports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from synapse_runtime.protocol import GenerateResponse


@runtime_checkable
class InferencePort(Protocol):
    """Port implemented by inference runtimes (vLLM, llama.cpp, SGLang)."""

    def load_model(
        self,
        model_path: str,
        expert_indices: list[int],
        seed: int = 0,
    ) -> None:
        """Load a model with specified experts into VRAM."""
        ...

    def generate(
        self,
        prompt_tokens: list[int],
        seed: int = 0,
        max_tokens: int = 256,
    ) -> GenerateResponse:
        """Generate tokens from a prompt."""
        ...

    @property
    def is_loaded(self) -> bool: ...

    @property
    def loaded_experts(self) -> list[int]: ...
