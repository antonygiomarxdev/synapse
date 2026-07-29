"""Bridge protocol dataclasses.

Pure domain types mirroring synapse-core/proto/runtime.proto 1:1.
Serialization has been moved to synapse_runtime.infrastructure.serializer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class LoadModelRequest:
    """Request to load a model with specific experts."""

    model_id: str
    expert_indices: list[int] = field(default_factory=list)

    MESSAGE_TYPE: ClassVar[int] = 1


@dataclass
class LoadModelResponse:
    """Response after model load attempt."""

    success: bool = False
    error: str = ""
    loaded_experts: int = 0

    MESSAGE_TYPE: ClassVar[int] = 2


@dataclass
class GenerateRequest:
    """Token generation request."""

    request_id: bytes = b""
    token_ids: list[int] = field(default_factory=list)
    seed: int = 0
    max_tokens: int = 0

    MESSAGE_TYPE: ClassVar[int] = 3


@dataclass
class GenerateResponse:
    """Token generation response."""

    request_id: bytes = b""
    token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    finished: bool = False

    MESSAGE_TYPE: ClassVar[int] = 4


@dataclass
class VerifyHashRequest:
    """SHA256 verification request."""

    model_id: str = ""
    expected_sha256: str = ""

    MESSAGE_TYPE: ClassVar[int] = 5


@dataclass
class VerifyHashResponse:
    """SHA256 verification response."""

    matches: bool = False
    actual_sha256: str = ""

    MESSAGE_TYPE: ClassVar[int] = 6


@dataclass
class VramQueryRequest:
    """VRAM query — no fields."""

    MESSAGE_TYPE: ClassVar[int] = 7


@dataclass
class VramQueryResponse:
    """VRAM query response."""

    total_mb: int = 0
    available_mb: int = 0

    MESSAGE_TYPE: ClassVar[int] = 8
