"""VRAM detection and optimal expert assignment for compute nodes."""

from __future__ import annotations

_BYTES_PER_MB: int = 1024 * 1024
_MB_PER_GB: int = 1024


def detect_vram() -> tuple[int, int]:
    """Detect GPU VRAM.

    Returns:
        (total_mb, available_mb) tuple. Both are 0 if no GPU is available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_mb = total_bytes // _BYTES_PER_MB
            free_mb = free_bytes // _BYTES_PER_MB
            return int(total_mb), int(free_mb)
    except ImportError:
        pass

    return 0, 0


def calculate_expert_count(
    total_mb: int,
    available_mb: int,
    expert_size_gb: float,
    shared_params_gb: float,
    max_experts: int | None = None,
) -> int:
    """Calculate how many experts fit in available VRAM.

    Formula:
        available_gb = available_mb / 1024
        experts = floor((available_gb - shared_params_gb) / expert_size_gb)

    Args:
        total_mb: Total GPU VRAM in MB.
        available_mb: Available VRAM in MB.
        expert_size_gb: Size of one expert in GB (4-bit quantized).
        shared_params_gb: Size of shared parameters in GB.
        max_experts: Optional cap (model's total expert count).

    Returns:
        Number of experts that fit. Minimum 0.
    """
    if available_mb == 0:
        return 0

    available_gb = available_mb / float(_MB_PER_GB)
    usable_gb = available_gb - shared_params_gb

    if usable_gb <= 0.0:
        return 0

    count = int(usable_gb / expert_size_gb)

    if max_experts is not None:
        count = min(count, max_experts)

    return max(count, 0)
