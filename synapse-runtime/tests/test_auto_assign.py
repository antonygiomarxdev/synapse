"""Tests for VRAM detection and auto-assignment logic."""

from unittest.mock import patch

from synapse_runtime.auto_assign import calculate_expert_count, detect_vram


class TestDetectVram:
    def test_returns_tuple_of_two_ints(self) -> None:
        """detect_vram returns (total_mb, available_mb) as integers."""
        total, available = detect_vram()
        assert isinstance(total, int)
        assert isinstance(available, int)
        assert total > 0
        assert available > 0
        assert available <= total

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info", return_value=(8589934592, 17179869184))
    def test_uses_torch_cuda_when_available(
        self, mock_mem: object, mock_avail: object
    ) -> None:
        """When CUDA is available, uses torch.cuda.mem_get_info."""
        total, available = detect_vram()
        assert total == 16384  # 17179869184 / (1024*1024)
        assert available == 8192  # 8589934592 / (1024*1024)

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_zero_when_no_cuda(self, mock_avail: object) -> None:
        """When CUDA is not available, returns (0, 0)."""
        total, available = detect_vram()
        assert total == 0
        assert available == 0


class TestCalculateExpertCount:
    def test_standard_mixtral_config(self) -> None:
        """Mixtral 8x7B: 3GB expert + 3GB shared, 16GB VRAM -> 4 experts."""
        count = calculate_expert_count(
            total_mb=16384,
            available_mb=12288,
            expert_size_gb=3.0,
            shared_params_gb=3.0,
        )
        # (12288/1024 - 3) / 3 = (12 - 3) / 3 = 3 experts
        assert count == 3

    def test_deepseek_lite_tiny_experts(self) -> None:
        """DeepSeek-V2 Lite: 0.15GB experts, 1GB shared, 8GB VRAM -> many experts."""
        count = calculate_expert_count(
            total_mb=8192,
            available_mb=6144,
            expert_size_gb=0.15,
            shared_params_gb=1.0,
        )
        # (6144/1024 - 1) / 0.15 = (6 - 1) / 0.15 = 33
        assert count == 33

    def test_insufficient_vram_returns_zero(self) -> None:
        """When VRAM is less than shared params, returns 0 experts."""
        count = calculate_expert_count(
            total_mb=4096,
            available_mb=2048,
            expert_size_gb=3.0,
            shared_params_gb=3.0,
        )
        assert count == 0

    def test_exact_fit_one_expert(self) -> None:
        """When VRAM fits exactly one expert, returns 1."""
        # available_mb = shared_gb + 1*expert_gb in MB
        # 3GB shared + 3GB expert = 6GB = 6144 MB
        count = calculate_expert_count(
            total_mb=8192,
            available_mb=6144,
            expert_size_gb=3.0,
            shared_params_gb=3.0,
        )
        assert count == 1

    def test_respects_max_experts_cap(self) -> None:
        """Expert count is capped at the model's total expert count."""
        count = calculate_expert_count(
            total_mb=49152,  # 48GB
            available_mb=40960,  # 40GB free
            expert_size_gb=0.15,
            shared_params_gb=1.0,
            max_experts=8,  # e.g., Mixtral only has 8
        )
        # (40960/1024 - 1) / 0.15 = (40 - 1) / 0.15 = 260, but capped at 8
        assert count == 8
