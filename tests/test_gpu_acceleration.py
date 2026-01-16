"""
GPU acceleration opt-in behavior tests.
"""

from __future__ import annotations

from wordcloud.utils.gpu_acceleration import GPUAccelerator, create_gpu_accelerator


def test_gpu_accelerator_opt_in_default() -> None:
    assert create_gpu_accelerator() is None


def test_gpu_accelerator_opt_in_disabled() -> None:
    assert create_gpu_accelerator(enabled=False, backend="auto") is None


def test_gpu_accelerator_opt_in_enabled() -> None:
    accelerator = create_gpu_accelerator(enabled=True, backend="auto")
    if accelerator is not None:
        assert accelerator.is_available()


def test_gpu_accelerator_disabled_instance() -> None:
    accelerator = GPUAccelerator(enabled=False, backend="auto")
    assert accelerator.is_available() is False
