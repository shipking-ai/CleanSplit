"""Compute-device detection. Torch is imported lazily so non-GPU code paths never require it."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass


@dataclass
class DeviceInfo:
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    compute_capability: str | None
    vram_total_mb: float | None
    vram_free_mb: float | None
    platform: str
    cpu_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def probe() -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo(False, None, False, None, None, None, None, None, platform.platform(), os.cpu_count() or 1)
    cuda = bool(torch.cuda.is_available())
    name = cap = total = free = None
    if cuda:
        name = torch.cuda.get_device_name(0)
        cap = "%d.%d" % torch.cuda.get_device_capability(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        total, free = total_b / 2**20, free_b / 2**20
    return DeviceInfo(
        True, torch.__version__, cuda, torch.version.cuda, name, cap, total, free, platform.platform(), os.cpu_count() or 1
    )


def resolve_device(requested: str = "auto") -> str:
    """'auto' -> cuda if present else cpu. An explicit 'cuda' that is unavailable is an error, not a silent fallback."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested
