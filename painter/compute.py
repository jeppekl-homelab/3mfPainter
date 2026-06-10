"""Runtime device detection and tuning for mesh analysis.

All heavy math (edge detection, segmentation, symmetry) goes through torch and
must call `get_device()` instead of assuming CUDA — the code runs on machines
with an RTX as well as iGPU-only laptops.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComputeProfile:
    device: torch.device
    name: str
    # batchstørrelse til store vektoriserede operationer (faces pr. chunk)
    chunk_size: int
    # float-præcision: fp32 overalt; CUDA kan bruge TF32 til matmul
    allow_tf32: bool

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


@functools.cache
def get_profile() -> ComputeProfile:
    """Detect the best available compute device and tune for it. Cached."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 2**30
        profile = ComputeProfile(
            device=torch.device("cuda"),
            name=f"{props.name} ({vram_gb:.0f} GB VRAM)",
            # GPU'en sluger gerne alt på én gang op til et VRAM-afhængigt loft
            chunk_size=int(min(vram_gb, 16) * 2_000_000),
            allow_tf32=True,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        n_threads = torch.get_num_threads()
        profile = ComputeProfile(
            device=torch.device("cpu"),
            name=f"CPU ({n_threads} threads)",
            # mindre chunks på CPU: bedre cache-lokalitet og responsivt UI
            chunk_size=500_000,
            allow_tf32=False,
        )
    print(f"Compute:  {profile.name} [{profile.device}]")
    return profile


def get_device() -> torch.device:
    return get_profile().device
