"""Compute backend selection: CuPy (GPU) when available, otherwise NumPy (CPU).

The whole engine is written against the common NumPy/CuPy array API, so the same
code runs on both. Set the environment variable GENEWEAVER_FORCE_CPU=1 to force CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Backend:
    xp: Any          # numpy or cupy module
    is_gpu: bool
    name: str

    def to_host(self, arr):
        """Return a NumPy array regardless of where `arr` lives."""
        if self.is_gpu:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)


def get_backend(prefer_gpu: bool = True) -> Backend:
    if prefer_gpu and os.environ.get("GENEWEAVER_FORCE_CPU") != "1":
        try:
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = props["name"]
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()
                return Backend(cp, True, f"GPU (CuPy) - {gpu_name}")
        except Exception:
            pass  # no CuPy / no CUDA device -> fall back to CPU
    return Backend(np, False, "CPU (NumPy)")
