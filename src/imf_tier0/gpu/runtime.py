from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from typing import Any


def select_batch_size(probe: Callable[[int], bool], candidates: Sequence[int]) -> int:
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("batch-size candidates must be positive")
    for candidate in sorted(set(candidates), reverse=True):
        if probe(candidate):
            return candidate
    raise RuntimeError("no candidate batch size fits")


def release_cuda(torch_module: Any | None = None) -> None:
    gc.collect()
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def reset_peak_vram(torch_module: Any | None = None) -> None:
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()


def peak_vram_bytes(torch_module: Any | None = None) -> dict[str, int]:
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        return {"allocated": 0, "reserved": 0}
    return {
        "allocated": int(torch_module.cuda.max_memory_allocated()),
        "reserved": int(torch_module.cuda.max_memory_reserved()),
    }

