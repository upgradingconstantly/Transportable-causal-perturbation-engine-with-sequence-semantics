"""Deterministic runtime helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

torch: Any
try:
    import torch
except ImportError:  # pragma: no cover - torch may be unavailable in thin environments.
    torch = None


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set global pseudo-random seeds across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - backend support differs across environments.
            pass


def deterministic_probe(seed: int) -> tuple[float, float]:
    """Return deterministic numeric samples for test assertions."""
    set_global_seed(seed)
    return (random.random(), float(np.random.rand(1)[0]))
