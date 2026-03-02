"""W&B scaffolding with offline-by-default behavior for local development."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

try:
    import wandb
except ImportError:  # pragma: no cover - wandb may be unavailable in thin environments.
    wandb = None


class WandbRunLike(Protocol):
    """Minimal run protocol consumed by TCPE."""

    def log(self, data: Mapping[str, Any]) -> None:
        """Log run metrics."""

    def finish(self) -> None:
        """Close the run cleanly."""


@dataclass
class NoOpWandbRun:
    """Local fallback run used when `wandb` is unavailable."""

    project: str
    run_name: str | None
    mode: str
    run_dir: Path

    def log(self, data: Mapping[str, Any]) -> None:
        _ = data

    def finish(self) -> None:
        return


def get_wandb_mode() -> str:
    """Resolve W&B mode from environment with offline default."""
    mode = os.getenv("WANDB_MODE", "offline").strip()
    return mode or "offline"


def init_wandb_run(
    project: str = "tcpe",
    run_name: str | None = None,
    run_dir: str | Path = "artifacts/wandb",
    config: Mapping[str, Any] | None = None,
) -> WandbRunLike:
    """Create a W&B run object that is safe for local development."""
    mode = get_wandb_mode()
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if wandb is None:
        return NoOpWandbRun(project=project, run_name=run_name, mode=mode, run_dir=output_dir)

    run = wandb.init(
        project=project,
        name=run_name,
        mode=mode,
        dir=str(output_dir),
        config=dict(config or {}),
        reinit=True,
    )
    return cast(WandbRunLike, run)
