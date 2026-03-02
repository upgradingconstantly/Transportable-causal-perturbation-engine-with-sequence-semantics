"""Shared protocol and metadata types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class RunMetadata:
    """Metadata carried through pipeline runs."""

    run_id: str
    phase: str
    artifact_dir: Path


class ModuleStub(Protocol):
    """Common status method for Phase 1 module stubs."""

    def status(self) -> str:
        """Return current module readiness."""
