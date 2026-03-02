"""Run-id and artifact layout utilities for TCPE Phase 2."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tcpe.config import TCPEConfig

RUN_ID_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ArtifactLayout:
    """Deterministic artifact directory layout for a run."""

    run_root: Path
    logs_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    metadata_dir: Path


def generate_run_id(config: TCPEConfig, command_group: str, provided: str | None = None) -> str:
    """Build a deterministic run id from config state unless user-provided."""
    if provided is not None and provided.strip() != "":
        return _sanitize_run_id(provided)

    digest_source = (
        f"{command_group}|{config.environment}|{config.runtime.seed}|"
        f"{config.dataset.primary_id}|{config.model.transport_variant}|{config.model.latent_dim}"
    )
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"{command_group}-{config.environment}-{digest}"


def build_artifact_layout(config: TCPEConfig, command_group: str, run_id: str) -> ArtifactLayout:
    """Resolve deterministic run directories from config + command group + run id."""
    run_root = config.paths.artifact_root / config.environment / command_group / run_id
    return ArtifactLayout(
        run_root=run_root,
        logs_dir=run_root / "logs",
        checkpoints_dir=run_root / "checkpoints",
        reports_dir=run_root / "reports",
        metadata_dir=run_root / "metadata",
    )


def ensure_artifact_layout(layout: ArtifactLayout) -> None:
    """Create all expected artifact directories."""
    for path in (
        layout.run_root,
        layout.logs_dir,
        layout.checkpoints_dir,
        layout.reports_dir,
        layout.metadata_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_run_manifest(
    layout: ArtifactLayout,
    config: TCPEConfig,
    command_group: str,
    run_id: str,
    config_path: Path,
) -> Path:
    """Persist run metadata for reproducibility."""
    ensure_artifact_layout(layout)
    manifest_path = layout.metadata_dir / "run_manifest.json"
    payload = {
        "command_group": command_group,
        "run_id": run_id,
        "environment": config.environment,
        "config_path": str(config_path),
        "config": config.model_dump(mode="json"),
        "artifact_layout": {
            "run_root": str(layout.run_root),
            "logs_dir": str(layout.logs_dir),
            "checkpoints_dir": str(layout.checkpoints_dir),
            "reports_dir": str(layout.reports_dir),
            "metadata_dir": str(layout.metadata_dir),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _sanitize_run_id(value: str) -> str:
    sanitized = RUN_ID_SANITIZE_PATTERN.sub("-", value.strip())
    sanitized = sanitized.strip("-.")
    if sanitized == "":
        raise ValueError("Run id cannot be empty after sanitization.")
    return sanitized
