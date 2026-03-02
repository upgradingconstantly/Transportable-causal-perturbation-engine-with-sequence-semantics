"""Config models and loader utilities for TCPE Phase 2."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, ValidationError, field_validator

EnvironmentName = Literal["local", "kaggle", "azure"]
ENVIRONMENT_NAMES: tuple[EnvironmentName, ...] = ("local", "kaggle", "azure")
DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
MODEL_VARIANTS = ("ot", "flow", "bridge")


class ConfigLoadError(ValueError):
    """Raised when configuration files cannot be loaded."""


class ConfigValidationError(ValueError):
    """Raised when loaded configuration fails typed validation."""


class PathsConfig(BaseModel):
    """Filesystem paths used by TCPE workflows."""

    artifact_root: Path = Path("artifacts")
    data_root: Path = Path("data")
    cache_root: Path = Path(".cache/tcpe")

    @field_validator("artifact_root", "data_root", "cache_root")
    @classmethod
    def validate_path_not_empty(cls, value: Path) -> Path:
        if str(value).strip() == "":
            raise ValueError("Path cannot be empty.")
        return value


class ResourceConfig(BaseModel):
    """Execution resource limits and hints."""

    max_ram_gb: PositiveInt = 16
    max_disk_gb: PositiveInt = 300
    num_workers: NonNegativeInt = 2
    device: Literal["cpu", "gpu", "auto"] = "auto"


class DatasetConfig(BaseModel):
    """Dataset identifiers used by command groups."""

    primary_id: str = "synthetic"
    secondary_ids: list[str] = Field(default_factory=list)

    @field_validator("primary_id")
    @classmethod
    def validate_primary_id(cls, value: str) -> str:
        return _validate_dataset_id(value)

    @field_validator("secondary_ids")
    @classmethod
    def validate_secondary_ids(cls, values: list[str]) -> list[str]:
        return [_validate_dataset_id(item) for item in values]


class ModelOptions(BaseModel):
    """Model options validated at startup."""

    transport_variant: Literal["ot", "flow", "bridge"] = "ot"
    latent_dim: PositiveInt = 128
    sequence_embedding_dim: PositiveInt = 256
    cell_embedding_dim: PositiveInt = 256


class RuntimeOptions(BaseModel):
    """Runtime settings for deterministic behavior and logging."""

    seed: NonNegativeInt = 42
    deterministic_torch: bool = True
    wandb_project: str = "tcpe"
    wandb_mode: Literal["offline", "online", "disabled"] = "offline"


class TCPEConfig(BaseModel):
    """Top-level, typed TCPE configuration."""

    environment: EnvironmentName = "local"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelOptions = Field(default_factory=ModelOptions)
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: EnvironmentName) -> EnvironmentName:
        if value not in ENVIRONMENT_NAMES:
            raise ValueError(f"Unsupported environment '{value}'.")
        return value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_ENV_DIR = PROJECT_ROOT / "configs" / "env"


def _validate_dataset_id(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("Dataset identifiers cannot be empty.")
    if DATASET_ID_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            f"Invalid dataset identifier '{value}'. Allowed characters: letters, numbers, _, -, ."
        )
    return candidate


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigLoadError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        parsed = yaml.safe_load(file_handle) or {}

    if not isinstance(parsed, dict):
        raise ConfigLoadError(f"Config file must contain a YAML mapping: {path}")
    if any(not isinstance(key, str) for key in parsed):
        raise ConfigLoadError(f"Config keys must be strings: {path}")

    return cast(dict[str, Any], parsed)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)

    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value

    return merged


def load_config(
    config_path: str | Path | None = None,
    environment: EnvironmentName = "local",
    env_dir: str | Path | None = None,
) -> TCPEConfig:
    """Load base config + environment overlay and validate with pydantic."""
    if environment not in ENVIRONMENT_NAMES:
        raise ConfigLoadError(
            f"Unsupported environment '{environment}'. Valid values: {', '.join(ENVIRONMENT_NAMES)}"
        )

    base_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    overlay_root = Path(env_dir) if env_dir is not None else DEFAULT_ENV_DIR
    overlay_path = overlay_root / f"{environment}.yaml"

    base_mapping = _load_yaml_mapping(base_path)
    overlay_mapping = _load_yaml_mapping(overlay_path)
    merged = _deep_merge(base_mapping, overlay_mapping)
    merged["environment"] = environment

    try:
        return TCPEConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc
