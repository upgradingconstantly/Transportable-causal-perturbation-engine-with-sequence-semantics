"""Transport module core with OT strategy and baseline-gated training for TCPE Phase 10."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import sparse
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tcpe.anndata_schema import (
    CELL_STATE_EMBEDDING_OBSM_KEY,
    NORMALIZED_LAYER_KEY,
    SEQUENCE_EMBEDDING_OBSM_KEY,
)
from tcpe.baselines import REQUIRED_BASELINE_NAMES, BaselineSuiteResult
from tcpe.runtime.wandb_scaffold import WandbRunLike, init_wandb_run

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

try:  # pragma: no cover - optional dependency in thin environments.
    import lightning.pytorch as _lightning  # noqa: F401

    _LIGHTNING_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in thin environments.
    try:
        import pytorch_lightning as _lightning  # noqa: F401

        _LIGHTNING_AVAILABLE = True
    except ImportError:
        _LIGHTNING_AVAILABLE = False

try:  # pragma: no cover - optional dependency in thin environments.
    import ot
except ImportError:  # pragma: no cover - optional dependency in thin environments.
    ot = None

TRANSPORT_VARIANT_NAME = "ot_sinkhorn"
TRANSPORT_CHECKPOINT_SCHEMA_VERSION = "transport_ot_v1"
FLOW_MATCHING_VARIANT_NAME = "flow_matching_scaffold"
SCHRODINGER_BRIDGE_VARIANT_NAME = "schrodinger_bridge_scaffold"
EXPERIMENTAL_TAG = "experimental"
CLOUD_RECOMMENDED_TAG = "cloud-recommended"
TransportVariant = Literal["ot", "flow", "bridge"]


class TransportError(RuntimeError):
    """Base transport error."""


class TransportInputError(TransportError):
    """Raised when transport inputs are malformed."""


class TransportCheckpointError(TransportError):
    """Raised when checkpoint read/write fails."""


class UnsupportedTransportPathError(TransportError):
    """Raised for scaffold strategies that are intentionally non-trainable locally."""


@dataclass(frozen=True)
class TransportVariantInfo:
    """Structured metadata for transport variant dispatch and status tags."""

    variant: TransportVariant
    strategy_name: str
    status_tags: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransportStrategy(Protocol):
    """Transport strategy interface for pluggable model variants."""

    variant_name: str

    def fit(
        self,
        *,
        source_expression: np.ndarray,
        target_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        wandb_run: WandbRunLike | None = None,
        enable_wandb: bool = False,
        wandb_project: str = "tcpe",
        wandb_run_name: str | None = None,
        wandb_dir: str | Path = "artifacts/wandb",
    ) -> TransportFitResult:
        """Fit model parameters on paired source/target cell distributions."""

    def predict_distribution(
        self,
        *,
        source_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        n_samples: int = 0,
    ) -> TransportPrediction:
        """Predict transported distribution mean and uncertainty."""

    def save(self, path: str | Path) -> Path:
        """Serialize model parameters and config."""

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> Self:
        """Load model from checkpoint."""


@dataclass(frozen=True)
class OTTransportConfig:
    """Configuration for OT transport strategy."""

    input_dim: int
    sequence_embedding_dim: int
    cell_state_embedding_dim: int
    latent_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 40
    batch_size: int = 64
    sinkhorn_epsilon: float = 0.1
    sinkhorn_n_iters: int = 30
    sinkhorn_weight: float = 0.1
    uncertainty_floor: float = 1e-4
    dropout: float = 0.0
    conditioning_operator: Literal["add"] = "add"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.sequence_embedding_dim <= 0:
            raise ValueError("sequence_embedding_dim must be positive.")
        if self.cell_state_embedding_dim <= 0:
            raise ValueError("cell_state_embedding_dim must be positive.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be positive.")
        if self.sinkhorn_n_iters <= 0:
            raise ValueError("sinkhorn_n_iters must be positive.")
        if self.sinkhorn_weight < 0:
            raise ValueError("sinkhorn_weight must be non-negative.")
        if self.uncertainty_floor <= 0:
            raise ValueError("uncertainty_floor must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.conditioning_operator != "add":
            raise ValueError(
                "Raw concatenation is not allowed for conditioning. "
                "Use separate projection heads combined by element-wise addition."
            )


@dataclass(frozen=True)
class TransportTrainingData:
    """Fully prepared arrays for transport model fit."""

    source_expression: np.ndarray
    target_expression: np.ndarray
    sequence_embedding: np.ndarray
    cell_state_embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_anndata(
        cls,
        adata: AnnData,
        *,
        expression_layer: str = NORMALIZED_LAYER_KEY,
        source_policy: Literal["control_mean", "identity"] = "control_mean",
    ) -> TransportTrainingData:
        target_expression = _resolve_expression_matrix(
            adata=adata,
            expression_layer=expression_layer,
        )
        sequence_embedding = _coerce_dense_float32(adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY])
        cell_state_embedding = _coerce_dense_float32(adata.obsm[CELL_STATE_EMBEDDING_OBSM_KEY])
        source_expression = _build_source_expression(
            adata=adata,
            target_expression=target_expression,
            source_policy=source_policy,
        )

        return cls(
            source_expression=source_expression,
            target_expression=target_expression,
            sequence_embedding=sequence_embedding,
            cell_state_embedding=cell_state_embedding,
            metadata={
                "expression_layer": expression_layer,
                "source_policy": source_policy,
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
            },
        )


@dataclass(frozen=True)
class TransportPrediction:
    """Transport model prediction payload."""

    mean: np.ndarray
    variance: np.ndarray
    sampled: np.ndarray | None = None


@dataclass(frozen=True)
class TransportFitResult:
    """Fit summary and scalar training histories."""

    variant_name: str
    n_epochs: int
    loss_history: list[float]
    mse_history: list[float]
    sinkhorn_history: list[float]
    sinkhorn_pot_history: list[float]
    log_history: list[dict[str, float]]
    checkpoint_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdditiveConditioning(nn.Module):
    """Conditioning block with separate projection heads and additive merge."""

    def __init__(
        self,
        *,
        sequence_embedding_dim: int,
        cell_state_embedding_dim: int,
        latent_dim: int,
        combine_operator: Literal["add"] = "add",
    ) -> None:
        super().__init__()
        if combine_operator != "add":
            raise ValueError(
                "Raw concatenation is not allowed for conditioning. "
                "Use separate projection heads combined by element-wise addition."
            )
        self.combine_operator = combine_operator
        self.sequence_projection = nn.Linear(sequence_embedding_dim, latent_dim)
        self.cell_state_projection = nn.Linear(cell_state_embedding_dim, latent_dim)

    def project_sequence(self, sequence_embedding: Tensor) -> Tensor:
        """Project sequence embeddings into latent conditioning space."""
        return cast(Tensor, self.sequence_projection(sequence_embedding))

    def project_cell_state(self, cell_state_embedding: Tensor) -> Tensor:
        """Project cell-state embeddings into latent conditioning space."""
        return cast(Tensor, self.cell_state_projection(cell_state_embedding))

    def forward(self, sequence_embedding: Tensor, cell_state_embedding: Tensor) -> Tensor:
        sequence_latent = self.project_sequence(sequence_embedding)
        cell_latent = self.project_cell_state(cell_state_embedding)
        return cast(Tensor, sequence_latent + cell_latent)


class LatentEncoder(nn.Module):
    """Latent encoder for source expression profiles."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, source_expression: Tensor) -> Tensor:
        return cast(Tensor, self.network(source_expression))


class OTTransportNetwork(nn.Module):
    """OT transport network with additive conditioning and uncertainty head."""

    def __init__(self, config: OTTransportConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_encoder = LatentEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.conditioning = AdditiveConditioning(
            sequence_embedding_dim=config.sequence_embedding_dim,
            cell_state_embedding_dim=config.cell_state_embedding_dim,
            latent_dim=config.latent_dim,
            combine_operator=config.conditioning_operator,
        )
        self.mean_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.log_variance_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        self.uncertainty_floor = config.uncertainty_floor

    def forward(
        self,
        source_expression: Tensor,
        sequence_embedding: Tensor,
        cell_state_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        base_latent = self.latent_encoder(source_expression)
        conditioning = self.conditioning(sequence_embedding, cell_state_embedding)
        conditioned_latent = base_latent + conditioning
        mean = self.mean_decoder(conditioned_latent)
        raw_variance = self.log_variance_decoder(conditioned_latent)
        variance = functional.softplus(raw_variance) + self.uncertainty_floor
        return mean, variance


class OTTransportLightningModule(nn.Module):
    """Lightning-style wrapper for OT transport model training and logging."""

    def __init__(self, config: OTTransportConfig) -> None:
        super().__init__()
        self.config = config
        self.network = OTTransportNetwork(config=config)
        self.log_history: list[dict[str, float]] = []
        self._last_step_metrics: dict[str, Tensor] = {}

    def forward(
        self,
        source_expression: Tensor,
        sequence_embedding: Tensor,
        cell_state_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return cast(
            tuple[Tensor, Tensor],
            self.network(
                source_expression=source_expression,
                sequence_embedding=sequence_embedding,
                cell_state_embedding=cell_state_embedding,
            ),
        )

    def compute_batch_loss(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        source_expression, target_expression, sequence_embedding, cell_state_embedding = batch
        predicted_mean, predicted_variance = self(
            source_expression=source_expression,
            sequence_embedding=sequence_embedding,
            cell_state_embedding=cell_state_embedding,
        )

        nll = 0.5 * torch.mean(
            ((target_expression - predicted_mean).pow(2) / predicted_variance)
            + torch.log(predicted_variance)
        )
        mse = functional.mse_loss(predicted_mean, target_expression)
        sinkhorn = _differentiable_sinkhorn_distance(
            predicted_mean,
            target_expression,
            epsilon=self.config.sinkhorn_epsilon,
            n_iters=self.config.sinkhorn_n_iters,
        )
        loss = nll + (self.config.sinkhorn_weight * sinkhorn)
        return loss, {"nll": nll.detach(), "mse": mse.detach(), "sinkhorn": sinkhorn.detach()}

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int = 0,
    ) -> Tensor:
        _ = batch_idx
        loss, metrics = self.compute_batch_loss(batch=batch)
        self._last_step_metrics = metrics
        if _LIGHTNING_AVAILABLE:
            logger = getattr(self, "log_dict", None)
            if callable(logger):
                logger(
                    {
                        "train_loss": loss,
                        "train_mse": metrics["mse"],
                        "train_sinkhorn": metrics["sinkhorn"],
                    },
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def log_metrics(self, payload: Mapping[str, float]) -> None:
        self.log_history.append({key: float(value) for key, value in payload.items()})


class OTTransportStrategyImpl:
    """OT transport strategy implementation (Sinkhorn regularization)."""

    variant_name = TRANSPORT_VARIANT_NAME

    def __init__(
        self,
        *,
        config: OTTransportConfig,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.seed = seed
        self.module = OTTransportLightningModule(config=config).to(self.device)
        self.local_log_history: list[dict[str, float]] = []

        np.random.seed(seed)
        torch.manual_seed(seed)

    def fit(
        self,
        *,
        source_expression: np.ndarray,
        target_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        wandb_run: WandbRunLike | None = None,
        enable_wandb: bool = False,
        wandb_project: str = "tcpe",
        wandb_run_name: str | None = None,
        wandb_dir: str | Path = "artifacts/wandb",
    ) -> TransportFitResult:
        arrays = _validate_and_prepare_arrays(
            source_expression=source_expression,
            target_expression=target_expression,
            sequence_embedding=sequence_embedding,
            cell_state_embedding=cell_state_embedding,
            config=self.config,
        )
        source_array, target_array, sequence_array, cell_state_array = arrays

        source_tensor = torch.from_numpy(source_array).to(self.device)
        target_tensor = torch.from_numpy(target_array).to(self.device)
        sequence_tensor = torch.from_numpy(sequence_array).to(self.device)
        cell_state_tensor = torch.from_numpy(cell_state_array).to(self.device)

        dataset = TensorDataset(source_tensor, target_tensor, sequence_tensor, cell_state_tensor)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(self.seed)
        batch_size = min(self.config.batch_size, int(source_array.shape[0]))
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
        )

        optimizer = self.module.configure_optimizers()
        self.local_log_history = []
        self.module.log_history = []

        active_run = wandb_run
        owns_run = False
        if active_run is None and enable_wandb:
            active_run = init_wandb_run(
                project=wandb_project,
                run_name=wandb_run_name,
                run_dir=wandb_dir,
                config={
                    "variant": self.variant_name,
                    "schema_version": TRANSPORT_CHECKPOINT_SCHEMA_VERSION,
                    **asdict(self.config),
                },
            )
            owns_run = True

        loss_history: list[float] = []
        mse_history: list[float] = []
        sinkhorn_history: list[float] = []
        sinkhorn_pot_history: list[float] = []

        self.module.train()
        try:
            for epoch in range(self.config.n_epochs):
                batch_losses: list[float] = []
                for raw_batch in loader:
                    source_batch = cast(Tensor, raw_batch[0]).to(self.device)
                    target_batch = cast(Tensor, raw_batch[1]).to(self.device)
                    sequence_batch = cast(Tensor, raw_batch[2]).to(self.device)
                    cell_state_batch = cast(Tensor, raw_batch[3]).to(self.device)
                    batch = (source_batch, target_batch, sequence_batch, cell_state_batch)

                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self.module.compute_batch_loss(batch=batch)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(float(loss.detach().cpu().item()))

                self.module.eval()
                with torch.no_grad():
                    predicted_mean, _ = self.module(
                        source_expression=source_tensor,
                        sequence_embedding=sequence_tensor,
                        cell_state_embedding=cell_state_tensor,
                    )
                    epoch_mse = functional.mse_loss(predicted_mean, target_tensor)
                    epoch_sinkhorn = _differentiable_sinkhorn_distance(
                        predicted_mean,
                        target_tensor,
                        epsilon=self.config.sinkhorn_epsilon,
                        n_iters=self.config.sinkhorn_n_iters,
                    )
                    epoch_sinkhorn_pot = _sinkhorn_distance_metric(
                        x=predicted_mean.detach().cpu().numpy(),
                        y=target_tensor.detach().cpu().numpy(),
                        epsilon=self.config.sinkhorn_epsilon,
                        n_iters=self.config.sinkhorn_n_iters,
                    )
                self.module.train()

                epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
                mse_value = float(epoch_mse.detach().cpu().item())
                sinkhorn_value = float(epoch_sinkhorn.detach().cpu().item())
                sinkhorn_pot_value = float(epoch_sinkhorn_pot)

                loss_history.append(epoch_loss)
                mse_history.append(mse_value)
                sinkhorn_history.append(sinkhorn_value)
                sinkhorn_pot_history.append(sinkhorn_pot_value)

                metrics = {
                    "epoch": float(epoch + 1),
                    "train_loss": epoch_loss,
                    "train_mse": mse_value,
                    "train_sinkhorn": sinkhorn_value,
                    "train_sinkhorn_pot": sinkhorn_pot_value,
                }
                self.local_log_history.append(metrics)
                self.module.log_metrics(metrics)
                if active_run is not None:
                    active_run.log(metrics)
        finally:
            if owns_run and active_run is not None:
                active_run.finish()

        return TransportFitResult(
            variant_name=self.variant_name,
            n_epochs=self.config.n_epochs,
            loss_history=loss_history,
            mse_history=mse_history,
            sinkhorn_history=sinkhorn_history,
            sinkhorn_pot_history=sinkhorn_pot_history,
            log_history=list(self.local_log_history),
            checkpoint_schema_version=TRANSPORT_CHECKPOINT_SCHEMA_VERSION,
        )

    def predict_distribution(
        self,
        *,
        source_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        n_samples: int = 0,
    ) -> TransportPrediction:
        source_array = _coerce_dense_float32(source_expression)
        sequence_array = _coerce_dense_float32(sequence_embedding)
        cell_state_array = _coerce_dense_float32(cell_state_embedding)
        _validate_predict_shapes(
            source_expression=source_array,
            sequence_embedding=sequence_array,
            cell_state_embedding=cell_state_array,
            config=self.config,
        )

        source_tensor = torch.from_numpy(source_array).to(self.device)
        sequence_tensor = torch.from_numpy(sequence_array).to(self.device)
        cell_state_tensor = torch.from_numpy(cell_state_array).to(self.device)

        self.module.eval()
        with torch.no_grad():
            mean_tensor, variance_tensor = self.module(
                source_expression=source_tensor,
                sequence_embedding=sequence_tensor,
                cell_state_embedding=cell_state_tensor,
            )

        mean = mean_tensor.detach().cpu().numpy().astype(np.float32)
        variance = variance_tensor.detach().cpu().numpy().astype(np.float32)
        sampled: np.ndarray | None = None
        if n_samples > 0:
            std_tensor = torch.sqrt(torch.clamp(variance_tensor, min=self.config.uncertainty_floor))
            noise = torch.randn(
                (n_samples, *mean_tensor.shape),
                device=mean_tensor.device,
                dtype=mean_tensor.dtype,
            )
            samples = mean_tensor.unsqueeze(0) + (noise * std_tensor.unsqueeze(0))
            sampled = samples.detach().cpu().numpy().astype(np.float32)

        return TransportPrediction(mean=mean, variance=variance, sampled=sampled)

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TRANSPORT_CHECKPOINT_SCHEMA_VERSION,
            "variant_name": self.variant_name,
            "config": asdict(self.config),
            "seed": self.seed,
            "state_dict": self.module.state_dict(),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> Self:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise TransportCheckpointError(f"Checkpoint does not exist: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(payload, dict):
            raise TransportCheckpointError("Checkpoint payload is malformed (expected dict).")
        schema_version = payload.get("schema_version")
        if schema_version != TRANSPORT_CHECKPOINT_SCHEMA_VERSION:
            raise TransportCheckpointError(
                f"Unexpected checkpoint schema version '{schema_version}'. "
                f"Expected '{TRANSPORT_CHECKPOINT_SCHEMA_VERSION}'."
            )

        raw_config = payload.get("config")
        if not isinstance(raw_config, dict):
            raise TransportCheckpointError("Checkpoint payload missing config mapping.")
        config = OTTransportConfig(**cast(dict[str, Any], raw_config))

        seed = int(payload.get("seed", 42))
        strategy = cls(config=config, device=map_location, seed=seed)

        state_dict_raw = payload.get("state_dict")
        if not isinstance(state_dict_raw, dict):
            raise TransportCheckpointError("Checkpoint payload missing state_dict mapping.")
        state_dict = cast(dict[str, Tensor], state_dict_raw)
        strategy.module.load_state_dict(state_dict)
        strategy.module.eval()
        return strategy


OTTransportStrategy = OTTransportStrategyImpl


class FlowMatchingTransportStrategy:
    """Flow-matching scaffold with interface compliance and explicit status tags."""

    variant_name = FLOW_MATCHING_VARIANT_NAME
    status_tags: tuple[str, str] = (EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG)

    def __init__(self) -> None:
        self.notes = (
            "Flow-matching transport is scaffolded for Phase 11 and intended for cloud execution. "
            "Full training/inference support is deferred."
        )

    def fit(
        self,
        *,
        source_expression: np.ndarray,
        target_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        wandb_run: WandbRunLike | None = None,
        enable_wandb: bool = False,
        wandb_project: str = "tcpe",
        wandb_run_name: str | None = None,
        wandb_dir: str | Path = "artifacts/wandb",
    ) -> TransportFitResult:
        _ = (
            source_expression,
            target_expression,
            sequence_embedding,
            cell_state_embedding,
            wandb_run,
            enable_wandb,
            wandb_project,
            wandb_run_name,
            wandb_dir,
        )
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=self.variant_name,
                operation="fit",
                notes=self.notes,
            )
        )

    def predict_distribution(
        self,
        *,
        source_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        n_samples: int = 0,
    ) -> TransportPrediction:
        _ = (source_expression, sequence_embedding, cell_state_embedding, n_samples)
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=self.variant_name,
                operation="predict_distribution",
                notes=self.notes,
            )
        )

    def save(self, path: str | Path) -> Path:
        _ = path
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=self.variant_name,
                operation="save",
                notes=self.notes,
            )
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> Self:
        _ = (path, map_location)
        strategy = cls()
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=strategy.variant_name,
                operation="load",
                notes=strategy.notes,
            )
        )


class SchrodingerBridgeLightningModule(nn.Module):
    """Skeleton Lightning-style module with manual optimization enabled."""

    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.manual_optimization = True


class SchrodingerBridgeTransportStrategy:
    """Schrodinger bridge scaffold with manual-optimization module placeholder."""

    variant_name = SCHRODINGER_BRIDGE_VARIANT_NAME
    status_tags: tuple[str, str] = (EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG)

    def __init__(self) -> None:
        self.lightning_module = SchrodingerBridgeLightningModule()
        self.notes = (
            "Schrodinger bridge transport is scaffolded for Phase 11 and intended for cloud "
            "execution with manual optimization. Full training/inference support is deferred."
        )

    def fit(
        self,
        *,
        source_expression: np.ndarray,
        target_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        wandb_run: WandbRunLike | None = None,
        enable_wandb: bool = False,
        wandb_project: str = "tcpe",
        wandb_run_name: str | None = None,
        wandb_dir: str | Path = "artifacts/wandb",
    ) -> TransportFitResult:
        _ = (
            source_expression,
            target_expression,
            sequence_embedding,
            cell_state_embedding,
            wandb_run,
            enable_wandb,
            wandb_project,
            wandb_run_name,
            wandb_dir,
        )
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=self.variant_name,
                operation="fit",
                notes=self.notes,
            )
        )

    def predict_distribution(
        self,
        *,
        source_expression: np.ndarray,
        sequence_embedding: np.ndarray,
        cell_state_embedding: np.ndarray,
        n_samples: int = 0,
    ) -> TransportPrediction:
        _ = (source_expression, sequence_embedding, cell_state_embedding, n_samples)
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=self.variant_name,
                operation="predict_distribution",
                notes=self.notes,
            )
        )

    def save(self, path: str | Path) -> Path:
        _ = path
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=self.variant_name,
                operation="save",
                notes=self.notes,
            )
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> Self:
        _ = (path, map_location)
        strategy = cls()
        raise UnsupportedTransportPathError(
            _unsupported_path_message(
                variant=strategy.variant_name,
                operation="load",
                notes=strategy.notes,
            )
        )


def describe_transport_variant(variant: TransportVariant) -> TransportVariantInfo:
    """Return explicit status tags and notes for configured transport variant."""
    if variant == "ot":
        return TransportVariantInfo(
            variant="ot",
            strategy_name=TRANSPORT_VARIANT_NAME,
            status_tags=(),
            notes="OT strategy is fully operational for local Phase 10/11 runs.",
        )
    if variant == "flow":
        return TransportVariantInfo(
            variant="flow",
            strategy_name=FLOW_MATCHING_VARIANT_NAME,
            status_tags=(EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG),
            notes=(
                "Flow-matching strategy is scaffolded only. "
                "Heavy training should run in cloud phases."
            ),
        )
    if variant == "bridge":
        return TransportVariantInfo(
            variant="bridge",
            strategy_name=SCHRODINGER_BRIDGE_VARIANT_NAME,
            status_tags=(EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG),
            notes=(
                "Schrodinger bridge strategy is scaffolded only with manual optimization enabled. "
                "Heavy training should run in cloud phases."
            ),
        )
    raise ValueError(f"Unsupported transport variant '{variant}'.")


def build_transport_strategy(
    *,
    variant: TransportVariant,
    ot_config: OTTransportConfig | None = None,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> TransportStrategy:
    """Instantiate transport strategy selected via config/CLI dispatch."""
    if variant == "ot":
        if ot_config is None:
            raise ValueError("OT transport variant requires `ot_config`.")
        return OTTransportStrategy(config=ot_config, device=device, seed=seed)
    if variant == "flow":
        return FlowMatchingTransportStrategy()
    if variant == "bridge":
        return SchrodingerBridgeTransportStrategy()
    raise ValueError(f"Unsupported transport variant '{variant}'.")


class TransportModule:
    """Transport orchestration with mandatory baseline gate and OT strategy dispatch."""

    def __init__(self) -> None:
        self._active_strategy: TransportStrategy | None = None

    def status(self) -> str:
        return "phase11_transport_family_ready"

    def strategy_info(self, variant: TransportVariant) -> TransportVariantInfo:
        """Expose variant status tags for CLI/reporting."""
        return describe_transport_variant(variant)

    def resolve_strategy(
        self,
        *,
        variant: TransportVariant,
        ot_config: OTTransportConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ) -> TransportStrategy:
        """Instantiate configured transport strategy without code changes."""
        return build_transport_strategy(
            variant=variant,
            ot_config=ot_config,
            device=device,
            seed=seed,
        )

    def assert_baselines_ready(
        self,
        baseline_suite: BaselineSuiteResult | Mapping[str, Any],
    ) -> None:
        """Raise if mandatory baselines have not been run before training."""
        names = set(_extract_baseline_names(baseline_suite))
        required = set(REQUIRED_BASELINE_NAMES)
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(
                "Training is blocked until all mandatory baselines are available. "
                f"Missing baselines: {', '.join(missing)}."
            )

    def can_train(self, baseline_suite: BaselineSuiteResult | Mapping[str, Any]) -> bool:
        """Boolean wrapper for baseline readiness checks."""
        try:
            self.assert_baselines_ready(baseline_suite)
        except RuntimeError:
            return False
        return True

    def train_ot(
        self,
        *,
        baseline_suite: BaselineSuiteResult | Mapping[str, Any],
        training_data: TransportTrainingData,
        config: OTTransportConfig,
        device: str | torch.device = "cpu",
        wandb_run: WandbRunLike | None = None,
        enable_wandb: bool = False,
        wandb_project: str = "tcpe",
        wandb_run_name: str | None = None,
        wandb_dir: str | Path = "artifacts/wandb",
    ) -> tuple[OTTransportStrategy, TransportFitResult]:
        """Train OT strategy only after baseline-gate validation passes."""
        self.assert_baselines_ready(baseline_suite)

        strategy = OTTransportStrategy(config=config, device=device)
        fit_result = strategy.fit(
            source_expression=training_data.source_expression,
            target_expression=training_data.target_expression,
            sequence_embedding=training_data.sequence_embedding,
            cell_state_embedding=training_data.cell_state_embedding,
            wandb_run=wandb_run,
            enable_wandb=enable_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_dir=wandb_dir,
        )
        self._active_strategy = strategy
        return strategy, fit_result

    def active_strategy(self) -> TransportStrategy | None:
        """Return most recently trained OT strategy."""
        return self._active_strategy


def _unsupported_path_message(*, variant: str, operation: str, notes: str) -> str:
    tags = f"{EXPERIMENTAL_TAG}, {CLOUD_RECOMMENDED_TAG}"
    return (
        f"Transport variant '{variant}' does not support '{operation}' in local Phase 11. "
        f"Status tags: {tags}. {notes}"
    )


def _extract_baseline_names(baseline_suite: BaselineSuiteResult | Mapping[str, Any]) -> list[str]:
    if isinstance(baseline_suite, BaselineSuiteResult):
        return baseline_suite.baseline_names()

    baselines_raw = baseline_suite.get("baselines")
    if not isinstance(baselines_raw, list):
        return []

    names: list[str] = []
    for item in baselines_raw:
        if isinstance(item, Mapping):
            name = item.get("baseline_name")
            if isinstance(name, str):
                names.append(name)
    return names


def _coerce_dense_float32(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    return cast(np.ndarray, np.asarray(dense, dtype=np.float32))


def _resolve_expression_matrix(adata: AnnData, expression_layer: str) -> np.ndarray:
    if expression_layer in adata.layers:
        matrix = adata.layers[expression_layer]
    elif expression_layer == "X":
        matrix = adata.X
    else:
        raise TransportInputError(
            f"Expression layer '{expression_layer}' not found in AnnData layers and is not 'X'."
        )
    return _coerce_dense_float32(matrix)


def _build_source_expression(
    *,
    adata: AnnData,
    target_expression: np.ndarray,
    source_policy: Literal["control_mean", "identity"],
) -> np.ndarray:
    if source_policy == "identity":
        return target_expression.copy()

    control_mask = _detect_control_mask(adata=adata)
    if int(np.sum(control_mask)) > 0:
        control_mean = target_expression[control_mask].mean(axis=0)
    else:
        control_mean = target_expression.mean(axis=0)
    repeated = np.repeat(control_mean[None, :], repeats=target_expression.shape[0], axis=0)
    return cast(np.ndarray, repeated.astype(np.float32))


def _detect_control_mask(adata: AnnData) -> np.ndarray:
    condition_mask = np.zeros((adata.n_obs,), dtype=bool)
    if "condition" in adata.obs.columns:
        condition_mask = adata.obs["condition"].astype(str).str.lower().eq("control").to_numpy()

    perturbation_mask = np.zeros((adata.n_obs,), dtype=bool)
    if "perturbation_id" in adata.obs.columns:
        perturbation_mask = (
            adata.obs["perturbation_id"]
            .astype(str)
            .str.lower()
            .isin({"ntc", "ctrl", "control", "p000"})
            .to_numpy()
        )
    return cast(np.ndarray, condition_mask | perturbation_mask)


def _validate_and_prepare_arrays(
    *,
    source_expression: np.ndarray,
    target_expression: np.ndarray,
    sequence_embedding: np.ndarray,
    cell_state_embedding: np.ndarray,
    config: OTTransportConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_array = _coerce_dense_float32(source_expression)
    target_array = _coerce_dense_float32(target_expression)
    sequence_array = _coerce_dense_float32(sequence_embedding)
    cell_state_array = _coerce_dense_float32(cell_state_embedding)

    if source_array.shape != target_array.shape:
        raise TransportInputError(
            f"source_expression shape {source_array.shape} must equal target_expression "
            f"shape {target_array.shape}."
        )
    if len(source_array.shape) != 2:
        raise TransportInputError("source_expression and target_expression must be 2D matrices.")
    n_obs, n_genes = source_array.shape
    if n_obs == 0:
        raise TransportInputError("Training arrays must include at least one cell.")
    if n_genes != config.input_dim:
        raise TransportInputError(
            f"input_dim mismatch: config expects {config.input_dim}, got {n_genes}."
        )

    _validate_predict_shapes(
        source_expression=source_array,
        sequence_embedding=sequence_array,
        cell_state_embedding=cell_state_array,
        config=config,
    )
    return source_array, target_array, sequence_array, cell_state_array


def _validate_predict_shapes(
    *,
    source_expression: np.ndarray,
    sequence_embedding: np.ndarray,
    cell_state_embedding: np.ndarray,
    config: OTTransportConfig,
) -> None:
    if len(source_expression.shape) != 2:
        raise TransportInputError("source_expression must be 2D [n_obs, n_genes].")
    n_obs = source_expression.shape[0]

    if sequence_embedding.shape != (n_obs, config.sequence_embedding_dim):
        raise TransportInputError(
            "sequence_embedding shape mismatch. Expected "
            f"({n_obs}, {config.sequence_embedding_dim}), got {sequence_embedding.shape}."
        )
    if cell_state_embedding.shape != (n_obs, config.cell_state_embedding_dim):
        raise TransportInputError(
            "cell_state_embedding shape mismatch. Expected "
            f"({n_obs}, {config.cell_state_embedding_dim}), got {cell_state_embedding.shape}."
        )


def _differentiable_sinkhorn_distance(
    x: Tensor,
    y: Tensor,
    *,
    epsilon: float,
    n_iters: int,
) -> Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.zeros((), dtype=x.dtype, device=x.device)

    cost = torch.cdist(x, y, p=2).pow(2)
    n_x = x.shape[0]
    n_y = y.shape[0]

    a = torch.full((n_x,), 1.0 / float(n_x), dtype=x.dtype, device=x.device)
    b = torch.full((n_y,), 1.0 / float(n_y), dtype=y.dtype, device=y.device)
    kernel = torch.exp(-cost / epsilon)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    tiny = torch.finfo(x.dtype).eps

    for _ in range(n_iters):
        u = a / torch.clamp(kernel @ v, min=tiny)
        v = b / torch.clamp(torch.transpose(kernel, 0, 1) @ u, min=tiny)

    transport_plan = u[:, None] * kernel * v[None, :]
    return torch.sum(transport_plan * cost)


def _sinkhorn_distance_metric(
    *,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    n_iters: int,
) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if ot is None:
        x_tensor = torch.from_numpy(x.astype(np.float32, copy=False))
        y_tensor = torch.from_numpy(y.astype(np.float32, copy=False))
        sinkhorn = _differentiable_sinkhorn_distance(
            x_tensor,
            y_tensor,
            epsilon=epsilon,
            n_iters=n_iters,
        )
        return float(sinkhorn.detach().cpu().item())

    a = np.full((x.shape[0],), 1.0 / float(x.shape[0]), dtype=np.float64)
    b = np.full((y.shape[0],), 1.0 / float(y.shape[0]), dtype=np.float64)
    cost = ot.dist(
        x.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        metric="sqeuclidean",
    )
    sinkhorn = ot.sinkhorn2(a, b, cost, reg=epsilon, numItermax=n_iters)
    return float(sinkhorn)
