"""TCPE CLI scaffold for Phase 2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tcpe.config import (
    DEFAULT_CONFIG_PATH,
    ENVIRONMENT_NAMES,
    ConfigLoadError,
    ConfigValidationError,
    EnvironmentName,
    load_config,
)
from tcpe.runtime import (
    build_artifact_layout,
    ensure_artifact_layout,
    generate_run_id,
    set_global_seed,
    write_run_manifest,
)

COMMAND_GROUPS: tuple[str, ...] = (
    "data",
    "embed",
    "train",
    "causal",
    "eval",
    "card",
    "pipeline",
    "cloud",
)
PIPELINE_STEP_CHOICES: tuple[str, ...] = (
    "ingest",
    "preprocess",
    "embed",
    "train",
    "causal",
    "evaluate",
    "card",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    parser = argparse.ArgumentParser(
        prog="tcpe",
        description="Transportable Causal Perturbation Engine (Phase 2 CLI scaffold).",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to base config YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    common.add_argument(
        "--env",
        choices=ENVIRONMENT_NAMES,
        default="local",
        help="Environment overlay to apply.",
    )
    common.add_argument(
        "--run-id",
        default=None,
        help="Optional run id override. If omitted, a deterministic run id is generated.",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print resolved run layout without writing directories.",
    )

    subparsers = parser.add_subparsers(dest="command_group", required=True)
    for name in COMMAND_GROUPS:
        group_parser = subparsers.add_parser(
            name,
            parents=[common],
            help=f"{name} command group",
            description=f"Phase 2 scaffold for `{name}` command group.",
        )
        if name == "pipeline":
            group_parser.add_argument(
                "--resume",
                action="store_true",
                help="Resume pipeline execution from the last saved pipeline checkpoint.",
            )
            group_parser.add_argument(
                "--dataset-id",
                default=None,
                help="Optional dataset id override for pipeline runs (default: config primary_id).",
            )
            group_parser.add_argument(
                "--synthetic-cells",
                type=int,
                default=120,
                help="Synthetic pipeline cell count when dataset-id is synthetic.",
            )
            group_parser.add_argument(
                "--synthetic-genes",
                type=int,
                default=60,
                help="Synthetic pipeline gene count when dataset-id is synthetic.",
            )
            group_parser.add_argument(
                "--synthetic-perturbations",
                type=int,
                default=8,
                help="Synthetic pipeline perturbation count when dataset-id is synthetic.",
            )
            group_parser.add_argument(
                "--stop-after",
                choices=PIPELINE_STEP_CHOICES,
                default=None,
                help="Optional step name where pipeline run should stop early.",
            )
            group_parser.add_argument(
                "--transport-epochs",
                type=int,
                default=5,
                help="Transport OT training epochs for pipeline runs.",
            )
            group_parser.add_argument(
                "--transport-latent-dim",
                type=int,
                default=None,
                help="Optional transport latent dimension override for pipeline runs.",
            )
            group_parser.add_argument(
                "--transport-hidden-dim",
                type=int,
                default=None,
                help="Optional transport hidden dimension override for pipeline runs.",
            )
            group_parser.add_argument(
                "--causal-max-hvgs",
                type=int,
                default=40,
                help="Causal-module max HVG count for pipeline runs.",
            )
            group_parser.add_argument(
                "--causal-bootstrap-iters",
                type=int,
                default=0,
                help="Causal bootstrap iterations for pipeline runs.",
            )
        if name == "cloud":
            group_parser.add_argument(
                "--preprocessed-h5ad",
                type=Path,
                default=None,
                help=(
                    "Optional externally preprocessed HVG-selected h5ad to seed a resume-ready "
                    "checkpoint bundle."
                ),
            )
            group_parser.add_argument(
                "--dataset-label",
                default="external_hvg_h5ad",
                help="Dataset label to record in the seeded checkpoint bundle.",
            )
            group_parser.add_argument(
                "--budget-spent-usd",
                type=float,
                default=0.0,
                help="Current spend to record in the Phase 17 budget log.",
            )
            group_parser.add_argument(
                "--simulate-interruption-after",
                choices=PIPELINE_STEP_CHOICES,
                default="train",
                help="Pipeline step used for the spot interruption recovery simulation.",
            )
        group_parser.set_defaults(handler=_run_group_scaffold)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


def _run_group_scaffold(args: argparse.Namespace) -> int:
    command_group = str(args.command_group)
    config_path = Path(args.config)
    environment = args.env
    run_id_override = args.run_id
    dry_run = bool(args.dry_run)
    resume = bool(getattr(args, "resume", False))

    try:
        config = load_config(
            config_path=config_path,
            environment=environment_name(environment),
        )
    except (ConfigLoadError, ConfigValidationError, ValueError) as exc:
        print(f"[tcpe] configuration error: {exc}", file=sys.stderr)
        return 2

    set_global_seed(config.runtime.seed, config.runtime.deterministic_torch)

    run_id = generate_run_id(config=config, command_group=command_group, provided=run_id_override)
    layout = build_artifact_layout(config=config, command_group=command_group, run_id=run_id)
    transport_dispatch: dict[str, Any] | None = None
    pipeline_result: dict[str, Any] | None = None
    cloud_handoff_result: dict[str, Any] | None = None
    if command_group == "train":
        from tcpe.transport import describe_transport_variant

        info = describe_transport_variant(config.model.transport_variant)
        transport_dispatch = info.to_dict()

    manifest_path = None
    if command_group == "cloud":
        from tcpe.cloud_handoff import CloudHandoffModule

        if not dry_run:
            ensure_artifact_layout(layout)
            manifest_path = write_run_manifest(
                layout=layout,
                config=config,
                command_group=command_group,
                run_id=run_id,
                config_path=config_path,
            )
        cloud_module = CloudHandoffModule()
        cloud_result = cloud_module.run(
            config=config,
            layout=layout,
            run_id=run_id,
            preprocessed_h5ad_path=cast(Path | None, getattr(args, "preprocessed_h5ad", None)),
            dataset_label=str(getattr(args, "dataset_label", "external_hvg_h5ad")),
            budget_spent_usd=float(getattr(args, "budget_spent_usd", 0.0)),
            simulate_interruption_after=cast(
                Any,
                getattr(args, "simulate_interruption_after", "train"),
            ),
            dry_run=dry_run,
        )
        cloud_handoff_result = cloud_result.to_dict()
    elif not dry_run:
        ensure_artifact_layout(layout)
        manifest_path = write_run_manifest(
            layout=layout,
            config=config,
            command_group=command_group,
            run_id=run_id,
            config_path=config_path,
        )
        if command_group == "pipeline":
            from tcpe.pipeline import PipelineConfig, PipelineModule

            dataset_override = cast(str | None, getattr(args, "dataset_id", None))
            dataset_id = (
                dataset_override if dataset_override is not None else str(config.dataset.primary_id)
            )
            latent_override = cast(int | None, getattr(args, "transport_latent_dim", None))
            hidden_override = cast(int | None, getattr(args, "transport_hidden_dim", None))
            stop_after = cast(str | None, getattr(args, "stop_after", None))
            transport_latent_dim = (
                int(config.model.latent_dim) if latent_override is None else int(latent_override)
            )
            transport_hidden_dim = (
                max(96, int(config.model.latent_dim) * 2)
                if hidden_override is None
                else int(hidden_override)
            )
            pipeline_config = PipelineConfig(
                dataset_id=dataset_id,
                synthetic_n_cells=int(getattr(args, "synthetic_cells", 120)),
                synthetic_n_genes=int(getattr(args, "synthetic_genes", 60)),
                synthetic_n_perturbations=int(getattr(args, "synthetic_perturbations", 8)),
                seed=int(config.runtime.seed),
                transport_latent_dim=transport_latent_dim,
                transport_hidden_dim=transport_hidden_dim,
                transport_epochs=int(getattr(args, "transport_epochs", 5)),
                causal_max_hvgs=int(getattr(args, "causal_max_hvgs", 40)),
                causal_bootstrap_iterations=int(getattr(args, "causal_bootstrap_iters", 0)),
                stop_after_step=cast(Any, stop_after),
            )
            pipeline_module = PipelineModule()
            result = pipeline_module.run(
                config=config,
                layout=layout,
                run_id=run_id,
                pipeline_config=pipeline_config,
                resume=resume,
            )
            pipeline_result = result.to_dict()

    payload = {
        "command_group": command_group,
        "environment": config.environment,
        "config_path": str(config_path),
        "run_id": run_id,
        "dry_run": dry_run,
        "artifact_layout": {
            "run_root": str(layout.run_root),
            "logs_dir": str(layout.logs_dir),
            "checkpoints_dir": str(layout.checkpoints_dir),
            "reports_dir": str(layout.reports_dir),
            "metadata_dir": str(layout.metadata_dir),
        },
        "transport_dispatch": transport_dispatch,
        "pipeline_result": pipeline_result,
        "cloud_handoff_result": cloud_handoff_result,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "message": (
            "Phase 17 cloud handoff plan generated."
            if command_group == "cloud"
            else (
                f"Phase 15 pipeline executed for '{command_group}'."
                if command_group == "pipeline" and not dry_run
                else f"Phase 2 scaffold executed for '{command_group}'."
            )
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def environment_name(value: str) -> EnvironmentName:
    """Cast CLI environment argument to the typed environment literal."""
    if value not in ENVIRONMENT_NAMES:
        joined = ", ".join(ENVIRONMENT_NAMES)
        raise ValueError(f"Unsupported environment '{value}'. Valid values: {joined}")
    return cast(EnvironmentName, value)
