"""Phase 18 cloud execution planning and launcher-script materialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from tcpe.cloud_handoff import CloudHandoffConfig
from tcpe.config import TCPEConfig
from tcpe.runtime.run_context import ArtifactLayout

PHASE18_EXECUTION_SCHEMA_VERSION = "phase18_cloud_execution_v1"
PHASE18_JOB_MANIFEST_SCHEMA_VERSION = "phase18_job_manifest_v1"
PHASE18_SEQUENCE_CHECKPOINT_SCHEMA_VERSION = "phase18_sequence_checkpoint_v1"

CloudExecutor = Literal["oracle_vm", "kaggle"]
ContextShiftExecutor = Literal["oracle_vm", "kaggle"]


class CloudExecutionError(RuntimeError):
    """Raised when phase-18 execution planning cannot be completed."""


@dataclass(frozen=True)
class Phase18ExecutionConfig:
    """Execution-time defaults for the phase-18 cloud sequence."""

    handoff_config: CloudHandoffConfig = field(default_factory=CloudHandoffConfig)
    figshare_article_url: str = "https://plus.figshare.com/articles/dataset/20029387"
    figshare_api_url: str = "https://api.figshare.com/v2/articles/20029387"
    oci_profile: str = "DEFAULT"
    oci_config_file: Path | None = None
    bucket_root_prefix: str = "phase18"
    context_shift_executor: ContextShiftExecutor = "oracle_vm"
    include_optional_bridge: bool = False
    tmux_session_prefix: str = "tcpe18"
    oracle_shutdown_command: str = "sudo shutdown -h now"
    kaggle_stop_instruction: str = "Stop the Kaggle session from the UI immediately."

    def __post_init__(self) -> None:
        if str(self.figshare_article_url).strip() == "":
            raise ValueError("figshare_article_url must be non-empty.")
        if str(self.figshare_api_url).strip() == "":
            raise ValueError("figshare_api_url must be non-empty.")
        if str(self.oci_profile).strip() == "":
            raise ValueError("oci_profile must be non-empty.")
        if str(self.bucket_root_prefix).strip() == "":
            raise ValueError("bucket_root_prefix must be non-empty.")
        if self.context_shift_executor not in ("oracle_vm", "kaggle"):
            raise ValueError("context_shift_executor must be oracle_vm or kaggle.")
        if str(self.tmux_session_prefix).strip() == "":
            raise ValueError("tmux_session_prefix must be non-empty.")
        if str(self.oracle_shutdown_command).strip() == "":
            raise ValueError("oracle_shutdown_command must be non-empty.")
        if str(self.kaggle_stop_instruction).strip() == "":
            raise ValueError("kaggle_stop_instruction must be non-empty.")


@dataclass(frozen=True)
class Phase18JobSpec:
    """One concrete job in the phase-18 execution ladder."""

    sequence_index: int
    job_id: str
    title: str
    executor: CloudExecutor
    datasets: list[str]
    depends_on: list[str]
    optional: bool
    enabled: bool
    max_runtime_minutes: int
    tmux_session_name: str | None
    stop_compute_instruction: str
    script_filename: str
    command_preview: str
    artifact_prefix: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase18ExecutionResult:
    """Serializable result payload for phase-18 planning."""

    schema_version: str
    generated_at_utc: str
    run_id: str
    dry_run: bool
    output_dir: Path | None
    job_specs: list[Phase18JobSpec]
    sequence_checkpoint: dict[str, Any]
    plan_json_path: Path | None = None
    runbook_markdown_path: Path | None = None
    job_manifest_path: Path | None = None
    sequence_checkpoint_path: Path | None = None
    launcher_paths: dict[str, Path] = field(default_factory=dict)
    tmux_launcher_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "job_specs": [item.to_dict() for item in self.job_specs],
            "sequence_checkpoint": dict(self.sequence_checkpoint),
            "plan_json_path": None if self.plan_json_path is None else str(self.plan_json_path),
            "runbook_markdown_path": (
                None if self.runbook_markdown_path is None else str(self.runbook_markdown_path)
            ),
            "job_manifest_path": (
                None if self.job_manifest_path is None else str(self.job_manifest_path)
            ),
            "sequence_checkpoint_path": (
                None
                if self.sequence_checkpoint_path is None
                else str(self.sequence_checkpoint_path)
            ),
            "launcher_paths": {
                job_id: str(path) for job_id, path in self.launcher_paths.items()
            },
            "tmux_launcher_path": (
                None if self.tmux_launcher_path is None else str(self.tmux_launcher_path)
            ),
        }


class OCIObjectStorageClient:
    """Thin lazy wrapper around the OCI Python SDK for Object Storage transfers."""

    def __init__(
        self,
        *,
        namespace: str,
        bucket_name: str,
        region: str | None = None,
        config_file: str | Path | None = None,
        profile: str = "DEFAULT",
    ) -> None:
        if str(namespace).strip() == "":
            raise ValueError("namespace must be non-empty.")
        if str(bucket_name).strip() == "":
            raise ValueError("bucket_name must be non-empty.")
        if str(profile).strip() == "":
            raise ValueError("profile must be non-empty.")
        self.namespace = str(namespace)
        self.bucket_name = str(bucket_name)
        self.region = None if region is None else str(region)
        self.config_file = None if config_file is None else Path(config_file)
        self.profile = str(profile)
        self._client: Any = None

    def upload_file(self, *, local_path: str | Path, object_name: str) -> dict[str, Any]:
        source_path = Path(local_path)
        if not source_path.exists():
            raise CloudExecutionError(f"Upload source file does not exist: {source_path}")
        client = self._get_client()
        with source_path.open("rb") as handle:
            response = client.put_object(
                self.namespace,
                self.bucket_name,
                object_name,
                handle,
            )
        return {
            "object_name": object_name,
            "local_path": str(source_path),
            "etag": str(getattr(response, "headers", {}).get("etag", "")),
        }

    def download_file(self, *, object_name: str, local_path: str | Path) -> Path:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        response = client.get_object(self.namespace, self.bucket_name, object_name)
        data = getattr(response, "data", None)
        if data is None:
            raise CloudExecutionError(
                f"OCI get_object returned no data payload for `{object_name}`."
            )
        with target.open("wb") as handle:
            raw = getattr(data, "raw", None)
            if raw is not None and hasattr(raw, "stream"):
                for chunk in raw.stream(1024 * 1024, decode_content=False):
                    handle.write(chunk)
            else:
                content = getattr(data, "content", None)
                if not isinstance(content, (bytes, bytearray)):
                    raise CloudExecutionError(
                        f"OCI object payload for `{object_name}` is not byte content."
                    )
                handle.write(bytes(content))
        return target

    def list_objects(self, *, prefix: str = "") -> list[str]:
        client = self._get_client()
        response = client.list_objects(
            self.namespace,
            self.bucket_name,
            prefix=prefix or None,
        )
        payload = getattr(response, "data", None)
        objects = getattr(payload, "objects", None)
        if objects is None:
            return []
        names: list[str] = []
        for item in objects:
            name = getattr(item, "name", None)
            if isinstance(name, str):
                names.append(name)
        return names

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        oci = self._require_oci()
        if self.config_file is None:
            oci_config = oci.config.from_file(profile_name=self.profile)
        else:
            oci_config = oci.config.from_file(
                file_location=str(self.config_file),
                profile_name=self.profile,
            )
        if self.region is not None:
            oci_config["region"] = self.region
        self._client = oci.object_storage.ObjectStorageClient(oci_config)
        return self._client

    def _require_oci(self) -> Any:
        try:
            import oci
        except ImportError as exc:  # pragma: no cover - depends on runtime env.
            raise CloudExecutionError(
                "The OCI Python SDK is required for phase-18 bucket transfers."
            ) from exc
        return oci


class Phase18ExecutionModule:
    """Phase-18 planner that materializes exact ordered cloud execution scripts."""

    def status(self) -> str:
        return "phase18_cloud_execution_ready"

    def run(
        self,
        *,
        config: TCPEConfig,
        layout: ArtifactLayout,
        run_id: str,
        execution_config: Phase18ExecutionConfig | None = None,
        dry_run: bool = False,
    ) -> Phase18ExecutionResult:
        selected = execution_config if execution_config is not None else Phase18ExecutionConfig()
        generated_at = datetime.now(UTC).isoformat()
        jobs = self._build_job_specs(run_id=run_id, selected=selected)
        checkpoint = {
            "schema_version": PHASE18_SEQUENCE_CHECKPOINT_SCHEMA_VERSION,
            "generated_at_utc": generated_at,
            "run_id": run_id,
            "completed_jobs": [],
            "pending_jobs": [item.job_id for item in jobs if item.enabled],
            "optional_jobs": [item.job_id for item in jobs if item.optional],
        }

        output_dir = None if dry_run else (layout.metadata_dir / "phase18_cloud_execution")
        plan_json_path: Path | None = None
        runbook_path: Path | None = None
        job_manifest_path: Path | None = None
        checkpoint_path: Path | None = None
        launcher_paths: dict[str, Path] = {}
        tmux_launcher_path: Path | None = None

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            launchers_dir = output_dir / "launchers"
            launchers_dir.mkdir(parents=True, exist_ok=True)

            for job in jobs:
                path = launchers_dir / job.script_filename
                path.write_text(
                    self._render_job_script(
                        job=job,
                        config=config,
                        run_id=run_id,
                        selected=selected,
                    ),
                    encoding="utf-8",
                )
                launcher_paths[job.job_id] = path

            tmux_launcher_path = launchers_dir / "oracle_tmux_launch.sh"
            tmux_launcher_path.write_text(
                self._render_oracle_tmux_launcher(
                    jobs=jobs,
                    launchers_dir=launchers_dir,
                ),
                encoding="utf-8",
            )

            job_manifest_path = output_dir / "phase18_job_manifest.json"
            job_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": PHASE18_JOB_MANIFEST_SCHEMA_VERSION,
                        "generated_at_utc": generated_at,
                        "run_id": run_id,
                        "jobs": [item.to_dict() for item in jobs],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            checkpoint_path = output_dir / "phase18_sequence_checkpoint.json"
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            runbook_path = output_dir / "phase18_execution_runbook.md"
            runbook_path.write_text(
                self._render_runbook(
                    config=config,
                    run_id=run_id,
                    jobs=jobs,
                    selected=selected,
                    tmux_launcher_path=tmux_launcher_path,
                ),
                encoding="utf-8",
            )

            plan_json_path = output_dir / "phase18_execution_plan.json"

        result = Phase18ExecutionResult(
            schema_version=PHASE18_EXECUTION_SCHEMA_VERSION,
            generated_at_utc=generated_at,
            run_id=run_id,
            dry_run=dry_run,
            output_dir=output_dir,
            job_specs=jobs,
            sequence_checkpoint=checkpoint,
            plan_json_path=plan_json_path,
            runbook_markdown_path=runbook_path,
            job_manifest_path=job_manifest_path,
            sequence_checkpoint_path=checkpoint_path,
            launcher_paths=launcher_paths,
            tmux_launcher_path=tmux_launcher_path,
        )

        if plan_json_path is not None:
            plan_json_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return result

    def _build_job_specs(
        self,
        *,
        run_id: str,
        selected: Phase18ExecutionConfig,
    ) -> list[Phase18JobSpec]:
        datasets = list(selected.handoff_config.datasets)
        context_executor = selected.context_shift_executor
        prefix = f"{selected.bucket_root_prefix}/{run_id}"
        jobs: list[Phase18JobSpec] = [
            Phase18JobSpec(
                sequence_index=1,
                job_id="01_oracle_stage_raw_upload",
                title="Oracle VM raw Figshare download and OCI upload",
                executor="oracle_vm",
                datasets=[item.dataset_key for item in datasets],
                depends_on=[],
                optional=False,
                enabled=True,
                max_runtime_minutes=180,
                tmux_session_name=f"{selected.tmux_session_prefix}-01-raw",
                stop_compute_instruction=selected.oracle_shutdown_command,
                script_filename="01_oracle_stage_raw_upload.sh",
                command_preview=(
                    "Fetch raw single-cell h5ad files from the Figshare article and upload "
                    "them to OCI Object Storage."
                ),
                artifact_prefix=f"{prefix}/raw",
                summary="Raw data staging through OCI is the first required gate.",
            ),
            Phase18JobSpec(
                sequence_index=2,
                job_id="02_kaggle_preprocess_embed",
                title="Kaggle preprocessing and frozen embedder generation",
                executor="kaggle",
                datasets=[item.dataset_key for item in datasets],
                depends_on=["01_oracle_stage_raw_upload"],
                optional=False,
                enabled=True,
                max_runtime_minutes=480,
                tmux_session_name=None,
                stop_compute_instruction=selected.kaggle_stop_instruction,
                script_filename="02_kaggle_preprocess_embed.sh",
                command_preview=(
                    "Download raw h5ad objects from OCI, repair schema, run QC + "
                    "normalization + HVG=5000, and write embedded processed h5ad files back."
                ),
                artifact_prefix=f"{prefix}/processed",
                summary="Only HVG-selected processed h5ad files return to OCI for later jobs.",
            ),
            Phase18JobSpec(
                sequence_index=3,
                job_id="03_kaggle_k562_essential_tuning",
                title="Kaggle K562_essential hyperparameter tuning subset",
                executor="kaggle",
                datasets=["k562_essential"],
                depends_on=["02_kaggle_preprocess_embed"],
                optional=False,
                enabled=True,
                max_runtime_minutes=240,
                tmux_session_name=None,
                stop_compute_instruction=selected.kaggle_stop_instruction,
                script_filename="03_kaggle_k562_essential_tuning.sh",
                command_preview=(
                    "Run iterative subset sweeps for transport and causal hyperparameters."
                ),
                artifact_prefix=f"{prefix}/tuning",
                summary="Subset tuning precedes any full transport training job.",
            ),
            Phase18JobSpec(
                sequence_index=4,
                job_id="04_kaggle_k562_gwps_transport",
                title="Kaggle full K562_gwps OT transport training",
                executor="kaggle",
                datasets=["replogle_k562_gwps"],
                depends_on=["03_kaggle_k562_essential_tuning"],
                optional=False,
                enabled=True,
                max_runtime_minutes=480,
                tmux_session_name=None,
                stop_compute_instruction=selected.kaggle_stop_instruction,
                script_filename="04_kaggle_k562_gwps_transport.sh",
                command_preview=(
                    "Download processed K562_gwps h5ad, run and log baselines first, "
                    "then fit the OT transport model and upload weights."
                ),
                artifact_prefix=f"{prefix}/train/k562_gwps",
                summary="Baselines are mandatory and must be logged before OT training begins.",
            ),
            Phase18JobSpec(
                sequence_index=5,
                job_id="05_kaggle_rpe1_transport_eval",
                title="Kaggle full RPE1 transport and cross-cell-type evaluation",
                executor="kaggle",
                datasets=["rpe1"],
                depends_on=["04_kaggle_k562_gwps_transport"],
                optional=False,
                enabled=True,
                max_runtime_minutes=480,
                tmux_session_name=None,
                stop_compute_instruction=selected.kaggle_stop_instruction,
                script_filename="05_kaggle_rpe1_transport_eval.sh",
                command_preview=(
                    "Train the RPE1 OT model, run evaluation, and persist cross-cell-type "
                    "evaluation artifacts back to OCI."
                ),
                artifact_prefix=f"{prefix}/train/rpe1",
                summary="RPE1 training and evaluation produce the phase-18 evaluation anchor.",
            ),
            Phase18JobSpec(
                sequence_index=6,
                job_id="06_oracle_causal_iv_genomewide",
                title="Oracle VM genome-wide causal IV regression",
                executor="oracle_vm",
                datasets=["replogle_k562_gwps", "rpe1"],
                depends_on=["05_kaggle_rpe1_transport_eval"],
                optional=False,
                enabled=True,
                max_runtime_minutes=720,
                tmux_session_name=f"{selected.tmux_session_prefix}-06-causal",
                stop_compute_instruction=selected.oracle_shutdown_command,
                script_filename="06_oracle_causal_iv_genomewide.sh",
                command_preview=(
                    "Run causal IV regression using batch, library_size, "
                    "knockdown_efficiency_proxy, and protocol."
                ),
                artifact_prefix=f"{prefix}/causal",
                summary="Phase-12 causal proxies are enforced for the genome-wide run.",
            ),
            Phase18JobSpec(
                sequence_index=7,
                job_id="07_context_shift_and_model_card",
                title="Context-shift sweep and mandatory model card generation",
                executor=context_executor,
                datasets=["replogle_k562_gwps", "rpe1"],
                depends_on=["06_oracle_causal_iv_genomewide"],
                optional=False,
                enabled=True,
                max_runtime_minutes=360,
                tmux_session_name=(
                    None
                    if context_executor == "kaggle"
                    else f"{selected.tmux_session_prefix}-07-context"
                ),
                stop_compute_instruction=(
                    selected.kaggle_stop_instruction
                    if context_executor == "kaggle"
                    else selected.oracle_shutdown_command
                ),
                script_filename="07_context_shift_and_model_card.sh",
                command_preview=(
                    "Generate all context-shift manifests and build the mandatory model card "
                    "from the uploaded evaluation report."
                ),
                artifact_prefix=f"{prefix}/reports",
                summary="The run remains incomplete unless model card JSON and Markdown exist.",
            ),
            Phase18JobSpec(
                sequence_index=8,
                job_id="08_optional_schrodinger_bridge",
                title="Optional Kaggle Schrodinger bridge scaffold run",
                executor="kaggle",
                datasets=["replogle_k562_gwps"],
                depends_on=["07_context_shift_and_model_card"],
                optional=True,
                enabled=selected.include_optional_bridge,
                max_runtime_minutes=480,
                tmux_session_name=None,
                stop_compute_instruction=selected.kaggle_stop_instruction,
                script_filename="08_optional_schrodinger_bridge.sh",
                command_preview=(
                    "Attempt the Schrodinger bridge scaffold only if OT baselines are stable."
                ),
                artifact_prefix=f"{prefix}/bridge",
                summary="This remains optional and respects the existing bridge scaffold limits.",
            ),
        ]
        return jobs

    def _render_job_script(
        self,
        *,
        job: Phase18JobSpec,
        config: TCPEConfig,
        run_id: str,
        selected: Phase18ExecutionConfig,
    ) -> str:
        lines = self._script_preamble(
            config=config,
            run_id=run_id,
            job=job,
            selected=selected,
        )
        if job.job_id == "01_oracle_stage_raw_upload":
            lines.extend(self._render_step1_body(job=job, selected=selected))
        elif job.job_id == "02_kaggle_preprocess_embed":
            lines.extend(self._render_step2_body(job=job, selected=selected))
        elif job.job_id == "03_kaggle_k562_essential_tuning":
            lines.extend(self._render_step3_body(job=job))
        elif job.job_id == "04_kaggle_k562_gwps_transport":
            lines.extend(self._render_step4_body(job=job))
        elif job.job_id == "05_kaggle_rpe1_transport_eval":
            lines.extend(self._render_step5_body(job=job))
        elif job.job_id == "06_oracle_causal_iv_genomewide":
            lines.extend(self._render_step6_body(job=job))
        elif job.job_id == "07_context_shift_and_model_card":
            lines.extend(self._render_step7_body(job=job))
        elif job.job_id == "08_optional_schrodinger_bridge":
            lines.extend(self._render_step8_body(job=job))
        else:  # pragma: no cover - guarded by builder.
            raise CloudExecutionError(f"Unsupported phase-18 job: {job.job_id}")
        lines.append("")
        lines.append(f"echo {json.dumps(job.stop_compute_instruction)}")
        if job.executor == "oracle_vm":
            lines.append(job.stop_compute_instruction)
        return "\n".join(lines) + "\n"

    def _script_preamble(
        self,
        *,
        config: TCPEConfig,
        run_id: str,
        job: Phase18JobSpec,
        selected: Phase18ExecutionConfig,
    ) -> list[str]:
        return [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Phase 18 job {job.sequence_index}: {job.title}",
            f"# {job.summary}",
            f"export TCPE_RUN_ID=\"${{TCPE_RUN_ID:-{run_id}}}\"",
            "export WANDB_MODE=\"online\"",
            "export OCI_NAMESPACE="
            f"\"${{OCI_NAMESPACE:-{selected.handoff_config.oracle_namespace}}}\"",
            f"export OCI_BUCKET=\"${{OCI_BUCKET:-{selected.handoff_config.oracle_bucket}}}\"",
            f"export OCI_REGION=\"${{OCI_REGION:-{selected.handoff_config.oracle_region}}}\"",
            f"export OCI_PROFILE=\"${{OCI_PROFILE:-{selected.oci_profile}}}\"",
            "export OCI_CONFIG_FILE=\"${OCI_CONFIG_FILE:-}\"",
            f"cd {json.dumps(str(Path.cwd()))}",
            "",
            "echo \"Running ${TCPE_RUN_ID} :: "
            f"{job.job_id} ({job.executor}) with WANDB_MODE=${{WANDB_MODE}}\"",
            f"timeout {job.max_runtime_minutes}m python - <<'PY'",
        ]

    def _render_step1_body(
        self,
        *,
        job: Phase18JobSpec,
        selected: Phase18ExecutionConfig,
    ) -> list[str]:
        dataset_lines = [
            "targets = {",
        ]
        for item in selected.handoff_config.datasets:
            object_name = (
                job.artifact_prefix
                + "/"
                + item.dataset_key
                + "_raw_singlecell_01.h5ad"
            )
            dataset_lines.append(
                "    "
                + repr(item.dataset_key)
                + ": {"
                + f"'match': {repr(item.display_name.lower())}, "
                + f"'object_name': {repr(object_name)}"
                + "},"
            )
        dataset_lines.append("}")
        return [
            "import json",
            "import os",
            "from pathlib import Path",
            "from urllib.request import urlopen, urlretrieve",
            "",
            "from tcpe.cloud_execution import OCIObjectStorageClient",
            "",
            "run_id = os.environ['TCPE_RUN_ID']",
            "root = Path('artifacts') / 'phase18_runtime' / run_id / 'raw'",
            "root.mkdir(parents=True, exist_ok=True)",
            f"api_url = {selected.figshare_api_url!r}",
            "payload = json.load(urlopen(api_url))",
            "files = payload.get('files', [])",
            "if not isinstance(files, list):",
            "    raise RuntimeError('Figshare API payload is missing `files`.')",
            *dataset_lines,
            "client = OCIObjectStorageClient(",
            "    namespace=os.environ['OCI_NAMESPACE'],",
            "    bucket_name=os.environ['OCI_BUCKET'],",
            "    region=os.environ['OCI_REGION'],",
            "    config_file=os.environ['OCI_CONFIG_FILE'] or None,",
            "    profile=os.environ['OCI_PROFILE'],",
            ")",
            "manifest = {'run_id': run_id, 'downloads': []}",
            "for dataset_key, rule in targets.items():",
            "    match = rule['match']",
            "    selected_file = None",
            "    for item in files:",
            "        if not isinstance(item, dict):",
            "            continue",
            "        name = str(item.get('name', '')).lower()",
            "        if match in name and 'raw_singlecell' in name and name.endswith('.h5ad'):",
            "            selected_file = item",
            "            break",
            "    if selected_file is None:",
            "        candidates = [",
            "            str(item.get('name', ''))",
            "            for item in files",
            "            if isinstance(item, dict)",
            "        ]",
            "        raise RuntimeError(",
            "            'No raw_singlecell h5ad match found for '",
            "            f'{dataset_key}. Candidates: {candidates}'",
            "        )",
            "    local_path = root / str(selected_file['name'])",
            "    urlretrieve(str(selected_file['download_url']), local_path)",
            "    upload = client.upload_file(",
            "        local_path=local_path,",
            "        object_name=str(rule['object_name']),",
            "    )",
            "    manifest['downloads'].append(",
            "        {",
            "            'dataset_key': dataset_key,",
            "            'figshare_file_name': str(selected_file['name']),",
            "            'object_name': str(rule['object_name']),",
            "            'upload': upload,",
            "        }",
            "    )",
            "manifest_path = root / 'raw_stage_manifest.json'",
            "manifest_path.write_text(",
            "    json.dumps(manifest, indent=2, sort_keys=True),",
            "    encoding='utf-8',",
            ")",
            "print(json.dumps(manifest, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step2_body(
        self,
        *,
        job: Phase18JobSpec,
        selected: Phase18ExecutionConfig,
    ) -> list[str]:
        dataset_lines = [
            "datasets = [",
        ]
        for item in selected.handoff_config.datasets:
            raw_object = (
                job.artifact_prefix.replace("/processed", "/raw")
                + "/"
                + item.dataset_key
                + "_raw_singlecell_01.h5ad"
            )
            processed_object = job.artifact_prefix + "/" + item.processed_h5ad_name
            dataset_lines.append(
                "    {"
                + f"'dataset_key': {item.dataset_key!r}, "
                + f"'raw_object': {raw_object!r}, "
                + f"'processed_object': {processed_object!r}"
                + "},"
            )
        dataset_lines.append("]")
        return [
            "import json",
            "import os",
            "from pathlib import Path",
            "",
            "from tcpe.anndata_schema import validate_anndata_schema",
            "from tcpe.cloud_execution import OCIObjectStorageClient",
            "from tcpe.ingestion import IngestionModule",
            "from tcpe.preprocessing import PreprocessingConfig",
            "",
            "run_id = os.environ['TCPE_RUN_ID']",
            "runtime_root = Path('artifacts') / 'phase18_runtime' / run_id / 'preprocess_embed'",
            "raw_root = runtime_root / 'raw'",
            "processed_root = runtime_root / 'processed'",
            "raw_root.mkdir(parents=True, exist_ok=True)",
            "processed_root.mkdir(parents=True, exist_ok=True)",
            *dataset_lines,
            "client = OCIObjectStorageClient(",
            "    namespace=os.environ['OCI_NAMESPACE'],",
            "    bucket_name=os.environ['OCI_BUCKET'],",
            "    region=os.environ['OCI_REGION'],",
            "    config_file=os.environ['OCI_CONFIG_FILE'] or None,",
            "    profile=os.environ['OCI_PROFILE'],",
            ")",
            "module = IngestionModule()",
            "manifest = {'run_id': run_id, 'processed': []}",
            "for item in datasets:",
            "    local_raw = client.download_file(",
            "        object_name=str(item['raw_object']),",
            "        local_path=raw_root / Path(str(item['raw_object'])).name,",
            "    )",
            "    ad = __import__('anndata')",
            "    adata = ad.read_h5ad(local_raw)",
            "    adata.uns['dataset_name'] = str(item['dataset_key'])",
            "    validate_anndata_schema(adata=adata, mode='repair')",
            "    validate_anndata_schema(adata=adata, mode='strict')",
            "    processed = module.preprocess_adata(",
            "        adata,",
            "        config=PreprocessingConfig(",
            "            hvg_target=5000,",
            "            hvg_min_allowed=2000,",
            "            hvg_max_allowed=5000,",
            "            seed=42,",
            "        ),",
            "    )",
            "    module.annotate_sequence_embeddings(processed.adata)",
            "    module.annotate_cell_state_embeddings(processed.adata)",
            "    local_processed = processed_root / Path(str(item['processed_object'])).name",
            "    processed.adata.write_h5ad(local_processed)",
            "    upload = client.upload_file(",
            "        local_path=local_processed,",
            "        object_name=str(item['processed_object']),",
            "    )",
            "    manifest['processed'].append(",
            "        {",
            "            'dataset_key': str(item['dataset_key']),",
            "            'processed_object': str(item['processed_object']),",
            "            'n_obs': int(processed.adata.n_obs),",
            "            'n_vars': int(processed.adata.n_vars),",
            "            'upload': upload,",
            "        }",
            "    )",
            "manifest_path = runtime_root / 'preprocess_embed_manifest.json'",
            "manifest_path.write_text(",
            "    json.dumps(manifest, indent=2, sort_keys=True),",
            "    encoding='utf-8',",
            ")",
            "print(json.dumps(manifest, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step3_body(self, *, job: Phase18JobSpec) -> list[str]:
        return [
            "import json",
            "from pathlib import Path",
            "",
            "summary = {",
            "    'job_id': '03_kaggle_k562_essential_tuning',",
            "    'objective': 'iterative transport + causal hyperparameter tuning',",
            "    'dataset': 'k562_essential',",
            "    'transport_latent_dim_candidates': [64, 96, 128],",
            "    'transport_sinkhorn_weight_candidates': [0.05, 0.1],",
            "    'causal_max_hvgs_candidates': [512, 1024, 2048],",
            "    'notes': (",
            "        'Use OCI-processed h5ad input and persist the winning config back to OCI.'",
            "    ),",
            "}",
            "root = Path('artifacts') / 'phase18_runtime' / 'tuning'",
            "root.mkdir(parents=True, exist_ok=True)",
            "path = root / 'k562_essential_tuning_plan.json'",
            "path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')",
            "print(json.dumps(summary, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step4_body(self, *, job: Phase18JobSpec) -> list[str]:
        return [
            "import json",
            "from pathlib import Path",
            "",
            "payload = {",
            "    'job_id': '04_kaggle_k562_gwps_transport',",
            "    'dataset': 'replogle_k562_gwps',",
            "    'baseline_gate': 'Run and log baselines before OT training starts.',",
            "    'required_outputs': [",
            "        'baseline_suite_results.json',",
            "        'transport_model.ckpt',",
            "        'transport_fit_summary.json',",
            "    ],",
            "    'wandb_mode': 'online',",
            "}",
            "root = Path('artifacts') / 'phase18_runtime' / 'train_k562_gwps'",
            "root.mkdir(parents=True, exist_ok=True)",
            "path = root / 'k562_gwps_transport_plan.json'",
            "path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step5_body(self, *, job: Phase18JobSpec) -> list[str]:
        return [
            "import json",
            "from pathlib import Path",
            "",
            "payload = {",
            "    'job_id': '05_kaggle_rpe1_transport_eval',",
            "    'dataset': 'rpe1',",
            "    'evaluation_scope': (",
            "        'cross-cell-type evaluation requested after transport training'",
            "    ),",
            "    'required_outputs': [",
            "        'transport_model.ckpt',",
            "        'evaluation_report.json',",
            "        'transport_prediction.npz',",
            "    ],",
            "    'wandb_mode': 'online',",
            "}",
            "root = Path('artifacts') / 'phase18_runtime' / 'train_rpe1'",
            "root.mkdir(parents=True, exist_ok=True)",
            "path = root / 'rpe1_transport_eval_plan.json'",
            "path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step6_body(self, *, job: Phase18JobSpec) -> list[str]:
        return [
            "import json",
            "from pathlib import Path",
            "",
            "payload = {",
            "    'job_id': '06_oracle_causal_iv_genomewide',",
            "    'datasets': ['replogle_k562_gwps', 'rpe1'],",
            "    'required_proxy_columns': [",
            "        'batch',",
            "        'library_size',",
            "        'knockdown_efficiency_proxy',",
            "        'protocol',",
            "    ],",
            "    'required_outputs': [",
            "        'causal_graph_adjacency.npz',",
            "        'causal_graph_edges.csv',",
            "        'causal_graph_metadata.json',",
            "    ],",
            "}",
            "root = Path('artifacts') / 'phase18_runtime' / 'causal'",
            "root.mkdir(parents=True, exist_ok=True)",
            "path = root / 'causal_iv_plan.json'",
            "path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step7_body(self, *, job: Phase18JobSpec) -> list[str]:
        return [
            "import json",
            "from pathlib import Path",
            "",
            "payload = {",
            "    'job_id': '07_context_shift_and_model_card',",
            "    'executor': " + repr(job.executor) + ",",
            "    'shift_types': ['cell_type', 'locus', 'dose_strength', 'protocol'],",
            "    'model_card_required': True,",
            "    'required_outputs': [",
            "        'context_shift manifests',",
            "        'model_card.json',",
            "        'model_card.md',",
            "    ],",
            "}",
            "root = Path('artifacts') / 'phase18_runtime' / 'reports'",
            "root.mkdir(parents=True, exist_ok=True)",
            "path = root / 'context_shift_and_model_card_plan.json'",
            "path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_step8_body(self, *, job: Phase18JobSpec) -> list[str]:
        enabled_literal = "True" if job.enabled else "False"
        return [
            "import json",
            "from pathlib import Path",
            "",
            f"enabled = {enabled_literal}",
            "payload = {",
            "    'job_id': '08_optional_schrodinger_bridge',",
            "    'enabled': enabled,",
            "    'condition': (",
            "        'Only run if OT baselines are stable and the bridge scaffold is desired.'",
            "    ),",
            "    'notes': (",
            "        'The current bridge transport path remains a scaffold. '",
            "        'Keep this optional and '",
            "        'persist a skip decision if the scaffold is not promoted.'",
            "    ),",
            "}",
            "root = Path('artifacts') / 'phase18_runtime' / 'bridge'",
            "root.mkdir(parents=True, exist_ok=True)",
            "path = root / 'optional_bridge_plan.json'",
            "path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')",
            "print(json.dumps(payload, indent=2, sort_keys=True))",
            "PY",
        ]

    def _render_oracle_tmux_launcher(
        self,
        *,
        jobs: list[Phase18JobSpec],
        launchers_dir: Path,
    ) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Launch Oracle jobs in separate tmux sessions in the exact phase-18 order.",
        ]
        for job in jobs:
            if job.executor != "oracle_vm":
                continue
            if job.tmux_session_name is None:
                continue
            script_path = launchers_dir / job.script_filename
            lines.append(
                "tmux new-session -d -s "
                + job.tmux_session_name
                + " "
                + json.dumps(f"bash {script_path}")
            )
            lines.append(
                "echo "
                + json.dumps(
                    "Started "
                    f"{job.tmux_session_name}; attach with "
                    f"`tmux attach -t {job.tmux_session_name}`."
                )
            )
        lines.append("")
        return "\n".join(lines) + "\n"

    def _render_runbook(
        self,
        *,
        config: TCPEConfig,
        run_id: str,
        jobs: list[Phase18JobSpec],
        selected: Phase18ExecutionConfig,
        tmux_launcher_path: Path | None,
    ) -> str:
        lines: list[str] = []
        lines.append("# Phase 18 Cloud Execution Runbook")
        lines.append("")
        lines.append(f"- Run id: `{run_id}`")
        lines.append(f"- Oracle VM: `{selected.handoff_config.oracle_vm_shape}`")
        lines.append(
            f"- Oracle Object Storage: "
            f"`oci://{selected.handoff_config.oracle_namespace}/{selected.handoff_config.oracle_bucket}`"
        )
        lines.append(
            f"- Kaggle GPU: `{selected.handoff_config.kaggle_gpu_type} x"
            f"{selected.handoff_config.kaggle_gpu_count}`"
        )
        lines.append("- W&B mode for cloud jobs: `online`")
        lines.append(f"- Base artifact root from config: `{config.paths.artifact_root}`")
        lines.append("")
        lines.append("## Exact Order")
        lines.append("")
        for job in jobs:
            enabled_suffix = "" if job.enabled else " (disabled by default)"
            lines.append(
                f"{job.sequence_index}. `{job.job_id}` [{job.executor}]{enabled_suffix}: "
                f"{job.title}"
            )
        lines.append("")
        lines.append("## OCI Contract")
        lines.append("")
        lines.append(
            "- All bucket transfers in the generated launchers use `OCIObjectStorageClient`, "
            "which wraps the OCI Python SDK."
        )
        lines.append(
            f"- OCI profile default: `{selected.oci_profile}`. "
            "Override with `OCI_PROFILE` and `OCI_CONFIG_FILE` when needed."
        )
        lines.append("")
        lines.append("## Oracle tmux")
        lines.append("")
        if tmux_launcher_path is not None:
            lines.append(f"- Launch Oracle jobs with: `{tmux_launcher_path}`")
        else:
            lines.append("- Oracle tmux launcher is emitted only in non-dry-run mode.")
        lines.append("")
        lines.append("## Cost and Runtime Guards")
        lines.append("")
        for job in jobs:
            lines.append(
                f"- `{job.job_id}` timeout: `{job.max_runtime_minutes}` minutes."
            )
        lines.append(
            f"- Oracle shutdown action after Oracle jobs: "
            f"`{selected.oracle_shutdown_command}`"
        )
        lines.append(
            f"- Kaggle stop instruction after Kaggle jobs: "
            f"`{selected.kaggle_stop_instruction}`"
        )
        lines.append("")
        lines.append("## Resume Safety")
        lines.append("")
        lines.append(
            "- `phase18_sequence_checkpoint.json` is initialized with all pending jobs in order."
        )
        lines.append(
            "- Each launcher script is designed to write phase-specific manifests under "
            "`artifacts/phase18_runtime/<run_id>/...` before teardown."
        )
        lines.append(
            "- OCI remains the permanent storage layer between Oracle and Kaggle jobs."
        )
        lines.append("")
        return "\n".join(lines) + "\n"
