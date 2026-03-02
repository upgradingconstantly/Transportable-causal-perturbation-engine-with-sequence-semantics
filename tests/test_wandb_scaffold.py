"""W&B scaffold tests."""

from pathlib import Path

from tcpe.runtime.wandb_scaffold import get_wandb_mode, init_wandb_run


def test_wandb_defaults_to_offline(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert get_wandb_mode() == "offline"

    run = init_wandb_run(project="tcpe-test", run_name="phase1", run_dir=tmp_path)
    run.log({"loss": 0.123})
    run.finish()

    assert tmp_path.exists()
