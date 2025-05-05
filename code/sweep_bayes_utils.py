# sweep_utils.py
from pathlib import Path
import yaml
import wandb
from typing import Dict, Any

# ------------------------------------------------------------------ #
# Bayesian sweep definition
# ------------------------------------------------------------------ #
_BAYESIAN_SWEEP = """
program: train.py
method: bayes
metric:
  name: epoch_val_loss          # <-- must match the key your trainer logs
  goal: minimize
parameters:
  lr:
    distribution: log_uniform_values
    min: 5e-7
    max: 1e-3

  cutoff:
    values: [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

  batch_size:
    values: [32, 64, 128]

  num_block:
    values: [2, 3, 4, 5, 6]

  attention_head:
    values: [8, 16, 32, 64]

  num_features:
    values: [64, 128, 256]

  epochs:
    values: [70, 80, 90, 100, 110, 120]

early_terminate:
  type: hyperband
  min_iter: 20
"""

def launch_bayesian_sweep(cfg_file: Path,
                          sweep_yaml: str | None = None,
                          project: str = "deta-bayes",
                          trials: int = 25) -> None:
    """
    Create a W&B Bayesian sweep and run <trials> local agents.

    Parameters
    ----------
    cfg_file : Path
        Path to your base YAML/JSON config (e.g. configs/imag_default.yaml)
    sweep_yaml : str | None
        Custom YAML string that overrides _BAYESIAN_SWEEP.
    project : str
        W&B project name
    trials : int
        Number of individual runs to execute locally.
    """
    sweep_def: Dict[str, Any] = yaml.safe_load(sweep_yaml or _BAYESIAN_SWEEP)
    sweep_def.setdefault("command",
                         ["python", "spectra_pol_yaml.py", "--config", str(cfg_file)])

    sweep_id = wandb.sweep(sweep_def, project=project)
    print(f"Created Bayesian sweep {sweep_id}")

    # Run the agents
    wandb.agent(sweep_id, project=project, count=trials)
