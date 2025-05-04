# sweep_utils.py
from pathlib import Path
import yaml
import wandb
from typing import Dict, Any

# ------------------------------------------------------------------ #
# Random-search sweep definition (edit ranges/values as you like)
# ------------------------------------------------------------------ #
_RANDOM_SWEEP = """
program: train.py
method: random
metric:
  name: epoch_val_loss          # <-- must match the key your trainer logs
  goal: minimize
parameters:
  # ⇩ continuous ranges (log-uniform is perfect for learning-rate)
  lr:
    distribution: uniform
    min: 5e-7
    max: 1e-3

  cutoff:
    distribution: uniform
    min: 3.0
    max: 7.0

  # ⇩ discrete sets
  batch_size:
    values: [32, 64, 128]

  num_block:
    values: [2, 3, 4, 5, 6]

  num_features:
    values: [64, 128, 256]
  
  epochs:
    values: [40, 50, 60, 70, 80, 90] 
"""

def launch_random_sweep(cfg_file: Path,
                        sweep_yaml: str | None = None,
                        project: str = "deta-random",
                        trials: int = 10) -> None:
    """
    Create a W&B random-search sweep and run <trials> local agents.

    Parameters
    ----------
    cfg_file : Path
        Path to your base YAML/JSON config (e.g. configs/imag_default.yaml)
    sweep_yaml : str | None
        Custom YAML string that overrides _RANDOM_SWEEP.
    project : str
        W&B project name
    trials : int
        Number of individual runs to execute locally.
    """
    sweep_def: Dict[str, Any] = yaml.safe_load(sweep_yaml or _RANDOM_SWEEP)
    # make sure every run receives the --config path
    sweep_def.setdefault("command",
                         ["python", "spectra_pol_yaml.py", "--config", str(cfg_file)])
    sweep_id = wandb.sweep(sweep_def, project=project)
    print(f"Created sweep {sweep_id}")

    # kick off <trials> runs on this machine (blocking call)
    wandb.agent(sweep_id, project=project, count=trials)

    sweep_def.setdefault(
        "command",
        ["python", "code/spectra_pol_yaml.py", "--config", str(cfg_file)]
    )