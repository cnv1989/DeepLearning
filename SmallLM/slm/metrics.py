"""Recording training metrics.

Every logged step is appended as one JSON line to <out>/metrics.jsonl, which
is easy to load later:

    import pandas as pd
    df = pd.read_json("out/metrics.jsonl", lines=True)
    df.plot(x="step", y=["train_loss", "val_loss"])

Pass --wandb to also send metrics to Weights & Biases (pip install wandb).
"""

import json
import os


class MetricsLogger:
    def __init__(self, out_dir, run_config=None, use_wandb=False, project="smalllm", enabled=True):
        self.enabled = enabled  # only the main process logs in multi-GPU runs
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self.wandb = None
        if enabled and use_wandb:
            import wandb

            wandb.init(project=project, config=run_config)
            self.wandb = wandb

    def log(self, step, **metrics):
        if not self.enabled:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()
