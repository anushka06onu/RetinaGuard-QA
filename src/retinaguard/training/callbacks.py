"""Training callbacks: Early Stopping, Checkpoint Saver, Metric History Logger."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


class EarlyStopping:
    """Early stopping monitor based on validation metric."""
    def __init__(self, patience: int = 7, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        score = val_score if self.mode == "max" else -val_score

        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


class ModelCheckpointSaver:
    """Saves best model weights and history to artifacts/models/."""
    def __init__(self, save_dir: Union[str, Path] = "artifacts/models", filename: str = "best.ckpt"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.save_dir / filename

    def save(self, state_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        payload = {"state_dict": state_dict, "metadata": metadata or {}}
        torch.save(payload, self.filepath)


class MetricHistoryLogger:
    """Logs training & validation loss and metric histories to CSV / JSON."""
    def __init__(self, output_path: Union[str, Path] = "artifacts/metrics/train_history.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}

    def log(self, epoch: int, train_loss: float, val_loss: float, val_macro_f1: float):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(round(train_loss, 4))
        self.history["val_loss"].append(round(val_loss, 4))
        self.history["val_macro_f1"].append(round(val_macro_f1, 4))
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
