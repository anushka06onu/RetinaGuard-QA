"""Training and validation epoch execution loops."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any] = None,
) -> float:
    """Run one training epoch over dataloader."""
    model.train()
    total_loss = 0.0
    n_batches = max(1, len(dataloader))

    for batch in dataloader:
        optimizer.zero_grad()
        images = batch["image"].to(device)

        # Move targets to device
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        out = model(images)
        if isinstance(out, dict):
            loss_dict = criterion(out, batch_dev)
            loss = loss_dict["loss_total"]
        else:
            loss = criterion(out, batch_dev["quality_target"])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_batches


def evaluate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run validation evaluation returning average loss, Macro-F1, all logits, and labels."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    n_batches = max(1, len(dataloader))

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            out = model(images)
            if isinstance(out, dict):
                loss_dict = criterion(out, batch_dev)
                loss = loss_dict["loss_total"]
                logits = out["quality_logits"].cpu().numpy()
            else:
                loss = criterion(out, batch_dev["quality_target"])
                logits = out.cpu().numpy()

            total_loss += loss.item()
            targets = batch["quality_target"].numpy()
            all_logits.append(logits)
            all_labels.append(targets)

    all_logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 3))
    all_labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))

    preds = np.argmax(all_logits_arr, axis=-1) if len(all_logits_arr) > 0 else np.zeros((0,))
    macro_f1 = float(f1_score(all_labels_arr, preds, average="macro", zero_division=0))

    return total_loss / n_batches, macro_f1, all_logits_arr, all_labels_arr
