"""Training and validation epoch execution loops."""

from typing import Any, Dict, Optional, Tuple

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
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Run validation evaluation returning average loss, metrics dictionary, all logits, and labels."""
    model.eval()
    total_loss = 0.0
    n_batches = max(1, len(dataloader))

    # Collectors for multi-head metrics
    collectors = {
        "quality": {"logits": [], "targets": [], "masks": []},
        "overall_quality": {"logits": [], "targets": [], "masks": []},
        "artifact": {"logits": [], "targets": [], "masks": []},
        "clarity": {"logits": [], "targets": [], "masks": []},
        "field_definition": {"logits": [], "targets": [], "masks": []},
    }

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

                for head_name in [
                    "quality",
                    "overall_quality",
                    "artifact",
                    "clarity",
                    "field_definition",
                ]:
                    logits_key = f"{head_name}_logits"
                    target_key = f"{head_name}_target"
                    mask_key = f"{head_name}_mask"
                    if logits_key in out and target_key in batch:
                        collectors[head_name]["logits"].append(out[logits_key].cpu().numpy())
                        collectors[head_name]["targets"].append(batch[target_key].numpy())
                        mask_val = (
                            batch[mask_key].numpy()
                            if mask_key in batch
                            else np.ones_like(batch[target_key].numpy())
                        )
                        collectors[head_name]["masks"].append(mask_val)
            else:
                loss = criterion(out, batch_dev["quality_target"])
                collectors["quality"]["logits"].append(out.cpu().numpy())
                collectors["quality"]["targets"].append(batch["quality_target"].numpy())
                collectors["quality"]["masks"].append(np.ones_like(batch["quality_target"].numpy()))

            total_loss += loss.item()

    # Calculate metrics per head
    metrics = {}
    for head_name, data in collectors.items():
        if data["logits"] and data["targets"]:
            logits_arr = np.concatenate(data["logits"], axis=0)
            targets_arr = np.concatenate(data["targets"], axis=0)
            masks_arr = (
                np.concatenate(data["masks"], axis=0)
                if data["masks"]
                else np.ones_like(targets_arr)
            )
            valid_idx = masks_arr > 0.5
            if np.any(valid_idx):
                valid_preds = np.argmax(logits_arr[valid_idx], axis=-1)
                valid_targets = targets_arr[valid_idx]
                f1 = float(f1_score(valid_targets, valid_preds, average="macro", zero_division=0))
                metrics[f"{head_name}_macro_f1"] = f1
            else:
                metrics[f"{head_name}_macro_f1"] = 0.0

    # Determine primary metric based strictly on presence of valid masked samples
    has_eyeq = (
        bool(np.any(np.concatenate(collectors["quality"]["masks"], axis=0) > 0.5))
        if collectors["quality"]["masks"]
        else False
    )
    has_deepdrid = (
        bool(np.any(np.concatenate(collectors["overall_quality"]["masks"], axis=0) > 0.5))
        if collectors["overall_quality"]["masks"]
        else False
    )

    if has_eyeq:
        primary_f1 = metrics.get("quality_macro_f1", 0.0)
    elif has_deepdrid:
        primary_f1 = metrics.get("overall_quality_macro_f1", 0.0)
    else:
        primary_f1 = 0.0

    metrics["primary_macro_f1"] = primary_f1

    all_logits_arr = (
        np.concatenate(collectors["quality"]["logits"], axis=0)
        if collectors["quality"]["logits"]
        else np.zeros((0, 3))
    )
    all_labels_arr = (
        np.concatenate(collectors["quality"]["targets"], axis=0)
        if collectors["quality"]["targets"]
        else np.zeros((0,))
    )

    return total_loss / n_batches, metrics, all_logits_arr, all_labels_arr
