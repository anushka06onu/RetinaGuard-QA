"""Training orchestration runner matching Phase 9 of blueprint with fail-fast data checks."""

from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from retinaguard.data.datasets import RetinalQualityDataset
from retinaguard.models.losses import MaskedMultiTaskLoss
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.training.callbacks import (
    EarlyStopping,
    MetricHistoryLogger,
    ModelCheckpointSaver,
)
from retinaguard.training.engine import evaluate_epoch, train_one_epoch
from retinaguard.utils.reproducibility import get_device, seed_everything


def run_training_experiment(
    config_path: Union[str, Path] = "configs/train_multitask.yaml",
    smoke_test: bool = False,
    fixture_mode: bool = False,
) -> Dict[str, Any]:
    """Execute training experiment defined by YAML config with strict data checks."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("experiment", {}).get("seed", 2026)
    seed_everything(seed)
    device = get_device()

    # Determine splits paths
    if fixture_mode:
        eyeq_train_p = Path("tests/fixtures/splits/eyeq_train.csv")
        eyeq_val_p = Path("tests/fixtures/splits/eyeq_val.csv")
        deepdrid_train_p = Path("tests/fixtures/splits/deepdrid_train.csv")
        deepdrid_val_p = Path("tests/fixtures/splits/deepdrid_val.csv")
        save_dir = Path("tests/fixtures/checkpoints")
        history_path = Path("tests/fixtures/train_history.json")
    else:
        eyeq_train_p = Path(
            cfg.get("data", {}).get("train_eyeq_split", "data/splits/eyeq_train.csv")
        )
        eyeq_val_p = Path(cfg.get("data", {}).get("val_eyeq_split", "data/splits/eyeq_val.csv"))
        deepdrid_train_p = Path(
            cfg.get("data", {}).get("train_deepdrid_split", "data/splits/deepdrid_train.csv")
        )
        deepdrid_val_p = Path(
            cfg.get("data", {}).get("val_deepdrid_split", "data/splits/deepdrid_val.csv")
        )
        save_dir = Path("artifacts/models")
        history_path = Path("artifacts/metrics/train_history.json")

    has_eyeq_train = eyeq_train_p.is_file()
    has_dd_train = deepdrid_train_p.is_file()
    has_eyeq_val = eyeq_val_p.is_file()
    has_dd_val = deepdrid_val_p.is_file()

    if not (has_eyeq_train or has_dd_train):
        raise FileNotFoundError(
            f"No training split manifests found at {eyeq_train_p} or {deepdrid_train_p}.\n"
            "Please run data preparation and split creation scripts."
        )

    if not (has_eyeq_val or has_dd_val):
        raise FileNotFoundError(
            f"No validation split manifests found at {eyeq_val_p} or {deepdrid_val_p}.\n"
            "Please run scripts/create_splits.py to generate verified validation splits."
        )

    batch_size = 4 if smoke_test else cfg.get("training", {}).get("batch_size", 32)
    epochs = 2 if smoke_test else cfg.get("training", {}).get("epochs", 40)
    img_size = cfg.get("training", {}).get("image_size", 384)

    # Build Training Dataset & Loader
    train_datasets = []
    if has_eyeq_train:
        train_datasets.append(
            (
                "eyeq",
                RetinalQualityDataset(
                    pd.read_csv(eyeq_train_p),
                    is_training=True,
                    image_size=img_size,
                    allow_synthetic_fallback=fixture_mode,
                ),
            )
        )
    if has_dd_train:
        train_datasets.append(
            (
                "deepdrid",
                RetinalQualityDataset(
                    pd.read_csv(deepdrid_train_p),
                    is_training=True,
                    image_size=img_size,
                    allow_synthetic_fallback=fixture_mode,
                ),
            )
        )

    if len(train_datasets) == 1:
        train_loader = DataLoader(
            train_datasets[0][1], batch_size=batch_size, shuffle=True, drop_last=False
        )
    else:
        n_eyeq = len(train_datasets[0][1])
        n_dd = len(train_datasets[1][1])
        weights_eyeq = [0.5 / max(1, n_eyeq)] * n_eyeq
        weights_dd = [0.5 / max(1, n_dd)] * n_dd
        combined_weights = torch.tensor(weights_eyeq + weights_dd, dtype=torch.float32)
        combined_train_ds = ConcatDataset([train_datasets[0][1], train_datasets[1][1]])
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=combined_weights, num_samples=len(combined_weights), replacement=True
        )
        train_loader = DataLoader(
            combined_train_ds, batch_size=batch_size, sampler=sampler, drop_last=False
        )

    # Build Validation Dataset & Loader
    val_datasets = []
    if has_eyeq_val:
        val_datasets.append(
            RetinalQualityDataset(
                pd.read_csv(eyeq_val_p),
                is_training=False,
                image_size=img_size,
                allow_synthetic_fallback=fixture_mode,
            )
        )
    if has_dd_val:
        val_datasets.append(
            RetinalQualityDataset(
                pd.read_csv(deepdrid_val_p),
                is_training=False,
                image_size=img_size,
                allow_synthetic_fallback=fixture_mode,
            )
        )

    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize Multi-Task Model & Loss
    pretrained = cfg.get("model", {}).get("pretrained", True) if not fixture_mode else False
    model = RetinaGuardMultiTaskModel(
        backbone_name=cfg.get("model", {}).get("backbone", "mobilenetv3_large_100"),
        pretrained=pretrained,
    ).to(device)

    criterion = MaskedMultiTaskLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("training", {}).get("learning_rate", 3e-4),
        weight_decay=cfg.get("training", {}).get("weight_decay", 1e-4),
    )

    early_stopping = EarlyStopping(
        patience=cfg.get("training", {}).get("early_stopping_patience", 7)
    )
    checkpoint_saver = ModelCheckpointSaver(save_dir=save_dir, filename="best.ckpt")
    history_logger = MetricHistoryLogger(output_path=history_path)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        val_f1 = (
            val_metrics.get("primary_macro_f1", val_metrics.get("quality_macro_f1", 0.0))
            if isinstance(val_metrics, dict)
            else float(val_metrics)
        )

        history_logger.log(epoch, train_loss, val_loss, val_metrics)

        if isinstance(val_metrics, dict):
            metric_summary = " | ".join(
                f"{k}: {v:.4f}" for k, v in val_metrics.items() if k != "primary_macro_f1"
            )
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {metric_summary}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
            )

        is_best = early_stopping(val_f1)
        if is_best:
            best_val_f1 = val_f1
            checkpoint_saver.save(
                model.state_dict(),
                {
                    "val_macro_f1": val_f1,
                    "val_metrics": (
                        val_metrics if isinstance(val_metrics, dict) else {"val_macro_f1": val_f1}
                    ),
                    "epoch": epoch,
                },
            )

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return {
        "best_val_macro_f1": best_val_f1,
        "epochs_trained": epoch,
        "checkpoint_path": str(checkpoint_saver.filepath),
    }
