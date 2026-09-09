"""Training orchestration runner matching Phase 9 of blueprint with fail-fast data checks."""

from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from src.retinaguard.data.datasets import RetinalQualityDataset
from src.retinaguard.models.losses import MaskedMultiTaskLoss
from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel
from src.retinaguard.training.callbacks import (
    EarlyStopping,
    MetricHistoryLogger,
    ModelCheckpointSaver,
)
from src.retinaguard.training.engine import evaluate_epoch, train_one_epoch
from src.retinaguard.utils.reproducibility import get_device, seed_everything


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
        train_split_p = Path("tests/fixtures/splits/eyeq_train.csv")
        val_split_p = Path("tests/fixtures/splits/eyeq_val.csv")
        save_dir = Path("tests/fixtures/checkpoints")
        history_path = Path("tests/fixtures/train_history.json")
    else:
        train_split_p = Path(cfg.get("data", {}).get("train_split", "data/splits/eyeq_train.csv"))
        val_split_p = Path(cfg.get("data", {}).get("val_split", "data/splits/eyeq_val.csv"))
        save_dir = Path("artifacts/models")
        history_path = Path("artifacts/metrics/train_history.json")

    if not train_split_p.is_file():
        raise FileNotFoundError(
            f"Training split manifest not found at: {train_split_p}.\n"
            "Please run scripts/prepare_eyeq.py and scripts/create_splits.py after obtaining datasets."
        )

    if not val_split_p.is_file():
        raise FileNotFoundError(
            f"Validation split manifest not found at: {val_split_p}.\n"
            "Please run scripts/create_splits.py to generate verified validation splits."
        )

    train_df = pd.read_csv(train_split_p)
    val_df = pd.read_csv(val_split_p)

    batch_size = 4 if smoke_test else cfg.get("training", {}).get("batch_size", 32)
    epochs = 2 if smoke_test else cfg.get("training", {}).get("epochs", 40)
    img_size = cfg.get("training", {}).get("image_size", 384)

    # Multi-task training dataset loading
    train_ds = RetinalQualityDataset(
        train_df, is_training=True, image_size=img_size, allow_synthetic_fallback=fixture_mode
    )
    val_ds = RetinalQualityDataset(
        val_df, is_training=False, image_size=img_size, allow_synthetic_fallback=fixture_mode
    )

    # If deepdrid development splits exist, concatenate for multi-task supervision
    deepdrid_train_p = (
        Path(cfg.get("data", {}).get("train_deepdrid_split", "data/splits/deepdrid_train.csv"))
        if not fixture_mode
        else Path("tests/fixtures/splits/deepdrid_train.csv")
    )
    if deepdrid_train_p.is_file():
        df_dd = pd.read_csv(deepdrid_train_p)
        train_ds_dd = RetinalQualityDataset(
            df_dd, is_training=True, image_size=img_size, allow_synthetic_fallback=fixture_mode
        )
        train_ds = ConcatDataset([train_ds, train_ds_dd])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
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
        val_loss, val_f1, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        history_logger.log(epoch, train_loss, val_loss, val_f1)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        is_best = early_stopping(val_f1)
        if is_best:
            best_val_f1 = val_f1
            checkpoint_saver.save(model.state_dict(), {"val_macro_f1": val_f1, "epoch": epoch})

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return {
        "best_val_macro_f1": best_val_f1,
        "epochs_trained": epoch,
        "checkpoint_path": str(checkpoint_saver.filepath),
    }
