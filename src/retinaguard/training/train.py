"""Training orchestration runner matching blueprint with strict data checks and rich provenance."""

import datetime
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from retinaguard.utils.hashing import compute_sha256
from retinaguard.utils.reproducibility import get_device, seed_everything


def get_git_commit() -> str:
    """Retrieve current Git commit SHA-256."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_training_experiment(
    config_path: Union[str, Path] = "configs/train_multitask.yaml",
    smoke_test: bool = False,
    fixture_mode: bool = False,
    override_seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute training experiment defined by YAML config with strict data checks and rich provenance."""
    from retinaguard.training.config_schema import validate_training_config

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not fixture_mode:
        # Strict validation of configuration schema
        validate_training_config(cfg)

    seed = (
        override_seed if override_seed is not None else cfg.get("experiment", {}).get("seed", 2026)
    )
    seed_everything(seed)
    device = get_device()

    # 1. Dataset Selection Configuration (Item 2 & 3)
    data_cfg = cfg.get("data", {})
    datasets_cfg = data_cfg.get("datasets", {})

    eyeq_enabled = (
        datasets_cfg.get("eyeq", {}).get("enabled", True) if "eyeq" in datasets_cfg else True
    )
    deepdrid_enabled = (
        datasets_cfg.get("deepdrid", {}).get("enabled", True)
        if "deepdrid" in datasets_cfg
        else True
    )

    # Determine splits paths
    if fixture_mode:
        eyeq_train_p = Path("tests/fixtures/splits/eyeq_train.csv")
        eyeq_val_p = Path("tests/fixtures/splits/eyeq_val.csv")
        deepdrid_train_p = Path("tests/fixtures/splits/deepdrid_train.csv")
        deepdrid_val_p = Path("tests/fixtures/splits/deepdrid_val.csv")
        save_dir = Path(output_dir) if output_dir else Path("tests/fixtures/checkpoints")
        history_path = save_dir / "train_history.json"
    else:
        eyeq_train_p = Path(
            datasets_cfg.get("eyeq", {}).get(
                "train_split", data_cfg.get("train_eyeq_split", "data/splits/eyeq_train.csv")
            )
        )
        eyeq_val_p = Path(
            datasets_cfg.get("eyeq", {}).get(
                "val_split", data_cfg.get("val_eyeq_split", "data/splits/eyeq_val.csv")
            )
        )
        deepdrid_train_p = Path(
            datasets_cfg.get("deepdrid", {}).get(
                "train_split",
                data_cfg.get("train_deepdrid_split", "data/splits/deepdrid_train.csv"),
            )
        )
        deepdrid_val_p = Path(
            datasets_cfg.get("deepdrid", {}).get(
                "val_split", data_cfg.get("val_deepdrid_split", "data/splits/deepdrid_val.csv")
            )
        )
        save_dir = Path(output_dir) if output_dir else Path("artifacts/models")
        history_path = (
            save_dir / "train_history.json"
            if output_dir
            else Path("artifacts/metrics/train_history.json")
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    has_eyeq_train = eyeq_enabled and eyeq_train_p.is_file()
    has_dd_train = deepdrid_enabled and deepdrid_train_p.is_file()
    has_eyeq_val = eyeq_enabled and eyeq_val_p.is_file()
    has_dd_val = deepdrid_enabled and deepdrid_val_p.is_file()

    if not (has_eyeq_train or has_dd_train):
        raise FileNotFoundError(
            f"No enabled training split manifests found (EyeQ: {eyeq_train_p} [enabled={eyeq_enabled}], "
            f"DeepDRiD: {deepdrid_train_p} [enabled={deepdrid_enabled}]).\n"
            "Please check dataset configuration and run preparation scripts."
        )

    if not (has_eyeq_val or has_dd_val):
        raise FileNotFoundError(
            f"No enabled validation split manifests found (EyeQ: {eyeq_val_p} [enabled={eyeq_enabled}], "
            f"DeepDRiD: {deepdrid_val_p} [enabled={deepdrid_enabled}]).\n"
            "Please run scripts/create_splits.py to generate verified validation splits."
        )

    training_dataset_names: List[str] = []
    if has_eyeq_train:
        training_dataset_names.append("EyeQ")
    if has_dd_train:
        training_dataset_names.append("DeepDRiD")

    # Record dataset split hashes for provenance (Item 12)
    train_split_hashes = {}
    val_split_hashes = {}
    if has_eyeq_train:
        train_split_hashes["eyeq"] = compute_sha256(eyeq_train_p)
    if has_dd_train:
        train_split_hashes["deepdrid"] = compute_sha256(deepdrid_train_p)
    if has_eyeq_val:
        val_split_hashes["eyeq"] = compute_sha256(eyeq_val_p)
    if has_dd_val:
        val_split_hashes["deepdrid"] = compute_sha256(deepdrid_val_p)

    train_cfg = cfg.get("training", {})
    batch_size = 4 if smoke_test else train_cfg.get("batch_size", 32)
    epochs = 2 if smoke_test else train_cfg.get("epochs", 40)
    img_size = train_cfg.get("image_size", 384)
    num_workers = 0 if fixture_mode else train_cfg.get("num_workers", 2)
    use_amp = train_cfg.get("mixed_precision", False) and device.type == "cuda"

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
            train_datasets[0][1],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        n_eyeq = len(train_datasets[0][1])
        n_dd = len(train_datasets[1][1])
        weights_eyeq = [0.5 / max(1, n_eyeq)] * n_eyeq
        weights_dd = [0.5 / max(1, n_dd)] * n_dd
        combined_weights = torch.tensor(weights_eyeq + weights_dd, dtype=torch.float32)
        combined_train_ds: torch.utils.data.Dataset = ConcatDataset(
            [train_datasets[0][1], train_datasets[1][1]]
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=combined_weights.tolist(), num_samples=len(combined_weights), replacement=True
        )
        train_loader = DataLoader(
            combined_train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
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
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize Model
    model_cfg = cfg.get("model", {})
    backbone_name = model_cfg.get("backbone", "mobilenetv3_large_100")
    pretrained = model_cfg.get("pretrained", True) if not fixture_mode else False
    dropout = model_cfg.get("dropout", 0.2)
    latent_dim = model_cfg.get("latent_dim", 128)

    model = RetinaGuardMultiTaskModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
        dropout=dropout,
        latent_dim=latent_dim,
    ).to(device)

    # Loss Configuration (Item 7)
    loss_cfg = cfg.get("loss", {})
    heads_cfg = model_cfg.get("heads", {})
    w_q = heads_cfg.get("quality", {}).get("weight", 1.0)
    w_oq = heads_cfg.get("overall_quality", {}).get("weight", 0.8)
    w_art = heads_cfg.get("artifact", {}).get("weight", 0.5)
    w_cla = heads_cfg.get("clarity", {}).get("weight", 0.5)
    w_fld = heads_cfg.get("field_definition", {}).get("weight", 0.5)
    label_smoothing = loss_cfg.get("label_smoothing", 0.05)

    criterion = MaskedMultiTaskLoss(
        weight_quality=w_q,
        weight_overall_quality=w_oq,
        weight_artifact=w_art,
        weight_clarity=w_cla,
        weight_field_def=w_fld,
        label_smoothing=label_smoothing,
    )

    # Optimizer Selection (Item 7)
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    optimizer: torch.optim.Optimizer
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning Rate Scheduler (Item 7)
    sched_name = train_cfg.get("scheduler", "cosine").lower()
    if sched_name == "cosine":
        scheduler: Optional[Any] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    early_stopping = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 7))
    checkpoint_saver = ModelCheckpointSaver(save_dir=save_dir, filename="best.ckpt")
    history_logger = MetricHistoryLogger(output_path=history_path)

    best_val_score = 0.0
    git_commit = get_git_commit()

    print(
        f"=== Starting Training [{cfg.get('experiment', {}).get('name', 'experiment')}] "
        f"Seed: {seed} | Backbone: {backbone_name} | Datasets: {training_dataset_names} ==="
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_metrics, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        val_score = (
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
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Obj Score: {val_score:.4f} | {metric_summary}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}"
            )

        is_best = early_stopping(val_score)
        if is_best:
            best_val_score = float(val_score or 0.0)

            # Rich Checkpoint Provenance (Item 12)
            checkpoint_metadata = {
                "architecture": "RetinaGuardMultiTaskModel",
                "backbone": backbone_name,
                "head_dimensions": {
                    "quality_head": 3,
                    "overall_quality_head": 2,
                    "artifact_head": 3,
                    "clarity_head": 3,
                    "field_def_head": 3,
                    "projection_head": latent_dim,
                },
                "class_mappings": {
                    "quality": ["good", "usable", "reject"],
                    "overall_quality": ["good", "poor_reject"],
                    "artifact": [0, 1, 2],
                    "clarity": [0, 1, 2],
                    "field_definition": [0, 1, 2],
                },
                "training_datasets": training_dataset_names,
                "seed": seed,
                "epoch": epoch,
                "selection_metric": train_cfg.get("selection_metric", "primary_macro_f1"),
                "best_val_metric": val_score,
                "val_metrics": (
                    val_metrics if isinstance(val_metrics, dict) else {"val_score": val_score}
                ),
                "git_commit": git_commit,
                "resolved_config": cfg,
                "train_split_hashes": train_split_hashes,
                "val_split_hashes": val_split_hashes,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "environment": {
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "device": str(device),
                },
                "saved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            checkpoint_saver.save(
                model.state_dict(),
                checkpoint_metadata,
            )

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return {
        "best_val_macro_f1": best_val_score,
        "epochs_trained": epoch,
        "checkpoint_path": str(checkpoint_saver.filepath),
        "training_datasets": training_dataset_names,
    }
