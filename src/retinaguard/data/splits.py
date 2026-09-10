"""Deterministic stratified group-aware patient splitting per Phase 4 of blueprint."""

from pathlib import Path
from typing import Dict, Union

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_grouped_train_val_split(
    manifest_df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 2026,
    patient_col: str = "patient_id",
    stratify_col: str = "quality_canonical",
) -> Dict[str, pd.DataFrame]:
    """Partition dataset strictly into Train and Val without dropping data or creating an unused test subset."""
    df = manifest_df.copy().reset_index(drop=True)
    df[patient_col] = df[patient_col].fillna(df["image_id"]).astype(str)
    strat_target = df[stratify_col].fillna("unlabeled").astype(str)

    n_splits = max(2, int(round(1.0 / max(val_ratio, 0.05))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_idx, val_idx = next(sgkf.split(df, y=strat_target, groups=df[patient_col]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    assert (
        len(df_train) + len(df_val) == len(df)
    ), f"Split size mismatch: {len(df_train)} + {len(df_val)} != {len(df)}"
    return {"train": df_train, "val": df_val}


def create_patient_grouped_splits(
    manifest_df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 2026,
    patient_col: str = "patient_id",
    stratify_col: str = "quality_canonical",
) -> Dict[str, pd.DataFrame]:
    """Partition dataset into Train, Val, and Test ensuring zero patient leakage and stratified quality balance."""
    df = manifest_df.copy().reset_index(drop=True)

    # Fill missing patient IDs with individual image IDs if unprovable
    df[patient_col] = df[patient_col].fillna(df["image_id"]).astype(str)

    # Stratification target (fill missing with 'unlabeled')
    strat_target = df[stratify_col].fillna("unlabeled").astype(str)

    # Use StratifiedGroupKFold with k=5 folds (~20% test, 20% val, 60% train) or k=10
    n_splits = max(3, int(round(1.0 / max(val_ratio, test_ratio, 0.1))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits_generator = sgkf.split(df, y=strat_target, groups=df[patient_col])
    train_val_idx, test_idx = next(splits_generator)

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Sub-split train_val into train and val using dedicated train/val function
    adj_val_ratio = val_ratio / max(1.0 - test_ratio, 1e-4)
    train_val_splits = create_grouped_train_val_split(
        df_train_val,
        val_ratio=adj_val_ratio,
        seed=seed,
        patient_col=patient_col,
        stratify_col=stratify_col,
    )

    return {
        "train": train_val_splits["train"],
        "val": train_val_splits["val"],
        "test": df_test,
    }


def save_split_manifests(
    splits: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path] = "data/splits",
    prefix: str = "eyeq",
) -> Dict[str, Path]:
    """Save split CSV files to data/splits/."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for split_name, split_df in splits.items():
        file_path = out_dir / f"{prefix}_{split_name}.csv"
        split_df.to_csv(file_path, index=False)
        saved_paths[split_name] = file_path

    return saved_paths
