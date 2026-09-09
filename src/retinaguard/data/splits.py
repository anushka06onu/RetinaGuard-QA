"""Deterministic group-aware patient splitting per Phase 4 of blueprint."""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def create_patient_grouped_splits(
    manifest_df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 2026,
    patient_col: str = "patient_id",
    stratify_col: str = "quality_canonical"
) -> Dict[str, pd.DataFrame]:
    """Partition dataset into Train, Val, and Test ensuring zero patient leakage."""
    df = manifest_df.copy()
    
    # Fill missing patient IDs with individual image IDs if unprovable
    df[patient_col] = df[patient_col].fillna(df["image_id"])

    # First split off Test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df[patient_col]))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Second split Train vs Val from remaining
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_ratio, random_state=seed)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val[patient_col]))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    return {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }


def save_split_manifests(
    splits: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path] = "data/splits",
    prefix: str = "eyeq"
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
