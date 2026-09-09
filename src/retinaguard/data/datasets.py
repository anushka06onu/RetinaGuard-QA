"""PyTorch Dataset loader supporting masked multi-task annotations."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import get_train_transforms, get_val_transforms


class RetinalQualityDataset(Dataset):
    """Dataset loader handling single-task (EyeQ) and multi-task (EyeQ + DeepDRiD) annotations.

    Label mapping:
    - quality: good -> 0, usable -> 1, reject -> 2
    - artifact: 0, 1, 2
    - clarity: 0, 1, 2
    - field_definition: 0, 1, 2
    """

    QUALITY_MAP = {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}

    def __init__(
        self,
        manifest_or_csv: Union[pd.DataFrame, str, Path],
        transform: Optional[Callable] = None,
        is_training: bool = False,
        image_size: int = 384,
        allow_synthetic_fallback: bool = False,
    ):
        if isinstance(manifest_or_csv, pd.DataFrame):
            self.df = manifest_or_csv.reset_index(drop=True)
        else:
            self.df = pd.read_csv(manifest_or_csv)

        self.allow_synthetic_fallback = allow_synthetic_fallback

        if transform is not None:
            self.transform = transform
        else:
            self.transform = (
                get_train_transforms(image_size) if is_training else get_val_transforms(image_size)
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        raw_path = row.get("path", "")
        img_path = Path(raw_path) if raw_path else None

        if img_path is not None and img_path.is_file():
            try:
                with Image.open(img_path) as image:
                    img = image.convert("RGB")
            except Exception as exc:
                if self.allow_synthetic_fallback:
                    img = Image.new("RGB", (384, 384), color=(180, 80, 30))
                else:
                    raise RuntimeError(f"Unreadable image file: {img_path}") from exc
        else:
            if self.allow_synthetic_fallback:
                img = Image.new("RGB", (384, 384), color=(180, 80, 30))
            else:
                raise FileNotFoundError(f"Missing image file: {img_path}")

        tensor = self.transform(img)

        # 1. Primary Quality Label & Mask
        q_label_raw = row.get("quality_canonical", row.get("quality_raw", None))
        if pd.notna(q_label_raw) and q_label_raw in self.QUALITY_MAP:
            quality_target = torch.tensor(self.QUALITY_MAP[q_label_raw], dtype=torch.long)
            quality_mask = torch.tensor(1.0, dtype=torch.float32)
        else:
            quality_target = torch.tensor(0, dtype=torch.long)
            quality_mask = torch.tensor(0.0, dtype=torch.float32)

        # 2. DeepDRiD Attribute Labels & Masks
        def parse_attr(val):
            if pd.notna(val):
                try:
                    return torch.tensor(int(val), dtype=torch.long), torch.tensor(
                        1.0, dtype=torch.float32
                    )
                except Exception:
                    pass
            return torch.tensor(0, dtype=torch.long), torch.tensor(0.0, dtype=torch.float32)

        art_target, art_mask = parse_attr(row.get("artifact", None))
        cla_target, cla_mask = parse_attr(row.get("clarity", None))
        fld_target, fld_mask = parse_attr(row.get("field_definition", None))

        return {
            "image": tensor,
            "image_id": str(row.get("image_id", idx)),
            "patient_id": str(row.get("patient_id", "")),
            "quality_target": quality_target,
            "quality_mask": quality_mask,
            "artifact_target": art_target,
            "artifact_mask": art_mask,
            "clarity_target": cla_target,
            "clarity_mask": cla_mask,
            "field_definition_target": fld_target,
            "field_definition_mask": fld_mask,
        }
