"""Canonical preprocessing and FOV detection matching Phase 5 of blueprint."""

import json
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


def crop_retinal_fov(
    image: Union[Image.Image, np.ndarray],
    threshold: int = 15,
    margin_ratio: float = 0.02
) -> Image.Image:
    """Detect circular retinal fundus field boundary and crop empty black outer borders.

    Retains the whole retinal field, pads to square instead of stretching.
    """
    if isinstance(image, Image.Image):
        np_img = np.array(image.convert("RGB"))
    else:
        np_img = np.array(image)

    # Compute luminance
    gray = 0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    mask = gray > threshold

    if not np.any(mask):
        return Image.fromarray(np_img)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w, _ = np_img.shape
    margin_h = int((rmax - rmin) * margin_ratio)
    margin_w = int((cmax - cmin) * margin_ratio)

    rmin = max(0, rmin - margin_h)
    rmax = min(h, rmax + margin_h)
    cmin = max(0, cmin - margin_w)
    cmax = min(w, cmax + margin_w)

    cropped = np_img[rmin:rmax, cmin:cmax]

    # Pad to square instead of stretching
    ch, cw, _ = cropped.shape
    if ch != cw:
        max_dim = max(ch, cw)
        padded = np.zeros((max_dim, max_dim, 3), dtype=cropped.dtype)
        y_off = (max_dim - ch) // 2
        x_off = (max_dim - cw) // 2
        padded[y_off:y_off + ch, x_off:x_off + cw] = cropped
        return Image.fromarray(padded)

    return Image.fromarray(cropped)


class CanonicalRetinalTransform:
    """Callable canonical FOV crop transform."""
    def __init__(self, threshold: int = 15, margin_ratio: float = 0.02):
        self.threshold = threshold
        self.margin_ratio = margin_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        return crop_retinal_fov(img, self.threshold, self.margin_ratio)


def get_train_transforms(image_size: int = 384) -> T.Compose:
    """Training-only transforms with quality-aware augmentations."""
    return T.Compose([
        CanonicalRetinalTransform(),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.04),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size: int = 384) -> T.Compose:
    """Deterministic validation, test, and production deployment transform."""
    return T.Compose([
        CanonicalRetinalTransform(),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image_canonical(
    image: Union[Image.Image, np.ndarray],
    image_size: int = 384
) -> torch.Tensor:
    """End-to-end preprocessing into a normalized tensor [1, 3, H, W]."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    transform = get_val_transforms(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def export_preprocessing_metadata(
    output_path: Union[str, Path] = "artifacts/models/preprocessing.json",
    image_size: int = 384
) -> Dict[str, Any]:
    """Export canonical preprocessing metadata so API reads from versioned artifact."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "version": "1.0.0",
        "image_size": [image_size, image_size],
        "crop_method": "canonical_fov_intensity_mask",
        "fov_threshold": 15,
        "margin_ratio": 0.02,
        "padding": "square_pad_preserve_aspect",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "color_space": "RGB"
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
