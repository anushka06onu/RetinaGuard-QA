"""Canonical preprocessing and FOV detection matching Phase 5 of blueprint."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def crop_retinal_fov(
    image: Union[Image.Image, np.ndarray], threshold: int = 15, margin_ratio: float = 0.02
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
    margin_h = int((rmax - rmin + 1) * margin_ratio)
    margin_w = int((cmax - cmin + 1) * margin_ratio)

    rmin = max(0, rmin - margin_h)
    rmax = min(h, rmax + 1 + margin_h)
    cmin = max(0, cmin - margin_w)
    cmax = min(w, cmax + 1 + margin_w)

    cropped = np_img[rmin:rmax, cmin:cmax]

    # Pad to square instead of stretching
    ch, cw, _ = cropped.shape
    if ch != cw:
        max_dim = max(ch, cw)
        padded = np.zeros((max_dim, max_dim, 3), dtype=cropped.dtype)
        y_off = (max_dim - ch) // 2
        x_off = (max_dim - cw) // 2
        padded[y_off : y_off + ch, x_off : x_off + cw] = cropped
        return Image.fromarray(padded)

    return Image.fromarray(cropped)


class CanonicalRetinalTransform:
    """Callable canonical FOV crop transform."""

    def __init__(self, threshold: int = 15, margin_ratio: float = 0.02):
        self.threshold = threshold
        self.margin_ratio = margin_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        return crop_retinal_fov(img, self.threshold, self.margin_ratio)


def get_train_transforms(image_size: Union[int, list, tuple] = 384) -> T.Compose:
    """Training augmentations preserving retinal diagnostic fidelity without altering quality classes.

    Uses strictly label-preserving spatial transforms (flips and bounded rotation)
    rather than ColorJitter, which directly modifies quality characteristics.
    """
    size_tuple = (
        (image_size[0], image_size[1])
        if isinstance(image_size, (list, tuple))
        else (int(image_size), int(image_size))
    )
    return T.Compose(
        [
            CanonicalRetinalTransform(),
            T.Resize(size_tuple, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transforms(
    image_size: Union[int, list, tuple] = 384,
    fov_threshold: int = 15,
    margin_ratio: float = 0.02,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
) -> T.Compose:
    """Deterministic validation, test, and production deployment transform."""
    size_tuple = (
        (image_size[0], image_size[1])
        if isinstance(image_size, (list, tuple))
        else (int(image_size), int(image_size))
    )
    return T.Compose(
        [
            CanonicalRetinalTransform(threshold=fov_threshold, margin_ratio=margin_ratio),
            T.Resize(size_tuple, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_transform_from_metadata(metadata: Dict[str, Any]) -> T.Compose:
    """Build preprocessing pipeline strictly from validated preprocessing metadata."""
    img_size = metadata.get("image_size", [384, 384])
    fov_th = metadata.get("fov_threshold", 15)
    margin_r = metadata.get("margin_ratio", 0.02)
    norm = metadata.get("normalization", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std = norm.get("std", [0.229, 0.224, 0.225])

    interp_name = metadata.get("interpolation", "bilinear").lower()
    interp_mode = (
        T.InterpolationMode.NEAREST if interp_name == "nearest" else T.InterpolationMode.BILINEAR
    )

    size_tuple = (
        (img_size[0], img_size[1])
        if isinstance(img_size, (list, tuple))
        else (int(img_size), int(img_size))
    )

    return T.Compose(
        [
            CanonicalRetinalTransform(threshold=fov_th, margin_ratio=margin_r),
            T.Resize(size_tuple, interpolation=interp_mode),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_image_canonical(
    image: Union[Image.Image, np.ndarray], image_size: Union[int, list, tuple] = 384
) -> torch.Tensor:
    """End-to-end preprocessing into a normalized tensor [1, 3, H, W]."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    transform = get_val_transforms(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def export_preprocessing_metadata(
    output_path: Union[str, Path] = "artifacts/models/preprocessing.json",
    image_size: int = 384,
    source_checkpoint_sha256: str = "unknown",
    status: str = "completed",
) -> Dict[str, Any]:
    """Export canonical preprocessing metadata so API and inference read from versioned artifact."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": "1.0.0",
        "version": "1.0.0",
        "class_order": ["good", "usable", "reject"],
        "image_size": [image_size, image_size],
        "crop_method": "canonical_fov_intensity_mask",
        "fov_threshold": 15,
        "margin_ratio": 0.02,
        "interpolation": "bilinear",
        "padding": "square_pad_preserve_aspect",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "color_space": "RGB",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "status": status,
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
