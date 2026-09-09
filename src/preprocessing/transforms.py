"""Canonical field-of-view (FOV) detection, circular mask cropping, and PyTorch transformation pipelines."""

from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def crop_retinal_fov(
    image: Union[Image.Image, np.ndarray], 
    threshold: int = 15, 
    margin_ratio: float = 0.02
) -> Image.Image:
    """Detect circular retinal fundus field boundary and crop black surrounding margins.

    Args:
        image: PIL Image or NumPy RGB array.
        threshold: Intensity threshold (0-255) to distinguish retinal tissue from background.
        margin_ratio: Margin to preserve around detected bounding box.

    Returns:
        Square-cropped PIL Image focused on the retinal mask.
    """
    if isinstance(image, Image.Image):
        np_img = np.array(image.convert("RGB"))
    else:
        np_img = image

    # Convert to grayscale luminance
    gray = np_img.mean(axis=2)
    mask = gray > threshold

    if not np.any(mask):
        # Image is entirely dark; return as is
        return Image.fromarray(np_img)

    # Find bounding box coordinates
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

    # Make square by padding if aspect ratio is distorted
    ch, cw, _ = cropped.shape
    if ch != cw:
        max_side = max(ch, cw)
        padded = np.zeros((max_side, max_side, 3), dtype=cropped.dtype)
        y_offset = (max_side - ch) // 2
        x_offset = (max_side - cw) // 2
        padded[y_offset:y_offset + ch, x_offset:x_offset + cw] = cropped
        return Image.fromarray(padded)

    return Image.fromarray(cropped)


class RetinalCanonicalCrop:
    """Callable transform for canonical FOV cropping."""
    def __init__(self, threshold: int = 15, margin_ratio: float = 0.02):
        self.threshold = threshold
        self.margin_ratio = margin_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        return crop_retinal_fov(img, self.threshold, self.margin_ratio)


def get_training_transforms(image_size: Tuple[int, int] = (384, 384)) -> T.Compose:
    """Returns data augmentation and normalization pipeline for model training."""
    return T.Compose([
        RetinalCanonicalCrop(),
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (384, 384)) -> T.Compose:
    """Returns deterministic validation and deployment preprocessing pipeline."""
    return T.Compose([
        RetinalCanonicalCrop(),
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_fundus_image(
    image: Union[Image.Image, np.ndarray], 
    target_size: Tuple[int, int] = (384, 384)
) -> torch.Tensor:
    """End-to-end preprocessing of an arbitrary fundus image into a normalized PyTorch tensor [1, 3, H, W]."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    transform = get_validation_transforms(target_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)
