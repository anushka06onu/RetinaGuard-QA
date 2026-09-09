"""Unit tests for retinal transforms, canonical FOV cropping, and synthetic corruption engine."""

import numpy as np
from PIL import Image
import torch
import pytest

from src.preprocessing.transforms import crop_retinal_fov, preprocess_fundus_image
from src.preprocessing.color_norm import graham_color_normalization, reinhard_color_transfer
from src.preprocessing.synthetic_degradations import (
    SyntheticCorruptionEngine,
    apply_gaussian_blur,
    apply_underexposure,
    apply_overexposure
)


def test_retinal_fov_crop():
    # Construct an image with a central circle and dark margins
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[50:250, 50:250, 0] = 200 # Red box/circle
    img = Image.fromarray(arr)

    cropped = crop_retinal_fov(img, threshold=10)
    w, h = cropped.size
    assert w <= 300 and h <= 300
    assert w == h # Check square padding output


def test_preprocess_fundus_image():
    img = Image.new("RGB", (200, 200), color=(180, 80, 30))
    tensor = preprocess_fundus_image(img, target_size=(384, 384))
    assert tensor.shape == (1, 3, 384, 384)
    assert isinstance(tensor, torch.Tensor)


def test_synthetic_corruptions():
    img = Image.new("RGB", (100, 100), color=(150, 70, 30))
    corruptions = SyntheticCorruptionEngine.get_all_corruption_names()
    assert len(corruptions) == 8

    for corr in corruptions:
        corrupted = SyntheticCorruptionEngine.apply(img, corr, severity=3)
        assert corrupted.size == (100, 100)


def test_color_normalizations():
    img = Image.new("RGB", (100, 100), color=(160, 60, 20))
    norm_graham = graham_color_normalization(img)
    assert norm_graham.size == (100, 100)

    norm_reinhard = reinhard_color_transfer(img)
    assert norm_reinhard.size == (100, 100)
