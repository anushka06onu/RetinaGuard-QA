"""Unit tests for Out-of-Distribution detection, Energy scores, and Visual Modality Gate."""

import numpy as np
from PIL import Image
import torch
import pytest

from src.ood.energy_score import compute_energy_score, is_ood_energy
from src.ood.mahalanobis import MahalanobisOODDetector
from src.ood.visual_gate import RetinalModalityGate, validate_retinal_modality


def test_energy_score():
    in_dist_logits = np.array([[5.0, 1.0, -2.0]])
    ood_logits = np.array([[0.1, -0.2, 0.0]])

    energy_id = compute_energy_score(in_dist_logits)
    energy_ood = compute_energy_score(ood_logits)

    # In-distribution logits should have higher (less negative) energy than flat OOD logits
    assert energy_id[0] > energy_ood[0]


def test_mahalanobis_detector():
    features = np.random.randn(50, 32)
    labels = np.random.randint(0, 3, 50)

    detector = MahalanobisOODDetector(num_classes=3, feature_dim=32)
    detector.fit(features, labels)

    dists = detector.compute_min_mahalanobis_distance(features[:5])
    assert len(dists) == 5
    assert (dists >= 0).all()


def test_retinal_modality_gate():
    # Valid fundus-like image (Red > Green > Blue)
    fundus_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    fundus_arr[:, :, 0] = 200
    fundus_arr[:, :, 1] = 80
    fundus_arr[:, :, 2] = 20
    res_fundus = validate_retinal_modality(Image.fromarray(fundus_arr))
    assert res_fundus["is_fundus_candidate"] is True

    # Grayscale / Chest X-Ray (Equal R, G, B)
    xray_arr = np.full((100, 100, 3), 120, dtype=np.uint8)
    res_xray = validate_retinal_modality(Image.fromarray(xray_arr))
    assert res_xray["is_fundus_candidate"] is False
    assert "Monochromatic" in res_xray["reason"]

    # Flat black / blank
    blank_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    res_blank = validate_retinal_modality(Image.fromarray(blank_arr))
    assert res_blank["is_fundus_candidate"] is False
