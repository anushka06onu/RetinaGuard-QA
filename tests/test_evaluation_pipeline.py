"""Unit tests for metrics, calibration, selective prediction, OOD, and corruptions."""

import numpy as np
from PIL import Image

from src.retinaguard.evaluation.calibration import (
    compute_brier_score,
    compute_ece,
)
from src.retinaguard.evaluation.corruptions import SyntheticCorruptionSuite
from src.retinaguard.evaluation.metrics import (
    compute_quality_metrics,
)
from src.retinaguard.evaluation.ood import (
    RetinalModalityValidator,
    compute_energy_score,
)
from src.retinaguard.evaluation.selective import (
    compute_risk_coverage_curve,
)


def test_quality_metrics():
    labels = np.array([0, 1, 2, 0, 1, 2])
    logits = np.array(
        [
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 1.0],
            [0.0, 1.0, 5.0],
            [4.0, 2.0, 1.0],
            [1.0, 4.0, 2.0],
            [0.0, 1.0, 4.0],
        ]
    )
    metrics = compute_quality_metrics(logits, labels, is_logits=True)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["quadratic_weighted_kappa"] == 1.0


def test_calibration_and_selective():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.85, 0.1, 0.05],
            [0.15, 0.7, 0.15],
            [0.05, 0.15, 0.8],
        ]
    )
    ece = compute_ece(probs, labels)
    assert 0.0 <= ece["ece"] <= 1.0

    brier = compute_brier_score(probs, labels)
    assert brier >= 0.0

    confs = np.max(probs, axis=-1)
    preds = np.argmax(probs, axis=-1)
    curve = compute_risk_coverage_curve(confs, preds, labels)
    assert curve["aurc"] >= 0.0


def test_ood_and_modality():
    logits_id = np.array([[4.0, 1.0, 0.0]])
    logits_ood = np.array([[0.1, 0.0, -0.1]])

    e_id = compute_energy_score(logits_id)
    e_ood = compute_energy_score(logits_ood)
    assert e_id[0] > e_ood[0]

    # Modality gate check
    fundus_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    fundus_arr[:, :, 0] = 200
    fundus_arr[:, :, 1] = 80
    fundus_arr[:, :, 2] = 20
    val = RetinalModalityValidator.validate(fundus_arr)
    assert val["is_fundus"] is True


def test_corruptions():
    img = Image.new("RGB", (100, 100), color=(180, 80, 20))
    corruptions = SyntheticCorruptionSuite.get_all_names()
    assert len(corruptions) == 10
    for c in corruptions:
        out = SyntheticCorruptionSuite.apply(img, c, severity=3)
        assert out.size == (100, 100)
