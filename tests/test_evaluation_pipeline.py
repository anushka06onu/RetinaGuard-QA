"""Unit tests for metrics, calibration, selective prediction, OOD, and corruptions."""

import numpy as np
from PIL import Image

from retinaguard.evaluation.calibration import (
    compute_brier_score,
    compute_ece,
)
from retinaguard.evaluation.corruptions import SyntheticCorruptionSuite
from retinaguard.evaluation.metrics import (
    compute_quality_metrics,
)
from retinaguard.evaluation.ood import (
    RetinalModalityValidator,
    compute_energy_score,
)
from retinaguard.evaluation.selective import (
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


def test_calibration_inference_energy_parity():
    test_logits = np.array([[3.5, 1.2, -0.8], [0.1, 0.2, 0.0]], dtype=np.float32)
    temp = 1.35

    # Calibration-time energy calculation
    cal_energy = compute_energy_score(test_logits, temperature=temp)

    # Inference-time energy calculation (per sample in loop)
    inf_energy_0 = float(compute_energy_score(test_logits[0:1], temperature=temp)[0])
    inf_energy_1 = float(compute_energy_score(test_logits[1:2], temperature=temp)[0])

    np.testing.assert_allclose(cal_energy[0], inf_energy_0, rtol=1e-6)
    np.testing.assert_allclose(cal_energy[1], inf_energy_1, rtol=1e-6)


def test_evaluation_ignores_masked_placeholder_records(tmp_path):
    import pandas as pd
    import torch

    from scripts.evaluate import evaluate_dataset_partition

    # 1. Create real temporary test images
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (100, 100), color=(180, 80, 30)).save(img1)
    Image.new("RGB", (100, 100), color=(180, 80, 30)).save(img2)

    # 2. Manifest with 1 valid attribute/quality sample, and 1 missing sample (mask=0)
    df = pd.DataFrame(
        [
            {
                "image_id": "img_valid",
                "patient_id": "p1",
                "path": str(img1),
                "dataset": "deepdrid",
                "overall_quality_canonical": "usable",  # class 1
                "artifact": 1,  # class 1
                "clarity": 0,  # class 0
                "field_definition": 0,  # class 0
            },
            {
                "image_id": "img_missing",
                "patient_id": "p2",
                "path": str(img2),
                "dataset": "deepdrid",
                # missing labels => target 0, mask 0.0
                "overall_quality_canonical": None,
                "artifact": None,
                "clarity": None,
                "field_definition": None,
            },
        ]
    )
    csv_file = tmp_path / "test_split.csv"
    df.to_csv(csv_file, index=False)

    # 3. Model outputs:
    # Sample 0 (valid): predicts correct class (overall: 1, artifact: 1, clarity: 0, field_def: 0)
    # Sample 1 (masked): intentionally predicts class 2 (mismatches placeholder target 0)
    class DummyMultiTaskModel(torch.nn.Module):
        def forward(self, x):
            overall_logits = torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
            artifact_logits = torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
            clarity_logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
            field_def_logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
            return {
                "overall_quality_logits": overall_logits,
                "artifact_logits": artifact_logits,
                "clarity_logits": clarity_logits,
                "field_definition_logits": field_def_logits,
            }

    model = DummyMultiTaskModel()
    metrics = evaluate_dataset_partition(
        model, str(csv_file), dataset_name="DeepDRiD (Test)", is_deepdrid=True
    )

    # Masked placeholder record (sample 1) must be ignored:
    assert metrics["num_samples"] == 1
    assert metrics["total_records_in_split"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["artifact_num_samples"] == 1
    assert metrics["artifact_macro_f1"] == 1.0
    assert metrics["clarity_num_samples"] == 1
    assert metrics["clarity_macro_f1"] == 1.0
    assert metrics["field_definition_num_samples"] == 1
    assert metrics["field_definition_macro_f1"] == 1.0
