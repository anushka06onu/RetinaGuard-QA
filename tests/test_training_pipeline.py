"""Unit and integration tests for training orchestration, callbacks, and selective prediction."""

import json
from pathlib import Path

import numpy as np
import torch

from retinaguard.evaluation.calibration import (
    compute_brier_score,
    compute_ece,
)
from retinaguard.evaluation.selective import (
    compute_risk_coverage_curve,
    evaluate_selective_abstention,
)
from retinaguard.training.callbacks import (
    EarlyStopping,
    MetricHistoryLogger,
    ModelCheckpointSaver,
)
from retinaguard.training.train import run_training_experiment


def test_training_callbacks_unit(tmp_path):
    # Test EarlyStopping
    es = EarlyStopping(patience=2, mode="max")
    assert es(0.5) is True  # new best
    assert es.best_score == 0.5
    assert es.early_stop is False
    assert es(0.4) is False  # worse, counter=1
    assert es.early_stop is False
    assert es(0.3) is False  # worse, counter=2
    assert es.early_stop is True

    # Test MetricHistoryLogger
    log_file = tmp_path / "history.json"
    logger = MetricHistoryLogger(log_file)
    logger.log(1, train_loss=0.8, val_loss=0.7, val_metrics={"macro_f1": 0.65})
    logger.log(2, train_loss=0.6, val_loss=0.5, val_metrics={"macro_f1": 0.75})
    assert log_file.is_file()
    with open(log_file, "r") as f:
        hist = json.load(f)
    assert len(hist["epoch"]) == 2

    # Test ModelCheckpointSaver
    ckpt_dir = tmp_path / "checkpoints"
    saver = ModelCheckpointSaver(ckpt_dir, filename="best.ckpt")
    saver.save({"weights": torch.tensor([1.0, 2.0])}, {"epoch": 2, "macro_f1": 0.75})
    saved_p = ckpt_dir / "best.ckpt"
    assert saved_p.is_file()
    state = torch.load(saved_p, map_location="cpu")
    assert "state_dict" in state
    assert state["metadata"]["macro_f1"] == 0.75


def test_selective_prediction_and_calibration_unit():
    probs = np.array([[0.95, 0.05], [0.45, 0.55], [0.85, 0.15], [0.40, 0.60]])
    confidences = np.array([0.95, 0.45, 0.85, 0.40])
    predictions = np.array([0, 1, 0, 1])
    labels = np.array([0, 0, 0, 1])

    res = evaluate_selective_abstention(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        min_confidence_threshold=0.80,
    )
    assert res["coverage"] == 0.5
    assert res["abstention_rate"] == 0.5
    assert res["selective_accuracy"] == 1.0

    rc_curve = compute_risk_coverage_curve(confidences, predictions, labels)
    assert "aurc" in rc_curve
    assert len(rc_curve["coverages"]) > 0

    # Calibration functions
    ece_dict = compute_ece(probs, labels, num_bins=5)
    assert 0.0 <= ece_dict["ece"] <= 1.0
    assert 0.0 <= ece_dict["mce"] <= 1.0
    brier = compute_brier_score(probs, labels)
    assert brier >= 0.0


def test_run_training_experiment_fixture_mode(tmp_path):
    res = run_training_experiment(
        config_path="configs/train_multitask.yaml",
        smoke_test=True,
        fixture_mode=True,
        override_seed=2026,
        output_dir=tmp_path,
    )
    assert "checkpoint_path" in res
    assert Path(res["checkpoint_path"]).is_file()
    assert res["epochs_trained"] == 2
