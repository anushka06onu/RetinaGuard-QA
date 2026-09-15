#!/usr/bin/env bash
# RetinaGuard-QA Master Reproduction Suite
set -e

echo "=========================================================="
echo "    RetinaGuard-QA: Master Evaluation & Reproduction      "
echo "=========================================================="

export PYTHONPATH=src:.

echo "[1/13] Auditing Dataset Splits and Leakage Isolation..."
python3 scripts/audit_dataset.py

echo "[2/13] Verifying Dataset Split Cryptographic Provenance..."
python3 scripts/verify_splits.py --provenance-file data/splits_provenance.json --splits-dir data/splits

echo "[3/13] Training 3-Seed Multi-Task Model Campaign..."
python3 scripts/run_campaign.py --config configs/train_multitask.yaml --seeds 2026 2027 2028

echo "[4/13] Running 3-Seed Zero-Shot Transfer Campaign..."
python3 scripts/run_campaign.py --config configs/train_multitask.yaml --seeds 2026 2027 2028 --zero-shot

echo "[5/13] Training Classical and Majority-Class Baselines..."
python3 scripts/train_baselines.py --eyeq-train data/splits/deepdrid_train.csv --eyeq-test data/splits/deepdrid_external_test.csv --deepdrid-test data/splits/deepdrid_external_test.csv

echo "[6/13] Running Architectural and Attribute Loss Ablations..."
python3 scripts/run_ablations.py --config configs/train_multitask.yaml

echo "[7/13] Exporting Selected Checkpoint to Production ONNX..."
python3 scripts/export_onnx.py --checkpoint artifacts/models/best.ckpt --output-onnx artifacts/models/model.onnx

echo "[8/13] Calibrating Probabilities and Decision Thresholds..."
python3 scripts/calibrate.py --checkpoint artifacts/models/best.ckpt --val-split data/splits/deepdrid_val.csv --task deepdrid_overall --onnx-model artifacts/models/model.onnx

echo "[9/13] Evaluating Held-Out Dataset Performance..."
python3 scripts/evaluate.py --checkpoint artifacts/models/best.ckpt --output-dir artifacts/metrics

echo "[10/13] Benchmarking Out-of-Distribution Detection..."
python3 scripts/benchmark_ood.py --checkpoint artifacts/models/best.ckpt --id-test-split data/splits/deepdrid_external_test.csv --task deepdrid_overall --include-synthetic-stress-test --output-file artifacts/metrics/ood.json

echo "[11/13] Benchmarking Optical Corruption Robustness..."
python3 scripts/benchmark_corruptions.py --checkpoint artifacts/models/best.ckpt --test-split data/splits/deepdrid_external_test.csv --task deepdrid_overall --output-dir artifacts/metrics

echo "[12/13] Benchmarking End-to-End Inference Latency..."
python3 scripts/benchmark_inference.py --model artifacts/models/model.onnx --test-split data/splits/deepdrid_external_test.csv --output-json artifacts/metrics/latency.json

echo "[13/13] Generating Publication and Report Figures..."
python3 scripts/generate_figures.py --metrics-dir artifacts/metrics --figures-dir artifacts/figures

echo "Verifying Final Artifact Manifest..."
python3 scripts/verify_artifacts.py --manifest artifacts/provenance/SHA256SUMS --artifacts-dir artifacts

echo "=========================================================="
echo "  All evaluations and parity benchmarks finished cleanly! "
echo "=========================================================="
