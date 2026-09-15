#!/usr/bin/env bash
# RetinaGuard-QA Master Reproduction Suite
set -e

echo "=========================================================="
echo "    RetinaGuard-QA: Master Evaluation & Reproduction      "
echo "=========================================================="

export PYTHONPATH=.

echo "[1/11] Auditing Dataset Splits and Leakage Isolation..."
python3 scripts/audit_dataset.py

echo "[2/11] Training 3-Seed Multi-Task Model Campaign..."
python3 scripts/run_campaign.py

echo "[3/11] Running 3-Seed Zero-Shot Transfer Campaign..."
python3 scripts/run_campaign.py --zero-shot

echo "[4/11] Training Classical and Majority-Class Baselines..."
python3 scripts/train_baselines.py

echo "[5/11] Running Architectural and Attribute Loss Ablations..."
python3 scripts/run_ablations.py

echo "[6/11] Exporting Selected Checkpoint to Production ONNX..."
python3 scripts/export_onnx.py --checkpoint artifacts/models/best.ckpt

echo "[7/11] Calibrating Probabilities and Decision Thresholds..."
python3 scripts/calibrate.py --checkpoint artifacts/models/best.ckpt --task eyeq_quality

echo "[8/11] Benchmarking Out-of-Distribution Detection..."
python3 scripts/benchmark_ood.py

echo "[9/11] Benchmarking Optical Corruption Robustness..."
python3 scripts/benchmark_corruptions.py

echo "[10/11] Benchmarking End-to-End Inference Latency..."
python3 scripts/benchmark_inference.py

echo "[11/11] Generating Publication and Report Figures..."
python3 scripts/generate_figures.py

echo "=========================================================="
echo "  All evaluations and parity benchmarks finished cleanly! "
echo "=========================================================="
