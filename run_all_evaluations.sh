#!/usr/bin/env bash
# RetinaGuard-QA Master Reproduction Suite
set -e

echo "=========================================================="
echo "    RetinaGuard-QA: Autonomous Evaluation & Benchmark     "
echo "=========================================================="

export PYTHONPATH=.

echo "[1/7] Running Classical Feature Baseline..."
python3 experiments/01_baseline_classical.py

echo "[2/7] Running Single-Task CNN Encoders..."
python3 experiments/02_train_backbones.py

echo "[3/7] Training Multi-Task Architecture & Exporting ONNX..."
python3 experiments/03_train_multitask.py

echo "[4/7] Running Uncertainty Calibration & Selective Risk-Coverage..."
python3 experiments/04_calibration_uncertainty.py

echo "[5/7] Running Cross-Dataset Transfer (EyeQ -> DeepDRiD)..."
python3 experiments/05_cross_dataset_eval.py

echo "[6/7] Running Controlled Optical Robustness Sweep..."
python3 experiments/06_robustness_sweep.py

echo "[7/7] Generating Component Ablation Summary..."
python3 experiments/07_ablation_study.py

echo "[8/8] Executing Complete Pytest Validation Suite..."
pytest tests/ -v

echo "=========================================================="
echo "  All evaluations and parity tests completed successfully!"
echo "=========================================================="
