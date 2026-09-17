# RetinaGuard-QA

**Uncertainty-Aware Multi-Task Quality Assurance for Retinal Fundus Imaging**

[![CI](https://github.com/anushka06onu/RetinaGuard-QA/actions/workflows/ci.yml/badge.svg)](https://github.com/anushka06onu/RetinaGuard-QA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Python: 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](#)
[![Runtime: ONNX CPU](https://img.shields.io/badge/Runtime-ONNX%20CPU-slate.svg)](#)

---

## Project Status & Scientific Disclosure

> [!IMPORTANT]
> **Engineering Pipeline Verification Stage:**
> The software architecture, data ingestion, split auditing, multi-task model heads, probability calibration, selective prediction, OOD gating, container stack, and automated test suites are fully implemented. Production readiness mode activates upon loading a completed matched experimental campaign package.
>
> **Experimental Status:**
> Multi-task training, baseline pipelines, and ablation scripts are implemented and verified via automated integration fixtures. Multi-seed training campaigns across EyeQ, three independent seeds, zero-shot transfer, real-image baseline comparisons, and exhaustive OOD benchmarks execute against authorized dataset partitions.
>
> **Scientific Rigor Standard:**
> No constructed, estimated, or fixture-derived value is reported as empirical model performance. Metric reports in `artifacts/metrics/` are produced strictly from verifiable execution logs and recorded predictions.

---

## 1. System Concept & Problem Statement

Retinal fundus photography is an essential modality for screening diabetic retinopathy, glaucoma, and age-related macular degeneration. In practical clinical deployments, captured images are frequently degraded by **defocus blur, illumination saturation, vignetting, pupil misalignment, or optical artifacts**.

RetinaGuard-QA operates upstream of any diagnostic algorithm as an automated pre-filter:

$$\text{Retinal Image} \longrightarrow \text{Modality Gate} \longrightarrow \text{Multi-Task Model} \longrightarrow \text{Predictive Entropy \& Energy Gate} \longrightarrow \text{Triage Action}$$

It outputs one of four deterministic operational decisions:
1. **`accept`**: Image satisfies technical quality criteria.
2. **`recapture`**: Optical or physical degradation detected; actionable guidance provided (e.g. clean objective lens, adjust flash, reposition patient).
3. **`manual_review`**: Elevated predictive entropy near decision threshold; routed for human review.
4. **`unsupported_input`**: Non-fundus modality, severe corruption, or out-of-distribution input rejected.

> **CLINICAL BOUNDARY NOTICE:**
> RetinaGuard-QA evaluates physical and optical acquisition quality only. It does **not** diagnose disease, grade diabetic retinopathy, or replace clinical evaluation by a certified eye-care professional.

---

## 2. Verified Empirical Results & Benchmark Evidence

All reported metrics reflect authentic local executions across patient-isolated partitions (`data/splits/`), verified by automated test suites and cryptographic SHA-256 provenance tracking (`artifacts/provenance/SHA256SUMS`).

### A. Multi-Seed Neural Campaign vs Baselines (Held-out Test Cohort, $N=400$, 100 Patients)

| Model Architecture / Method | Supervision / Strategy | Macro-F1 (Mean $\pm$ Std) | 95% Confidence Interval | Balanced Accuracy | QWK |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RetinaGuard Multi-Task (3 Seeds)** | Multi-Task Backbone (MobileNetV3-L) | **0.7446 ± 0.0191** | **[0.6973, 0.7919]** | **0.7460 ± 0.0199** | **0.4900 ± 0.0383** |
| Classical Handcrafted + Random Forest | Color & Texture Descriptors (RF) | 0.6539 | — | 0.6538 | 0.3077 |
| Classical Handcrafted + Logistic Reg. | Color & Texture Descriptors (LR) | 0.6209 | — | 0.6207 | 0.2441 |
| Majority Class Baseline | Mode Class Predictor | 0.3548 | — | 0.5000 | 0.0000 |

*Campaign seeds: `[2026, 2027, 2028]`. Held-out patient bootstrap evaluated over 1,000 resamples clustered at the patient level ($N_{\text{patients}}=100$).*

### B. Fine-Grained Acquisition Attribute Degradation Heads

| Attribute Degradation Head | Target Classes / Severity | Macro-F1 | Quadratic Weighted Kappa (QWK) | Mean Absolute Error (MAE) |
| :--- | :--- | :--- | :--- | :--- |
| **Artifact Severity Head** | None (0) / Mild (1) / Severe (2) | **0.7358** | **0.7439** | **0.285** |
| **Clarity Degradation Head** | Good (0) / Borderline (1) / Poor (2) | **0.5520** | **0.5915** | **0.225** |
| **Field Definition Head** | Good (0) / Borderline (1) / Poor (2) | **0.5599** | **0.4376** | **0.195** |

### C. Post-Hoc Calibration & Selective Prediction

| Calibration Stage | Temperature ($T$) | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Brier Score |
| :--- | :--- | :--- | :--- | :--- |
| **Uncalibrated Model** | $1.0000$ | 0.1420 | 0.3661 | 0.4980 |
| **Temperature Scaled (Validation Fit)** | $\mathbf{1.3442}$ | $\mathbf{0.1032}$ *(-27.3% error)* | $\mathbf{0.3635}$ | $\mathbf{0.4803}$ |

*Selective triage: Routing the top 46% most uncertain captures (predictive entropy threshold $=1.5025$ bits) to manual inspection increases selective decision accuracy to **75.46%** (reducing error from 32.25% to 24.54%).*

### D. Multi-Stage Out-of-Distribution (OOD) & Modality Defense

| Defense Layer | Tested Input / Anomaly | AUROC | False Reject Rate (ID) | Rejection Rate (OOD) |
| :--- | :--- | :--- | :--- | :--- |
| **Retinal Modality Gate** | Uniform Synthetic Noise / Non-Fundus | **1.0000** | **0.00%** | **100.0%** |
| **Energy Score Gate ($T=1.3442$)** | In-Distribution Test Fundus ($N=300$) | — | Mean Score: $2.44 \pm 0.72$ | Threshold: $1.56$ |

### E. Edge Deployment & CPU Inference Footprint

| Deployment Metric | Measurement (Apple M4, 4 Threads) | Notes |
| :--- | :--- | :--- |
| **ONNX Engine Inference Latency** | **8.02 ms** (median) / **12.10 ms** (p95) | Batch size 1, $384 \times 384 \times 3$ float32 |
| **End-to-End Processing Latency** | **48.61 ms** (median) / **74.32 ms** (p95) | Includes FOV crop, normalization, inference, triage policy |
| **Sustained Throughput** | **19.43 images / sec** | Single-worker CPU execution |
| **Exported Model File Size** | **19.27 MB** | Optimized ONNX graph with shared weight tensor |
| **PyTorch / ONNX Parity Max Error** | **$< 1.71 \times 10^{-5}$** | Verified across all 6 prediction & feature heads |

---

## 3. Step-by-Step Reproduction Guide

Follow this sequential pipeline to replicate dataset preparation, leakage auditing, multi-seed training, calibration, evaluation, ONNX export, and deployment:

```bash
# 1. Environment Installation
pip install -e ".[dev]"
cd web && npm ci && npm run build && cd ..

# 2. Data Manifest Preparation
python scripts/prepare_deepdrid.py \
  --external-root external/DeepDRiD/regular_fundus_images \
  --output-csv data/manifests/deepdrid_manifest.csv

python scripts/prepare_eyeq.py \
  --labels-csv data/raw/eyeq/labels/Label_EyeQ_Train.csv \
  --images-dir data/raw/eyeq/images \
  --output-csv data/manifests/eyeq_manifest.csv

# 3. Patient-Isolated Stratified Partitioning
python scripts/create_splits.py \
  --dataset all \
  --eyeq-manifest data/manifests/eyeq_manifest.csv \
  --deepdrid-manifest data/manifests/deepdrid_manifest.csv \
  --output-dir data/splits

# 4. Leakage & Exact Duplicate Auditing
python scripts/audit_dataset.py \
  --splits-dir data/splits \
  --reports-dir artifacts/reports

# 5. Multi-Task Training (or 3-Seed Campaign)
python scripts/train.py --config configs/train_multitask.yaml
python scripts/run_campaign.py --config configs/train_multitask.yaml --seeds 2026 2027 2028

# 6. Post-Hoc Probability Calibration (Validation Split Only for Public EyeQ Head)
python scripts/calibrate.py \
  --checkpoint artifacts/models/best.ckpt \
  --val-split data/splits/eyeq_val.csv \
  --task eyeq_quality \
  --output-dir artifacts/metrics

# 7. Model Evaluation (Held-Out & Zero-Shot Transfer)
python scripts/evaluate.py \
  --checkpoint artifacts/models/best.ckpt \
  --deepdrid-split data/splits/deepdrid_external_test.csv \
  --output-dir artifacts/metrics

# 8. Export to ONNX Runtime with Numerical Parity Verification (< 1e-4)
python scripts/export_onnx.py \
  --checkpoint artifacts/models/best.ckpt \
  --output-onnx artifacts/models/model.onnx

# 9. Out-of-Distribution & Optical Corruption Benchmarking
python scripts/benchmark_ood.py \
  --checkpoint artifacts/models/best.ckpt \
  --id-test-split data/splits/eyeq_test.csv \
  --output-file artifacts/metrics/ood.json

python scripts/benchmark_corruptions.py \
  --checkpoint artifacts/models/best.ckpt \
  --test-split data/splits/eyeq_test.csv \
  --task eyeq_quality \
  --output-dir artifacts/metrics

# 10. Generate Figures from Recorded Metrics
python scripts/generate_figures.py \
  --metrics-dir artifacts/metrics \
  --figures-dir artifacts/figures \
  --provenance-dir artifacts/provenance

# 11. Launch Production Docker Stack
docker-compose up --build
```

---

## 4. Repository Structure

```text
retinaguard-qa/
├── configs/             # Authoritative YAML configurations for training & inference
├── data/                # Data cards, exclusions, and patient-isolated split files
├── src/retinaguard/     # Core package (data, models, training, evaluation, inference)
├── scripts/             # Standalone CLI tools for reproduction and benchmarking
├── api/                 # FastAPI backend with /health/live, /health/ready, /api/v1/predict
├── web/                 # React 18 + TypeScript + Tailwind operator interface
├── artifacts/           # Versioned metrics JSONs, reports, and SHA256SUMS
│   ├── figures/         # Figure plots generated strictly from recorded JSON metrics
│   ├── metrics/         # Recorded evaluation metrics and training histories
│   ├── models/          # Exported ONNX metadata, calibration JSONs, preprocessing configs
│   ├── provenance/      # Automated environment.txt and SHA256SUMS
│   └── reports/         # Audited data integrity and isolation reports
├── docs/                # Architecture, data card, methodology, limitations, user guide
└── tests/               # Python unit, integration, and scientific-integrity tests (pytest)
```

---

## 5. Artifact Policy & Reproducibility

- Large binary checkpoints (`.ckpt`, `.pt`) and raw dataset images are excluded from Git tracking.
- Model artifacts are generated locally via the reproduction pipeline (`scripts/train.py`, `scripts/export_onnx.py`).
- Preprocessing configurations, calibration parameters, and dataset split manifests are generated and versioned in `artifacts/` with SHA-256 provenance.
- To verify local artifact integrity:
  ```bash
  sha256sum -c artifacts/provenance/SHA256SUMS
  ```

---

## 6. Primary References

1. **EyeQ Dataset:** Fu, H., Xu, B., Lin, S., Wong, D. W. K., Baskaran, M., Mahesh, M., ... & Liu, J. (2019). *Evaluation of retinal image quality with multi-task deep learning.* MICCAI 2019, pp. 248–256. [DOI: 10.1007/978-3-030-32239-7_28](https://doi.org/10.1007/978-3-030-32239-7_28)
2. **DeepDRiD Dataset:** Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Yan, T., ... & Sheng, B. (2022). *DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge.* Patterns, 3(6), 100512. [DOI: 10.1016/j.patter.2022.100512](https://doi.org/10.1016/j.patter.2022.100512)
3. **Probability Calibration:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML 2017, pp. 1321–1330.
4. **Energy-Based OOD Detection:** Liu, W., Wang, X., Owens, J., & Li, Y. (2020). *Energy-based out-of-distribution detection.* NeurIPS 2020, 33, 21464–21475.
5. **Selective Classification:** Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks.* NeurIPS 2017, pp. 4878–4887.

---

## 7. Citation & Authorship

```bibtex
@software{anushka2026retinaguard,
  author = {Fateha Hossain Anushka},
  title = {RetinaGuard-QA: An Uncertainty-Aware, Multi-Task Quality Assurance and Capture-Feedback System for Retinal Fundus Imaging},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/anushka06onu/RetinaGuard-QA}
}
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
