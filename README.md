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
> The software architecture, data ingestion, split auditing, multi-task model heads, probability calibration, selective prediction, OOD gating, and deployment stack are implemented and covered by automated test suites; final-model deployment validation remains pending conclusion of full-scale dataset training.
>
> **Experimental Status:**
> Preliminary multi-task engineering runs have been executed on the verified DeepDRiD dataset partitions. Full multi-seed campaigns across EyeQ, three independent training seeds, zero-shot transfer, real-image baseline comparisons, and exhaustive OOD benchmarks are currently in progress on authorized datasets.
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

## 2. Experimental Protocols & Evaluation Methodology

### A. Experimental Benchmark Design

The experimental protocol evaluates multi-task representations against single-task and classical baselines across strictly patient-isolated splits:

| Model / Comparison | Cohort / Split | Target Tasks | Metric Objectives |
| :--- | :--- | :--- | :--- |
| **Majority Class Baseline** | EyeQ Official Test / DeepDRiD Test | Quality Grade | Macro-F1, Accuracy |
| **Classical Texture/Color Baseline** | EyeQ Official Test / DeepDRiD Test | Feature Extractor + Classifier | Macro-F1, AUROC |
| **MobileNetV3 Single-Task** | EyeQ Official Test | 3-Class Quality | Macro-F1, Balanced Acc, Latency |
| **EfficientNet-B0 Single-Task** | EyeQ Official Test | 3-Class Quality | Macro-F1, Balanced Acc, Latency |
| **RetinaGuard-QA Multi-Task (3 Seeds)** | EyeQ Official Test ($N=15{,}700$) | Quality + Acquisition Attributes | Macro-F1, ECE, Selective Precision |
| **DeepDRiD Supervised Held-Out** | DeepDRiD Test ($N=400$, 100 patients) | Binary Overall Quality + Attributes | Binary Macro-F1, Ordinal F1 |
| **Zero-Shot External Transfer** | DeepDRiD External ($N=400$, 100 patients) | Binary Acceptable vs Reject Transfer | External AUROC, Macro-F1 |

### B. Scientific Rigor & Evidence Integrity Standards

1. **Patient-Isolated Splitting:** Patient identifiers are partitioned deterministically (`sha256(patient_id + seed)`) with zero cross-split overlap verified cryptographically by `scripts/audit_dataset.py`.
2. **Post-Hoc Probability Calibration:** Temperature scaling parameters are fit strictly on held-out validation sets.
3. **Selective Prediction:** Risk-coverage profiling establishes abstention thresholds for borderline images without test-set tuning.
4. **No Fabricated or Hardcoded Data:** Metric summaries in `artifacts/metrics/` are generated directly from execution runs.


---

## 3. Step-by-Step Reproduction Guide

Follow this sequential pipeline to replicate dataset preparation, leakage auditing, multi-seed training, calibration, evaluation, ONNX export, and deployment:

```bash
# 1. Environment Installation
pip install -e ".[dev]"
cd web && npm install && npm run build && cd ..

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

# 6. Post-Hoc Probability Calibration (Validation Split Only)
python scripts/calibrate.py \
  --checkpoint artifacts/models/best.ckpt \
  --val-split data/splits/deepdrid_val.csv \
  --task deepdrid_overall \
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
  --id-test-split data/splits/deepdrid_external_test.csv \
  --output-file artifacts/metrics/ood.json

python scripts/benchmark_corruptions.py \
  --checkpoint artifacts/models/best.ckpt \
  --test-split data/splits/deepdrid_external_test.csv \
  --task deepdrid_overall \
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
├── api/                 # FastAPI backend with /health/live, /health/ready, /api/predict
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

- Large binary checkpoints (`.ckpt`, `.pt`) are excluded from Git tracking.
- Model artifacts are generated locally via the reproduction pipeline (`scripts/train.py`, `scripts/export_onnx.py`).
- Preprocessing configurations, calibration parameters, and dataset split manifests are tracked directly in the repository with SHA-256 provenance.
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
