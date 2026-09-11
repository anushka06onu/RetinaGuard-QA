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
> The software architecture, data ingestion, split auditing, multi-task model heads, probability calibration, selective prediction, OOD gating, and deployment stack are fully implemented and verified with automated test suites.
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

## 2. Experimental Protocols & Current Evidence

### A. Experimental Benchmark Design

When executed across full cohorts, the experimental protocol evaluates models over patient-isolated splits:

| Model / Comparison | Cohort / Split | Target Tasks | Status |
| :--- | :--- | :--- | :---: |
| **Majority Class Baseline** | EyeQ Test ($N=2,400$) | 3-class Quality | Planned |
| **Classical Features (Random Forest / LR)** | EyeQ Test ($N=2,400$) | Texture & Contrast Features | Planned |
| **MobileNetV3 Single-Task** | EyeQ Test ($N=2,400$) | Quality Grade | Planned |
| **EfficientNet-B0 Single-Task** | EyeQ Test ($N=2,400$) | Quality Grade | Planned |
| **RetinaGuard-QA Multi-Task (3 Seeds)** | EyeQ Test ($N=2,400$) | Quality + Acquisition Attributes | In Progress |
| **DeepDRiD Held-Out Supervised** | DeepDRiD Test ($N=400$) | Overall Quality + 3 Ordinal Attributes | In Progress |
| **DeepDRiD Zero-Shot External Transfer** | DeepDRiD External ($N=400$) | Binary Acceptable vs Reject | In Progress |

### B. Preliminary Engineering Execution on DeepDRiD

A preliminary multi-task training run was conducted on the 2,000 verified DeepDRiD challenge images across the official training, validation, and held-out test partitions:

- **Held-Out Test Partition ($N=400$ images, 100 patients):**
  - **Macro-F1 (Preliminary 3-Class Head):** 0.6968 (95% CI: [0.6450, 0.7469])
  - **Balanced Accuracy:** 0.6997
  - **Artifact Attribute Macro-F1:** 0.7377
  - *Note:* This preliminary run utilized the earlier 3-output head. The corrected binary overall-quality head (`0: Good`, `1: Poor/Reject`, `nn.Linear(512, 2)`) is now implemented for the final campaign.
  - Full per-class records are documented in [artifacts/metrics/held_out_deepdrid.json](file:///Users/fatehahossainanushka/RetinaGuard-QA/artifacts/metrics/held_out_deepdrid.json).

---

## 3. Step-by-Step Reproduction Guide

Follow this sequential pipeline to replicate dataset preparation, leakage auditing, multi-seed training, calibration, evaluation, ONNX export, and deployment:

```bash
# 1. Environment Installation
pip install -e ".[dev]"
cd web && npm install && npm run build && cd ..

# 2. Data Manifest Preparation
python scripts/prepare_deepdrid.py --data_dir data/raw/deepdrid
python scripts/prepare_eyeq.py --data_dir data/raw/eyeq

# 3. Patient-Isolated Stratified Partitioning
python scripts/create_splits.py \
  --dataset all \
  --eyeq-manifest data/manifests/eyeq_manifest.csv \
  --deepdrid-manifest data/manifests/deepdrid_manifest.csv \
  --output-dir data/splits

# 4. Leakage & Exact Duplicate Auditing
python scripts/audit_dataset.py --splits_dir data/splits

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
  --output-file artifacts/metrics/corruptions.json

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
└── tests/               # 39 unit, API integration, and scientific rigor tests (pytest)
```

---

## 5. Artifact Policy & Checkpoint Availability

- Large binary checkpoints (`.ckpt`, `.pt`) are excluded from Git history per repository data policy.
- Checkpoint releases and exported ONNX models are hosted externally via GitHub Releases:
  - **Release Repository:** `https://github.com/anushka06onu/RetinaGuard-QA/releases`
  - **Exported ONNX Model:** `model.onnx` (SHA-256: `cb9d189f92091f916d91657cd6fd23279261c65f83db5dd235639c03a7f8388c`)
  - **Source PyTorch Checkpoint:** `best.ckpt` (SHA-256: `563db501aed717055360759b7483b155de1003fc82f55756d6d8ac4c8c24d91d`)
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
  version = {1.0.0},
  url = {https://github.com/anushka06onu/RetinaGuard-QA}
}
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
