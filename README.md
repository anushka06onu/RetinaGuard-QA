# RetinaGuard-QA

**Uncertainty-Aware and Device-Robust Quality Control for Retinal Fundus Images**

[![CI](https://github.com/anushka06onu/RetinaGuard-QA/actions/workflows/ci.yml/badge.svg)](https://github.com/anushka06onu/RetinaGuard-QA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Python: 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](#)
[![Runtime: ONNX CPU](https://img.shields.io/badge/Runtime-ONNX%20CPU-slate.svg)](#)

---

## Project Status

> [!IMPORTANT]
> **Engineering Pipeline Verification Stage:** The engineering pipeline and software architecture are under active research and development. The repository currently includes testing fixtures and rigorous test suites for validating data ingestion, model heads, temperature calibration, OOD detection, inference, and the web interface. Real empirical results on the EyeQ and DeepDRiD datasets will be reported only after executing the frozen experimental protocol on legally acquired data. No fixture-derived result is presented as empirical research evidence.

---

## 1. Problem & Motivation

Retinal fundus photography is used globally to screen for diabetic retinopathy, glaucoma, and macular degeneration. In practical clinical deployments, captured images are frequently degraded by **defocus blur, xenon flash saturation, underexposure, vignetting, or patient motion artifacts**.

An algorithm should **never analyze an image that is inadequate or outside its known distribution**. RetinaGuard-QA operates strictly before diagnostic review as an automated pre-filter:

$$\text{Retinal Image} \longrightarrow \text{Modality Gate} \longrightarrow \text{Multi-Task Quality Model} \longrightarrow \text{Uncertainty \& OOD Check} \longrightarrow \text{Triage Action}$$

It returns one of three clear operational actions:
1. **Accept**: The model classified this image as technically acceptable under its experimental quality-assessment protocol. Clinical suitability still requires qualified review.
2. **Recapture**: Acquisition artifact or clarity degradation detected; capture guidance provided.
3. **Manual review**: High epistemic uncertainty near decision boundary or unfamiliar optical distribution.

> **CLINICAL BOUNDARY NOTICE:** RetinaGuard-QA assesses physical and optical acquisition quality only. It does **not** diagnose disease, predict DR grade, or replace clinical evaluation by a certified eye-care professional.

---

## 2. Planned Empirical Benchmark Protocol

Once training on the full authorized datasets completes, results will be populated across 3 independent seeds with 95% patient-clustered bootstrap confidence intervals:

| Model | Test Source | Macro-F1 | Balanced Acc | QWK | ECE ($\downarrow$) | Parameters | CPU p95 Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Classical Features (Random Forest) | EyeQ Internal | TBD | TBD | TBD | TBD | — | TBD |
| MobileNetV3-Small (Single-Task) | EyeQ Internal | TBD | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 (Single-Task) | EyeQ Internal | TBD | TBD | TBD | TBD | TBD | TBD |
| **RetinaGuard-QA (Multi-Task)** | **EyeQ Internal** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| **RetinaGuard-QA (Multi-Task)** | **DeepDRiD External** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## 3. System Architecture & Reproduction

```text
retinaguard-qa/
├── configs/             # YAML configurations for training, data, calibration, and corruptions
├── data/                # Data cards, checksum manifests, and immutable patient splits
├── src/retinaguard/     # Core library: data, models, training, evaluation, inference, utils
├── scripts/             # Execution scripts for auditing, training, calibration, and benchmarking
├── api/                 # FastAPI backend with /health, /model-info, and /predict
├── web/                 # React, TypeScript, and Tailwind CSS operator web interface
├── artifacts/           # Versioned ONNX models, JSON metrics, and markdown reports
├── docs/                # Data card, model card, methodology, limitations, and ethics
└── tests/               # Unit, data leakage, scientific rigor, and integration test suite
```

### Complete Reproduction Commands

```bash
# 1. Install dependencies
make install

# 2. Run data audit & leakage checks
make audit-data

# 3. Generate immutable patient-isolated splits
make split-data

# 4. Train multi-task quality model
python scripts/train.py --config configs/train_multitask.yaml

# 5. Calibrate probabilities & evaluate selective prediction
python scripts/calibrate.py

# 6. Run controlled optical corruption sweep (10 types x 5 severities)
python scripts/benchmark_corruptions.py

# 7. Export to ONNX Runtime and benchmark CPU latency
python scripts/export_onnx.py
python scripts/benchmark_inference.py

# 8. Run test suite
make test

# 9. Start FastAPI backend & React web portal
make api
make web
```

---

## 4. Documentation & Cards

- [Data Card](docs/data-card.md) - Dataset provenance, licensing, patient groupings, and audit protocols.
- [Model Card](docs/model-card.md) - Model architecture, training objective, intended use, and limitations.
- [Methodology](docs/methodology.md) - Mathematical formulations for calibration, selective prediction, and masked losses.
- [Limitations & Failure Modes](docs/limitations.md) - Detailed failure analysis and known edge cases.
- [Ethics & Intended Use](docs/ethics-and-intended-use.md) - Clinical boundaries, privacy protections, and safety gates.

---

## 5. Citation & License

```bibtex
@software{retinaguard_qa_2026,
  author = {RetinaGuard-QA Research Team},
  title = {RetinaGuard-QA: Uncertainty-Aware and Device-Robust Quality Control for Retinal Fundus Images},
  year = {2026},
  url = {https://github.com/anushka06onu/RetinaGuard-QA}
}
```

Licensed under the [MIT License](LICENSE).
