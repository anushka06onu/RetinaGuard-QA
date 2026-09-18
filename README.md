# RetinaGuard-QA

**Uncertainty-Aware Multi-Task Quality Assurance for Retinal Fundus Imaging**

[![CI](https://github.com/anushka06onu/RetinaGuard-QA/actions/workflows/ci.yml/badge.svg)](https://github.com/anushka06onu/RetinaGuard-QA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)
[![Python: 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](#)
[![Runtime: ONNX CPU](https://img.shields.io/badge/Runtime-ONNX%20CPU-slate.svg)](#)

---

## Project Status & Scientific Scope

> [!IMPORTANT]
> **DeepDRiD-Focused Empirical Release (v1.0.0):**
> This repository implements and evaluates a multi-task quality assessment model trained and validated on the **DeepDRiD** fundus benchmark across three independent random seeds (`[2026, 2027, 2028]`).
>
> - **Primary Deployed Task:** Binary overall acquisition quality assessment (`good` vs `poor_or_reject`) via the trained `overall_quality_logits` head.
> - **Auxiliary Acquisition Attributes:** Multiclass severity classification for **Artifacts** (`none`, `mild`, `severe`), **Clarity** (`good`, `borderline`, `poor`), and **Field Definition** (`good`, `borderline`, `poor`).
> - **Scientific Rigor Standard:** Zero metric fabrication. All metric reports in `artifacts/metrics/`, figures in `artifacts/figures/`, and tables below strictly originate from verified local execution logs over patient-isolated splits (`data/splits/`) with SHA-256 provenance tracking (`artifacts/provenance/SHA256SUMS`).

---

## 1. System Concept & Problem Statement

Retinal fundus photography is an essential modality for screening diabetic retinopathy, glaucoma, and age-related macular degeneration. In practical clinical deployments, captured images are frequently degraded by **defocus blur, illumination saturation, vignetting, pupil misalignment, or optical artifacts**.

RetinaGuard-QA operates upstream of any diagnostic algorithm as an automated quality assurance pre-filter:

$$\text{Retinal Image} \longrightarrow \text{Modality Gate} \longrightarrow \text{Multi-Task Model} \longrightarrow \text{Predictive Entropy Gate} \longrightarrow \text{Triage Action}$$

It outputs one of four deterministic operational decisions:
1. **`accept`**: Image satisfies technical quality criteria (`good` overall quality).
2. **`recapture`**: Optical or physical degradation detected (`poor_or_reject`); actionable attribute guidance provided (e.g. clean objective lens, adjust flash, reposition patient).
3. **`manual_review`**: Elevated predictive entropy near decision boundary ($> 0.811$ bits for binary task); routed for human review.
4. **`unsupported_input`**: Non-fundus modality, synthetic noise, or severe optical corruption rejected by the modality gate.

> [!NOTE]
> **CLINICAL BOUNDARY NOTICE:**
> RetinaGuard-QA evaluates physical and optical image acquisition quality only. It does **not** diagnose disease, grade diabetic retinopathy, or replace clinical evaluation by a certified eye-care professional.

---

## 2. Verified Empirical Results & Benchmark Evidence

All reported metrics reflect authentic local executions across patient-isolated partitions (`data/splits/`), verified by automated test suites and cryptographic SHA-256 provenance tracking (`artifacts/provenance/SHA256SUMS`).

### A. Deep-Learning Baseline & Multi-Task Comparison (Held-out Test Split, $N=400$, 100 Patients)

| Model Architecture / Method | Supervision / Strategy | Macro-F1 (Mean $\pm$ Std) | 95% Confidence Interval | Balanced Accuracy | QWK |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed Multi-Task MobileNetV3 (3 Seeds)** | Overall Quality + 3 Auxiliary Heads ($\lambda_{\text{aux}}=0.3$) | **0.7446 ± 0.0191** | **[0.6973, 0.7919]** | **0.7460 ± 0.0199** | **0.4900 ± 0.0383** |
| Single-Task MobileNetV3 (3 Seeds) | Overall Quality Only ($\lambda_{\text{aux}}=0.0$) | 0.7419 ± 0.0053 | [0.7287, 0.7551] | 0.7425 ± 0.0050 | 0.4850 ± 0.0100 |
| Classical Handcrafted + Random Forest | Color & Texture Descriptors (RF) | 0.6539 | — | 0.6538 | 0.3077 |
| Classical Handcrafted + Logistic Reg. | Color & Texture Descriptors (LR) | 0.6209 | — | 0.6207 | 0.2441 |
| Majority Class Baseline | Mode Class Predictor | 0.3548 | — | 0.5000 | 0.0000 |

*Campaign seeds: `[2026, 2027, 2028]`. Selected deployed checkpoint (Seed 2026) achieves held-out **Macro-F1: 0.7264**, **Balanced Accuracy: 0.7285**.*

### B. Controlled Ablation Study (3 Seeds per Variant)

To isolate architectural and training contributions, 4 configurations were trained and evaluated across identical data splits, random seeds, optimizer budgets, and image resolutions ($384 \times 384$):

| Ablation Configuration | Description / Purpose | Test Macro-F1 (Mean $\pm$ Std) | Test Accuracy (Mean $\pm$ Std) | Best Val Macro-F1 (Mean $\pm$ Std) |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed Multi-Task Model** | Full multitask ($\lambda_{\text{aux}}=0.3$, label smoothing $=0.1$) | **0.7446 ± 0.0191** | **0.7467 ± 0.0188** | 0.6351 ± 0.0165 |
| **Single-Task Overall Quality** | Single-task baseline ($\lambda_{\text{aux}}=0.0$) | 0.7419 ± 0.0053 | 0.7425 ± 0.0050 | 0.4586 ± 0.0321 |
| **No Label Smoothing** | Full multitask without label smoothing ($ls=0.0$) | 0.7319 ± 0.0226 | 0.7333 ± 0.0240 | 0.6418 ± 0.0100 |
| **Auxiliary Weight 1.0** | Full multitask with equal loss weights ($\lambda_{\text{aux}}=1.0$) | 0.7178 ± 0.0340 | 0.7217 ± 0.0338 | 0.6318 ± 0.0114 |

*Findings: Multi-task learning matches/slightly improves overall quality classification while yielding three fine-grained acquisition attribute heads with zero latency overhead. Label smoothing improves generalization on fundus imagery.*

### C. Fine-Grained Acquisition Attribute Degradation Heads

| Attribute Degradation Head | Severity Classes | Macro-F1 | Quadratic Weighted Kappa (QWK) | Mean Absolute Error (MAE) |
| :--- | :--- | :--- | :--- | :--- |
| **Artifact Severity Head** | None (0) / Mild (1) / Severe (2) | **0.7358** | **0.7439** | **0.285** |
| **Clarity Degradation Head** | Good (0) / Borderline (1) / Poor (2) | **0.5520** | **0.5915** | **0.225** |
| **Field Definition Head** | Good (0) / Borderline (1) / Poor (2) | **0.5599** | **0.4376** | **0.195** |

### D. Post-Hoc Probability Calibration & Selective Prediction

Calibration evaluated on the trained binary head `overall_quality_logits` using `deepdrid_val.csv`:

| Calibration Stage | Temperature ($T$) | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Brier Score |
| :--- | :--- | :--- | :--- | :--- |
| **Uncalibrated Model** | $1.0000$ | 0.1295 | 0.2078 | 0.3667 |
| **Temperature Scaled (Validation Fit)** | $\mathbf{1.3956}$ | $\mathbf{0.1018}$ *(-21.4% error)* | $\mathbf{0.2064}$ | $\mathbf{0.3533}$ |

*Selective prediction: Evaluated across coverage levels $0.50 \to 1.00$ with Area Under the Risk-Coverage Curve ($\text{AURC}) = \mathbf{0.1729}$.*

### E. Input Validation & Modality Defense

| Defense Layer | Tested Input / Anomaly | AUROC | False Reject Rate (ID) | Rejection Rate (OOD) |
| :--- | :--- | :--- | :--- | :--- |
| **Retinal Modality Gate** | Uniform Synthetic Noise / Non-Fundus | **1.0000** | **0.00%** | **100.0%** |
| **Free-Energy Metric ($T=1.3956$)** | Synthetic Noise Stress Test | 0.1048 | — | *Transparent baseline (unregularized)* |

*Note: Free energy on standard architectures without explicit energy regularization does not provide standalone OOD separation; the heuristic retinal modality gate acts as the primary input-validity defense.*

### F. Edge Deployment & CPU Inference Footprint

| Deployment Metric | Measurement (Apple M4, 4 Threads) | Notes |
| :--- | :--- | :--- |
| **ONNX Engine Inference Latency** | **8.42 ms** (median) / **13.56 ms** (p95) | Batch size 1, $384 \times 384 \times 3$ float32 |
| **End-to-End Processing Latency** | **47.84 ms** (median) / **78.29 ms** (p95) | Includes FOV crop, normalization, inference, triage policy |
| **Sustained Throughput** | **20.14 images / sec** | Single-worker CPU execution |
| **Exported Model File Size** | **19.27 MB** | Optimized ONNX graph with shared weight tensor |
| **PyTorch / ONNX Parity Max Error** | **$< 1.99 \times 10^{-5}$** | Verified across all 4 prediction heads |

---

## 3. Data Integrity & Leakage Auditing

- **Patient-Isolated Partitioning:** All splits (`data/splits/`) partition patients strictly without cross-split overlap (0 patient ID leakage). Patient identifiers were derived from canonical dataset filenames.
- **Perceptual-Hash Collision Audit:** Evaluated perceptual hash collision check across splits: **0 unresolved cross-split collision candidates**.

---

## 4. Step-by-Step Reproduction Guide

```bash
# 1. Environment Installation
pip install -e ".[dev]"
cd web && npm ci && npm run build && cd ..

# 2. Data Manifest Preparation
python scripts/prepare_deepdrid.py \
  --external-root external/DeepDRiD/regular_fundus_images \
  --output-csv data/manifests/deepdrid_manifest.csv

# 3. Patient-Isolated Stratified Partitioning
python scripts/create_splits.py \
  --dataset deepdrid \
  --deepdrid-manifest data/manifests/deepdrid_manifest.csv \
  --output-dir data/splits

# 4. Leakage & Collision Auditing
python scripts/audit_dataset.py \
  --splits-dir data/splits \
  --reports-dir artifacts/reports

# 5. Multi-Seed Training Campaign
python scripts/run_campaign.py --config configs/train_multitask.yaml --seeds 2026 2027 2028

# 6. Post-Hoc Probability Calibration on Trained Binary Head
python scripts/calibrate.py \
  --checkpoint artifacts/models/best.ckpt \
  --val-split data/splits/deepdrid_val.csv \
  --task deepdrid_overall \
  --output-dir artifacts/metrics

# 7. Model Evaluation on Held-Out External Test Set
python scripts/evaluate.py \
  --checkpoint artifacts/models/best.ckpt \
  --deepdrid-split data/splits/deepdrid_external_test.csv \
  --output-dir artifacts/metrics

# 8. Export to ONNX Runtime with Parity Verification (< 1e-4)
python scripts/export_onnx.py \
  --checkpoint artifacts/models/best.ckpt \
  --output-onnx artifacts/models/model.onnx

# 9. Ablation Study Execution (4 Configurations x 3 Seeds)
python scripts/run_ablations.py \
  --base-config configs/train_multitask.yaml \
  --deepdrid-split data/splits/deepdrid_external_test.csv \
  --seeds 2026 2027 2028 \
  --output-dir artifacts/metrics

# 10. Out-of-Distribution & Optical Corruption Benchmarking
python scripts/benchmark_ood.py \
  --checkpoint artifacts/models/best.ckpt \
  --id-test-split data/splits/deepdrid_external_test.csv \
  --output-file artifacts/metrics/ood.json

python scripts/benchmark_corruptions.py \
  --checkpoint artifacts/models/best.ckpt \
  --test-split data/splits/deepdrid_external_test.csv \
  --task deepdrid_overall \
  --output-dir artifacts/metrics

# 11. Generate Scientific Figures & Checksums
python scripts/generate_figures.py \
  --metrics-dir artifacts/metrics \
  --figures-dir artifacts/figures

python scripts/verify_artifacts.py --generate
python scripts/verify_artifacts.py

# 12. Launch Production Docker Stack
docker-compose up --build
```

---

## 5. Repository Structure

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
│   ├── metrics/         # Recorded evaluation metrics, ablations, and training histories
│   ├── models/          # Exported ONNX model, calibration JSONs, preprocessing configs
│   ├── provenance/      # Automated environment.txt and SHA256SUMS
│   └── reports/         # Audited data integrity and isolation reports
├── docs/                # Architecture, data card, methodology, limitations, user guide
└── tests/               # Python unit, integration, and scientific-integrity tests (pytest)
```

---

## 6. Primary References

1. **DeepDRiD Dataset:** Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Yan, T., ... & Sheng, B. (2022). *DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge.* Patterns, 3(6), 100512. [DOI: 10.1016/j.patter.2022.100512](https://doi.org/10.1016/j.patter.2022.100512)
2. **Probability Calibration:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML 2017, pp. 1321–1330.
3. **Energy-Based OOD Detection:** Liu, W., Wang, X., Owens, J., & Li, Y. (2020). *Energy-based out-of-distribution detection.* NeurIPS 2020, 33, 21464–21475.
4. **Selective Classification:** Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks.* NeurIPS 2017, pp. 4878–4887.

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

