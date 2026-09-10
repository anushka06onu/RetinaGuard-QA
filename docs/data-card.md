# Dataset Card: RetinaGuard-QA Benchmark Cohorts

## 1. Provenance & Access Protocols

### Primary Development Cohort: EyeQ
- **Origin:** EyeQ dataset (Fu et al., MICCAI 2019 / IEEE TMI 2021), derived from a multi-center EyePACS diabetic retinopathy screening program.
- **Repository:** `https://github.com/HzFu/EyeQ.git` (accessed September 10, 2026).
- **Access Route:** The official EyeQ repository distributes public quality labels (`Label_EyeQ_train.csv`, `Label_EyeQ_test.csv`) and preprocessing scripts under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. Underlying image pixels are obtained through the linked EyePACS / Kaggle research data route (`kaggle competitions download -c diabetic-retinopathy-detection`).
- **Images & Resolution:** 28,792 color fundus photographs acquired across heterogeneous clinical fundus cameras (Centervue DRS, Canon CR-2, Topcon NW-400).
- **Label Taxonomy:** 3-class consensus grading:
  - `Good` (58.4%): High dynamic range, distinct foveal reflex, sharp retinal vascular margins.
  - `Usable` (22.4%): Minor peripheral artifacts or mild defocus; macula and optic disc remain interpretable.
  - `Reject` (19.2%): Inadequate illumination, severe blur, or extensive field loss impeding clinical assessment.

### External & Multi-Task Generalization Cohort: DeepDRiD
- **Origin:** Deep Diabetic Retinopathy Image Dataset Challenge (Liu et al., *Patterns* 2022, ISBI 2020 Challenge 5).
- **Repository:** `https://github.com/deepdrdoc/DeepDRiD.git` (accessed September 10, 2026).
- **Structure & Folds:** 2,000 real regular color fundus images across official published folds:
  - Training fold (`regular-fundus-training`): 1,200 images, 300 patients.
  - Validation fold (`regular-fundus-validation`): 400 images, 100 patients.
  - External Evaluation fold (`Online-Challenge1&2-Evaluation`): 400 images, 100 patients.
- **Multi-Task Attributes:** Dual-view regular fundus images annotated for `Overall quality` (good, usable, reject) and ordinal attributes: `Artifact` (0-2), `Clarity` (0-2), and `Field definition` (0-2).

### Out-of-Distribution (OOD) Cohorts
- **Ultra-Widefield (UWF) Fundus:** 100+ images from DeepDRiD Sub-Challenge 3 (`external/DeepDRiD/ultra-widefield_images/`) serving as domain-shift OOD evaluation against narrow-field Color Fundus models.
- **Synthetic Corruptions:** Controlled 10-type benchmark suite (defocus blur, motion blur, brightness shifts, contrast reduction, Gaussian noise, lens flare, salt-and-pepper, compression artifacts, gamma distortion, hue shift).
- **Non-Retinal / Invalid Modalities:** Natural scene images (ImageNet-style), blank/dark captures, and grayscale artifacts audited with strict separate reporting.

---

## 3. Experimental Evaluation Taxonomy

To maintain scientific precision and avoid conflating supervised multi-task learning with zero-shot domain transfer, the experimental evaluation is partitioned into three distinct benchmark protocols:

| Experiment Protocol | Training Cohort | Test Cohort | Scientific Meaning |
|---|---|---|---|
| **EyeQ Baseline** | EyeQ Train | EyeQ Test | **Internal performance** on primary screening distribution |
| **Zero-Shot Transfer** | EyeQ Train only | DeepDRiD External Evaluation | **External generalization** without target-domain adaptation |
| **Multi-Task Model** | EyeQ Train + DeepDRiD Train | DeepDRiD Held-Out Evaluation | **Multi-dataset supervised performance** on shared quality & attributes |

---

## 4. Integrity & Leakage Prevention Audit
- **Cryptographic Audit:** Every raw image is indexed with a unique SHA-256 hash.
- **Subject Leakage Isolation:** All images sharing patient identifiers or capture dates are assigned atomically to single partitions.
- **Exclusion Protocol:** Corrupt headers, zero-byte files, and unreadable images are logged to `data/manifests/*_exclusions.csv` and audited in `artifacts/reports/data_audit.json`.

