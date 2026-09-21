# Dataset Card: RetinaGuard-QA Benchmark Cohorts

## 1. Primary Development & Evaluation Benchmark: DeepDRiD

The released RetinaGuard-QA system is trained and evaluated strictly on the **DeepDRiD** (Diabetic Retinopathy—Grading and Image Quality Estimation Challenge) benchmark dataset.

- **Citation:** Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Yan, T., ... & Sheng, B. (2022). *DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge.* Patterns, 3(6), 100512. [DOI: 10.1016/j.patter.2022.100512](https://doi.org/10.1016/j.patter.2022.100512)
- **Repository:** `https://github.com/deepdrdoc/DeepDRiD.git`
- **Modality:** Color Retinal Fundus Photographs (Standard 45°/50° Field of View).
- **Structure & Partitions:** 2,000 real regular color fundus images across official challenge partitions, preserving strict patient isolation and official partition boundaries:
  - **Training partition (`train.csv` / `regular-fundus-training`):** 1,200 images from 300 unique patients (4 images/patient: two eyes, two fields).
  - **Validation partition (`val.csv` / `regular-fundus-validation`):** 400 images from 100 unique patients.
  - **Official Evaluation partition (`test.csv` / `Online-Challenge1&2-Evaluation`):** 400 images from 100 unique patients.
- **Label Taxonomy:**
  - **Overall Quality (`overall_quality_logits`):** Strictly binary classification (`0: Good`, `1: Poor/Reject`).
  - **Artifact Severity (`artifact_logits`):** 3-level ordinal scale (`0: None/Minimal`, `1: Moderate`, `2: Severe`).
  - **Clarity Severity (`clarity_logits`):** 3-level ordinal scale (`0: Normal/Sharp`, `1: Mild Blur`, `2: Severe Blur`).
  - **Field Definition (`field_definition_logits`):** 3-level ordinal scale (`0: Standard Centering`, `1: Mild Truncation`, `2: Severe Misalignment`).

---

## 2. Robustness, Corruptions & Stress-Testing Suites

- **Synthetic-Noise OOD Stress Testing:** Evaluated against synthetic uniform-noise input distributions to benchmark input-validation gating and energy-score distribution logging.
- **Optical & Acquisition Corruption Suite:** Evaluated across 10 optical and digital corruption types at 5 severity levels:
  1. Gaussian blur
  2. Defocus blur
  3. Motion blur
  4. Brightness increase (overexposure)
  5. Brightness decrease (underexposure)
  6. Contrast reduction
  7. Gaussian noise
  8. Salt-and-pepper noise
  9. JPEG compression artifacts
  10. Pixel dropout / lens smearing

---

## 3. Data Integrity, Provenance & Leakage Auditing

- **Official Partition Preservation:** DeepDRiD challenge partitions are preserved without data leakage or post-hoc data mixing.
- **Patient Isolation:** Patient grouping guarantees zero patient overlap across training, validation, and evaluation splits (300 / 100 / 100 patient allocation).
- **Cryptographic & Perceptual De-duplication:** Verified with SHA-256 hashes and perceptual hashing (pHash) to ensure 0 within-split duplicates and 0 cross-split duplicates.
- **Provenance Manifest:** Cryptographic split fingerprints and dataset distributions are permanently tracked in `data/splits_provenance.json` and verified automatically by `scripts/verify_splits.py`.

---

## 4. Future External Validation Cohorts

The following cohort is planned for future multi-center cross-camera generalizability studies but is **not part of the current v1.0.x released model training or evaluation**:

- **EyeQ Benchmark Cohort:** Derived from EyePACS multi-center diabetic retinopathy screening (Fu et al., MICCAI 2019). Contains 28,792 fundus images categorized into 3 quality tiers (`Good`, `Usable`, `Reject`). Reserved for future prospective cross-cohort domain adaptation studies.
