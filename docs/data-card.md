# Dataset Card: RetinaGuard-QA Benchmark Cohorts

## 1. Provenance & Primary References

### Primary Development Cohort: EyeQ
- **Citation:** Fu, H., Xu, B., Lin, S., Wong, D. W. K., Baskaran, M., Mahesh, M., ... & Liu, J. (2019). *Evaluation of retinal image quality with multi-task deep learning.* International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 248–256. [DOI: 10.1007/978-3-030-32239-7_28](https://doi.org/10.1007/978-3-030-32239-7_28)
- **Repository:** `https://github.com/HzFu/EyeQ.git`
- **Source Cohort:** Derived from EyePACS multi-center diabetic retinopathy screening images.
- **Access & License:** Quality labels (`Label_EyeQ_train.csv`, `Label_EyeQ_test.csv`) distributed under CC BY-NC-SA 4.0. Raw images obtained via EyePACS research data protocol.
- **Samples & Classes:** 28,792 color fundus photographs acquired across heterogeneous fundus cameras (Centervue DRS, Canon CR-2, Topcon NW-400):
  - `Good` (58.4%): Sharp vascular margins, clear optic disc, distinct foveal reflex.
  - `Usable` (22.4%): Mild peripheral artifacts or illumination drop; central macula and optic disc remain interpretable.
  - `Reject` (19.2%): Severe defocus blur, extreme exposure failure, or extensive field loss impeding clinical evaluation.

### External & Multi-Task Generalization Cohort: DeepDRiD
- **Citation:** Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Yan, T., ... & Sheng, B. (2022). *DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge.* Patterns, 3(6), 100512. [DOI: 10.1016/j.patter.2022.100512](https://doi.org/10.1016/j.patter.2022.100512)
- **Repository:** `https://github.com/deepdrdoc/DeepDRiD.git`
- **Structure & Partitions:** 2,000 real regular color fundus images across official challenge partitions:
  - Training partition (`regular-fundus-training`): 1,200 images, 300 patients.
  - Validation partition (`regular-fundus-validation`): 400 images, 100 patients.
  - External evaluation partition (`Online-Challenge1&2-Evaluation`): 400 images, 100 patients.
- **Label Taxonomy:**
  - `Overall Quality`: Strictly binary (`0: Good`, `1: Poor/Reject`).
  - `Artifact`: Ordinal 3-level scale (`0: None/Minimal`, `1: Moderate`, `2: Severe`).
  - `Clarity`: Ordinal 3-level scale (`0: Normal/Sharp`, `1: Mild Blur`, `2: Severe Blur`).
  - `Field Definition`: Ordinal 3-level scale (`0: Standard Centering`, `1: Mild Truncation`, `2: Severe Misalignment`).

---

## 2. Out-of-Distribution (OOD) Cohorts

- **Near-OOD:** Ultra-Widefield (UWF) fundus images (DeepDRiD Challenge 3), representing a large optical domain shift from standard 45°/50° color fundus photographs.
- **Far-OOD:** Non-retinal natural scene images and invalid optical captures tested to verify energy-based abstention.
- **Synthetic Corruptions Suite:** 10 corruption types across 5 severities:
  1. Gaussian blur
  2. Defocus blur
  3. Motion blur
  4. Brightness increase
  5. Brightness decrease / underexposure
  6. Contrast reduction
  7. Gaussian noise
  8. Salt-and-pepper noise
  9. JPEG compression artifacts
  10. Pixel dropout / lens smearing

---

## 3. Experimental Protocols

| Protocol | Training Set | Evaluation Set | Scientific Purpose |
| :--- | :--- | :--- | :--- |
| **EyeQ Baseline** | EyeQ Train (only) | EyeQ Test | Single-task baseline on primary screening distribution. |
| **Multi-Task Supervised** | EyeQ Train + DeepDRiD Train | DeepDRiD Test | Multi-dataset supervised evaluation of quality & attributes. |
| **Zero-Shot Transfer** | EyeQ Train (only) | DeepDRiD External Test | Unsupervised cross-dataset domain transfer without target tuning. |

---

## 4. Integrity, De-duplication & Leakage Auditing

- **Exact Duplicate Prevention:** Cryptographic SHA-256 computation over every image verifies 0 within-split and 0 cross-split duplicates.
- **Patient Isolation:** Patient grouping guarantees zero patient overlap across training, validation, and test partitions.
- **Audit Reports:** Provenance-bound JSON records generated via `scripts/audit_dataset.py` in `artifacts/reports/data_audit.json`.
