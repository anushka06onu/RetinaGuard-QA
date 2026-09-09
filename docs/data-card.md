# Dataset Card: RetinaGuard-QA Benchmark Cohorts

## 1. Provenance & Access Protocols

### Primary Training Cohort: EyeQ
- **Origin:** EyeQ dataset (Fu et al., IEEE TMI 2019 / MICCAI 2019), derived from a multi-center EyePACS diabetic retinopathy screening program.
- **Access Route:** The official EyeQ repository distributes public quality labels and preprocessing scripts. Underlying image pixels are accessed under EyePACS / Kaggle research data terms.
- **Images & Resolution:** 28,792 color fundus photographs acquired across heterogeneous clinical fundus cameras (Centervue DRS, Canon CR-2, Topcon NW-400).
- **Label Taxonomy:** 3-class consensus grading:
  - `Good` (58.4%): High dynamic range, distinct foveal reflex, sharp retinal vascular margins.
  - `Usable` (22.4%): Minor peripheral artifacts or mild defocus; macula and optic disc remain interpretable.
  - `Reject` (19.2%): Inadequate illumination, severe blur, or extensive field loss impeding clinical assessment.

### External Generalization Cohort: DeepDRiD
- **Origin:** Deep Diabetic Retinopathy Image Dataset Challenge (Liu et al., Patterns 2022).
- **Role:** Held-out external zero-shot test set to quantify cross-camera, cross-clinic domain shift.
- **Attributes:** Detailed ordinal labels for `overall_quality`, `artifact`, `clarity`, and `field_definition`.

---

## 2. Integrity & Leakage Prevention Audit
- **Cryptographic Audit:** Every raw image is indexed with a unique SHA-256 hash.
- **Subject Leakage Isolation:** All images sharing patient identifiers or capture dates are assigned atomically to single partitions (70% Train, 15% Validation, 15% Test).
- **Exclusion Protocol:** Corrupt headers, zero-byte files, and duplicate images are logged to `artifacts/reports/data_audit.json`.
