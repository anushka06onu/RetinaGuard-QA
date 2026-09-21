# Model Card: RetinaGuard-QA Multi-Task Quality Gate

## Model Details
- **Model Name:** RetinaGuard-QA Multi-Task Retinal Quality & Attribute Assessment Model
- **Version:** v1.0.3
- **Architecture:** Shared lightweight CNN backbone (`mobilenetv3_large_100`) with four multi-task classification heads.
- **Runtime:** CPU-optimized ONNX Runtime (`onnxruntime_cpu`) and PyTorch 2.0+.
- **Input Specification:** Single color retinal fundus image ($384 \times 384 \times 3$, RGB channel ordering, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`).
- **Trained & Deployed Model Outputs:**
  1. `overall_quality_logits` (2 classes: `Good`, `Poor-Reject`) [Primary Deployed Triage Head]
  2. `artifact_logits` (3 ordinal levels: `0=None/Minimal`, `1=Moderate`, `2=Severe`)
  3. `clarity_logits` (3 ordinal levels: `0=Normal/Sharp`, `1=Mild Blur`, `2=Severe Blur`)
  4. `field_definition_logits` (3 ordinal levels: `0=Standard Centering`, `1=Mild Truncation`, `2=Severe Misalignment`)
- **Downstream Decision Outputs:**
  - Calibrated Confidence: Post-hoc temperature-scaled max softmax probability ($T = 1.3956$)
  - Predictive Shannon Entropy: $H(p) \in [0.0, 1.0\text{ bit}]$ for binary overall quality
  - Out-of-Distribution Energy Score: Research logging ($E(x) = -T \cdot \log \sum \exp(z_i / T)$)
  - Actionable Triage Recommendation: `Accept`, `Recapture`, `Manual review`, `Unsupported input`

---

## Intended Use & Target Workflow
- **Clinical Setting:** Pre-diagnostic edge triage pre-filter running directly on retinal camera workstations or clinic capture terminals before image ingestion by downstream AI diagnostic pipelines or telehealth ophthalmologists.
- **Target Users:** Retinal screening photographers, primary care technicians, optometrists, and telemedicine triage operators.
- **Operational Goal:** Flag optical artifacts, defocus blur, and illumination defects at acquisition time while the patient is still seated in front of the camera, preventing ungradable remote tele-ophthalmology consults.

---

## Prohibited & Out-of-Scope Uses
- **No Disease Diagnosis:** Strictly prohibited from diagnosing diabetic retinopathy, age-related macular degeneration, glaucoma, or any ocular or systemic condition.
- **No Treatment Advice:** The system must never recommend medication, surgery, or clinical follow-up intervals for pathology.
- **No Autonomous Diagnostic Clearance:** An `Accept` grade indicates optical quality sufficiency, not absence of ocular disease.

---

## Benchmark Datasets & Development Cohort
1. **DeepDRiD Dataset:**
   - Source: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge (Patterns 2022) with binary overall quality and fine-grained sub-attribute labels (Artifact, Clarity, Field Definition).
   - Partitions: 1,200 training images (300 patients), 400 validation images (100 patients), and 400 official evaluation images (100 patients).
   - License: CC BY-NC-ND 4.0 / Challenge Data Agreement.
2. **Synthetic-Noise Stress-Testing:**
   - Evaluated against synthetic uniform-noise inputs to benchmark input validation and out-of-distribution research logging behavior.

---

## Patient-Isolated Splitting Methodology
- **Split Strategy:** Deterministic hash-based patient grouping (`sha256(patient_id + seed)`).
- **Leakage Safeguards:**
  - Strict disjoint patient partitioning between Train, Validation, and Test sets.
  - Zero eye-level or temporal sequence leakage across splits.
  - Cross-split near-duplicate checking using perceptual hashing (pHash) and exact SHA-256 digests.
  - Fail-fast validation enforced by `scripts/verify_splits.py` with cryptographic provenance records in `data/splits_provenance.json`.

---

## Preprocessing & Optical FOV Standardization
1. **Canonical Field-of-View (FOV) Detection:**
   - Intensity thresholding ($\text{threshold} = 15$) on green channel to locate circular fundus mask.
   - Bounding box extraction with safety margin ratio ($0.02$) and exclusive pixel boundaries (`[rmin:rmax+1, cmin:cmax+1]`).
2. **Standardized Resizing & Rescaling:**
   - High-quality bilinear interpolation with anti-aliasing to $384 \times 384$ px.
   - Label-preserving geometric data augmentation during training (small rotation $\pm 10^\circ$, horizontal flip, bounded random scaling).

---

## Uncertainty Quantification & Selective Prediction
- **Post-Hoc Calibration:** Temperature scaling on held-out validation logits ($T = 1.3956$) to optimize Expected Calibration Error (ECE) and negative log-likelihood.
- **Selective Abstention:** Rejection of high-entropy predictions ($H(p) > 0.9943\text{ bits}$), routing borderline cases to `Manual review` to prevent erroneous automatic acceptance.
- **OOD Energy Logging:** Energy-based out-of-distribution logging computed and tracked for research analysis.

---

## Cryptographic Artifact Provenance
Every production deployment artifact is bound to source training and export digests:
- Preprocessing Contract: `artifacts/models/preprocessing.json`
- Calibration Metadata: `artifacts/models/calibration_metadata.json`
- Production ONNX Runtime Model: `artifacts/models/model.onnx`
- Audit Reports: `artifacts/reports/cross_split_isolation_audit.json`, `artifacts/reports/data_flow_report.json`
- Checksums Manifest: `artifacts/provenance/SHA256SUMS`
