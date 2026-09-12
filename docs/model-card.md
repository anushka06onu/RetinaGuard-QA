# Model Card: RetinaGuard-QA Multi-Task Quality Gate

## Model Details
- **Model Name:** RetinaGuard-QA Multi-Task Retinal Quality & Attribute Assessment Model
- **Version:** v0.2.0
- **Architecture:** Shared lightweight CNN backbone (`mobilenetv3_large_100` / `efficientnet_b0`) with masked multi-task classification heads.
- **Runtime:** CPU-optimized ONNX Runtime (`onnxruntime_cpu`) and PyTorch 2.0+.
- **Input Specification:** Single color retinal fundus image ($384 \times 384 \times 3$, RGB channel ordering, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`).
- **Primary Model Outputs:**
  1. `quality_logits` (3 classes: `Good`, `Usable`, `Reject`)
  2. `overall_quality_logits` (2 classes: `Good`, `Poor-Reject`)
  3. `artifact_logits` (3 ordinal levels: `0=None`, `1=Mild`, `2=Severe`)
  4. `clarity_logits` (3 ordinal levels: `0=High`, `1=Moderate`, `2=Low`)
  5. `field_definition_logits` (3 ordinal levels: `0=Adequate`, `1=Acceptable`, `2=Poor`)
- **Downstream Decision Outputs:**
  - Calibrated Confidence (Post-hoc temperature-scaled max softmax probability)
  - Predictive Shannon Entropy ($[0.0, 1.585\text{ bits}]$)
  - Out-of-Distribution Energy Score ($E(x) = -T \cdot \log \sum \exp(z_i / T)$)
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

## Datasets & Licensing
1. **EyeQ Dataset:**
   - Source: High-resolution fundus photographs from the EyePACS screening programme with three-class quality annotations (`Good`, `Usable`, `Reject`).
   - License: Research use per EyePACS / EyeQ data agreement.
2. **DeepDRiD Dataset:**
   - Source: Second Diabetic Retinopathy Image Dataset Challenge with binary overall quality and fine-grained sub-attribute labels (Artifact, Clarity, Field Definition).
   - License: CC BY-NC-ND 4.0 / Challenge Data Agreement.
3. **Out-of-Distribution Benchmark Sets:**
   - Non-fundus natural imagery (ImageNet validation subsets, CIFAR-10) and non-retinal medical modalities (OCT, Chest X-ray).

---

## Patient-Isolated Splitting Methodology
- **Split Strategy:** Deterministic hash-based patient grouping (`sha256(patient_id + seed)`).
- **Leakage Safeguards:**
  - Strict disjoint patient partitioning between Train, Validation, and Test sets.
  - Zero eye-level or temporal sequence leakage across splits.
  - Cross-split and cross-dataset near-duplicate checking using perceptual hashing (pHash) and exact SHA-256 digests.
  - Fail-fast validation enforced by `scripts/audit_dataset.py` with cryptographic audit reports in `artifacts/reports/`.

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
- **Post-Hoc Calibration:** Temperature scaling on held-out validation logits to optimize Expected Calibration Error (ECE) and negative log-likelihood.
- **Selective Abstention:** Rejection of low-confidence or high-entropy predictions ($H(p) > 1.3474\text{ bits}$), routing borderline cases to `Manual review` to maintain target precision ($\ge 95\%$) on accepted samples.
- **OOD Energy Gating:** Energy-based out-of-distribution detection with configurable directionality (`lower_is_ood` / `higher_is_ood`).

---

## Cryptographic Artifact Provenance
Every production deployment artifact is bound to source training and export digests:
- Preprocessing Contract: `artifacts/models/preprocessing.json`
- Calibration Metadata: `artifacts/models/calibration_metadata.json`
- Production ONNX Runtime Model: `artifacts/models/model.onnx`
- Audit Reports: `artifacts/reports/cross_split_isolation_audit.json`, `artifacts/reports/data_flow_report.json`
- Checksums Manifest: `SHA256SUMS`

