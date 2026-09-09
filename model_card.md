# Model Card: RetinaGuardNet (Multi-Task Quality Gate)

## Model Overview
- **Model Name:** RetinaGuardNet
- **Architecture:** Lightweight multi-task CNN (Default: `mobilenetv3_large_100` / `efficientnet_b0` backbone with specialized quality, defect, score, and feature-embedding heads).
- **Primary Task:** Automated technical acquisition quality assessment, actionable acquisition defect identification, calibrated uncertainty quantification, and out-of-distribution (OOD) modality gating for color retinal fundus photographs.
- **Model Version:** v1.0.0
- **Input:** Single color retinal fundus image (RGB, $384 \times 384$ px standard canonical field).
- **Outputs:**
  1. **Quality Grade:** Categorical probability distribution across `[Good, Usable, Reject]`.
  2. **Actionable Defects:** Multi-label sigmoid probabilities across:
     - `Severe Blur / Focus Loss`
     - `Underexposure`
     - `Overexposure`
     - `Uneven Illumination / Shadowing`
     - `Field Truncation / Off-Center`
     - `Optical Artifacts / Lens Smear`
  3. **Continuous Quality Index:** Estimated normalized visual fidelity score $[0.0, 100.0]$.
  4. **Uncertainty & OOD Measures:**
     - Calibrated Softmax Uncertainty (Shannon Entropy & Max-Probability Margin)
     - Energy-Based OOD Score & Mahalanobis distance in latent space.

---

## Intended Use & Deployment Context
- **Target Users:** Screening operators, primary care technicians, and automated AI triage pipelines.
- **Deployment Stage:** Upstream quality gate **prior** to any clinical disease classification (e.g., Diabetic Retinopathy, Glaucoma, AMD) or human expert review.
- **Decision Engine Actions:**
  - `Accept`: Image meets clinical standards; forward to diagnosis pipeline.
  - `Usable with Warning`: Minor peripheral degradation; sufficient for macular/optic disc review.
  - `Recapture with Guidance`: Low technical quality; returns specific physical adjustments (e.g. refocus, adjust flash intensity, reposition fixation target).
  - `Manual Review`: High model uncertainty / ambiguous boundary cases requiring human operator judgment.
  - `Unsupported / OOD`: Input rejected as non-fundus, severely corrupt, or from unknown optical modalities.

---

## Training Data & Evaluation Protocol
- **Primary Training Cohort:** EyeQ dataset (standardized partition strictly isolating patient IDs).
- **External Validation:** DeepDRiD Challenge Quality Sub-dataset (assessing cross-camera and cross-ethnicity domain shifts).
- **Synthetic Robustness Suite:** Controlled 5-level severity evaluations over 8 distinct optical corruptions (Gaussian Blur, Motion Blur, Exposure Shift, Vignetting, Compression, Additive Sensor Noise, Color Temperature Shift).
- **Evaluation Metrics:** Macro-F1, Cohen's Weighted Kappa ($\kappa_w$), Expected Calibration Error (ECE), Brier Score, AUROC, Risk-Coverage Curves, and CPU Inference Latency (ONNX Runtime).

---

## Out-of-Scope Use
- **Disease Diagnosis:** RetinaGuardNet does NOT predict diabetic retinopathy grade, glaucoma, hypertensive retinopathy, or systemic biomarkers.
- **Surgical Guidance:** Not validated for intra-operative imaging or OCT-A volumes.
- **Replacement for Ophthalmologist:** Does not constitute medical diagnosis or advice.
