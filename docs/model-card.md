# Model Card: RetinaGuard-QA Multi-Task Quality Gate

## Model Details
- **Architecture:** Shared lightweight CNN backbone (`mobilenetv3_large_100` / `efficientnet_b0`) with masked multi-task classification heads.
- **Model Version:** v1.0.0
- **Input:** Single color retinal fundus image ($384 \times 384$ px RGB tensor, normalized with ImageNet statistics).
- **Primary Outputs:**
  1. Quality Grade Softmax Probabilities: `[Good, Usable, Reject]`
  2. Calibrated Top-1 Confidence
  3. Predictive Shannon Entropy $[0.0, 1.58\text{ bits}]$
  4. Out-of-Distribution Energy Score
  5. Decoded Attribute Scores: Artifact, Clarity, Field Definition
  6. Actionable Triage Decision: `Accept`, `Recapture`, `Manual review`, `Unsupported input`

---

## Intended Use
- **Deployment Location:** Edge triage pre-filter running directly on clinic capture workstations before downstream AI diagnostic pipelines.
- **Target User:** Retinal screening photographers, optometrists, primary care technicians.

---

## Prohibited & Out-of-Scope Use
- **Not a Diagnostic Tool:** Does not predict diabetic retinopathy grade, glaucoma, AMD, or systemic vascular conditions.
- **No Direct Treatment Dispatch:** Triage actions relate solely to optical image quality and recapture recommendations.
