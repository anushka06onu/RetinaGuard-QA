# Ethical Principles & Clinical Boundaries

## Core Mission
RetinaGuard-QA is engineered as a **quality-assurance pre-filter** to improve the reliability, trustworthiness, and safety of clinical artificial intelligence deployments in tele-ophthalmology and primary screening.

---

## Explicit Clinical Boundary Disclaimers
1. **Not a Medical Device:** RetinaGuard-QA has not undergone formal regulatory certification (such as FDA 510(k), CE-MDR Class IIa) as a diagnostic medical device.
2. **Quality Control Only:** The software assesses physical and sensor-level image quality (blur, illumination, centering, field loss, motion, artifacts). It does **not** evaluate disease status, diabetic retinopathy severity, cup-to-disc ratio, macular edema, or any pathological lesion.
3. **Mandatory Human Oversight:** Any image flagged with high predictive uncertainty or classified as `Reject` must be reviewed by a certified ophthalmic photographer, clinician, or optometry staff.

---

## Data Privacy & Confidentiality
- **Zero Default Persistence:** Uploaded image buffers are processed in-memory during HTTP inference and immediately cleared from memory upon response generation.
- **No PHI Collection:** The system neither requires nor stores Protected Health Information (PHI) such as patient names, MRNs, national IDs, or clinic addresses.
- **Auditing & Logging:** Diagnostic error metrics are logged pseudonymously with cryptographic run hashes.

---

## Bias & Demographic Fairness Considerations
- Retinal pigmentation varies with systemic melanin levels (e.g. fundus background coloration in diverse ethnic populations).
- Quality models trained exclusively on homogeneous populations may misinterpret dark choroidal pigmentation as underexposure.
- RetinaGuard-QA integrates canonical illumination normalization and evaluates performance across multi-ethnic cross-dataset cohorts (EyeQ, DeepDRiD) to mitigate demographic and camera-specific bias.
