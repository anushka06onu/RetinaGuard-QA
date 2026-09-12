# System Limitations & Boundary Conditions

Reviewers, clinical collaborators, and deployment engineers should explicitly account for the following limitations:

## 1. No Prospective Clinical Validation
The system is an investigational research prototype evaluated on retrospective public benchmarks (EyeQ, DeepDRiD). It has not undergone prospective real-world clinical trials or regulatory device clearance.

## 2. Dataset Shift Across Populations and Disease Severities
Performance may fluctuate when deployed in clinical environments whose patient demographics, disease prevalence (e.g., severe proliferative diabetic retinopathy vs. healthy screening cohorts), or ocular comorbidities differ from the training distribution.

## 3. Ground-Truth Inter-Rater Subjectivity
Quality grading in public datasets reflects human annotator subjectivity. Ambiguity near the `Good` vs. `Usable` and `Usable` vs. `Reject` boundaries exhibits known inter-grader discordance ($\approx 10\text{--}15\%$). RetinaGuard-QA handles boundary uncertainty via selective abstention (`Manual review`).

## 4. Optical Hardware & Camera Generalization
The canonical FOV cropping and feature representations are tuned for standard $45^\circ / 50^\circ$ tabletop fundus cameras (Canon CR-2, Topcon TRC, Zeiss). Generalization to ultra-widefield scanning laser ophthalmoscopes (e.g., Optos $200^\circ$) or smartphone-based adapters is not guaranteed.

## 5. Confounding Media Opacities vs. Acquisition Defocus
Dense cataracts, corneal opacities, vitreous hemorrhages, and asteroid hyalosis scatter optical light internally, causing optical degradation identical to operator misfocus or inadequate illumination. The model detects technical image degradation but cannot discern ocular pathology from operator error.

## 6. Statistical Retinal Modality & Anatomical Gating
Modality checks and OOD detection use statistical energy scoring and RGB channel heuristics. While effective against non-fundus natural imagery and non-retinal medical modalities, they cannot guarantee detection of all out-of-distribution inputs.

## 7. No Disease Diagnosis
Technical image-quality assessment produces zero diagnostic signal regarding diabetic retinopathy, glaucoma, AMD, or other vascular/retinal disorders. A grade of `Good` indicates adequate optical clarity, not an absence of retinal pathology.

## 8. No Replacement for Certified Staff
The system serves purely as an edge-based pre-capture quality filter and operator feedback tool. It cannot substitute for clinical oversight by certified optometrists, ophthalmic photographers, or ophthalmologists.

## 9. Limited Demographic and Sociodemographic Metadata
Publicly available retinal quality benchmarks lack granular demographic, racial, and socioeconomic stratification metadata, precluding exhaustive subgroup fairness validation.

## 10. Post-Deployment Calibration Drift & Threshold Local Validation
Temperature scaling calibration parameters and selective prediction abstention thresholds are data-dependent and may drift under deployment distribution shifts. Local validation and threshold tuning are mandatory prior to clinical workflow integration.

