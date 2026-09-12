# Ethics & Intended Use Statement

## Intended Use
RetinaGuard-QA is a research prototype intended to investigate automated technical image-quality assistance. It is designed to assist clinical staff and triage workflows by evaluating technical fundus image quality before diagnostic interpretation.

## Prohibited Clinical Claims
- **No Diagnostic Claim:** The system does NOT diagnose diabetic retinopathy, macular degeneration, glaucoma, or any systemic or retinal disease.
- **No Treatment Advice:** The system must never be used to recommend treatment, medication, or surgical intervention.
- **Not a Clinician Replacement:** An automated quality gate does not substitute for clinical examination or judgment by a certified ophthalmologist or optometrist.

## Explicit Research & Deployment Limitations
Reviewers and prospective users should note the following constraints:
1. **No Prospective Clinical Validation:** The system has not been evaluated in prospective real-world clinical trials.
2. **Dataset Shift:** Performance may vary significantly when exposed to patient populations or disease stages differing from the training corpora.
3. **Dataset-Label Subjectivity:** Image quality labels in underlying datasets reflect human annotator subjectivity and consensus grading variance.
4. **Camera & Hardware Generalization:** Generalization to unvalidated fundus camera hardware, optics, or field-of-view geometries is not guaranteed.
5. **Heuristic Modality Filtering:** Modality and anatomical checks are statistical/heuristic and cannot guarantee detection of all out-of-distribution inputs.
6. **No Disease Diagnosis:** Technical adequacy assessment does not provide any diagnostic signal regarding ocular pathology.
7. **No Replacement for Trained Staff:** Cannot replace trained ophthalmic photographers, technicians, or clinicians.
8. **Limited Demographic Metadata:** Benchmark datasets have limited demographic, racial, and socioeconomic stratification metadata.
9. **Post-Deployment Calibration Drift:** Uncertainty and probability calibration may degrade over time under clinical distribution shifts.
10. **Setting-Specific Threshold Validation:** Decision thresholds and reject/review policies require local clinical recalibration and validation prior to any operational deployment.

## Privacy & Transient Ingestion
- Uploaded image buffers are processed entirely in-memory and discarded upon completion of inference.
- No Protected Health Information (PHI), patient names, or hospital metadata are logged or stored.

