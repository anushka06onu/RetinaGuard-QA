# System Limitations & Boundary Conditions

## 1. Confounding Pathologies vs. Acquisition Defocus
- **Media Opacities:** Dense nuclear cataracts, asteroid hyalosis, and vitreous hemorrhages scatter optical rays internally, creating visual blur indistinguishable from camera misfocus. The system flags the technical degradation, but cannot attribute its etiology to the camera versus ocular pathology.

## 2. Ultra-Widefield & Smartphone Fundoscopy
- The canonical FOV detector is optimized for standard $45^\circ / 50^\circ$ tabletop fundus cameras. Images from $200^\circ$ scanning laser ophthalmoscopes (e.g. Optos) or handheld smartphone adapters exhibit distinct geometric distortions that may trigger elevated uncertainty.

## 3. Ground-Truth Inter-Rater Subjectivity
- Mild quality degradation near the `Good` vs. `Usable` boundary carries inherent clinician disagreement ($\approx 10-15\%$). RetinaGuard-QA handles this ambiguity through selective abstention (`Manual review`).
