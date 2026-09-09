# Known Limitations & Failure Modes

## 1. Domain & Optical Device Shift
- **Field-of-View (FOV) Variations:** RetinaGuard-QA is calibrated primarily for standard 45° and 50° non-mydriatic fundus cameras. Ultra-widefield fundus photography (e.g. 200° Optos Daytona / California) exhibits peripheral distortion and eyelid artifacts not fully captured by canonical circular masking.
- **Smartphone / Handheld Fundus Cameras:** Handheld ophthalmoscopes produce irregular illumination glare and variable magnification ratios that may yield elevated uncertainty scores.

## 2. Pathological Confounders vs. Acquisition Defects
- **Dense Cataracts & Vitreous Hemorrhage:** Severe ocular media opacities physically prevent light transmission to the retina, presenting visual blur that mimics optical defocus. The system detects the optical degradation, but cannot distinguish camera misfocus from internal media opacities.
- **Extensive Laser Photocoagulation:** Dense laser burn scars or subretinal fibrosis may produce high-frequency edge gradients that can influence continuous sharpness estimators.

## 3. Epistemic Uncertainty & Rare Artifacts
- Rare lens reflections (crescent artifacts, dust on objective lens, patient eyelash obstruction) may fall outside the training manifold. In these instances, the Out-of-Distribution (OOD) score or high entropy flag triggers a `Manual Review` directive rather than an overconfident classification.

## 4. Multi-Label Defect Granularity
- While synthetic corruptions allow fine-grained isolation of degradation axes, real-world clinician ground-truth defect labels (such as fine distinction between haze vs mild illumination unevenness) contain subjective inter-rater variability.
