# Dataset Card: EyeQ (Internal Cohort)

## Dataset Summary
- **Dataset Name:** EyeQ (Quality Assessment of Retinal Fundus Images)
- **Source Repository:** https://github.com/hzfu/eyeq
- **Associated Publication:** Fu et al., "Evaluation of Retinal Image Quality Assessment: A Large-scale Dataset and Benchmark", IEEE TMI 2019 / MICCAI 2019.
- **Total Images:** 28,792 color fundus photographs (subset of EyePACS).
- **Label Taxonomy:** 3-class grading (`Good`, `Usable`, `Reject`).
- **Defect Annotations:** Illumination defects (underexposure/overexposure), blur/defocus, and field obstruction artifacts.

## Quality Label Distribution (Approximate)
- **Good:** ~16,817 (58.4%)
- **Usable:** ~6,449 (22.4%)
- **Reject:** ~5,526 (19.2%)

## Splitting & Integrity Strategy
- Partitioned via grouped patient hashing into 70% Train, 15% Validation, 15% Test.
- No patient identity crosses partition boundaries.
