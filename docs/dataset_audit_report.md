# Dataset Audit & Patient Isolation Report

## Methodology
Data leakage across training and test splits represents one of the most pervasive failure modes in medical machine learning. When multiple retinal photographs originating from the same subject (e.g. left eye vs. right eye, or temporal follow-up sessions) are split randomly, deep networks can memorize idiosyncratic anatomical signatures (such as specific vessel branching patterns, choroidal pigmentation, or optic disc tilt), yielding inflated test metrics that collapse on external cohorts.

## Audit Safeguards Implemented
1. **Cryptographic SHA-256 Hashing**: Identifies identical duplicate images across partitions.
2. **Perceptual Hash (pHash) Clustering**: Detects near-duplicates differing only by slight compression, crop, or rotation.
3. **Regex-Based Patient Extraction**: Automatically parses patient IDs from file nomenclature across EyeQ, DeepDRiD, and clinical datasets.
4. **Grouped Split Manifest Generator**: Enforces 0% patient identity overlap between training, validation, and testing sets.
