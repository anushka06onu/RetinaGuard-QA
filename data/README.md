# Data Directory & Manifest Protocol

This directory contains dataset documentation, checksum manifests, and patient-safe split indices used by RetinaGuard-QA.

## Directory Structure
- `dataset_cards/`: Standardized dataset cards describing origins, ethics, licenses, acquisition devices, and quality distributions.
- `split_manifests/`: Cryptographically locked CSV/JSON manifests defining `train`, `val`, and `test` partitions without patient overlap.
- `synthetic/`: Generated controlled corruption benchmarks.

## Patient Leakage Isolation
Medical imaging benchmarks frequently suffer from patient-level data leakage when multiple images from the same subject/eye are assigned to both training and test sets. 
RetinaGuard-QA enforces strict patient grouping: all images sharing the same patient identifier or capture session are locked into a single partition.
