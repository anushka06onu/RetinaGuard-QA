# Data Directory & Manifest Protocol

This directory contains dataset documentation, checksum manifests, and patient-safe split indices used by RetinaGuard-QA.

## Directory Structure
- `dataset_cards/`: Standardized dataset cards describing origins, ethics, licenses, acquisition devices, and quality distributions (EyeQ, DeepDRiD).
- `manifests/`: Raw-to-canonical dataset mapping manifests and exclusions (`deepdrid_manifest.csv`, `eyeq_manifest.csv`, `deepdrid_manifest.metadata.json`, etc.).
- `splits/`: Patient-isolated CSV partitions (`deepdrid_train.csv`, `deepdrid_val.csv`, `deepdrid_external_test.csv`, `eyeq_train.csv`, `eyeq_val.csv`, `eyeq_test.csv`).
- `splits_provenance.json`: Cryptographically locked split provenance manifest recording seeds, partition algorithms, label hashes, record/patient counts, and class distributions.

## Patient Leakage Isolation
Medical imaging benchmarks frequently suffer from patient-level data leakage when multiple images from the same subject/eye are assigned to both training and test sets. 
RetinaGuard-QA enforces strict patient grouping: all images sharing the same patient identifier or capture session are locked into a single partition. Zero patient ID overlap, SHA-256 duplicate overlap, and perceptual duplicate overlap are validated by automated audits.
