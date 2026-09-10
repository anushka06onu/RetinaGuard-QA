# RetinaGuard-QA Cross-Split Isolation & Data Audit Report

## Summary
- **Schema Version:** 1.0
- **Generated At UTC:** 2026-09-10T21:51:03.689040+00:00
- **Git Commit:** `2649d84ab437ce25a970de3c892d6c84b329e098`
- **Audit Script SHA-256:** `404a4574b9f995223129a449e11479116a472e3b75409e83cf161e5d1e3deba9`
- **Mapping Schema SHA-256:** `73798da8cea7949158c54113ef5712f4af95c0e62d640d33f360febe2ec6dc03`
- **Isolation Status:** ✅ PASSED (Zero Leakage)
- **Splits Evaluated:** deepdrid_train, deepdrid_external_test, deepdrid_val

## Partition Summary & Cryptographic Hashes
| Split | Split File SHA-256 | Images | Unique Patients | Unique Hashes | Intra-Split Dups |
|---|---|:---:|:---:|:---:|:---:|
| **deepdrid_train** | `d7d85b2b8a8ca99e...` | 1200 | 300 | 1200 | 0 |
| **deepdrid_external_test** | `ed61d43d2b41dda3...` | 400 | 100 | 400 | 0 |
| **deepdrid_val** | `c7d600f26a5624d5...` | 400 | 100 | 400 | 0 |

## Patient Isolation Verification
- ✅ **Zero Patient Leakage:** No patient identities cross partition boundaries (0% patient leakage across all splits).

## Cryptographic Hash (SHA-256) Isolation Verification
- ✅ **Zero Image Leakage:** No duplicate files or identical images cross partition boundaries.

## Comparisons Performed
- `patient_id_cross_split`
- `sha256_cross_split`
- `intra_split_duplicates`
- `image_file_existence_and_hash_verification`
- `label_distribution_verification`
