# RetinaGuard-QA Cross-Split Isolation & Data Audit Report

## Summary
- **Schema Version:** 1.0
- **Generated At UTC:** 2026-09-10T22:02:17.817611+00:00
- **Git Commit:** `cb3c1c6ffcce8c658d3322032506f7babc7a28d0`
- **Audit Script SHA-256:** `1e6e65f11d4181a5faf33f65822b7bd8bbf9e5735be1bda5b6544dff91b3dcaf`
- **Mapping Schema SHA-256:** `73798da8cea7949158c54113ef5712f4af95c0e62d640d33f360febe2ec6dc03`
- **Cross-Split Isolation:** ✅ PASSED (Zero Leakage)
- **Intra-Split Uniqueness:** ✅ PASSED (No Duplicates)
- **Image Integrity:** ✅ PASSED (All Verified)
- **Overall Audit Status:** ✅ PASSED
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
