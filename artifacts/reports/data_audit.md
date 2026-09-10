# RetinaGuard-QA Data & Leakage Audit Report

## Summary
- **Isolation Status:** ✅ PASSED (Zero Leakage)
- **Splits Evaluated:** deepdrid_train, deepdrid_external_test, deepdrid_val

## Partition Counts
| Split | Total Images | Unique Patients |
|---|---|---|
| deepdrid_train | 1200 | 300 |
| deepdrid_external_test | 400 | 100 |
| deepdrid_val | 400 | 100 |

## Patient Leakage Details
No patient identities cross partition boundaries (0% patient leakage).

## Hash (SHA-256) Leakage Details
No duplicate files or identical images cross partition boundaries.
