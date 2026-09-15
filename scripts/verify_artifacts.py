#!/usr/bin/env python3
"""Verify integrity, cryptographic checksums, and lineage consistency of repository artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


MANDATORY_ARTIFACTS = [
    "models/model.onnx",
    "models/model.onnx.data",
    "models/preprocessing.json",
    "models/calibration_metadata.json",
    "models/onnx_manifest.json",
    "metrics/onnx_parity.json",
    "reports/data_audit.json",
    "reports/cross_split_isolation_audit.json",
    "reports/data_flow_report.json",
    "provenance/environment.txt",
]


def verify_checksum_manifest(
    manifest_path: Path,
    artifacts_dir: Path,
    repo_root: Path,
    require_mandatory: bool = True,
    check_lineage: bool = True,
) -> Dict[str, Any]:
    """Verify all listed hashes in SHA256SUMS and validate artifact consistency."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checksum manifest not found: {manifest_path}")

    report: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "total_listed_files": 0,
        "valid_files": 0,
        "mismatched_files": 0,
        "missing_files": 0,
        "unlisted_mandatory_files": [],
        "lineage_errors": [],
        "errors": [],
        "passed": False,
    }

    listed_rel_paths: List[str] = []

    with open(manifest_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                report["errors"].append(f"Line {line_num}: Malformed line '{line}'")
                continue

            expected_sha, rel_file = parts[0], parts[1].strip()
            listed_rel_paths.append(rel_file)
            report["total_listed_files"] += 1

            # Resolve file path: check artifacts_dir first, then repo_root
            candidate_p = artifacts_dir / rel_file
            if not candidate_p.is_file():
                candidate_p = repo_root / rel_file

            if not candidate_p.is_file():
                report["missing_files"] += 1
                report["errors"].append(f"Missing file: {rel_file}")
                continue

            actual_sha = compute_sha256(candidate_p)
            if actual_sha == expected_sha:
                report["valid_files"] += 1
            else:
                report["mismatched_files"] += 1
                report["errors"].append(
                    f"Checksum mismatch for '{rel_file}': actual {actual_sha}, expected {expected_sha}"
                )

    # Check mandatory files listed
    if require_mandatory:
        for mand in MANDATORY_ARTIFACTS:
            if mand not in listed_rel_paths:
                # Check if it's in repo root or artifacts
                mand_p = artifacts_dir / mand
                if not mand_p.is_file():
                    mand_p = repo_root / mand
                if mand_p.is_file():
                    report["unlisted_mandatory_files"].append(mand)
                    report["errors"].append(
                        f"Mandatory artifact '{mand}' exists but is missing from SHA256SUMS"
                    )

    # Lineage and cross-artifact consistency checks
    if check_lineage:
        onnx_p = artifacts_dir / "models/model.onnx"
        manifest_p = artifacts_dir / "models/onnx_manifest.json"
        calib_p = artifacts_dir / "models/calibration_metadata.json"
        parity_p = artifacts_dir / "metrics/onnx_parity.json"

        if onnx_p.is_file() and manifest_p.is_file():
            actual_onnx_sha = compute_sha256(onnx_p)
            try:
                with open(manifest_p) as f:
                    m_data = json.load(f)
                rec_sha = m_data.get("onnx_file_sha256")
                if rec_sha and rec_sha != actual_onnx_sha:
                    err = f"ONNX manifest hash mismatch: recorded {rec_sha}, actual model.onnx {actual_onnx_sha}"
                    report["lineage_errors"].append(err)
                    report["errors"].append(err)
            except Exception as exc:
                report["lineage_errors"].append(f"Failed to parse onnx_manifest.json: {exc}")

        if onnx_p.is_file() and calib_p.is_file():
            actual_onnx_sha = compute_sha256(onnx_p)
            try:
                with open(calib_p) as f:
                    c_data = json.load(f)
                calib_onnx_sha = c_data.get("onnx_model_sha256")
                if calib_onnx_sha and calib_onnx_sha != actual_onnx_sha:
                    err = f"Calibration metadata ONNX hash mismatch: recorded {calib_onnx_sha}, actual model.onnx {actual_onnx_sha}"
                    report["lineage_errors"].append(err)
                    report["errors"].append(err)
            except Exception as exc:
                report["lineage_errors"].append(f"Failed to parse calibration_metadata.json: {exc}")

        if parity_p.is_file():
            try:
                with open(parity_p) as f:
                    p_data = json.load(f)
                if not p_data.get("parity_passed", False):
                    err = "ONNX parity verification record metrics/onnx_parity.json indicates parity_passed is False"
                    report["lineage_errors"].append(err)
                    report["errors"].append(err)
            except Exception as exc:
                report["lineage_errors"].append(f"Failed to parse onnx_parity.json: {exc}")

    report["passed"] = (
        report["mismatched_files"] == 0
        and report["missing_files"] == 0
        and len(report["unlisted_mandatory_files"]) == 0
        and len(report["lineage_errors"]) == 0
        and len(report["errors"]) == 0
    )

    return report


def generate_checksum_manifest(
    artifacts_dir: Path,
    output_path: Path,
) -> int:
    """Generate canonical SHA256SUMS file covering all relevant artifacts."""
    files_to_hash: List[Path] = []

    search_subdirs = ["models", "metrics", "reports", "figures", "provenance", "campaigns"]
    for sub in search_subdirs:
        sub_p = artifacts_dir / sub
        if sub_p.is_dir():
            for p in sorted(sub_p.rglob("*")):
                if p.is_file() and p.name != "SHA256SUMS" and not p.name.startswith("."):
                    files_to_hash.append(p)

    # Sort deterministically
    files_to_hash = sorted(
        list(set(files_to_hash)), key=lambda x: str(x.relative_to(artifacts_dir))
    )

    lines = []
    for f in files_to_hash:
        rel_str = str(f.relative_to(artifacts_dir))
        sha = compute_sha256(f)
        lines.append(f"{sha}  {rel_str}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        out.writelines(lines)

    print(f"Wrote {len(lines)} artifact hashes to {output_path}")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or regenerate artifact checksum manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/provenance/SHA256SUMS"),
        help="Path to SHA256SUMS",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root directory of artifacts",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root directory",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate SHA256SUMS manifest instead of verifying",
    )
    args = parser.parse_args()

    if args.generate:
        generate_checksum_manifest(args.artifacts_dir, args.manifest)
        print("Regeneration complete.")
        return 0

    print(f"Verifying artifacts using manifest '{args.manifest}'...")
    try:
        report = verify_checksum_manifest(args.manifest, args.artifacts_dir, args.repo_root)
    except Exception as exc:
        print(f"FAILED: Verification error: {exc}")
        return 1

    print(f"Total listed:    {report['total_listed_files']}")
    print(f"Valid hashes:    {report['valid_files']}")
    print(f"Mismatches:      {report['mismatched_files']}")
    print(f"Missing files:   {report['missing_files']}")

    if report["passed"]:
        print("SUCCESS: All artifact checksums and lineage invariants verified.")
        return 0
    else:
        print("FAILED: Artifact verification failed with the following errors:")
        for err in report["errors"]:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
