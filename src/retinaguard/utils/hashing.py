"""Cryptographic hashing utilities for dataset integrity and artifact verification."""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(file_path: Union[str, Path], expected_sha256: str) -> bool:
    """Verify that file checksum matches the expected SHA-256."""
    actual_sha = compute_sha256(file_path)
    return actual_sha.lower() == expected_sha256.lower()
