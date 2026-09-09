"""Cryptographic and perceptual hash auditing for exact and near-duplicate detection."""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from PIL import Image
import imagehash


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute the cryptographic hash of a file on disk.

    Args:
        file_path: Path to the target image or file.
        algorithm: Hashing algorithm ('sha256' or 'md5').

    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_perceptual_hash(image_or_path: Union[str, Path, Image.Image], hash_size: int = 8) -> imagehash.ImageHash:
    """Compute perceptual hash (pHash) for near-duplicate and rotation invariant detection."""
    if isinstance(image_or_path, (str, Path)):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB")
    return imagehash.phash(img, hash_size=hash_size)


def compute_phash_distance(hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
    """Compute Hamming distance between two perceptual hashes."""
    return hash1 - hash2


def find_duplicate_images(
    file_paths: List[Union[str, Path]], 
    check_perceptual: bool = True,
    phash_threshold: int = 4
) -> Dict[str, Union[List[List[str]], Dict[str, List[str]]]]:
    """Scan a collection of image paths to identify exact and near-duplicate images.

    Args:
        file_paths: List of file paths to inspect.
        check_perceptual: Whether to compute pHash clusters.
        phash_threshold: Max Hamming distance to classify as near-duplicate.

    Returns:
        Dictionary with exact duplicates and near-duplicate clusters.
    """
    exact_hashes: Dict[str, List[str]] = {}
    phashes: List[Tuple[str, imagehash.ImageHash]] = []

    for p in file_paths:
        p_str = str(p)
        try:
            sha = compute_file_hash(p_str)
            exact_hashes.setdefault(sha, []).append(p_str)
            
            if check_perceptual:
                ph = compute_perceptual_hash(p_str)
                phashes.append((p_str, ph))
        except Exception as e:
            continue

    exact_duplicates = [paths for paths in exact_hashes.values() if len(paths) > 1]

    near_duplicates = []
    if check_perceptual and len(phashes) > 1:
        n = len(phashes)
        visited = set()
        for i in range(n):
            if i in visited:
                continue
            cluster = [phashes[i][0]]
            for j in range(i + 1, n):
                if j in visited:
                    continue
                dist = compute_phash_distance(phashes[i][1], phashes[j][1])
                if dist <= phash_threshold:
                    cluster.append(phashes[j][0])
                    visited.add(j)
            if len(cluster) > 1:
                visited.add(i)
                near_duplicates.append(cluster)

    return {
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "total_scanned": len(file_paths),
        "unique_exact_count": len(exact_hashes)
    }
