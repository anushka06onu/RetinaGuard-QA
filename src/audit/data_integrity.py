"""Data integrity inspection for image formats, bit-depths, channels, and corrupt headers."""

from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image, ImageOps
import numpy as np


def inspect_image_file(file_path: Union[str, Path]) -> Dict[str, Union[bool, int, float, str, Optional[str]]]:
    """Inspect an individual image file for physical integrity, color channels, and basic metrics."""
    path = Path(file_path)
    result = {
        "file_name": path.name,
        "is_valid": False,
        "width": 0,
        "height": 0,
        "aspect_ratio": 0.0,
        "channels": 0,
        "mode": "",
        "format": "",
        "file_size_bytes": 0,
        "is_corrupt": False,
        "error_message": None
    }

    if not path.exists():
        result["error_message"] = "File does not exist"
        result["is_corrupt"] = True
        return result

    result["file_size_bytes"] = path.stat().st_size
    if result["file_size_bytes"] == 0:
        result["error_message"] = "File is zero bytes (empty)"
        result["is_corrupt"] = True
        return result

    try:
        with Image.open(path) as img:
            img.verify()

        # Reopen after verify()
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            w, h = img.size
            result["width"] = w
            result["height"] = h
            result["aspect_ratio"] = round(w / max(1, h), 4)
            result["mode"] = img.mode
            result["format"] = img.format or path.suffix.replace(".", "").upper()
            result["channels"] = len(img.getbands())
            result["is_valid"] = True

    except Exception as e:
        result["is_corrupt"] = True
        result["error_message"] = str(e)

    return result


def validate_dataset_directory(
    image_dir: Union[str, Path], 
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
) -> Dict[str, Union[int, List[Dict]]]:
    """Inspect all images in a directory, checking for corruptions and size statistics."""
    dir_path = Path(image_dir)
    image_files = [p for p in dir_path.rglob("*") if p.suffix.lower() in extensions]

    valid_count = 0
    corrupt_files = []
    resolutions = []

    for img_p in image_files:
        info = inspect_image_file(img_p)
        if info["is_valid"]:
            valid_count += 1
            resolutions.append((info["width"], info["height"]))
        else:
            corrupt_files.append(info)

    return {
        "total_scanned": len(image_files),
        "valid_count": valid_count,
        "corrupt_count": len(corrupt_files),
        "corrupt_details": corrupt_files,
        "sample_resolutions": resolutions[:10]
    }
