"""Data adapters constructing canonical manifest CSVs per blueprint Section 8."""

import re
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from PIL import Image

from src.retinaguard.utils.hashing import compute_sha256

MANIFEST_COLUMNS = [
    "dataset",
    "image_id",
    "patient_id",
    "eye",
    "path",
    "width",
    "height",
    "sha256",
    "quality_raw",
    "quality_canonical",
    "overall_quality_raw",
    "overall_quality_canonical",
    "artifact",
    "clarity",
    "field_definition",
    "source_split",
    "label_available",
]

EXCLUSION_COLUMNS = [
    "dataset",
    "image_id",
    "path",
    "reason",
    "detail",
    "source_split",
]


def extract_patient_and_eye(filename_or_id: str) -> tuple[Optional[str], Optional[str]]:
    """Extract patient ID and eye orientation (left/right) from common fundus filename schemas."""
    stem = Path(filename_or_id).stem
    eye = None
    if "left" in stem.lower() or "_os" in stem.lower() or "os_" in stem.lower():
        eye = "left"
    elif "right" in stem.lower() or "_od" in stem.lower() or "od_" in stem.lower():
        eye = "right"

    # Match patient ID (digits or prefixed string)
    match = re.match(
        r"^([a-zA-Z0-9]+)_(left|right|OD|OS|1|2|macula|optic_disc)", stem, re.IGNORECASE
    )
    if match:
        return match.group(1), eye

    match_leading = re.match(r"^([a-zA-Z]*\d+)", stem)
    if match_leading:
        return match_leading.group(1), eye

    return stem, eye


def parse_eyeq_metadata(
    csv_path: Union[str, Path],
    images_dir: Union[str, Path],
    source_split: str = "train",
    exclusions_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Parse EyeQ raw labels CSV into canonical schema with explicit exclusion auditing."""
    df_raw = pd.read_csv(csv_path)
    img_dir = Path(images_dir)
    rows = []
    exclusions = []

    # EyeQ label map: 0 -> good, 1 -> usable, 2 -> reject
    label_map = {
        0: "good",
        1: "usable",
        2: "reject",
        "0": "good",
        "1": "usable",
        "2": "reject",
        "good": "good",
        "usable": "usable",
        "reject": "reject",
        "Good": "good",
        "Usable": "usable",
        "Reject": "reject",
    }

    for _, r in df_raw.iterrows():
        img_name = str(r.get("image", r.get("image_id", r.iloc[0])))
        if not img_name.endswith((".jpeg", ".jpg", ".png")):
            img_name = f"{img_name}.jpeg"

        img_path = img_dir / img_name
        rel_path = str(img_path)
        p_id, eye = extract_patient_and_eye(img_name)

        if not img_path.is_file():
            exclusions.append(
                {
                    "dataset": "eyeq",
                    "image_id": Path(img_name).stem,
                    "path": rel_path,
                    "reason": "missing_image",
                    "detail": f"File does not exist at {img_path}",
                    "source_split": source_split,
                }
            )
            continue

        try:
            sha = compute_sha256(img_path)
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as exc:
            exclusions.append(
                {
                    "dataset": "eyeq",
                    "image_id": Path(img_name).stem,
                    "path": rel_path,
                    "reason": "unreadable_image",
                    "detail": f"{type(exc).__name__}: {str(exc)}",
                    "source_split": source_split,
                }
            )
            continue

        raw_label = r.get("quality", r.get("label", None))
        if pd.notna(raw_label):
            if raw_label in label_map:
                canonical_label = label_map[raw_label]
            else:
                raise ValueError(
                    f"Unrecognized EyeQ quality label '{raw_label}' for image '{img_name}' in {csv_path}"
                )
        else:
            canonical_label = None

        rows.append(
            {
                "dataset": "eyeq",
                "image_id": Path(img_name).stem,
                "patient_id": p_id,
                "eye": eye,
                "path": rel_path,
                "width": w,
                "height": h,
                "sha256": sha,
                "quality_raw": raw_label,
                "quality_canonical": canonical_label,
                "overall_quality_raw": None,
                "overall_quality_canonical": None,
                "artifact": None,
                "clarity": None,
                "field_definition": None,
                "source_split": source_split,
                "label_available": canonical_label is not None,
            }
        )

    if exclusions_csv is not None or len(exclusions) > 0:
        ex_path = Path(exclusions_csv or "data/manifests/eyeq_exclusions.csv")
        ex_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS).to_csv(ex_path, index=False)

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def parse_deepdrid_metadata(
    csv_path: Union[str, Path],
    images_dir: Union[str, Path],
    source_split: str = "train",
    exclusions_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Parse DeepDRiD raw fold labels CSV into canonical schema with strict label validation."""
    df_raw = pd.read_csv(csv_path)
    img_dir = Path(images_dir)
    rows = []
    exclusions = []

    # Overall Quality canonical mapping: 0 -> good, 1 -> usable, 2 -> reject
    oq_map = {
        0: "good",
        1: "usable",
        2: "reject",
        "0": "good",
        "1": "usable",
        "2": "reject",
        "Good": "good",
        "Usable": "usable",
        "Reject": "reject",
        "good": "good",
        "usable": "usable",
        "reject": "reject",
    }

    # Ordinal Attribute mappings (0=none/clear/adequate, 1=mild/moderate, 2=severe/poor)
    # Supports 3-level (0, 1, 2) and 5-level (0-4) raw ratings per deepdrid_label_mapping.yaml
    attr_map = {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        "0": 0,
        "1": 1,
        "2": 1,
        "3": 2,
        "4": 2,
    }
    clarity_fld_map = {
        0: 0,
        1: 0,
        2: 1,
        3: 2,
        4: 2,
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 2,
    }

    for _, r in df_raw.iterrows():
        # Match flexible column header variations
        img_name = str(
            r.get(
                "image_id",
                r.get(
                    "image",
                    r.get(
                        "Image_id",
                        r.get("Image", r.iloc[0]),
                    ),
                ),
            )
        )
        if not img_name.endswith((".jpg", ".jpeg", ".png")):
            img_name = f"{img_name}.jpg"

        img_path = img_dir / img_name
        rel_path = str(img_path)
        p_id, eye = extract_patient_and_eye(img_name)

        if not img_path.is_file():
            exclusions.append(
                {
                    "dataset": "deepdrid",
                    "image_id": Path(img_name).stem,
                    "path": rel_path,
                    "reason": "missing_image",
                    "detail": f"File does not exist at {img_path}",
                    "source_split": source_split,
                }
            )
            continue

        try:
            sha = compute_sha256(img_path)
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as exc:
            exclusions.append(
                {
                    "dataset": "deepdrid",
                    "image_id": Path(img_name).stem,
                    "path": rel_path,
                    "reason": "unreadable_image",
                    "detail": f"{type(exc).__name__}: {str(exc)}",
                    "source_split": source_split,
                }
            )
            continue

        raw_oq = r.get(
            "overall_quality",
            r.get("Overall quality", r.get("Overall_quality", r.get("quality", None))),
        )
        if pd.notna(raw_oq):
            if raw_oq in oq_map:
                canonical_oq = oq_map[raw_oq]
            else:
                raise ValueError(
                    f"Unrecognized DeepDRiD overall quality '{raw_oq}' for image '{img_name}' in {csv_path}"
                )
        else:
            canonical_oq = None

        def parse_deepdrid_attr(val, col_name, mapping):
            if pd.notna(val):
                if val in mapping:
                    return mapping[val]
                raise ValueError(
                    f"Unrecognized DeepDRiD {col_name} '{val}' for image '{img_name}' in {csv_path}"
                )
            return None

        artifact = parse_deepdrid_attr(
            r.get("artifact", r.get("Artifact", r.get("artifacts", None))), "artifact", attr_map
        )
        clarity = parse_deepdrid_attr(
            r.get("clarity", r.get("Clarity", None)), "clarity", clarity_fld_map
        )
        field_def = parse_deepdrid_attr(
            r.get("field_definition", r.get("Field definition", r.get("Field_definition", None))),
            "field_definition",
            clarity_fld_map,
        )

        rows.append(
            {
                "dataset": "deepdrid",
                "image_id": Path(img_name).stem,
                "patient_id": p_id,
                "eye": eye,
                "path": rel_path,
                "width": w,
                "height": h,
                "sha256": sha,
                "quality_raw": None,
                "quality_canonical": None,
                "overall_quality_raw": raw_oq,
                "overall_quality_canonical": canonical_oq,
                "artifact": artifact,
                "clarity": clarity,
                "field_definition": field_def,
                "source_split": source_split,
                "label_available": canonical_oq is not None,
            }
        )

    if exclusions_csv is not None or len(exclusions) > 0:
        ex_path = Path(exclusions_csv or "data/manifests/deepdrid_exclusions.csv")
        ex_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS).to_csv(ex_path, index=False)

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_canonical_manifest(dfs: List[pd.DataFrame], output_csv: Union[str, Path]) -> pd.DataFrame:
    """Concatenate and write canonical manifest CSV."""
    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.concat(dfs, ignore_index=True)
    manifest.to_csv(out_p, index=False)
    return manifest
