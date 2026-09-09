"""Patient ID extraction and partition leakage verification."""

import re
from pathlib import Path
from typing import Dict, List, Set, Union
import pandas as pd


def extract_patient_id_from_filename(filename_or_path: Union[str, Path]) -> str:
    """Extract patient/subject identifier from common retinal dataset filename conventions.

    Examples:
        - EyePACS / EyeQ: '10_left.jpeg' -> '10'
        - DeepDRiD: '001_1_left.jpg' -> '001'
        - Generic: 'patient_42_OD.png' -> 'patient_42'
    """
    stem = Path(filename_or_path).stem
    # Match pattern like '1234_left' or '1234_right'
    match = re.match(r"^([a-zA-Z0-9]+)_(left|right|OD|OS|1|2|macula|optic_disc)", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Match leading digits or patient ID prefixes like 'P001'
    match_leading = re.match(r"^([a-zA-Z]*\d+)", stem)
    if match_leading:
        return match_leading.group(1)
        
    return stem


def verify_patient_split_isolation(
    splits: Dict[str, List[str]], 
    patient_extractor=extract_patient_id_from_filename
) -> Dict[str, Union[bool, Dict[str, int], Dict[str, List[str]]]]:
    """Verify that no patient identity appears across multiple dataset partitions (train, val, test).

    Args:
        splits: Dictionary mapping split names (e.g. 'train', 'val', 'test') to lists of filenames/paths.
        patient_extractor: Function mapping file path/name to patient ID string.

    Returns:
        Audit report containing pass/fail boolean, patient counts per split, and intersecting leaks.
    """
    patient_sets: Dict[str, Set[str]] = {}
    for split_name, files in splits.items():
        p_ids = {patient_extractor(f) for f in files}
        patient_sets[split_name] = p_ids

    leakages: Dict[str, List[str]] = {}
    split_names = list(splits.keys())
    has_leak = False

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            intersection = patient_sets[s1].intersection(patient_sets[s2])
            if intersection:
                has_leak = True
                leakages[f"{s1}_vs_{s2}"] = sorted(list(intersection))

    return {
        "isolation_passed": not has_leak,
        "patient_counts": {s: len(p_ids) for s, p_ids in patient_sets.items()},
        "image_counts": {s: len(files) for s, files in splits.items()},
        "leaked_patient_ids": leakages
    }
