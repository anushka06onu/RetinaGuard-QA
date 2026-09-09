"""Systematic robustness corruption sweep benchmarking model resilience under optical degradations."""

from typing import Dict, List, Callable, Tuple
import numpy as np
from PIL import Image
import torch

from src.preprocessing.synthetic_degradations import SyntheticCorruptionEngine
from .metrics_engine import compute_multiclass_metrics


def run_robustness_sweep(
    model: torch.nn.Module,
    test_images: List[Image.Image],
    test_labels: np.ndarray,
    transform_fn: Callable[[Image.Image], torch.Tensor],
    device: str = "cpu"
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Evaluate quality prediction degradation across 8 corruption types and 5 severity levels.

    Returns:
        Nested dict: {corruption_type: {severity_1..5: metrics_dict}}
    """
    model.eval()
    model.to(device)
    results = {}
    corruptions = SyntheticCorruptionEngine.get_all_corruption_names()

    for corr_name in corruptions:
        results[corr_name] = {}
        for sev in [1, 2, 3, 4, 5]:
            corrupted_tensors = []
            for img in test_images:
                c_img = SyntheticCorruptionEngine.apply(img, corr_name, severity=sev)
                tensor = transform_fn(c_img)
                corrupted_tensors.append(tensor)

            batch = torch.stack(corrupted_tensors).to(device)
            with torch.no_grad():
                out = model(batch)
                logits = out["grade_logits"].cpu().numpy()

            metrics = compute_multiclass_metrics(logits, test_labels, is_probabilities=False)
            results[corr_name][sev] = {
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "cohen_weighted_kappa": metrics["cohen_weighted_kappa"]
            }

    return results


def compute_relative_robustness(
    clean_metrics: Dict[str, float], 
    corrupted_results: Dict[str, Dict[int, Dict[str, float]]]
) -> Dict[str, float]:
    """Compute Relative Robustness Index (RRI = mean performance across corruptions / clean performance)."""
    clean_f1 = clean_metrics["macro_f1"]
    all_corrupted_f1s = []

    for corr_name, sev_dict in corrupted_results.items():
        for sev, metrics in sev_dict.items():
            all_corrupted_f1s.append(metrics["macro_f1"])

    mean_corr_f1 = float(np.mean(all_corrupted_f1s))
    rri = float(mean_corr_f1 / max(1e-4, clean_f1))

    return {
        "clean_macro_f1": clean_f1,
        "mean_corrupted_macro_f1": mean_corr_f1,
        "relative_robustness_index": rri,
        "absolute_performance_drop": clean_f1 - mean_corr_f1
    }
