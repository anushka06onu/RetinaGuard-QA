"""Experiment 06: Controlled Synthetic Robustness Corruption Sweep (8 Types x 5 Severities)."""

import numpy as np
from PIL import Image
import torch

from src.models.multi_task_head import RetinaGuardNet
from src.preprocessing.transforms import preprocess_fundus_image
from src.evaluation.robustness_benchmark import run_robustness_sweep, compute_relative_robustness


def create_clean_test_cohort(n_samples: int = 30):
    images = []
    labels = []
    for i in range(n_samples):
        grade = i % 3
        arr = np.zeros((384, 384, 3), dtype=np.uint8)
        y, x = np.ogrid[:384, :384]
        mask = (x - 192)**2 + (y - 192)**2 < 170**2
        arr[mask, 0] = 200 if grade == 0 else 140 if grade == 1 else 60
        arr[mask, 1] = 90 if grade == 0 else 60 if grade == 1 else 25
        arr[mask, 2] = 30 if grade == 0 else 20 if grade == 1 else 10
        images.append(Image.fromarray(arr))
        labels.append(grade)
    return images, np.array(labels)


def run_robustness_experiment():
    print("=== Running Experiment 06: Controlled Optical Robustness Sweep ===")
    model = RetinaGuardNet(pretrained=False)
    model.eval()

    def transform_fn(img):
        return preprocess_fundus_image(img).squeeze(0)

    clean_images, labels = create_clean_test_cohort(30)
    
    # Baseline clean performance
    with torch.no_grad():
        batch = torch.stack([transform_fn(img) for img in clean_images])
        clean_logits = model(batch)["grade_logits"].cpu().numpy()
    
    from src.evaluation.metrics_engine import compute_multiclass_metrics
    clean_metrics = compute_multiclass_metrics(clean_logits, labels)
    print(f"Clean Benchmark Macro-F1: {clean_metrics['macro_f1']:.4f}")

    # Robustness sweep
    sweep_results = run_robustness_sweep(model, clean_images, labels, transform_fn)
    
    print("\nCorruption Robustness Summary (Macro-F1 across Severity 1 -> 5):")
    print("-" * 65)
    print(f"{'Corruption Type':<30} | Sev 1  | Sev 3  | Sev 5")
    print("-" * 65)
    for corr_name, sevs in sweep_results.items():
        print(f"{corr_name:<30} | {sevs[1]['macro_f1']:.3f}  | {sevs[3]['macro_f1']:.3f}  | {sevs[5]['macro_f1']:.3f}")
    print("-" * 65)

    rob_index = compute_relative_robustness(clean_metrics, sweep_results)
    print(f"\nRelative Robustness Index (RRI): {rob_index['relative_robustness_index']:.4f}")
    print(f"Average Robustness Drop:       {rob_index['absolute_performance_drop']:.4f}")

    return sweep_results, rob_index


if __name__ == "__main__":
    run_robustness_experiment()
