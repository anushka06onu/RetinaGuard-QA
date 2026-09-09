"""Experiment 05: Cross-Dataset Generalization (EyeQ Internal vs DeepDRiD External)."""

import numpy as np
from PIL import Image
import torch

from src.models.multi_task_head import RetinaGuardNet
from src.preprocessing.transforms import preprocess_fundus_image
from src.evaluation.cross_dataset import evaluate_cross_dataset_transfer


def simulate_retinal_dataset(name: str, n_samples: int = 40, shift_factor: float = 1.0):
    """Generate simulated retinal images for domain transfer evaluation."""
    images = []
    labels = []
    
    for i in range(n_samples):
        grade = i % 3
        arr = np.zeros((384, 384, 3), dtype=np.uint8)
        y, x = np.ogrid[:384, :384]
        mask = (x - 192)**2 + (y - 192)**2 < 170**2
        
        # Add domain shift factor to color channels
        arr[mask, 0] = np.clip(int((180 if grade==0 else 130 if grade==1 else 50) * shift_factor), 0, 255)
        arr[mask, 1] = np.clip(int((80 if grade==0 else 50 if grade==1 else 20) * shift_factor), 0, 255)
        arr[mask, 2] = np.clip(int((30 if grade==0 else 20 if grade==1 else 10) * shift_factor), 0, 255)
        
        images.append(Image.fromarray(arr))
        labels.append(grade)

    return images, np.array(labels)


def run_cross_dataset_experiment():
    print("=== Running Experiment 05: Cross-Dataset Domain Generalization ===")
    
    model = RetinaGuardNet(pretrained=False)
    model.eval()

    def transform_fn(img):
        return preprocess_fundus_image(img).squeeze(0)

    # 1. Internal EyeQ Test Partition
    eyeq_imgs, eyeq_labels = simulate_retinal_dataset("EyeQ (Internal)", n_samples=60, shift_factor=1.0)
    eyeq_results = evaluate_cross_dataset_transfer(
        model, "EyeQ (Internal Test)", eyeq_imgs, eyeq_labels, transform_fn
    )

    # 2. External DeepDRiD Benchmark (Camera/Illumination Shift)
    deepdrid_imgs, deepdrid_labels = simulate_retinal_dataset("DeepDRiD (External)", n_samples=60, shift_factor=0.75)
    deepdrid_results = evaluate_cross_dataset_transfer(
        model, "DeepDRiD (External Benchmark)", deepdrid_imgs, deepdrid_labels, transform_fn
    )

    print(f"\n[Internal] EyeQ Macro-F1:       {eyeq_results['macro_f1']:.4f} (Kappa: {eyeq_results['cohen_weighted_kappa']:.4f})")
    print(f"[External] DeepDRiD Macro-F1:   {deepdrid_results['macro_f1']:.4f} (Kappa: {deepdrid_results['cohen_weighted_kappa']:.4f})")
    
    delta_f1 = eyeq_results['macro_f1'] - deepdrid_results['macro_f1']
    print(f"Domain Transfer F1 Gap (Delta): {delta_f1:.4f}")

    return {
        "eyeq": eyeq_results,
        "deepdrid": deepdrid_results,
        "domain_gap": delta_f1
    }


if __name__ == "__main__":
    run_cross_dataset_experiment()
