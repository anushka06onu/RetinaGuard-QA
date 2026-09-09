"""Experiment 01: Classical Feature-Based Image Quality Assessment (IQA) Baselines."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

from src.models.baselines import ClassicalQualityClassifier
from src.evaluation.metrics_engine import compute_multiclass_metrics


def generate_synthetic_benchmark_cohort(n_samples: int = 150):
    """Generate realistic synthetic fundus samples across 3 quality grades for benchmark verification."""
    images = []
    labels = []
    
    for i in range(n_samples):
        # Grade 0: Good (sharp edges, healthy illumination, orange-red fundus profile)
        # Grade 1: Usable (mild blur or slight illumination non-uniformity)
        # Grade 2: Reject (severe blur, dark, or saturated)
        grade = i % 3
        
        arr = np.zeros((384, 384, 3), dtype=np.uint8)
        # Create circular fundus mask
        y, x = np.ogrid[:384, :384]
        dist = np.sqrt((x - 192)**2 + (y - 192)**2)
        mask = dist < 170

        if grade == 0:
            arr[mask, 0] = np.random.randint(180, 230)
            arr[mask, 1] = np.random.randint(70, 110)
            arr[mask, 2] = np.random.randint(20, 50)
            # Add vessel-like texture
            arr[150:240, 150:240, 1] += 30
        elif grade == 1:
            arr[mask, 0] = np.random.randint(120, 170)
            arr[mask, 1] = np.random.randint(40, 80)
            arr[mask, 2] = np.random.randint(10, 40)
        else: # Reject
            arr[mask, 0] = np.random.randint(30, 70)
            arr[mask, 1] = np.random.randint(10, 30)
            arr[mask, 2] = np.random.randint(5, 20)

        img = Image.fromarray(arr)
        images.append(img)
        labels.append(grade)

    return images, np.array(labels)


def main():
    print("=== Running Experiment 01: Classical Feature Baselines ===")
    images, labels = generate_synthetic_benchmark_cohort(180)
    
    # Split train/test
    train_imgs, test_imgs = images[:120], images[120:]
    train_y, test_y = labels[:120], labels[120:]

    clf = ClassicalQualityClassifier(classifier_type="random_forest")
    clf.fit(train_imgs, train_y)
    
    preds = clf.predict(test_imgs)
    probs = clf.predict_proba(test_imgs)

    metrics = compute_multiclass_metrics(probs, test_y, is_probabilities=True)
    print(f"Classical RF Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Balanced Accuracy:    {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa:        {metrics['cohen_weighted_kappa']:.4f}")
    print("Confusion Matrix:\n", np.array(metrics['confusion_matrix']))


if __name__ == "__main__":
    main()
