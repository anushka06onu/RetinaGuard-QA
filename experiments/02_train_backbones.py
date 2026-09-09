"""Experiment 02: Single-Task Deep CNN Encoder Baselines (MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny)."""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.models.baselines import SingleTaskQualityClassifier
from src.evaluation.metrics_engine import compute_multiclass_metrics


def run_single_task_benchmark(backbone_names=["mobilenetv3_large_100", "efficientnet_b0"], num_epochs=3):
    print("=== Running Experiment 02: Single-Task Deep Encoders ===")
    
    # Synthetic batch generator for architecture comparison
    X = torch.randn(60, 3, 384, 384)
    y = torch.randint(0, 3, (60,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=12, shuffle=True)

    results = {}
    for name in backbone_names:
        print(f"\nEvaluating Backbone: {name}")
        model = SingleTaskQualityClassifier(backbone_name=name, pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        t0 = time.time()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(X).cpu().numpy()
        metrics = compute_multiclass_metrics(test_logits, y.numpy(), is_probabilities=False)
        metrics["training_time_sec"] = round(time.time() - t0, 2)
        results[name] = metrics
        print(f"  --> {name} Final Macro-F1: {metrics['macro_f1']:.4f}")

    return results


if __name__ == "__main__":
    run_single_task_benchmark()
