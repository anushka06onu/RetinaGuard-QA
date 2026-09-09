"""Experiment 03: Multi-Task Quality Architecture Training and Model Export (RetinaGuardNet)."""

from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.multi_task_head import RetinaGuardNet, MultiTaskLoss
from src.inference.onnx_exporter import export_to_onnx, verify_onnx_numerical_parity
from src.evaluation.metrics_engine import compute_multiclass_metrics, compute_defect_metrics


def train_and_export_retinaguard(
    backbone_name: str = "mobilenetv3_large_100",
    epochs: int = 5,
    batch_size: int = 16,
    save_dir: str = "results/models"
):
    print("=== Running Experiment 03: Multi-Task RetinaGuardNet Training & Export ===")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize model and multi-task loss
    model = RetinaGuardNet(backbone_name=backbone_name, pretrained=False)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Simulated synthetic training dataset
    num_samples = 120
    X = torch.randn(num_samples, 3, 384, 384)
    y_grade = torch.randint(0, 3, (num_samples,))
    y_defects = (torch.rand(num_samples, 6) > 0.7).float()
    y_score = torch.rand(num_samples, 1) * 100.0

    dataset = TensorDataset(X, y_grade, y_defects, y_score)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    t_start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_x, b_grade, b_defects, b_score in loader:
            optimizer.zero_grad()
            preds = model(b_x)
            loss_dict = criterion(preds, b_grade, b_defects, b_score)
            loss_dict["loss_total"].backward()
            optimizer.step()
            epoch_loss += loss_dict["loss_total"].item()

        print(f"Epoch {epoch+1}/{epochs} - Total Loss: {epoch_loss/len(loader):.4f}")

    train_time = time.time() - t_start
    print(f"Training completed in {train_time:.2f}s")

    # Save PyTorch Checkpoint
    pt_path = save_path / "retinaguard_net.pt"
    torch.save({"model_state_dict": model.state_dict(), "version": "1.0.0"}, pt_path)
    print(f"Saved PyTorch Checkpoint to: {pt_path}")

    # Export to ONNX
    onnx_path = save_path / "retinaguard_net.onnx"
    export_to_onnx(model, onnx_path)
    print(f"Exported ONNX Model to: {onnx_path}")

    # Verify ONNX Parity
    parity = verify_onnx_numerical_parity(model, onnx_path)
    print(f"ONNX Parity Verified: {parity['is_parity_verified']} (Max Error: {parity['max_absolute_error']:.6f})")

    return model, pt_path, onnx_path


if __name__ == "__main__":
    train_and_export_retinaguard()
