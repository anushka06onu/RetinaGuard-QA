"""Zero-shot cross-dataset transfer evaluation engine (Internal EyeQ -> External DeepDRiD)."""

from typing import Dict, List, Callable, Tuple, Any
import numpy as np
from PIL import Image
import torch

from .metrics_engine import compute_multiclass_metrics

def evaluate_cross_dataset_transfer(
    model: torch.nn.Module,
    dataset_name: str,
    images: List[Image.Image],
    labels: np.ndarray,
    transform_fn: Callable[[Image.Image], torch.Tensor],
    batch_size: int = 32,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Run zero-shot evaluation on an external out-of-domain retinal dataset."""
    model.eval()
    model.to(device)

    all_logits = []
    n = len(images)

    for i in range(0, n, batch_size):
        batch_imgs = images[i:i + batch_size]
        tensors = [transform_fn(img) for img in batch_imgs]
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            out = model(batch_tensor)
            logits = out["grade_logits"].cpu().numpy()
            all_logits.append(logits)

    all_logits = np.concatenate(all_logits, axis=0)
    metrics = compute_multiclass_metrics(all_logits, labels, is_probabilities=False)
    metrics["dataset_name"] = dataset_name
    metrics["sample_size"] = n

    return metrics
