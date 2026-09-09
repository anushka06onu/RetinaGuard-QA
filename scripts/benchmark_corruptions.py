"""Run controlled optical corruption robustness sweep (10 types x 5 severities)."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel
from src.retinaguard.evaluation.corruptions import SyntheticCorruptionSuite
from src.retinaguard.data.preprocessing import preprocess_image_canonical
from src.retinaguard.evaluation.metrics import compute_quality_metrics


def main():
    parser = argparse.ArgumentParser(description="Run synthetic corruption robustness benchmark.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/best.ckpt")
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print("=== Running Controlled Optical Corruption Benchmark (10 Types x 5 Severities) ===")
    model = RetinaGuardMultiTaskModel(pretrained=False)
    if Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
    model.eval()

    # Generate synthetic clean test set
    n_samples = 30
    clean_images = []
    labels = []
    for i in range(n_samples):
        arr = np.zeros((384, 384, 3), dtype=np.uint8)
        grade = i % 3
        arr[:, :, 0] = 200 if grade == 0 else 130 if grade == 1 else 60
        arr[:, :, 1] = 90 if grade == 0 else 60 if grade == 1 else 20
        arr[:, :, 2] = 30 if grade == 0 else 20 if grade == 1 else 10
        clean_images.append(Image.fromarray(arr))
        labels.append(grade)
    labels = np.array(labels)

    corruptions = SyntheticCorruptionSuite.get_all_names()
    results = {}

    for c_name in corruptions:
        results[c_name] = {}
        for sev in [1, 2, 3, 4, 5]:
            tensors = []
            for img in clean_images:
                c_img = SyntheticCorruptionSuite.apply(img, c_name, severity=sev)
                t = preprocess_image_canonical(c_img).squeeze(0)
                tensors.append(t)
            batch = torch.stack(tensors)
            with torch.no_grad():
                logits = model(batch)["quality_logits"].cpu().numpy()
            m = compute_quality_metrics(logits, labels, is_logits=True)
            results[c_name][str(sev)] = {
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "quadratic_weighted_kappa": m["quadratic_weighted_kappa"]
            }

    with open(out_p / "corruptions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluated {len(corruptions)} corruption types. Saved results to: {out_p / 'corruptions.json'}")


if __name__ == "__main__":
    main()
