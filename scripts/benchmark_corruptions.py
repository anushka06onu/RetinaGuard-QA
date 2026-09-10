"""Run controlled optical corruption robustness sweep (10 types x 5 severities) on genuine held-out test data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from retinaguard.data.preprocessing import preprocess_image_canonical
from retinaguard.evaluation.corruptions import SyntheticCorruptionSuite
from retinaguard.evaluation.metrics import compute_quality_metrics
from retinaguard.models.multitask import RetinaGuardMultiTaskModel


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic corruption robustness benchmark on real test data."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="data/splits/eyeq_test.csv",
        help="Path to EyeQ test split CSV",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/metrics")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum test samples to evaluate per corruption",
    )
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {args.checkpoint}. Robustness benchmark requires trained model weights."
        )

    test_p = Path(args.test_split)
    if not test_p.is_file():
        raise FileNotFoundError(
            f"Test split manifest not found at {args.test_split}. Robustness benchmark requires real test image split."
        )

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Optical Corruption Robustness Benchmark on {test_p} ===")
    model = RetinaGuardMultiTaskModel(pretrained=False)
    state = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    test_df = pd.read_csv(test_p)
    if len(test_df) == 0:
        raise ValueError(f"Test split {test_p} is empty.")

    # Quality label map
    q_map = {"good": 0, "usable": 1, "reject": 2, 0: 0, 1: 1, 2: 2}

    # Load clean real images
    clean_images = []
    labels = []
    sample_df = test_df.head(args.max_samples)

    for _, row in sample_df.iterrows():
        img_path = Path(row["path"])
        if not img_path.is_file():
            raise FileNotFoundError(
                f"Test image not found at {img_path}. Corruption benchmark requires real readable images."
            )
        with Image.open(img_path) as img:
            clean_images.append(img.convert("RGB"))
        labels.append(q_map[row["quality_canonical"]])

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
                "quadratic_weighted_kappa": m["quadratic_weighted_kappa"],
            }

    with open(out_p / "corruptions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"Evaluated {len(corruptions)} corruption types across 5 severities. Saved to: {out_p / 'corruptions.json'}"
    )


if __name__ == "__main__":
    main()
