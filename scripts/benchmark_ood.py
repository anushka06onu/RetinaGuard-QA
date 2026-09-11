"""Out-of-Distribution (OOD) and Modality Gate benchmark suite.

Evaluates Free Energy OOD scoring (AUROC, AUPRC, FPR@95%TPR) and Modality Validator.
Requires genuine In-Distribution (ID) fundus images. Fails fast if ID images are absent.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from retinaguard.data.preprocessing import preprocess_image_canonical
from retinaguard.evaluation.ood import (
    RetinalModalityValidator,
    compute_energy_score,
)
from retinaguard.models.multitask import RetinaGuardMultiTaskModel


def compute_fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate False Positive Rate at 95% True Positive Rate."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return float(fpr[idx[0]])
    return 1.0


def load_images_from_manifest(
    manifest_csv: Path, max_samples: int = 0
) -> List[Tuple[str, Image.Image]]:
    """Load genuine images from a split/manifest CSV."""
    if not manifest_csv.is_file():
        return []

    df = pd.read_csv(manifest_csv)
    records = []
    for _, row in df.iterrows():
        p_str = str(row.get("path", row.get("image_path", "")))
        img_p = Path(p_str)
        if img_p.is_file():
            try:
                with Image.open(img_p) as img:
                    img_id = str(row.get("image_id", img_p.stem))
                    records.append((img_id, img.convert("RGB")))
                if 0 < max_samples <= len(records):
                    break
            except Exception:
                continue
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OOD detection (AUROC, AUPRC, FPR@95%TPR) and modality gate."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--id-test-split",
        type=str,
        default="data/splits/deepdrid_external_test.csv",
        help="Path to In-Distribution (ID) test split CSV",
    )
    parser.add_argument(
        "--near-ood-manifest",
        type=str,
        default=None,
        help="Path to Near-OOD manifest CSV (e.g. Ultra-Widefield fundus)",
    )
    parser.add_argument(
        "--far-ood-manifest",
        type=str,
        default=None,
        help="Path to Far-OOD manifest CSV (e.g. natural images/non-retinal)",
    )
    parser.add_argument(
        "--include-synthetic-stress-test",
        action="store_true",
        default=False,
        help="Include synthetic corruption stress patterns (clearly labeled as synthetic)",
    )
    parser.add_argument(
        "--calibration-meta",
        type=str,
        default="artifacts/models/calibration_metadata.json",
        help="Path to calibration metadata containing validation-fitted OOD threshold",
    )
    parser.add_argument("--output-file", type=str, default="artifacts/metrics/ood.json")
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print("=== Running OOD Detection & Modality Gate Benchmark ===")
    model = RetinaGuardMultiTaskModel(pretrained=False)
    state = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    # 1. Load In-Distribution (ID) fundus images
    id_records = load_images_from_manifest(Path(args.id_test_split), max_samples=args.max_samples)
    if len(id_records) == 0:
        raise FileNotFoundError(
            f"No readable In-Distribution images found in {args.id_test_split}. "
            "A valid ID dataset is strictly required."
        )

    print(f"Loaded {len(id_records)} genuine In-Distribution (ID) images.")

    # Compute ID energy scores
    id_scores = []
    id_modality_passes = 0

    with torch.no_grad():
        for _, img in id_records:
            modality_res = RetinalModalityValidator.validate(img)
            if modality_res.get("is_fundus", False):
                id_modality_passes += 1

            tensor = preprocess_image_canonical(img, image_size=384)
            outputs = model(tensor)
            logits = outputs["quality_logits"].numpy()[0]
            # Higher energy score -> more in-distribution
            e_score = compute_energy_score(logits, temperature=1.0)
            id_scores.append(float(e_score))

    id_scores_arr = np.array(id_scores)
    id_modality_frr = 1.0 - (id_modality_passes / len(id_records))

    benchmarks = {
        "in_distribution": {
            "dataset_split": args.id_test_split,
            "num_samples": len(id_records),
            "mean_energy_score": float(np.mean(id_scores_arr)),
            "std_energy_score": float(np.std(id_scores_arr)),
            "modality_false_reject_rate": float(id_modality_frr),
        }
    }

    # 2. Near-OOD evaluation if manifest provided
    if args.near_ood_manifest and Path(args.near_ood_manifest).is_file():
        near_records = load_images_from_manifest(
            Path(args.near_ood_manifest), max_samples=args.max_samples
        )
        if near_records:
            near_scores = []
            with torch.no_grad():
                for _, img in near_records:
                    tensor = preprocess_image_canonical(img, image_size=384)
                    outputs = model(tensor)
                    logits = outputs["quality_logits"].numpy()[0]
                    near_scores.append(float(compute_energy_score(logits, temperature=1.0)))

            near_scores_arr = np.array(near_scores)
            y_true = np.concatenate([np.ones(len(id_scores_arr)), np.zeros(len(near_scores_arr))])
            y_scores = np.concatenate([id_scores_arr, near_scores_arr])

            benchmarks["near_ood_manifest"] = {
                "manifest": args.near_ood_manifest,
                "num_samples": len(near_records),
                "auroc": float(roc_auc_score(y_true, y_scores)),
                "fpr_at_95_tpr": float(compute_fpr_at_95_tpr(y_true, y_scores)),
            }

    # 3. Far-OOD evaluation if manifest provided
    if args.far_ood_manifest and Path(args.far_ood_manifest).is_file():
        far_records = load_images_from_manifest(
            Path(args.far_ood_manifest), max_samples=args.max_samples
        )
        if far_records:
            far_scores = []
            with torch.no_grad():
                for _, img in far_records:
                    tensor = preprocess_image_canonical(img, image_size=384)
                    outputs = model(tensor)
                    logits = outputs["quality_logits"].numpy()[0]
                    far_scores.append(float(compute_energy_score(logits, temperature=1.0)))

            far_scores_arr = np.array(far_scores)
            y_true = np.concatenate([np.ones(len(id_scores_arr)), np.zeros(len(far_scores_arr))])
            y_scores = np.concatenate([id_scores_arr, far_scores_arr])

            benchmarks["far_ood_manifest"] = {
                "manifest": args.far_ood_manifest,
                "num_samples": len(far_records),
                "auroc": float(roc_auc_score(y_true, y_scores)),
                "fpr_at_95_tpr": float(compute_fpr_at_95_tpr(y_true, y_scores)),
            }

    # 4. Controlled Synthetic Stress Test (Labeled explicitly as synthetic)
    if args.include_synthetic_stress_test:
        rng = np.random.RandomState(2026)
        synthetic_scores = []
        for _ in range(len(id_records)):
            arr = rng.randint(0, 256, (384, 384, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            tensor = preprocess_image_canonical(img, image_size=384)
            with torch.no_grad():
                outputs = model(tensor)
                logits = outputs["quality_logits"].numpy()[0]
                synthetic_scores.append(float(compute_energy_score(logits, temperature=1.0)))

        syn_arr = np.array(synthetic_scores)
        y_true = np.concatenate([np.ones(len(id_scores_arr)), np.zeros(len(syn_arr))])
        y_scores = np.concatenate([id_scores_arr, syn_arr])

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        benchmarks["synthetic_noise_stress_test"] = {
            "type": "synthetic_uniform_noise",
            "num_samples": len(syn_arr),
            "auroc": float(roc_auc_score(y_true, y_scores)),
            "auprc": float(auc(recall, precision)),
            "fpr_at_95_tpr": float(compute_fpr_at_95_tpr(y_true, y_scores)),
        }

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"Saved OOD benchmark results to {out_file}")


if __name__ == "__main__":
    main()
