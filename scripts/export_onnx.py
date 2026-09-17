"""Export trained model checkpoint to ONNX format with strict multi-head numerical parity verification."""

import argparse
import datetime
import json
import sys
from pathlib import Path

# Ensure local source package is prioritized over any site-packages install
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import onnx
import onnxruntime as ort
import torch

from retinaguard.data.preprocessing import export_preprocessing_metadata
from retinaguard.models.multitask import RetinaGuardMultiTaskModel
from retinaguard.utils.hashing import compute_sha256


class OnnxMultiTaskWrapper(torch.nn.Module):
    def __init__(self, model: RetinaGuardMultiTaskModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return (
            out["quality_logits"],
            out["overall_quality_logits"],
            out["artifact_logits"],
            out["clarity_logits"],
            out["field_definition_logits"],
            out["latent_features"],
        )


def main():
    parser = argparse.ArgumentParser(description="Export RetinaGuard checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained PyTorch checkpoint (.ckpt)"
    )
    parser.add_argument("--output-onnx", type=str, default="artifacts/models/model.onnx")
    parser.add_argument("--opset-version", type=int, default=18)
    parser.add_argument(
        "--primary-task",
        type=str,
        default=None,
        help="Primary task name (e.g. 'deepdrid_overall' or 'eyeq_quality')",
    )
    parser.add_argument(
        "--primary-output",
        type=str,
        default=None,
        help="Primary output head name (e.g. 'overall_quality_logits' or 'quality_logits')",
    )
    parser.add_argument(
        "--class-order",
        type=str,
        default=None,
        help="Comma-separated class names (e.g. 'good,poor_or_reject' or 'good,usable,reject')",
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=5e-4,
        help="Maximum allowed absolute numerical difference between PyTorch and ONNX Runtime",
    )
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(
            f"A trained checkpoint is required for ONNX export. Checkpoint not found: {args.checkpoint}"
        )

    out_p = Path(args.output_onnx)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ckpt_sha = compute_sha256(ckpt_p)

    try:
        ckpt_state = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_state = torch.load(ckpt_p, map_location="cpu")
    ckpt_meta = ckpt_state.get("metadata", {}) if isinstance(ckpt_state, dict) else {}
    img_size = ckpt_meta.get("resolved_config", {}).get("training", {}).get("image_size", 384)
    training_datasets = ckpt_meta.get("training_datasets", ["DeepDRiD"])

    # Determine trained heads strictly based on enabled datasets
    if "trained_heads" in ckpt_meta:
        trained_heads = list(ckpt_meta["trained_heads"])
    else:
        trained_heads = []
        if "EyeQ" in training_datasets:
            trained_heads.append("quality_logits")
        if "DeepDRiD" in training_datasets:
            trained_heads.extend(
                [
                    "overall_quality_logits",
                    "artifact_logits",
                    "clarity_logits",
                    "field_definition_logits",
                ]
            )
        if not trained_heads:
            trained_heads = ["overall_quality_logits"]

    # Determine primary head & task
    if args.primary_output:
        primary_output = args.primary_output
    elif "primary_head" in ckpt_meta:
        primary_output = ckpt_meta["primary_head"]
    elif "overall_quality_logits" in trained_heads and "quality_logits" not in trained_heads:
        primary_output = "overall_quality_logits"
    elif "quality_logits" in trained_heads:
        primary_output = "quality_logits"
    else:
        primary_output = trained_heads[0]

    if args.primary_task:
        primary_task = args.primary_task
    elif primary_output == "overall_quality_logits":
        primary_task = "deepdrid_overall"
    elif primary_output == "quality_logits":
        primary_task = "eyeq_quality"
    else:
        primary_task = "quality"

    if args.class_order:
        class_order = [c.strip() for c in args.class_order.split(",") if c.strip()]
    elif primary_output == "overall_quality_logits":
        class_order = ["good", "poor_or_reject"]
    elif primary_output == "quality_logits":
        class_order = ["good", "usable", "reject"]
    else:
        class_order = ["class_0", "class_1"]

    print(
        f"=== Exporting Trained RetinaGuard Checkpoint ({ckpt_p}) to ONNX ({out_p}) [Size: {img_size}x{img_size}] ==="
    )
    print(
        f"Configured Primary Head: '{primary_output}' (Task: '{primary_task}', Classes: {class_order})"
    )
    print(f"Trained Heads: {trained_heads}")
    model = RetinaGuardMultiTaskModel.from_checkpoint_metadata(ckpt_p)
    model.eval()

    wrapper = OnnxMultiTaskWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)
    input_names = ["input_image"]
    output_names = [
        "quality_logits",
        "overall_quality_logits",
        "artifact_logits",
        "clarity_logits",
        "field_definition_logits",
        "latent_features",
    ]

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_image": {0: "batch_size"}},
    )

    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph successfully verified by onnx.checker!")

    # Numerical Parity Check Across All Heads (Item 14)
    with torch.no_grad():
        pt_outs = wrapper(dummy_input)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(out_p), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    ort_outs = session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})

    per_head_errors = {}
    overall_max_error = 0.0

    for i, name in enumerate(output_names):
        pt_arr = pt_outs[i].cpu().numpy()
        ort_arr = ort_outs[i]
        abs_diff = np.abs(pt_arr - ort_arr)
        max_err = float(np.max(abs_diff))
        mean_err = float(np.mean(abs_diff))
        per_head_errors[name] = {
            "max_absolute_error": max_err,
            "mean_absolute_error": mean_err,
            "shape": list(ort_arr.shape),
            "parity_passed": max_err <= args.parity_tolerance,
        }
        overall_max_error = max(overall_max_error, max_err)
        print(
            f"Head '{name:25s}': Max Error = {max_err:.6e}, Mean Error = {mean_err:.6e} "
            f"[{'PASS' if max_err <= args.parity_tolerance else 'FAIL'}]"
        )

    onnx_sha = compute_sha256(out_p)

    # Export sidecar manifest (Item 11 & Item 16)
    sidecar_manifest = {
        "schema_version": "1.0.0",
        "status": "completed",
        "source_checkpoint_path": str(ckpt_p),
        "source_checkpoint_sha256": ckpt_sha,
        "onnx_model_sha256": onnx_sha,
        "image_size": [img_size, img_size],
        "primary_task": primary_task,
        "primary_output": primary_output,
        "num_classes": len(class_order),
        "class_order": class_order,
        "trained_heads": trained_heads,
        "training_datasets": training_datasets,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    sidecar_path = out_p.parent / "onnx_manifest.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_manifest, f, indent=2)
    print(f"Exported ONNX manifest sidecar to {sidecar_path}")

    parity_report = {
        "status": "completed_parity",
        "eligible_as_deployment_evidence": True,
        "eligible_as_predictive_performance_result": False,
        "overall_max_error": overall_max_error,
        "parity_tolerance": args.parity_tolerance,
        "parity_passed": bool(overall_max_error <= args.parity_tolerance),
        "heads": per_head_errors,
        "opset_version": args.opset_version,
        "checkpoint_path": str(ckpt_p),
        "checkpoint_sha256": ckpt_sha,
        "onnx_model_path": str(out_p),
        "onnx_model_sha256": onnx_sha,
        "software_versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    parity_path = Path("artifacts/metrics/onnx_parity.json")
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parity_path, "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2)
    print(f"Exported parity report to {parity_path}")

    if overall_max_error > args.parity_tolerance:
        raise RuntimeError(
            f"ONNX parity check FAILED! Overall maximum absolute error ({overall_max_error:.6e}) "
            f"exceeds tolerance ({args.parity_tolerance:.6e})."
        )

    # Export canonical preprocessing metadata JSON bound to checkpoint (Item 15)
    export_preprocessing_metadata(
        out_p.parent / "preprocessing.json",
        image_size=img_size,
        source_checkpoint_sha256=ckpt_sha,
        status="completed",
    )
    print("Exported preprocessing metadata to artifacts/models/preprocessing.json")


if __name__ == "__main__":
    main()
