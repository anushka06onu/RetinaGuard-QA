"""Export trained model checkpoint to ONNX format with strict multi-head numerical parity verification."""

import argparse
import datetime
import json
from pathlib import Path

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
        "--parity-tolerance",
        type=float,
        default=1e-4,
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

    print(f"=== Exporting Trained RetinaGuard Checkpoint ({ckpt_p}) to ONNX ({out_p}) ===")
    model = RetinaGuardMultiTaskModel(pretrained=False)
    state = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    model.eval()

    wrapper = OnnxMultiTaskWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 384, 384)
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

    ckpt_sha = compute_sha256(ckpt_p)
    onnx_sha = compute_sha256(out_p)

    parity_report = {
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

    # Export canonical preprocessing metadata JSON
    export_preprocessing_metadata(out_p.parent / "preprocessing.json", image_size=384)
    print("Exported preprocessing metadata to artifacts/models/preprocessing.json")


if __name__ == "__main__":
    main()
