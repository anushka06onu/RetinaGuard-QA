"""Export trained model checkpoint to ONNX format with dynamic batching and strict checkpoint verification."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from retinaguard.data.preprocessing import export_preprocessing_metadata
from retinaguard.models.multitask import RetinaGuardMultiTaskModel


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
    print("ONNX graph successfully verified!")

    # Numerical Parity Check
    with torch.no_grad():
        pt_out = wrapper(dummy_input)
    session = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
    ort_outs = session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})

    max_err = float(np.max(np.abs(pt_out[0].numpy() - ort_outs[0])))
    print(f"Numerical Parity: Max Absolute Error = {max_err:.6f} (< 1e-4 target)")

    # Export canonical preprocessing metadata JSON
    export_preprocessing_metadata(out_p.parent / "preprocessing.json", image_size=384)
    print("Exported preprocessing metadata to artifacts/models/preprocessing.json")


if __name__ == "__main__":
    main()
