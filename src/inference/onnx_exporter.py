"""Export PyTorch RetinaGuardNet models to optimized ONNX format and verify numerical parity."""

from pathlib import Path
from typing import Tuple, Union, Dict
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.models.multi_task_head import RetinaGuardNet


class OnnxExportableWrapper(torch.nn.Module):
    """Wrapper that formats multi-task outputs into explicit tensor tuples for ONNX export."""

    def __init__(self, model: RetinaGuardNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        # Returns: (calibrated_grade_logits, defect_logits, quality_score, latent_features)
        return (
            out["calibrated_grade_logits"],
            out["defect_logits"],
            out["quality_score"],
            out["latent_features"]
        )


def export_to_onnx(
    model: RetinaGuardNet,
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 384, 384),
    opset_version: int = 18
) -> Path:
    """Export RetinaGuardNet to ONNX format with dynamic batching."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = OnnxExportableWrapper(model)
    wrapper.eval()
    dummy_input = torch.randn(*input_shape)

    input_names = ["input_image"]
    output_names = ["grade_logits", "defect_logits", "quality_score", "latent_features"]
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "grade_logits": {0: "batch_size"},
        "defect_logits": {0: "batch_size"},
        "quality_score": {0: "batch_size"},
        "latent_features": {0: "batch_size"}
    }

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # Validate exported ONNX graph
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)

    return out_p


def verify_onnx_numerical_parity(
    model: RetinaGuardNet,
    onnx_path: Union[str, Path],
    rtol: float = 1e-3,
    atol: float = 1e-4
) -> Dict[str, Union[bool, float]]:
    """Test maximum absolute discrepancy between PyTorch and ONNX Runtime outputs."""
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        pt_out = model(dummy_input)
        pt_grade = pt_out["calibrated_grade_logits"].cpu().numpy()
        pt_defect = pt_out["defect_logits"].cpu().numpy()
        pt_score = pt_out["quality_score"].cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)

    diff_grade = float(np.max(np.abs(pt_grade - ort_outs[0])))
    diff_defect = float(np.max(np.abs(pt_defect - ort_outs[1])))
    diff_score = float(np.max(np.abs(pt_score - ort_outs[2])))

    max_diff = max(diff_grade, diff_defect, diff_score)
    is_valid = max_diff < atol

    return {
        "is_parity_verified": is_valid,
        "max_absolute_error": max_diff,
        "grade_error": diff_grade,
        "defect_error": diff_defect,
        "score_error": diff_score
    }
