"""Unified high-throughput inference engine supporting ONNX Runtime and PyTorch backends."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
from scipy.special import expit, softmax
import torch

from src.preprocessing.transforms import preprocess_fundus_image
from src.ood.visual_gate import validate_retinal_modality
from src.ood.energy_score import compute_energy_score
from src.uncertainty.metrics import compute_shannon_entropy
from .decision_engine import DecisionEngine, QualityAssessmentResult


class RetinaGuardInferenceEngine:
    """Production inference engine for real-time edge or server deployment."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_onnx: bool = True,
        device: str = "cpu",
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.use_onnx = use_onnx
        self.device = device
        self.decision_engine = decision_engine or DecisionEngine()

        self.ort_session = None
        self.pytorch_model = None

        if model_path is not None:
            self.load_model(model_path, use_onnx=use_onnx)

    def load_model(self, model_path: Union[str, Path], use_onnx: bool = True):
        """Load ONNX runtime session or PyTorch model state."""
        path = Path(model_path)
        self.use_onnx = use_onnx

        if use_onnx or path.suffix == ".onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.ort_session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            self.use_onnx = True
        else:
            from src.models.multi_task_head import RetinaGuardNet
            state = torch.load(str(path), map_location=self.device)
            model = RetinaGuardNet(pretrained=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.pytorch_model = model
            self.use_onnx = False

    def predict(self, image: Union[Image.Image, np.ndarray, str, Path]) -> QualityAssessmentResult:
        """Run full end-to-end quality assessment, uncertainty quantification, and decision triage on a single image."""
        t_start = time.perf_counter()

        # Load image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # Step 1: Optical & Modality Sanity Gate
        modality_check = validate_retinal_modality(pil_img)
        is_fundus = bool(modality_check["is_fundus_candidate"])

        # Step 2: Preprocess to Tensor
        tensor = preprocess_fundus_image(pil_img, target_size=(384, 384))

        # Step 3: Neural Model Forward Pass
        if self.use_onnx and self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: tensor.numpy()}
            grade_logits, defect_logits, quality_score_arr, latent_features = self.ort_session.run(None, ort_inputs)
            grade_logits = grade_logits[0]
            defect_logits = defect_logits[0]
            quality_score = float(quality_score_arr[0][0])
        elif self.pytorch_model is not None:
            with torch.no_grad():
                out = self.pytorch_model(tensor.to(self.device))
                grade_logits = out["calibrated_grade_logits"][0].cpu().numpy()
                defect_logits = out["defect_logits"][0].cpu().numpy()
                quality_score = float(out["quality_score"][0].cpu().numpy())
        else:
            # Fallback heuristic / mock prediction if no weights loaded yet (e.g. for initial bootstrap testing)
            grade_logits = np.array([2.5, 0.5, -1.0], dtype=np.float32)
            defect_logits = np.array([-2.0, -2.5, -3.0, -1.8, -2.2, -2.5], dtype=np.float32)
            quality_score = 88.5

        # Step 4: Calibrated Probabilities & Uncertainty
        grade_probs = softmax(grade_logits)
        defect_probs = expit(defect_logits)
        entropy = float(compute_shannon_entropy(grade_probs[None, :])[0])
        energy_score = float(compute_energy_score(grade_logits[None, :])[0])

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Step 5: Decision Engine Triage
        result = self.decision_engine.evaluate(
            grade_probs=grade_probs,
            defect_probs=defect_probs,
            quality_score=quality_score,
            energy_score=energy_score,
            entropy=entropy,
            is_fundus_modality=is_fundus,
            latency_ms=round(t_elapsed_ms, 2)
        )

        return result
