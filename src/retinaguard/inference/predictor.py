"""Production ONNX Runtime & PyTorch Predictor matching Phase 17 and Phase 24."""

import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from scipy.special import softmax

from src.retinaguard.data.preprocessing import preprocess_image_canonical
from src.retinaguard.evaluation.ood import (
    RetinalModalityValidator,
    compute_energy_score,
)

from .decision_policy import DecisionPolicyEngine
from .schemas import PredictionResponse


def compute_entropy(probs: np.ndarray, base: float = 2.0) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * (np.log(p) / np.log(base))))


class RetinaGuardPredictor:
    """End-to-end CPU inference engine for single fundus image quality assurance."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = "artifacts/models/model.onnx",
        preprocessing_config_path: Optional[
            Union[str, Path]
        ] = "artifacts/models/preprocessing.json",
        calibration_config_path: Optional[
            Union[str, Path]
        ] = "artifacts/models/calibration_metadata.json",
        policy_engine: Optional[DecisionPolicyEngine] = None,
        image_size: int = 384,
    ):
        self.image_size = image_size
        self.temperature = 1.0
        self.policy_engine = policy_engine or DecisionPolicyEngine()
        self.ort_session = None
        self.pt_model = None

        # Load authoritative preprocessing config if present
        if preprocessing_config_path:
            p_path = Path(preprocessing_config_path)
            if p_path.is_file():
                with open(p_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                img_sz = cfg.get("image_size", self.image_size)
                if isinstance(img_sz, int) and img_sz > 0:
                    self.image_size = img_sz
                elif (
                    isinstance(img_sz, (list, tuple))
                    and len(img_sz) == 2
                    and all(isinstance(x, int) and x > 0 for x in img_sz)
                ):
                    self.image_size = img_sz[0]
                else:
                    raise ValueError(
                        f"Invalid image_size in {preprocessing_config_path}: {img_sz}. Must be positive integer or [H, W] pair."
                    )

                if "normalization" in cfg:
                    norm = cfg["normalization"]
                    if (
                        "mean" in norm
                        and (not isinstance(norm["mean"], list) or len(norm["mean"]) != 3)
                    ) or (
                        "std" in norm
                        and (not isinstance(norm["std"], list) or len(norm["std"]) != 3)
                    ):
                        raise ValueError(
                            f"Invalid normalization parameters in {preprocessing_config_path}: must have 3-channel mean and std."
                        )

                if "class_order" in cfg:
                    if list(cfg["class_order"]) != ["good", "usable", "reject"]:
                        raise ValueError(
                            f"Invalid class_order in {preprocessing_config_path}: {cfg['class_order']}. Expected ['good', 'usable', 'reject']."
                        )

        # Load authoritative calibration temperature if present
        if calibration_config_path:
            cal_path = Path(calibration_config_path)
            if cal_path.is_file():
                with open(cal_path, "r", encoding="utf-8") as f:
                    cal_cfg = json.load(f)
                temp = float(cal_cfg.get("temperature", cal_cfg.get("optimal_temperature", 1.0)))
                if temp <= 0.0 or not np.isfinite(temp):
                    raise ValueError(
                        f"Invalid temperature in {calibration_config_path}: {temp}. Temperature must be > 0."
                    )
                self.temperature = temp

        if model_path is not None and Path(model_path).is_file():
            self.load_model(model_path)

    def load_model(self, model_path: Union[str, Path]):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if path.suffix == ".onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.ort_session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        else:
            from src.retinaguard.models.multitask import RetinaGuardMultiTaskModel

            model = RetinaGuardMultiTaskModel(pretrained=False)
            state = torch.load(str(path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            model.eval()
            self.pt_model = model

    def predict(self, image: Union[Image.Image, np.ndarray, str, Path]) -> PredictionResponse:
        t0 = time.perf_counter()

        if self.ort_session is None and self.pt_model is None:
            raise RuntimeError(
                "No model loaded in RetinaGuardPredictor. Inference requires a valid ONNX or PyTorch model."
            )

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # 1. Modality Validation Check
        modality_check = RetinalModalityValidator.validate(pil_img)
        is_valid_modality = bool(modality_check["is_fundus"])

        # 2. Canonical Preprocessing
        tensor = preprocess_image_canonical(pil_img, image_size=self.image_size)

        # 3. Model Inference
        attributes_raw = {"artifact": 0, "clarity": 0, "field_definition": 0}
        if self.ort_session is not None:
            inp_name = self.ort_session.get_inputs()[0].name
            ort_outs = self.ort_session.run(None, {inp_name: tensor.numpy()})
            # Outputs: [quality_logits, overall_quality, artifact, clarity, field_def, latent]
            q_logits = ort_outs[0][0][:3]
            if len(ort_outs) > 2:
                attributes_raw["artifact"] = int(np.argmax(ort_outs[2][0]))
            if len(ort_outs) > 3:
                attributes_raw["clarity"] = int(np.argmax(ort_outs[3][0]))
            if len(ort_outs) > 4:
                attributes_raw["field_definition"] = int(np.argmax(ort_outs[4][0]))
        else:
            with torch.no_grad():
                out = self.pt_model(tensor)
                q_logits = out["calibrated_quality_logits"][0].cpu().numpy()
                attributes_raw["artifact"] = int(torch.argmax(out["artifact_logits"][0]).item())
                attributes_raw["clarity"] = int(torch.argmax(out["clarity_logits"][0]).item())
                attributes_raw["field_definition"] = int(
                    torch.argmax(out["field_definition_logits"][0]).item()
                )

        # 4. Probabilities & Temperature Scaling
        scaled_logits = q_logits / max(self.temperature, 1e-4)
        probs_arr = softmax(scaled_logits, axis=-1)
        probs_dict = {
            "good": float(probs_arr[0]),
            "usable": float(probs_arr[1]),
            "reject": float(probs_arr[2]),
        }
        uncertainty = compute_entropy(probs_arr)
        ood_score = float(compute_energy_score(q_logits[None, :])[0])

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Evaluate Decision Policy
        return self.policy_engine.evaluate(
            probs=probs_dict,
            uncertainty=uncertainty,
            ood_score=ood_score,
            is_valid_modality=is_valid_modality,
            attributes_raw=attributes_raw,
            latency_ms=latency_ms,
        )
