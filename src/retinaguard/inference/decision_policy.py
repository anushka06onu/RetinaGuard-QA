"""Actionable decision policy and feedback formulation matching Phase 12 and 15."""

from typing import Dict, List, Optional
from .schemas import DecisionAction, QualityProbabilities, QualityAttributes, PredictionResponse


class DecisionPolicyEngine:
    """Triage decision engine enforcing uncertainty bounds, OOD gating, and actionable capture advice."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.85,  # bits
        ood_energy_threshold: float = 1.0,
        model_version: str = "1.0.0"
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.ood_energy_threshold = ood_energy_threshold
        self.model_version = model_version

    def formulate_feedback(
        self,
        decision: DecisionAction,
        quality: str,
        clarity_val: Optional[int],
        artifact_val: Optional[int],
        field_def_val: Optional[int]
    ) -> List[str]:
        feedback = []
        if decision == DecisionAction.UNSUPPORTED_INPUT:
            return ["Invalid image format or non-fundus modality. Please submit a valid color retinal fundus image."]

        if decision == DecisionAction.MANUAL_REVIEW:
            feedback.append("High model uncertainty or borderline acquisition quality. Request qualified clinician review.")

        if clarity_val is not None and clarity_val >= 2:
            feedback.append("Possible blur or focus defect detected. Stabilize and refocus camera before recapture.")

        if artifact_val is not None and artifact_val >= 2:
            feedback.append("Possible acquisition artifact detected. Inspect the lens, alignment, and field before recapture.")

        if field_def_val is not None and field_def_val >= 2:
            feedback.append("Possible field centering/definition defect. Adjust patient fixation before recapture.")

        if quality == "reject" and not feedback:
            feedback.append("Overall technical acquisition quality is inadequate. Recapture with proper optical alignment.")

        if not feedback:
            feedback.append("The model classified this image as technically acceptable under its experimental quality-assessment protocol. Clinical suitability still requires qualified review.")

        return feedback

    def evaluate(
        self,
        probs: Dict[str, float],
        uncertainty: float,
        ood_score: float,
        is_valid_modality: bool = True,
        attributes_raw: Optional[Dict[str, int]] = None,
        latency_ms: Optional[float] = None
    ) -> PredictionResponse:
        pred_class = max(probs, key=probs.get)
        cal_conf = probs[pred_class]

        attr_decodings = {
            "artifact": {0: "none", 1: "mild", 2: "severe"}.get(attributes_raw.get("artifact", 0) if attributes_raw else 0, None),
            "clarity": {0: "high", 1: "moderate", 2: "low"}.get(attributes_raw.get("clarity", 0) if attributes_raw else 0, None),
            "field_definition": {0: "adequate", 1: "incomplete", 2: "poor"}.get(attributes_raw.get("field_definition", 0) if attributes_raw else 0, None)
        }

        # Blueprint decision hierarchy
        if not is_valid_modality:
            decision = DecisionAction.UNSUPPORTED_INPUT
        elif ood_score < self.ood_energy_threshold:
            decision = DecisionAction.MANUAL_REVIEW
        elif uncertainty > self.uncertainty_threshold:
            decision = DecisionAction.MANUAL_REVIEW
        elif pred_class == "reject":
            decision = DecisionAction.RECAPTURE
        elif pred_class == "usable":
            decision = DecisionAction.MANUAL_REVIEW
        else:
            decision = DecisionAction.ACCEPT

        feedback = self.formulate_feedback(
            decision,
            pred_class,
            attributes_raw.get("clarity") if attributes_raw else None,
            attributes_raw.get("artifact") if attributes_raw else None,
            attributes_raw.get("field_definition") if attributes_raw else None
        )

        return PredictionResponse(
            model_version=self.model_version,
            quality=pred_class,
            probabilities=QualityProbabilities(
                good=round(probs.get("good", 0.0), 4),
                usable=round(probs.get("usable", 0.0), 4),
                reject=round(probs.get("reject", 0.0), 4)
            ),
            calibrated_confidence=round(cal_conf, 4),
            uncertainty=round(uncertainty, 4),
            ood_score=round(ood_score, 4),
            decision=decision,
            quality_attributes=QualityAttributes(**attr_decodings),
            feedback=feedback,
            latency_ms=round(latency_ms, 2) if latency_ms is not None else None
        )
